"""The Forum — a BettaFish-style moderated debate that produces the Pressure Brief.

Unlike the persona debate in `engine/debate.py`, the Forum's agents are
distinguished by *capability channel* (social sentiment / news / market), and a
**moderator carries the benchmark's verdict into the room** — for an efficient
sector it steers the discussion to the present read and away from a confident
forecast. The output is a `PressureBrief`: the hero of a Monitor run.

The roster is ~50 agents (see `roster_size`), which forces three scale decisions
documented at their definitions: prior context is capped (`_PRIOR_TURNS`), the
judge reads only the extremes verbatim (`_JUDGE_VERBATIM`), and round 2 invites
only the divergent few (`_REBUTTAL_AGENTS`) rather than everyone again.

Offline-first: every LLM call goes through `engine.llm`, RAG through the injected
RagEngine, and the social layer through the frozen snapshot.

Testing: patch `llm.complete` for headless runs. A run **with an event sink**
streams instead, so patch `llm.stream` too — `tests/test_forum._patch_llm` does
both, because patching only `complete` lets a streamed test make real calls.
"""
from __future__ import annotations

from typing import Callable, Optional

from ph_economic_ai.engine import llm
from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.auto_assemble import (
    SECTOR_SOURCES, SectorContext, auto_assemble)
from ph_economic_ai.engine.debate import (
    Agent, AgentResponse, _extract_electricity_change, _extract_percent,
    _extract_price, _parse_think)
from ph_economic_ai.engine.pressure_brief import PressureBrief, SectorReading

# Per-sector estimate parsing, agreement band, and the "flat" threshold.
_EXTRACTORS: dict[str, Callable] = {
    'gas': _extract_price, 'food': _extract_percent,
    'electricity': _extract_electricity_change,
}
# How many prior turns an agent is shown, and how many statements the judge reads
# verbatim. Both exist because the roster is ~50: unbounded context would overflow
# the local models and slow every successive call.
_PRIOR_TURNS = 6
_JUDGE_VERBATIM = 8
#: Agents invited to rebut the moderator in round 2 (the most divergent).
_REBUTTAL_AGENTS = 4

_BAND = {'gas': 0.20, 'food': 0.3, 'electricity': 0.10}   # "agree" if within this of the mean
_FLAT = {'gas': 0.05, 'food': 0.05, 'electricity': 0.02}  # |estimate| below this reads as flat
# Magnitude-guard band per sector — how far a consensus may sit from the anchor
# before it is more likely a weak-model error than a real signal (engine.anchoring).
_TOLERANCE = {
    'gas': anchoring.FUEL_TOLERANCE_PHP_L,
    'food': anchoring.FOOD_TOLERANCE_PCT,
    'electricity': anchoring.ELECTRICITY_TOLERANCE_PHP_KWH,
}
_EST_LINE = {
    'gas': 'ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L',
    'food': 'ESTIMATE: +X.X% or ESTIMATE: -X.X%',
    'electricity': 'ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh',
}

# Capability channels: each agent stays strictly in its lane so the three do NOT
# converge on the same paragraph — the point of a forum over a single model.
_CHANNEL_TEMPLATES = {
    'social': ('You speak ONLY to public attention and mood — what Filipinos are '
               'searching for and posting about {sector} prices RIGHT NOW (search-interest '
               'spikes, public complaints). Do NOT analyse markets or policy; that is other '
               'agents\' job. If the retrieved context shows no fresh social signal, say so '
               'plainly and keep your read tentative rather than inventing a mood.'),
    'news': ('You speak ONLY to reported events — announcements, rate changes, and official '
             'actions in the news about Philippine {sector} prices RIGHT NOW. Do NOT restate '
             'social mood or raw market prices; name the concrete event and its source.'),
    'market': ('You speak ONLY to the underlying market — the oil, FX, and spot-price moves '
               'driving {sector} cost RIGHT NOW. Do NOT restate news or sentiment; give the '
               'mechanism and the numbers.'),
}

# The named cast — (name, occupation, vantage) per sector x channel.
#
# The channel fixes WHICH EVIDENCE an agent may speak to (social / news / market);
# the vantage fixes WHOSE ANGLE it speaks from. Both matter at this roster size: 17
# agents sharing one channel prompt would return 17 near-identical paragraphs, which
# is a more expensive single model rather than a forum. The vantage is appended to
# the channel prompt so each agent has a concrete, different thing to look for.
_PERSONAS: dict[str, dict[str, list[tuple[str, str, str]]]] = {
    'gas': {
        'social': [
            ('Andrea Lim', 'Commuter Sentiment Analyst', 'daily commuters and fare complaints'),
            ('Joel Bautista', 'Jeepney Operator Rep', 'jeepney and PUV operators\' boundary costs'),
            ('Kim Salvador', 'Delivery Rider Organiser', 'motorcycle couriers and rider earnings'),
            ('Tessa Bagunu', 'Provincial Motorist Voice', 'motorists outside Metro Manila, where pumps differ'),
            ('Ryan Delos Reyes', 'Ride-Hail Driver Advocate', 'TNVS drivers and per-trip fuel burn'),
            ('Marilou Ibarra', 'Coastal Fuel Watcher', 'fisherfolk buying diesel for boats'),
        ],
        'news': [
            ('Rafael Cruz', 'Energy Desk Reporter', 'weekly oil-price adjustment announcements'),
            ('Nina Alcantara', 'DOE Beat Reporter', 'Department of Energy bulletins and advisories'),
            ('Ben Ocampo', 'Transport Sector Correspondent', 'fare petitions and subsidy releases'),
            ('Grace Manalo', 'Business Wire Editor', 'oil-company pricing statements'),
            ('Dodong Rivera', 'Regional Stringer', 'provincial pump-price reports'),
        ],
        'market': [
            ('Diego Ocampo', 'Crude & FX Trader', 'Brent and the peso'),
            ('Sofia Lazaro', 'Refined Product Analyst', 'MOPS gasoline and diesel cracks'),
            ('Ariel Tan', 'Downstream Margin Analyst', 'retailer margins and pass-through lag'),
            ('Hector Villamor', 'Freight & Shipping Analyst', 'tanker and freight cost into landed price'),
            ('Cielo Ramos', 'Retail Pricing Analyst', 'station-level price dispersion'),
            ('Ferdie Yap', 'Excise & Tax Analyst', 'excise, VAT and the fixed tax wedge'),
        ],
    },
    'food': {
        'social': [
            ('Bea Villanueva', 'Palengke Sentiment Analyst', 'wet-market shoppers and stall talk'),
            ('Lito Enriquez', 'Sari-Sari Store Owner', 'neighbourhood retail prices and repacking'),
            ('Divina Castro', 'Household Budget Voice', 'families substituting cheaper staples'),
            ('Aris Panganiban', 'Urban Poor Advocate', 'rice affordability at the bottom decile'),
            ('Chef Ramon Sy', 'Restaurant Buyer', 'bulk buyers seeing supplier quotes move'),
            ('Nanay Puring', 'Farmer Sentiment Reader', 'growers\' farmgate complaints'),
        ],
        'news': [
            ('Marco Reyes', 'Agriculture Correspondent', 'DA and NFA announcements'),
            ('Ivy Trinidad', 'Food Policy Reporter', 'import permits, tariffs and price caps'),
            ('Ramil Bacani', 'Weather Desk Reporter', 'PAGASA typhoon and drought impact on supply'),
            ('Karen Uy', 'Trade Beat Reporter', 'rice and vegetable import announcements'),
            ('Boy Fernandez', 'Regional Agri Stringer', 'provincial harvest and shipment reports'),
        ],
        'market': [
            ('Nadia Chua', 'Agri-Commodities Analyst', 'global rice, wheat, corn and soy'),
            ('Paolo Lim', 'Rice Import Analyst', 'Vietnamese and Thai landed rice cost'),
            ('Rowena Dizon', 'Vegetable Supply Analyst', 'Benguet and Mindanao supply volumes'),
            ('Jun Escalante', 'Protein Market Analyst', 'pork, chicken and fish price channels'),
            ('Miles Abad', 'Agri Logistics Analyst', 'trucking and cold-chain cost'),
            ('Trina Gomez', 'Farm Input Analyst', 'fertiliser and feed cost feeding into farmgate'),
        ],
    },
    'electricity': {
        'social': [
            ('Paolo Mendoza', 'Household Bill Watcher', 'residential bill shock complaints'),
            ('Erlinda Vega', 'Small Business Owner', 'sari-sari and carinderia power costs'),
            ('Migs Fontanilla', 'Condo Resident Rep', 'metered urban tenants'),
            ('Lolo Berting', 'Electric Coop Member', 'provincial co-op consumers outside Meralco'),
            ('Angel Ferrer', 'Industrial User Rep', 'factories on large-load tariffs'),
        ],
        'news': [
            ('Ligaya Torres', 'Utilities Correspondent', 'Meralco rate announcements'),
            ('Dennis Aquino', 'ERC Beat Reporter', 'Energy Regulatory Commission rulings'),
            ('Pia Mercado', 'Power Policy Reporter', 'subsidy and lifeline-rate policy'),
            ('Onin Guzman', 'Grid & Outage Reporter', 'NGCP alerts and yellow/red alerts'),
            ('Sam Delfin', 'Energy Business Reporter', 'generator and IPP statements'),
        ],
        'market': [
            ('Enzo Garcia', 'Power Market Analyst', 'WESM spot prices'),
            ('Belle Navarro', 'Natural Gas Analyst', 'Malampaya depletion and LNG import cost'),
            ('Tonio Sarmiento', 'Coal Import Analyst', 'Indonesian coal and the FX on it'),
            ('Rica Peralta', 'Generation Mix Analyst', 'the fuel mix behind the generation charge'),
            ('Gil Antonio', 'Transmission Cost Analyst', 'NGCP wheeling and ancillary charges'),
            ('Faye Rosales', 'FX Pass-Through Analyst', 'the peso on dollar-denominated fuel'),
        ],
    },
}

_MODERATOR_SYSTEM = (
    'You are the forum moderator for a Philippine price-pressure debate. You do not '
    'estimate prices yourself. You summarise the round, name the main disagreement, '
    'and enforce the benchmark note: keep the agents on the PRESENT read and stop any '
    'agent that drifts into a confident forward forecast. One short paragraph.'
)
_JUDGE_SYSTEM = (
    'You are the forum judge. You have just heard specialist analysts — a social-mood '
    'reader, a news reader, and a market reader — debate the CURRENT pressure on one '
    'Philippine price. Do NOT introduce new facts. Weigh what they said, trusting '
    'concrete cited data over vibes, resolve their disagreement into a SINGLE present '
    'read, and stay honest to the benchmark note (a present read, never a confident '
    'forecast). One or two sentences, then the estimate line.'
)
_SYNTH_SYSTEM = (
    'You are a Philippine macro analyst writing the present-pressure summary a '
    'household would read. Write 2-3 present-tense sentences on current pressure '
    'across the sectors. This is a nowcast — describe now, do not forecast.'
)


def _capability_agents(sector: str, per_channel: Optional[int] = None) -> list[Agent]:
    """The sector's cast, interleaved by channel.

    `per_channel` caps how many personas each channel contributes — the knob for
    trading debate breadth against wall-clock time on a single local GPU. None
    means the full roster.

    Agents are interleaved (social, news, market, social, …) rather than grouped,
    so a run that is watched live shows the three evidence lanes from the first
    few cards instead of six consecutive social reads.
    """
    srcs = SECTOR_SOURCES.get(sector, {})
    personas = _PERSONAS.get(sector, {})
    by_channel: dict[str, list[Agent]] = {}
    for channel, tmpl in _CHANNEL_TEMPLATES.items():
        entries = personas.get(channel) or [(channel.title(), 'Analyst', 'the general picture')]
        if per_channel is not None:
            entries = entries[:max(1, per_channel)]
        built = []
        for name, occupation, vantage in entries:
            built.append(Agent(
                name=name, role=occupation,
                system_prompt=(f'You are {name}, a {occupation} in the Philippines. '
                               + tmpl.format(sector=sector)
                               + f' Your particular vantage point is {vantage}; speak from '
                                 'it concretely and do not restate what a colleague on the '
                                 'same beat would already have said. '
                               + 'End your response with a CAUSAL CHAIN line and then: '
                               + _EST_LINE[sector]),
                rag_sources=srcs.get(channel, []),
                tier=llm.FAST))               # all forum agents share the fast tier
        by_channel[channel] = built

    # round-robin interleave
    agents: list[Agent] = []
    for i in range(max((len(v) for v in by_channel.values()), default=0)):
        for channel in _CHANNEL_TEMPLATES:
            if i < len(by_channel.get(channel, [])):
                agents.append(by_channel[channel][i])
    return agents


def roster_size(sectors=('gas', 'food', 'electricity'),
                per_channel: Optional[int] = None) -> int:
    """Total agents that would debate — the honest call-budget input."""
    return sum(len(_capability_agents(s, per_channel)) for s in sectors)


def _latest_per_agent(history: list[AgentResponse]) -> list[AgentResponse]:
    """One response per agent — its highest round.

    With adaptive rebuttals the last round holds only the divergent few, so
    "the final round" is no longer the same set as "every agent's final word".
    Consensus, confidence and the judge all need the latter.
    """
    latest: dict[str, AgentResponse] = {}
    for r in history:
        cur = latest.get(r.agent_name)
        if cur is None or r.round_num >= cur.round_num:
            latest[r.agent_name] = r
    return list(latest.values())


def _direction(sector: str, value: Optional[float]) -> str:
    if value is None:
        return 'unknown'
    if abs(value) < _FLAT.get(sector, 0.05):
        return 'flat'
    return 'rising' if value > 0 else 'easing'


def _driver_text(statement: str) -> Optional[str]:
    """The causal-chain line only — without the trailing ESTIMATE line."""
    if 'CAUSAL CHAIN:' not in statement:
        return None
    part = statement.split('CAUSAL CHAIN:')[-1].split('ESTIMATE:')[0]
    return part.strip()[:160] or None


class Forum:
    """Runs the moderated present-pressure debate for each sector."""

    def __init__(self, rag, contexts: list[SectorContext], as_of: str, window: str,
                 rounds: int = 2, deep_tier: str = llm.DEEP,
                 per_channel: Optional[int] = None,
                 rebuttal_agents: int = _REBUTTAL_AGENTS):
        self._rag = rag
        self._contexts = contexts
        self._as_of = as_of
        self._window = window
        self._rounds = max(1, rounds)
        self._deep = deep_tier
        self._per_channel = per_channel          # None = full ~50-agent roster
        self._rebuttal_agents = max(0, rebuttal_agents)
        self._on_event = None

    # ── prompts ───────────────────────────────────────────────────────────────

    def _rag_text(self, agent: Agent, query: str) -> tuple[str, list[str]]:
        """Retrieved context AND the sources it actually came from.

        Returning the *used* sources (rather than the agent's configured wishlist)
        is what keeps the citation honest: a channel whose feeds returned nothing
        must not appear on the card or in the debate map as though it was read.
        """
        try:
            chunks = self._rag.query(query, top_k=4, sources=agent.rag_sources)
        except Exception:
            chunks = []
        used = sorted({c['source'] for c in chunks if c.get('source')})
        text = '\n'.join(f"[{c['source']}] {c['text'][:280]}" for c in chunks) \
            or 'No context retrieved for your channel.'
        return text, used

    def _agent_prompt(self, agent: Agent, ctx: SectorContext,
                      history: list[AgentResponse], steer: str
                      ) -> tuple[list[dict], list[str]]:
        query = f"Current {ctx.sector} price pressure in the Philippines, {self._window}."
        # Only the most recent turns. Passing the whole history does not scale: with a
        # 50-agent roster the last speaker would receive ~3,400 tokens of prior
        # statements, which overflows a 3b context window and makes every successive
        # call slower than the one before it. A forum participant attends to the last
        # few speakers anyway.
        recent = history[-_PRIOR_TURNS:]
        prior = '\n'.join(f"{r.agent_name}: {r.statement[:280]}" for r in recent)
        if len(history) > len(recent):
            prior = f"({len(history) - len(recent)} earlier turns omitted)\n" + prior
        sc = ctx.social_counts or {}
        social_note = (f"Social posts mentioning {ctx.sector} in the snapshot — today "
                       f"{sc.get('today', 0)}, this week {sc.get('this_week', 0)}, "
                       f"this month {sc.get('this_month', 0)}.\n\n")
        rag_text, used = self._rag_text(agent, query)
        challenge = ('Do NOT repeat what earlier agents already said. Add only what YOUR '
                     'channel sees that they missed, or disagree and say why.\n') if prior else ''
        user = (
            f"BENCHMARK NOTE: {ctx.verdict_note}\n\n"
            f"As of {self._as_of} ({self._window}). Sector: {ctx.sector} "
            f"(report in {ctx.unit}).\n\n"
            f"{social_note}"
            f"Retrieved context:\n{rag_text}\n\n"
            + (f"Moderator steer: {steer}\n\n" if steer else '')
            + (f"Prior statements:\n{prior}\n\n" if prior else '')
            + challenge
            + "Give a short present-tense read from your channel only. End with:\n"
            "CAUSAL CHAIN: <trigger> → <effect> → <household impact>\n"
            + _EST_LINE[ctx.sector]
        )
        return ([{'role': 'system', 'content': agent.system_prompt},
                 {'role': 'user', 'content': user}], used)

    # ── loop ──────────────────────────────────────────────────────────────────

    def _moderate(self, ctx: SectorContext, recent: list[AgentResponse]) -> str:
        transcript = '\n'.join(
            f"{r.agent_name}: {r.statement[:220]} (est {r.price_estimate})"
            for r in recent)
        msgs = [
            {'role': 'system', 'content': _MODERATOR_SYSTEM},
            {'role': 'user', 'content': (
                f"Sector: {ctx.sector}. Benchmark note: {ctx.verdict_note}\n\n"
                f"Round statements:\n{transcript}\n\n"
                "Summarise the present pressure, name the disagreement, and give a "
                "one-line steer for the next round.")},
        ]
        try:
            return llm.complete(msgs, tier=self._deep, max_tokens=220).strip()
        except Exception:
            return ''

    def _judge_sector(self, ctx: SectorContext, finals: list[AgentResponse]):
        """Synthesise the debate into one present read (like the swarm's master
        judge) — resolving disagreement rather than averaging. Returns
        (estimate | None, verdict_text).

        `finals` is each agent's LATEST turn, so a rebuttal supersedes that agent's
        opening statement and every agent is represented exactly once.
        """
        # 50 full statements would overflow the deep tier's context, so the tail of a
        # large roster is compressed to name/estimate lines. The extremes are what a
        # judge resolving a disagreement needs in full, so those are kept verbatim.
        if len(finals) > _JUDGE_VERBATIM:
            with_est = [r for r in finals if r.price_estimate is not None]
            rest = [r for r in finals if r.price_estimate is None]
            by_est = sorted(with_est, key=lambda r: r.price_estimate)
            half = _JUDGE_VERBATIM // 2
            verbatim = by_est[:half] + by_est[-half:] if len(by_est) > _JUDGE_VERBATIM else by_est
            v_names = {id(r) for r in verbatim}
            lines = [f"{r.agent_name} ({r.round_num=}): {r.statement[:280]} "
                     f"[est {r.price_estimate}]" for r in verbatim]
            summary = ', '.join(f"{r.agent_name} {r.price_estimate}"
                                for r in with_est if id(r) not in v_names)
            if summary:
                lines.append(f"Other analysts' estimates: {summary}")
            if rest:
                lines.append(f"{len(rest)} analysts gave no parsable estimate.")
            transcript = '\n'.join(lines)
        else:
            transcript = '\n'.join(
                f"{r.agent_name}: {r.statement[:280]} [est {r.price_estimate}]"
                for r in finals)
        msgs = [
            {'role': 'system', 'content': _JUDGE_SYSTEM},
            {'role': 'user', 'content': (
                f"Sector: {ctx.sector} (report in {ctx.unit}). "
                f"Benchmark note: {ctx.verdict_note}\n\n"
                f"Analyst statements:\n{transcript}\n\n"
                "Weigh the analysts, resolve their disagreement, and give the single "
                "present read. End with:\n" + _EST_LINE[ctx.sector])},
        ]
        try:
            text = llm.complete(msgs, tier=self._deep, max_tokens=280)
        except Exception:
            return None, ''
        _, statement = _parse_think(text)
        return _EXTRACTORS[ctx.sector](statement), statement.strip()

    def _emit(self, kind: str, data: dict):
        if self._on_event:
            try:
                self._on_event(kind, data)
            except Exception:
                pass

    def _speak(self, agent: Agent, msgs: list[dict], sector: str, rnd: int) -> str:
        """One agent's turn.

        Streams **only when something is listening**. Tokens exist to fill a card
        live; with no event sink there is nobody to show them to, so a headless or
        batch run takes the plain `complete` path — cheaper, and it keeps
        `llm.complete` the single seam the module's tests monkeypatch.

        On a single local GPU the run is serial and takes minutes either way.
        Streaming does not make it faster; it makes progress continuously visible
        instead of leaving the user in front of a frozen panel. The accumulated
        text is identical either way — `complete` is literally
        ''.join(stream(...)) — so the debate itself is unchanged.
        """
        if self._on_event is None:
            try:
                return llm.complete(msgs, tier=agent.tier, max_tokens=500)
            except Exception:
                return ''
        parts: list[str] = []
        try:
            for chunk in llm.stream(msgs, tier=agent.tier, max_tokens=500):
                if not chunk:
                    continue
                parts.append(chunk)
                self._emit('agent_token', {'name': agent.name, 'sector': sector,
                                           'round': rnd, 'text': chunk})
        except Exception:
            # A mid-stream failure keeps whatever arrived; a dropped agent must not
            # abort the sector, exactly as before when this used `complete`.
            pass
        return ''.join(parts)

    def _divergent(self, ctx: SectorContext, responses: list[AgentResponse],
                   agents: list[Agent], k: int) -> list[Agent]:
        """The k agents whose estimates sit furthest from the round's centre.

        With a 50-agent roster, a second full round would roughly double a run that
        is already minutes long, and most of those turns would be an agent
        restating itself. The moderator's steer is only worth spending calls on
        where there is real disagreement, so round 2 is the outliers answering it.
        Agents that produced no parsable estimate are skipped: they have nothing to
        defend and are usually a dropped call.
        """
        est = {r.agent_name: r.price_estimate for r in responses
               if r.price_estimate is not None}
        if len(est) < 2:
            return []
        centre = sum(est.values()) / len(est)
        ranked = sorted(est, key=lambda n: -abs(est[n] - centre))[:max(0, k)]
        by_name = {a.name: a for a in agents}
        return [by_name[n] for n in ranked if n in by_name]

    def _run_sector(self, ctx: SectorContext) -> SectorReading:
        agents = _capability_agents(ctx.sector, self._per_channel)
        extractor = _EXTRACTORS[ctx.sector]
        history: list[AgentResponse] = []
        cited: set[str] = set()          # sources genuinely retrieved this sector
        steer = ''
        speaking = agents
        for rnd in range(1, self._rounds + 1):
            for agent in speaking:
                self._emit('agent_start', {'name': agent.name, 'occupation': agent.role,
                                           'sector': ctx.sector, 'round': rnd})
                msgs, used = self._agent_prompt(agent, ctx, history, steer)
                cited |= set(used)
                text = self._speak(agent, msgs, ctx.sector, rnd)
                thinking, statement = _parse_think(text)
                resp = AgentResponse(agent_name=agent.name, round_num=rnd,
                                     thinking=thinking, statement=statement,
                                     price_estimate=extractor(statement))
                history.append(resp)
                self._emit('agent_message', {
                    'name': agent.name, 'occupation': agent.role, 'sector': ctx.sector,
                    'round': rnd, 'message': statement,
                    'estimate': resp.price_estimate, 'unit': ctx.unit,
                    'sources': used})          # what it actually read, not its wishlist
            if rnd < self._rounds:                       # moderate BETWEEN rounds only
                this_round = [r for r in history if r.round_num == rnd]
                steer = self._moderate(ctx, this_round)
                speaking = self._divergent(ctx, this_round, agents, self._rebuttal_agents)
                self._emit('moderator', {'sector': ctx.sector, 'text': steer,
                                         'rebutting': [a.name for a in speaking]})
                if not speaking:
                    break                    # nothing to argue about; don't burn calls
        # Each agent's LATEST turn. Not "the last round": round 2 is only the
        # divergent few, so filtering by round number would silently reduce the
        # consensus to those 4 and discard the other 46 reads.
        finals = _latest_per_agent(history)
        judged, verdict = self._judge_sector(ctx, finals)
        self._emit('judge', {'sector': ctx.sector, 'text': verdict,
                             'estimate': judged, 'unit': ctx.unit})
        return self._aggregate(ctx, history, judged=judged, cited=cited)

    def _aggregate(self, ctx: SectorContext, history: list[AgentResponse],
                   judged: Optional[float] = None,
                   cited: Optional[set] = None) -> SectorReading:
        finals = _latest_per_agent(history)
        ests = [r.price_estimate for r in finals if r.price_estimate is not None]
        confidence = 0
        if ests:
            # Agreement measured on the raw agent estimates, corroboration-scaled:
            # a lone estimate can't be "100% agreed".
            band = _BAND.get(ctx.sector, 0.2)
            n = len(ests)
            centre = sum(ests) / n
            within = sum(1 for e in ests if abs(e - centre) <= band)
            confidence = int((within / n) * 100 * min(n, 2) / 2)
        # Point estimate: the JUDGE's synthesis (resolving disagreement), falling
        # back to the agent mean only if the judge produced no number.
        raw = judged if judged is not None else (sum(ests) / len(ests) if ests else None)
        avg = raw
        # Magnitude guard (§6.6): clamp toward the sector anchor, keeping direction —
        # this stops a "+5%/month" food read (a YoY-leak error) from reaching the card.
        if raw is not None and ctx.anchor is not None:
            try:
                avg = anchoring.reconcile_estimate(
                    raw, ctx.anchor, tolerance=_TOLERANCE.get(ctx.sector, 2.0)).value
            except Exception:
                avg = raw
        drivers = [d for r in finals if (d := _driver_text(r.statement))][:3]
        # Cite only what was actually retrieved — an empty list is the honest answer
        # when no feed returned anything, and is what the card/graph then show.
        sources = sorted(cited) if cited else []
        return SectorReading(
            sector=ctx.sector, direction=_direction(ctx.sector, avg),
            estimate=(round(avg, 2) if avg is not None else None),
            unit=ctx.unit, confidence=confidence, drivers=drivers, sources=sources)

    def _synthesize(self, readings: list[SectorReading]) -> str:
        body = '\n'.join(
            f"{r.sector}: {r.direction}, est {r.estimate} {r.unit}, "
            f"agreement {r.confidence}%" for r in readings)
        try:
            return llm.complete(
                [{'role': 'system', 'content': _SYNTH_SYSTEM},
                 {'role': 'user', 'content': f"Present readings:\n{body}\n\n"
                  "Write the 2-3 sentence present-pressure summary."}],
                tier=self._deep, max_tokens=200).strip()
        except Exception:
            return ''

    def run(self, on_event: Optional[Callable[[str, dict], None]] = None) -> PressureBrief:
        """on_event(kind, data): 'agent_start' / 'agent_message' / 'moderator'."""
        self._on_event = on_event
        readings = [self._run_sector(ctx) for ctx in self._contexts]
        return PressureBrief(as_of=self._as_of, window=self._window,
                             readings=readings, narrative=self._synthesize(readings))


def run_monitor(rag, corpus_dir=None, as_of=None, window: str = 'this_week',
                sectors=('gas', 'food', 'electricity'), rounds: int = 2, live: bool = True,
                on_event: Optional[Callable[[str, dict], None]] = None,
                per_channel: Optional[int] = None,
                rebuttal_agents: int = _REBUTTAL_AGENTS) -> PressureBrief:
    """One-click entry point: (optionally) refresh the social snapshot live, assemble
    the present context, then debate it into a Pressure Brief. `live` makes the
    Monitor hybrid — it pulls fresh search-interest/social text when possible and falls
    back to the frozen snapshot otherwise; the validated benchmark is never touched."""
    from ph_economic_ai.engine.social_snapshot import CORPUS_DIR
    cdir = corpus_dir or CORPUS_DIR
    if live:
        try:
            from ph_economic_ai.engine.live_social import refresh_social_snapshot
            refresh_social_snapshot(cdir)      # best-effort; frozen fallback on any miss
        except Exception:
            pass
    assembled = auto_assemble(rag=rag, as_of=as_of, window=window, sectors=sectors,
                              corpus_dir=cdir)
    forum = Forum(rag, assembled.contexts, as_of=assembled.as_of,
                  window=assembled.window, rounds=rounds,
                  per_channel=per_channel, rebuttal_agents=rebuttal_agents)
    return forum.run(on_event=on_event)
