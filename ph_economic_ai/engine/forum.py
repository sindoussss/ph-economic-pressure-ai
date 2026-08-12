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

import statistics
from typing import Callable, Optional

from ph_economic_ai.engine import llm
from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.auto_assemble import (
    SECTOR_SOURCES, SectorContext, auto_assemble, sector_corpus)
from ph_economic_ai.engine.debate import (
    Agent, AgentResponse, _MAX_REALISTIC_ELEC_PHP_KWH, _MAX_REALISTIC_FOOD_PCT,
    _MAX_REALISTIC_FUEL_PHP_L, _extract_electricity_change, _extract_percent,
    ESTIMATE_LINE, _extract_price, _parse_think, unfilled_scaffold,
    _extract_category_percents)
from ph_economic_ai.engine.pressure_brief import PressureBrief, SectorReading

# Per-sector estimate parsing, agreement band, and the "flat" threshold.
_EXTRACTORS: dict[str, Callable] = {
    'gas': _extract_price, 'food': _extract_percent,
    'electricity': _extract_electricity_change,
}

# Parse-sanity ceilings, shared with engine.debate. These are not economic
# judgements — they reject values that can only be a misparse, overwhelmingly the
# model quoting an absolute PRICE where a CHANGE was asked for. A live run had two
# agents return +150.00/L and +100.00/L, which the unguarded gas extractor passed
# straight onto their cards.
_PLAUSIBLE = {
    'gas': _MAX_REALISTIC_FUEL_PHP_L,
    'food': _MAX_REALISTIC_FOOD_PCT,
    'electricity': _MAX_REALISTIC_ELEC_PHP_KWH,
}


def _extract_guarded(sector: str, statement: str) -> tuple[Optional[float], Optional[float]]:
    """Parse an estimate into (accepted, rejected).

    Splitting the two keeps the card honest in the way the swarm already is: an
    agent that produced nothing and an agent whose number we threw away are
    different events, and collapsing both to "no estimate" hides the second. The
    rejected value is carried through so the UI can say the number was discarded
    rather than silently dropping it.

    Asymmetry worth knowing: `_extract_percent` and `_extract_electricity_change`
    already apply their own ceilings internally and return None, so for food and
    electricity an implausible value arrives here as "no estimate" and cannot be
    reported as rejected. Only `_extract_price` (gas) hands the raw value over —
    which is why gas was the sector that put +150.00/L on a card. The guard below
    closes that hole; unifying the *reporting* would mean changing extractors the
    swarm also depends on, so it is deliberately left alone.
    """
    value = _EXTRACTORS[sector](statement)
    if value is None:
        return None, None
    ceiling = _PLAUSIBLE.get(sector)
    if ceiling is not None and abs(value) > ceiling:
        return None, value
    return value, None


# How many prior turns an agent is shown, and how many statements the judge reads
# verbatim. Both exist because the roster is ~50: unbounded context would overflow
# the local models and slow every successive call.
_PRIOR_TURNS = 6
_JUDGE_VERBATIM = 8
#: CAP on how many agents may revise in round 2. Not a target: only agents whose
#: estimate sits OUTSIDE the agreement band are invited, and if fewer than this
#: many are outside, fewer are called.
#:
#: Was 4, which meant 46 of a 50-agent roster never saw the group's centre and
#: were locked to a round-1 answer given before any feedback existed. Agreement
#: was then measured over a roster that had mostly never been told what the room
#: thought. Inviting the agents the metric counts as disagreeing is not pressure:
#: those are the only ones whose revision can move the number, and the Delphi line
#: offers them "revise, or hold and cite" symmetrically.
_REBUTTAL_AGENTS = 16

# "Agree" if within this of the centre. Gas was 0.20, which is FINER than the
# 0.5-peso grid the models actually emit on: measured over every live estimate
# this session, 9 of 11 distinct values sat on that grid, so two agents giving the
# closest DIFFERENT answers the model can express were always scored as
# disagreeing. A band narrower than the instrument's resolution is an exact-match
# test wearing a tolerance's clothes.
#
# 0.50 is not chosen because it flatters the number. It is the widest band at
# which BOTH genuine-split controls still read 50 percent (clusters 1.5 and 0.9
# apart). At 1.00 the 0.9-apart control jumps to 100 percent, merging agents who
# really do disagree, so 1.00 is disqualified despite giving better live figures.
# The stopping rule is where the control breaks, not where the output looks good.
#
# food and electricity are UNCHANGED: their output quantum has not been measured,
# and moving them on gas's evidence would be the unprincipled version of this
# change. See `04 Engineering/Multi-Agent System` for the band study.
_BAND = {'gas': 0.50, 'food': 0.3, 'electricity': 0.10}
_FLAT = {'gas': 0.05, 'food': 0.05, 'electricity': 0.02}  # |estimate| below this reads as flat
# Magnitude-guard band per sector — how far a consensus may sit from the anchor
# before it is more likely a weak-model error than a real signal (engine.anchoring).
_TOLERANCE = {
    'gas': anchoring.FUEL_TOLERANCE_PHP_L,
    'food': anchoring.FOOD_TOLERANCE_PCT,
    'electricity': anchoring.ELECTRICITY_TOLERANCE_PHP_KWH,
}
# The final line every agent must emit, shared with `engine.debate` and
# `engine.swarm` rather than copied.
#
# This file's own `_EST_LINE` was the ORIGINAL fix for a template a small model
# copies verbatim, and it was right: a live 17-agent run had agents answer
# "ESTIMATE: +PHP X.XX/L", placeholder and all. The remedy then had to be
# rediscovered three more times, in the swarm's chain line, its estimate line and
# its DIRECTION menu, because it lived only here. It now lives in `debate`, the
# layer all three import, and this name is an alias so the Forum's call sites stay
# readable.
_EST_LINE = ESTIMATE_LINE

#: Six worked-example lines the food judge must append after its ESTIMATE
#: line, one per PSA sub-category. Same instructions-not-template pattern
#: _EST_LINE already established (RSK-012's lesson): a small model copies a
#: bare template verbatim, so every line needs its own worked example.
_FOOD_CATEGORY_LINES = {
    'rice': ('RICE: <the percent month-on-month CHANGE you expect for rice specifically, '
             'signed> (worked example: "RICE: +0.2%" or "RICE: -0.1%". Write your own '
             'number; never write X.X.)'),
    'meat': ('MEAT: <the percent month-on-month CHANGE you expect for meat specifically, '
             'signed> (worked example: "MEAT: +0.3%" or "MEAT: -0.2%". Write your own '
             'number; never write X.X.)'),
    'fish': ('FISH: <the percent month-on-month CHANGE you expect for fish and seafood '
             'specifically, signed> (worked example: "FISH: +0.8%" or "FISH: -0.4%". '
             'Write your own number; never write X.X.)'),
    'dairy_eggs': ('DAIRY_EGGS: <the percent month-on-month CHANGE you expect for milk, '
                  'dairy and eggs specifically, signed> (worked example: '
                  '"DAIRY_EGGS: +0.1%" or "DAIRY_EGGS: -0.1%". Write your own number; '
                  'never write X.X.)'),
    'vegetables': ('VEGETABLES: <the percent month-on-month CHANGE you expect for '
                   'vegetables specifically, signed> (worked example: '
                   '"VEGETABLES: +0.5%" or "VEGETABLES: -0.3%". Write your own number; '
                   'never write X.X.)'),
    'sugar': ('SUGAR: <the percent month-on-month CHANGE you expect for sugar and '
             'confectionery specifically, signed> (worked example: "SUGAR: +0.1%" or '
             '"SUGAR: -0.1%". Write your own number; never write X.X.)'),
}

# Capability channels: each agent stays strictly in its lane so the three do NOT
# converge on the same paragraph — the point of a forum over a single model.
_CHANNEL_TEMPLATES = {
    'social': ('You speak ONLY to public attention and mood — what Filipinos are '
               'searching for and posting about {sector} prices RIGHT NOW (search-interest '
               'spikes, public complaints). Do NOT analyse markets or policy; that is other '
               'agents\' job. If the retrieved context shows no fresh social signal, say so '
               'plainly and keep your read tentative rather than inventing a mood.\n'
               'CRITICAL: attention is NOT direction. Search interest measures how worried '
               'people are, not which way prices move, and this was tested: on this '
               'project\'s own benchmark these exact search terms do NOT predict fuel or '
               'food inflation any better than a naive baseline. Report WHAT people are '
               'attending to and how intensely. Never infer a price direction from mention '
               'volume: falling mentions do not mean falling prices, and a spike does not '
               'mean a rise.'),
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
def _next_change_label(sector: str) -> str:
    """The dated event this sector's pressure actually shows up in.

    Fuel moves on the weekly DOE adjustment; the CPI-derived sectors move on the
    PSA release. Naming it keeps the judge anchored to a real date without turning
    a present read into a forecast, and it answers what users actually asked: not
    only how much, but when.
    """
    from ph_economic_ai.engine import price_calendar
    try:
        event = (price_calendar.describe_next_fuel_adjustment() if sector == 'gas'
                 else price_calendar.describe_next_cpi_release())
        return f"{event['label']} ({event['when']})"
    except Exception:
        return 'the next scheduled adjustment'


_JUDGE_SYSTEM = (
    'You are the forum judge. You have just heard specialist analysts — a social-mood '
    'reader, a news reader, and a market reader — debate the CURRENT pressure on one '
    'Philippine price. Weigh what they said, trusting '
    'concrete cited data over vibes, resolve their disagreement into a SINGLE present '
    'read, and stay honest to the benchmark note (a present read, never a confident '
    'forecast). One or two sentences, then the estimate line.\n'
    'WEIGHTING RULE: the social-mood reader speaks to attention, not to direction. '
    'Search interest and mention volume were tested on this project\'s own benchmark '
    'and do NOT predict inflation better than a naive baseline. Social input may '
    'describe what the public is worried about; it may NOT on its own decide whether '
    'your estimate is positive or negative. A direction must rest on the news reader\'s '
    'concrete events or the market reader\'s price mechanics. If both of those are '
    'silent or conflicting, say the read is weak and keep the estimate near zero rather '
    'than borrowing a direction from mood.\n'
    'EVIDENCE RULE: you are given the retrieved evidence the analysts drew on. Use '
    'it to CHECK them, not to replace them. If an analyst asserts a figure or event '
    'the evidence does not support, say so and discount that analyst rather than '
    'averaging it in. You may NOT introduce a driver no analyst raised: the analysts '
    'do the analysis and you resolve it. If the evidence contradicts the whole panel, '
    'report the disagreement rather than substituting your own read.\n'
    'BACKWARD-LOOKING TRAP: an adjustment that has already been announced and taken '
    'effect is not pressure building, it is a past event already in the price. If the '
    'analysts are describing a change that has already happened, say so, and judge the '
    'pressure building on top of it rather than restating it.'
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
                # Per CHANNEL, deliberately, and not the whole sector corpus.
                #
                # Sharing all sector sources was tried and reverted. The swarm's
                # roles are functional (Forecaster, Critic, Synthesizer) so they
                # all benefit from all the evidence, but a forum channel is
                # evidence-TYPED: the social agent is instructed "do NOT analyse
                # markets; that is other agents' job", so handing it market data
                # tells it to read what it may not use, and its card then cites
                # BusinessWorld for a mood reading. `tests/test_forum` pins this:
                # a lane whose feeds return nothing must cite nothing.
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
    """The causal-chain line only — without the trailing ESTIMATE line.

    None when the agent returned the template rather than filling it in. This
    text is shown to the reader as the DRIVER behind a sector's number, and the
    Forum asks for the chain as `<trigger> → <effect> → <household impact>`;
    a small model that echoes that back would have put those three words on the
    card as if they were findings. Measured in the swarm on 2026-07-29, where
    eleven of twenty agents returned the bracketed template intact.
    """
    if 'CAUSAL CHAIN:' not in statement:
        return None
    part = statement.split('CAUSAL CHAIN:')[-1].split('ESTIMATE:')[0]
    if unfilled_scaffold(f'CAUSAL CHAIN: {part}'):
        return None
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
        # Kept so the response can carry what was actually read (RSK-019).
        self._last_retrieval = [{'source': c.get('source', '?'), 'text': c.get('text', '')}
                                for c in (chunks or [])]
        used = sorted({c['source'] for c in chunks if c.get('source')})
        text = '\n'.join(f"[{c['source']}] {c['text'][:280]}" for c in chunks) \
            or 'No context retrieved for your channel.'
        return text, used

    def _agent_prompt(self, agent: Agent, ctx: SectorContext,
                      history: list[AgentResponse], steer: str
                      ) -> tuple[list[dict], list[str]]:
        query = f"Current {ctx.sector} price pressure in the Philippines, {self._window}."
        # Round 1 is DELIBERATELY BLIND: an agent forming its opening read sees no
        # other agent's statement.
        #
        # Two reasons, one scientific and one practical. Agreement between agents
        # that have read each other is herding, not corroboration — and the
        # confidence number on the card is computed from exactly that agreement, so
        # showing prior statements makes it measure the wrong thing. Practically, a
        # 3b model shown neighbouring text copies it: a live 17-agent run returned
        # only 8 distinct openings out of 20, with a DOE beat reporter reciting the
        # social lane's "no recent chatter" verbatim.
        #
        # Prior statements appear only for a rebuttal (round > 1), where responding
        # to what others said is the entire point, and even then capped at
        # `_PRIOR_TURNS`.
        prior = ''
        if steer and history:
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
            # Small local models echo their input. Live runs opened with "BENCHMARK
            # NOTE:", "Retrieved context:" and even a neighbour's name — the prompt
            # read back rather than answered — so the scaffolding is named here and
            # ruled out explicitly.
            + "Write in your OWN words as this analyst. Do not repeat these headings "
              "('BENCHMARK NOTE', 'Retrieved context', 'Prior statements'), do not "
              "quote the context verbatim, and do not begin with another analyst's "
              "name.\n"
            # Figures from model memory are the worst output this panel can
            # produce, because they are specific, confident and wrong. A live run
            # had a coal-import analyst report "1 USD = 50.5 PHP, a 0.5% decline
            # from last week" when the true rate was 61.61 and no FX figure was in
            # his context at all. The invented number is more precise than the
            # honest answer would have been, which is exactly why a reader
            # believes it.
            + "NUMBERS RULE: state a specific figure (a price, a rate, a "
              "percentage) ONLY if it appears in your retrieved context above. If "
              "your channel gives you no figure for something, say so plainly — "
              "'no current reading in my channel' is a valid and useful "
              "observation. Never recall a number from memory or estimate one to "
              "fill a gap.\n"
            + "Give a short present-tense read from your channel only, then end with "
              "exactly these two lines:\n"
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
            return llm.complete(msgs, tier=self._deep, max_tokens=220,
                                seed=llm.derive_seed(self._as_of, ctx.sector,
                                                     'moderator')).strip()
        except Exception:
            return ''

    def _judge_sector(self, ctx: SectorContext, finals: list[AgentResponse]):
        """Synthesise the debate into one present read (like the swarm's master
        judge) — resolving disagreement rather than averaging. Returns
        (estimate | None, verdict_text, subcategories).

        `subcategories` is the food judge's six PSA sub-category reads (rice,
        meat, fish, dairy_eggs, vegetables, sugar), parsed from its closing
        lines — empty for every other sector, which never asks for them.

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
        # The judge reads the same corpus the analysts did, so it can verify a
        # claim instead of averaging an unverified one. Deliberately the SECTOR
        # corpus rather than a fresh query: the judge checks what the room was
        # working from, it does not go looking for something new.
        judge_evidence = ''
        try:
            chunks = self._rag.query(f'{ctx.sector} price Philippines', top_k=6,
                                     sources=sector_corpus(ctx.sector)) or []
            judge_evidence = '\n'.join(
                f"[{c.get('source', '?')}] {c.get('text', '')[:240]}" for c in chunks)
        except Exception:
            judge_evidence = ''

        # Food-only: six worked-example lines asking for a per-category read
        # alongside the blended ESTIMATE. Gas and electricity have no PSA
        # sub-categories, so their prompt is unchanged.
        category_lines = (
            '\n' + '\n'.join(_FOOD_CATEGORY_LINES.values())
            if ctx.sector == 'food' else ''
        )
        msgs = [
            {'role': 'system', 'content': _JUDGE_SYSTEM},
            {'role': 'user', 'content': (
                f"Sector: {ctx.sector} (report in {ctx.unit}). "
                f"This pressure lands at the next scheduled change on "
                f"{_next_change_label(ctx.sector)}. That date is a published schedule, "
                f"not something you are predicting. Read the pressure building into it; "
                f"do not restate a change already in effect.\n"
                f"Benchmark note: {ctx.verdict_note}\n\n"
                f"Analyst statements:\n{transcript}\n\n"
                + (f"Retrieved evidence (for CHECKING the analysts, not for adding "
                   f"new drivers):\n{judge_evidence}\n\n" if judge_evidence else "")
                + "Weigh the analysts, resolve their disagreement, and give the single "
                "present read. End with:\n" + _EST_LINE[ctx.sector] + category_lines)},
        ]
        try:
            text = llm.complete(msgs, tier=self._deep,
                                max_tokens=280 + (120 if ctx.sector == 'food' else 0),
                                seed=llm.derive_seed(self._as_of, ctx.sector, 'judge'))
        except Exception:
            return None, '', {}
        _, statement = _parse_think(text)
        # The judge is guarded too: it reads agent numbers, so it can repeat an
        # implausible one back.
        accepted, _ = _extract_guarded(ctx.sector, statement)
        subcategories = _extract_category_percents(statement) if ctx.sector == 'food' else {}
        return accepted, statement.strip(), subcategories

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
        # Per-agent derived seed: the same scenario reproduces this agent's turn
        # exactly, while a different agent (or a different as-of date) still gets a
        # different sample. A constant seed would collapse the roster to one voice;
        # no seed at all is what made two runs ten minutes apart disagree.
        seed = llm.derive_seed(self._as_of, self._window, sector, agent.name, rnd)

        if self._on_event is None:
            try:
                return llm.complete(msgs, tier=agent.tier, max_tokens=500, seed=seed)
            except Exception:
                return ''
        parts: list[str] = []
        try:
            for chunk in llm.stream(msgs, tier=agent.tier, max_tokens=500, seed=seed):
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

    @staticmethod
    def _delphi_line(ctx: SectorContext, responses: list[AgentResponse]) -> str:
        """Structured feedback for the rebuttal round: the group's own centre.

        This is the Delphi mechanism, and it is deliberately different from the
        shared-anchor injection that was tested and rejected. There, an external
        number was planted before any reasoning happened and five of seven agents
        echoed it back. Here, the centre IS the round's output, shown after every
        agent has already committed a first answer, and the outlier's options are
        stated symmetrically: move, or stay and cite. Convergence through that
        choice is earned; parroting a pre-seeded number is not. The placebo test
        for this distinction lives in the anchor experiment notes.
        """
        ests = [r.price_estimate for r in responses if r.price_estimate is not None]
        if len(ests) < 2:
            return ''
        centre = statistics.median(ests)
        return (f"The group's centre estimate this round is {centre:+.2f} {ctx.unit}. "
                f"If you keep an estimate far from it, cite the specific figure or "
                f"event that justifies the distance; otherwise revise toward the "
                f"centre.")

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
        # Medoid, matching the agreement metric: the centre must be a value an
        # agent actually gave. The mean is dragged by one extreme agent, which
        # both understates its own divergence and overstates everyone else's.
        values = list(est.values())
        centre = min(values, key=lambda c: (sum(abs(v - c) for v in values), c))
        # Invite exactly those the metric counts as disagreeing. An agent already
        # inside the band has nothing to revise, so calling it spends a turn to
        # learn nothing; an agent outside it is the only kind whose second thought
        # can change the reading.
        band = _BAND.get(ctx.sector, 0.2)
        outside = [n for n in est if abs(est[n] - centre) > band]
        ranked = sorted(outside, key=lambda n: -abs(est[n] - centre))[:max(0, k)]
        by_name = {a.name: a for a in agents}
        return [by_name[n] for n in ranked if n in by_name]

    def _run_sector(self, ctx: SectorContext) -> SectorReading:
        agents = _capability_agents(ctx.sector, self._per_channel)
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
                accepted, rejected = _extract_guarded(ctx.sector, statement)
                resp = AgentResponse(agent_name=agent.name, round_num=rnd,
                                     thinking=thinking, statement=statement,
                                     price_estimate=accepted,
                                     retrieval=list(getattr(self, '_last_retrieval', [])))
                history.append(resp)
                self._emit('agent_message', {
                    'name': agent.name, 'occupation': agent.role, 'sector': ctx.sector,
                    'round': rnd, 'message': statement,
                    'estimate': accepted, 'rejected_estimate': rejected,
                    'unit': ctx.unit,
                    'sources': used})          # what it actually read, not its wishlist
            if rnd < self._rounds:                       # moderate BETWEEN rounds only
                this_round = [r for r in history if r.round_num == rnd]
                steer = self._moderate(ctx, this_round)
                steer = (steer + ' ' + self._delphi_line(ctx, this_round)).strip()
                speaking = self._divergent(ctx, this_round, agents, self._rebuttal_agents)
                self._emit('moderator', {'sector': ctx.sector, 'text': steer,
                                         'rebutting': [a.name for a in speaking]})
                if not speaking:
                    break                    # nothing to argue about; don't burn calls
        # Each agent's LATEST turn. Not "the last round": round 2 is only the
        # divergent few, so filtering by round number would silently reduce the
        # consensus to those 4 and discard the other 46 reads.
        finals = _latest_per_agent(history)
        judged, verdict, subcategories = self._judge_sector(ctx, finals)
        self._emit('judge', {'sector': ctx.sector, 'text': verdict,
                             'estimate': judged, 'unit': ctx.unit})
        return self._aggregate(ctx, history, judged=judged, cited=cited,
                               subcategories=subcategories)

    def _aggregate(self, ctx: SectorContext, history: list[AgentResponse],
                   judged: Optional[float] = None,
                   cited: Optional[set] = None,
                   subcategories: Optional[dict] = None) -> SectorReading:
        finals = _latest_per_agent(history)
        ests = [r.price_estimate for r in finals if r.price_estimate is not None]
        confidence = 0
        if ests:
            # Agreement measured on the raw agent estimates, corroboration-scaled:
            # a lone estimate can't be "100% agreed".
            #
            # Centre on the MEDOID: the actual estimate closest to all the others.
            # Two artefacts led here. The mean is dragged by one extreme agent
            # (six agents near -0.6 plus one at -2.15 scored 28% around the mean,
            # 85% around the median). The abstract median then showed its own
            # failure on live data: agents quote values on a rough 0.5-peso grid,
            # and with estimates [1.0 x3, 1.5 x2, ...] the median landed at 1.25,
            # a number NO agent said, in the gap between sub-clusters, where the
            # +/-band reaches nobody and agreement read 0% against six near-equal
            # reads. The centre of a consensus measure must be a value someone
            # actually estimated. On a genuine split the medoid still scores low,
            # so this removes quantisation artefacts without widening the band.
            band = _BAND.get(ctx.sector, 0.2)
            n = len(ests)
            centre = min(ests, key=lambda c: (sum(abs(e - c) for e in ests), c))
            within = sum(1 for e in ests if abs(e - centre) <= band)
            confidence = int((within / n) * 100 * min(n, 2) / 2)
        direction_agreement = 0
        if ests:
            flat = _FLAT.get(ctx.sector, 0.05)
            signs = [0 if abs(e) < flat else (1 if e > 0 else -1) for e in ests]
            lead = max(set(signs), key=signs.count)
            direction_agreement = int(signs.count(lead) / len(signs) * 100
                                      * min(len(signs), 2) / 2)
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
            unit=ctx.unit, confidence=confidence,
            direction_agreement=direction_agreement,
            # The raw estimates, not the summary of them. The card shows the
            # distinct values and their span; a percentage alone cannot be
            # checked against anything else on the screen.
            estimates=[round(float(e), 2) for e in ests],
            drivers=drivers, sources=sources,
            subcategories=subcategories or {})

    def _synthesize(self, readings: list[SectorReading]) -> str:
        body = '\n'.join(
            f"{r.sector}: {r.direction}, est {r.estimate} {r.unit}, "
            f"agreement {r.confidence}%" for r in readings)
        try:
            return llm.complete(
                [{'role': 'system', 'content': _SYNTH_SYSTEM},
                 {'role': 'user', 'content': f"Present readings:\n{body}\n\n"
                  "Write the 2-3 sentence present-pressure summary."}],
                tier=self._deep, max_tokens=200,
                seed=llm.derive_seed(self._as_of, 'synthesis')).strip()
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
