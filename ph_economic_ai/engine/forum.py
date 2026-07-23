"""The Forum — a BettaFish-style moderated debate that produces the Pressure Brief.

Unlike the persona debate in `engine/debate.py`, the Forum's agents are
distinguished by *capability channel* (social sentiment / news / market), and a
**moderator carries the benchmark's verdict into the room** — for an efficient
sector it steers the discussion to the present read and away from a confident
forecast. The output is a `PressureBrief`: the hero of a Monitor run.

Offline-first: every LLM call goes through `engine.llm`, RAG through the injected
RagEngine, and the social layer through the frozen snapshot. Testable by
monkeypatching `llm.complete`.
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
_BAND = {'gas': 0.20, 'food': 0.3, 'electricity': 0.10}   # "agree" if within this of the mean
_FLAT = {'gas': 0.05, 'food': 0.05, 'electricity': 0.02}  # |estimate| below this reads as flat
_EST_LINE = {
    'gas': 'ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L',
    'food': 'ESTIMATE: +X.X% or ESTIMATE: -X.X%',
    'electricity': 'ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh',
}

# Capability channels: the instruction each channel follows (persona-agnostic).
_CHANNEL_TEMPLATES = {
    'social': ('You gauge how Filipinos are reacting RIGHT NOW to {sector} prices, '
               'using the frozen social posts and search-interest snapshot. State the '
               'CURRENT direction of pressure and why.'),
    'news': ('You summarise what the news is reporting RIGHT NOW about Philippine '
             '{sector} prices — announcements, rate changes, events. Present pressure only.'),
    'market': ('You read the underlying market drivers (oil, FX, spot rates) behind '
               'the CURRENT {sector} pressure. Present read, not a forecast.'),
}

# The named cast — (name, occupation) per sector x channel. Display/flavour only;
# it does not change what an agent does, only who the user sees speaking.
_PERSONAS: dict[str, dict[str, tuple[str, str]]] = {
    'gas': {
        'social': ('Andrea Lim', 'Commuter Sentiment Analyst'),
        'news':   ('Rafael Cruz', 'Energy Desk Reporter'),
        'market': ('Diego Ocampo', 'Crude & FX Trader'),
    },
    'food': {
        'social': ('Bea Villanueva', 'Palengke Sentiment Analyst'),
        'news':   ('Marco Reyes', 'Agriculture Correspondent'),
        'market': ('Nadia Chua', 'Agri-Commodities Analyst'),
    },
    'electricity': {
        'social': ('Paolo Mendoza', 'Household Bill Watcher'),
        'news':   ('Ligaya Torres', 'Utilities Correspondent'),
        'market': ('Enzo Garcia', 'Power Market Analyst'),
    },
}

_MODERATOR_SYSTEM = (
    'You are the forum moderator for a Philippine price-pressure debate. You do not '
    'estimate prices yourself. You summarise the round, name the main disagreement, '
    'and enforce the benchmark note: keep the agents on the PRESENT read and stop any '
    'agent that drifts into a confident forward forecast. One short paragraph.'
)
_SYNTH_SYSTEM = (
    'You are a Philippine macro analyst writing the present-pressure summary a '
    'household would read. Write 2-3 present-tense sentences on current pressure '
    'across the sectors. This is a nowcast — describe now, do not forecast.'
)


def _capability_agents(sector: str) -> list[Agent]:
    srcs = SECTOR_SOURCES.get(sector, {})
    personas = _PERSONAS.get(sector, {})
    agents = []
    for channel, tmpl in _CHANNEL_TEMPLATES.items():
        name, occupation = personas.get(channel, (channel.title(), 'Analyst'))
        agents.append(Agent(
            name=name, role=occupation,
            system_prompt=(f'You are {name}, a {occupation} in the Philippines. '
                           + tmpl.format(sector=sector)
                           + ' End your response with a CAUSAL CHAIN line and then: '
                           + _EST_LINE[sector]),
            rag_sources=srcs.get(channel, []),
            tier=llm.FAST, is_mini=(channel != 'market')))
    return agents


def _direction(sector: str, value: Optional[float]) -> str:
    if value is None:
        return 'unknown'
    if abs(value) < _FLAT.get(sector, 0.05):
        return 'flat'
    return 'rising' if value > 0 else 'easing'


class Forum:
    """Runs the moderated present-pressure debate for each sector."""

    def __init__(self, rag, contexts: list[SectorContext], as_of: str, window: str,
                 rounds: int = 2, deep_tier: str = llm.DEEP):
        self._rag = rag
        self._contexts = contexts
        self._as_of = as_of
        self._window = window
        self._rounds = max(1, rounds)
        self._deep = deep_tier
        self._on_event = None

    # ── prompts ───────────────────────────────────────────────────────────────

    def _rag_text(self, agent: Agent, query: str) -> str:
        try:
            chunks = self._rag.query(query, top_k=4, sources=agent.rag_sources)
        except Exception:
            chunks = []
        return '\n'.join(f"[{c['source']}] {c['text'][:280]}" for c in chunks) \
            or 'No frozen context retrieved.'

    def _agent_prompt(self, agent: Agent, ctx: SectorContext,
                      history: list[AgentResponse], steer: str) -> list[dict]:
        query = f"Current {ctx.sector} price pressure in the Philippines, {self._window}."
        prior = '\n'.join(f"{r.agent_name}: {r.statement[:280]}" for r in history)
        user = (
            f"BENCHMARK NOTE: {ctx.verdict_note}\n\n"
            f"As of {self._as_of} ({self._window}). Sector: {ctx.sector} "
            f"(report in {ctx.unit}).\n\n"
            f"Frozen context:\n{self._rag_text(agent, query)}\n\n"
            + (f"Moderator steer: {steer}\n\n" if steer else '')
            + (f"Prior statements:\n{prior}\n\n" if prior else '')
            + "Give a short present-tense read. End with:\n"
            "CAUSAL CHAIN: [signal] → [effect] → [household impact]\n"
            + _EST_LINE[ctx.sector]
        )
        return [{'role': 'system', 'content': agent.system_prompt},
                {'role': 'user', 'content': user}]

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

    def _emit(self, kind: str, data: dict):
        if self._on_event:
            try:
                self._on_event(kind, data)
            except Exception:
                pass

    def _run_sector(self, ctx: SectorContext) -> SectorReading:
        agents = _capability_agents(ctx.sector)
        extractor = _EXTRACTORS[ctx.sector]
        history: list[AgentResponse] = []
        steer = ''
        for rnd in range(1, self._rounds + 1):
            for agent in agents:
                self._emit('agent_start', {'name': agent.name, 'occupation': agent.role,
                                           'sector': ctx.sector, 'round': rnd})
                try:
                    text = llm.complete(self._agent_prompt(agent, ctx, history, steer),
                                        tier=agent.tier, max_tokens=500)
                except Exception:
                    text = ''
                thinking, statement = _parse_think(text)
                resp = AgentResponse(agent_name=agent.name, round_num=rnd,
                                     thinking=thinking, statement=statement,
                                     price_estimate=extractor(statement))
                history.append(resp)
                self._emit('agent_message', {
                    'name': agent.name, 'occupation': agent.role, 'sector': ctx.sector,
                    'round': rnd, 'message': statement,
                    'estimate': resp.price_estimate, 'unit': ctx.unit})
            if rnd < self._rounds:                       # moderate BETWEEN rounds only
                steer = self._moderate(ctx, [r for r in history if r.round_num == rnd])
                self._emit('moderator', {'sector': ctx.sector, 'text': steer})
        return self._aggregate(ctx, history)

    def _aggregate(self, ctx: SectorContext, history: list[AgentResponse]) -> SectorReading:
        final = max((r.round_num for r in history), default=0)
        finals = [r for r in history if r.round_num == final]
        ests = [r.price_estimate for r in finals if r.price_estimate is not None]
        if ests:
            avg = sum(ests) / len(ests)
            if ctx.anchor is not None:
                try:
                    avg = anchoring.reconcile_estimate(avg, ctx.anchor).value
                except Exception:
                    pass
            band = _BAND.get(ctx.sector, 0.2)
            confidence = int(sum(1 for e in ests if abs(e - avg) <= band) / len(ests) * 100)
        else:
            avg, confidence = None, 0
        drivers = [r.statement.split('CAUSAL CHAIN:')[-1].strip()[:160]
                   for r in finals if 'CAUSAL CHAIN:' in r.statement][:3]
        sources = sorted({s for a in _capability_agents(ctx.sector) for s in a.rag_sources})
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
                sectors=('gas', 'food', 'electricity'), rounds: int = 2,
                on_event: Optional[Callable[[str, str], None]] = None) -> PressureBrief:
    """One-click entry point: assemble the present context, then debate it into a
    Pressure Brief. This is what the "Run" button calls (Stage 1 of the Monitor)."""
    kwargs = {} if corpus_dir is None else {'corpus_dir': corpus_dir}
    assembled = auto_assemble(rag=rag, as_of=as_of, window=window, sectors=sectors, **kwargs)
    forum = Forum(rag, assembled.contexts, as_of=assembled.as_of,
                  window=assembled.window, rounds=rounds)
    return forum.run(on_event=on_event)
