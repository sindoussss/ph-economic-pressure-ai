from __future__ import annotations

import concurrent.futures
import re
import statistics
import threading
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import AgentResponse, _parse_think, _extract_price
from ph_economic_ai.engine.live_data import LiveDataBrief


# Fallback price used when live fetch fails (₱/L, NCR unleaded 91 avg).
# Only needs updating if the live fetcher stops working for an extended period.
_FALLBACK_RETAIL_PRICE_PHP: float = 98.82  # NCR Unleaded 91 avg May 20 2026

_PRICE_FETCH_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


def fetch_live_retail_price() -> float:
    """Fetch current NCR retail gasoline price from fuelprice.ph (DOE-sourced).

    Parses brand-average prices for all fuel types, takes the median of values
    in the 60–150 range. Falls back to _FALLBACK_RETAIL_PRICE_PHP on any error.
    """
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            'https://www.fuelprice.ph/',
            headers=_PRICE_FETCH_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Matches "avg ₱98.82/L" or "avg P98.82/L" — fuelprice.ph brand averages
        hits = [
            float(m) for m in re.findall(
                r'avg\s*[^\d]*(\d{2,3}(?:\.\d{1,2})?)\s*/[Ll]', text
            )
            if 60.0 <= float(m) <= 150.0
        ]
        if hits:
            hits.sort()
            return hits[len(hits) // 2]  # median across fuel types
    except Exception:
        pass
    return _FALLBACK_RETAIL_PRICE_PHP

# ── Region list ───────────────────────────────────────────────────────────────
REGIONS: list[str] = [
    'NCR',
    'Central Luzon',
    'Western Visayas',
    'Davao Region',
]

# Pairs: (0,1) and (2,3) → 2 regional judges
REGION_PAIRS: list[tuple[int, int]] = [(0, 1), (2, 3)]

# ── Tier assignments ──────────────────────────────────────────────────────────
# Every bulk agent runs on the fast tier and only the judges get the deep one.
# That looks blunter than the old five-way 3b/7b/14b split, but free-tier
# tokens-per-minute — not requests-per-day — is the binding constraint, and the
# deep tier's TPM ceiling cannot absorb 32 agent calls carrying RAG context.
# Spending it on the 7 judge calls buys more: the judges are what actually
# determine the master verdict.
_ROLE_TIERS: dict[str, str] = {
    'Forecaster':        llm.FAST,
    'DataExtractor':     llm.FAST,
    'Synthesizer':       llm.FAST,
    'Critic':            llm.FAST,
    'ConfidenceScorer':  llm.FAST,
}
_JUDGE_TIER = llm.DEEP

# RegionalJudge.run makes three calls (two defences + a synthesis); MasterJudge
# makes one. Named so expected_call_counts() stays honest if either changes.
_REGIONAL_JUDGE_CALLS = 3
_MASTER_JUDGE_CALLS = 1

# Reserved completion length per call. Module-level so the ablation harness can
# vary it: completions are ~24K of a run's ~44K fast-tier tokens, making this
# the single biggest lever on free-tier run time.
#
# The judge budgets are deliberately generous. A reasoning model on the deep
# tier (deepseek-r1 and friends) spends hundreds of tokens thinking before it
# writes anything, and if the cap lands mid-thought the reply is truncated
# before the ESTIMATE line — which does not error, it just silently yields a
# verdict with no estimate. This is a cap, not a target: models that finish
# early cost nothing extra.
_AGENT_MAX_TOKENS = 750
_JUDGE_MAX_TOKENS = 1800
_MASTER_MAX_TOKENS = 2000
_MAX_REALISTIC_FUEL_CHANGE = 8.0

# Role processing order within a round (Critic and ConfidenceScorer last so they
# can score agents they've already seen)
_ROLE_ORDER = ['Forecaster', 'DataExtractor', 'Synthesizer', 'Critic', 'ConfidenceScorer']


def expected_call_counts() -> dict[str, int]:
    """How many LLM calls one swarm run costs, derived from the swarm's shape.

    Single source of truth for anything that needs the number — the setup
    screen's time estimate, and free-tier quota planning. Derived rather than
    hardcoded because a stale constant is exactly how the old estimate came to
    claim "~371 calls" for a run that actually makes 39.
    """
    alive = len(_ROLE_ORDER)
    per_group = 0
    for _round_num, n_eliminate in _BRACKET:
        per_group += alive
        alive -= n_eliminate

    fast = per_group * len(REGIONS)
    deep = len(REGION_PAIRS) * _REGIONAL_JUDGE_CALLS + _MASTER_JUDGE_CALLS
    return {'fast': fast, 'deep': deep, 'total': fast + deep}


def group_critical_path() -> int:
    """Sequential call-depth of one group, in call durations.

    Round 1 is sequential by design (each agent reads its peers' answers in
    order), so it costs one duration per agent. Later rounds fan out across a
    thread pool and cost roughly one duration each regardless of width.
    """
    if not _BRACKET:
        return 0
    first_round_agents = len(_ROLE_ORDER)
    later_rounds = len(_BRACKET) - 1
    return first_round_agents + later_rounds


def _is_realistic_fuel_change(value: Optional[float]) -> bool:
    return value is not None and abs(value) <= _MAX_REALISTIC_FUEL_CHANGE


def _extract_fuel_change(text: str) -> Optional[float]:
    """Extract a signed PHP/L fuel price change, rejecting absolute-price parses."""
    estimate = None
    estimate_lines = re.findall(
        r'ESTIMATE\s*:\s*([+\-])\s*(?:₱|PHP|P|â‚±)?\s*(\d+(?:\.\d+)?)\s*/?\s*L?',
        text,
        flags=re.IGNORECASE,
    )
    if estimate_lines:
        sign, raw = estimate_lines[-1]
        estimate = (-1 if sign == '-' else 1) * float(raw)
    else:
        estimate = _extract_price(text)
    if not _is_realistic_fuel_change(estimate):
        return None
    return estimate


def _robust_confidence_pct(estimates: list[float], final_estimate: Optional[float]) -> int:
    """Confidence from agreement around the final estimate, robust to outliers.

    The old calculation used standard deviation across all intermediate values,
    so one bad parse like -92.30 could force a 10-15% confidence floor. This
    version discards impossible fuel changes and scores how tightly the usable
    estimates cluster around the master verdict.
    """
    valid = [e for e in estimates if _is_realistic_fuel_change(e)]
    if _is_realistic_fuel_change(final_estimate):
        center = float(final_estimate)
    elif valid:
        center = statistics.median(valid)
    else:
        return 0

    if len(valid) < 2:
        return 65 if valid else 0

    close = [e for e in valid if abs(e - center) <= 1.00]
    near = [e for e in valid if abs(e - center) <= 1.50]
    usable = close or near or valid
    spread = statistics.pstdev(usable) if len(usable) > 1 else 0.0

    agreement_score = len(close) / len(valid)
    near_score = len(near) / len(valid)
    spread_score = max(0.0, 1.0 - min(spread / 1.50, 1.0))
    confidence = 0.50 * agreement_score + 0.25 * near_score + 0.25 * spread_score
    return max(10, min(95, int(round(confidence * 100))))


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SwarmAgent:
    name: str
    role: str
    tier: str                  # llm.FAST | llm.DEEP — resolved to a model at call time
    group_id: int
    region_name: str
    system_prompt: str
    rag_sources: list[str]
    is_alive: bool = True
    combined_score: float = 0.0


@dataclass
class GroupSurvivor:
    group_id: int
    region_name: str
    response: AgentResponse
    combined_score: float
    agent_role: str
    agent_model: str


@dataclass
class RegionalVerdict:
    judge_id: int
    region_pair: tuple[str, str]
    estimate: Optional[float]
    confidence: float
    reasoning: str
    survivor_names: tuple[str, str]


@dataclass
class MasterVerdict:
    final_estimate: Optional[float]
    confidence_pct: int
    dissenting_regions: list[str]
    reasoning: str
    regional_verdicts: list[RegionalVerdict]
    regional_estimates: Optional[dict] = None  # {region_name: Optional[float]}
    all_responses: list = None   # list[AgentResponse] from all group arenas

    def __post_init__(self):
        if self.all_responses is None:
            self.all_responses = []


# ── RAG source assignments per role ───────────────────────────────────────────
_ROLE_RAG: dict[str, list[str]] = {
    'Forecaster':       ['DOEBulletin', 'PHRetailFuel', 'YahooFinanceCrude', 'YahooFinanceForex'],
    'DataExtractor':    ['DOEBulletin', 'PHRetailFuel', 'YahooFinanceCrude', 'ManilaBulletin'],
    'Synthesizer':      ['PHRetailFuel', 'neda_2024_2026', 'YahooFinanceForex'],
    'Critic':           ['DOEBulletin', 'BusinessWorld', 'ManilaBulletin'],
    'ConfidenceScorer': ['neda_2024_2026', 'BusinessWorld'],
}


# ── All 17 PH administrative regions ─────────────────────────────────────────
# anchor: index into swarm group survivors (0=NCR, 1=Central Luzon, 2=W.Visayas, 3=Davao)
# multiplier: freight/logistics premium over NCR (applied to price change magnitude)
ALL_REGIONS: list[dict] = [
    # Luzon
    {'name': 'NCR',               'code': 'NCR',   'multiplier': 1.00, 'anchor': 0,
     'nx': 0.64, 'ny': 0.33, 'isle': 'L'},
    {'name': 'Ilocos Region',     'code': 'I',     'multiplier': 1.05, 'anchor': 0,
     'nx': 0.36, 'ny': 0.14, 'isle': 'L'},
    {'name': 'Cagayan Valley',    'code': 'II',    'multiplier': 1.06, 'anchor': 0,
     'nx': 0.72, 'ny': 0.09, 'isle': 'L'},
    {'name': 'Central Luzon',     'code': 'III',   'multiplier': 1.02, 'anchor': 1,
     'nx': 0.60, 'ny': 0.25, 'isle': 'L'},
    {'name': 'CALABARZON',        'code': 'IVA',   'multiplier': 1.03, 'anchor': 1,
     'nx': 0.70, 'ny': 0.40, 'isle': 'L'},
    {'name': 'MIMAROPA',          'code': 'IVB',   'multiplier': 1.08, 'anchor': 1,
     'nx': 0.44, 'ny': 0.44, 'isle': 'L'},
    {'name': 'Bicol Region',      'code': 'V',     'multiplier': 1.06, 'anchor': 1,
     'nx': 0.82, 'ny': 0.44, 'isle': 'L'},
    {'name': 'CAR',               'code': 'CAR',   'multiplier': 1.08, 'anchor': 0,
     'nx': 0.60, 'ny': 0.15, 'isle': 'L'},
    # Visayas
    {'name': 'Western Visayas',   'code': 'VI',    'multiplier': 1.05, 'anchor': 2,
     'nx': 0.36, 'ny': 0.56, 'isle': 'V'},
    {'name': 'Central Visayas',   'code': 'VII',   'multiplier': 1.04, 'anchor': 2,
     'nx': 0.62, 'ny': 0.57, 'isle': 'V'},
    {'name': 'Eastern Visayas',   'code': 'VIII',  'multiplier': 1.07, 'anchor': 2,
     'nx': 0.82, 'ny': 0.53, 'isle': 'V'},
    # Mindanao
    {'name': 'Zamboanga',         'code': 'IX',    'multiplier': 1.08, 'anchor': 3,
     'nx': 0.26, 'ny': 0.70, 'isle': 'M'},
    {'name': 'Northern Mindanao', 'code': 'X',     'multiplier': 1.06, 'anchor': 3,
     'nx': 0.56, 'ny': 0.70, 'isle': 'M'},
    {'name': 'Caraga',            'code': 'XIII',  'multiplier': 1.07, 'anchor': 3,
     'nx': 0.82, 'ny': 0.74, 'isle': 'M'},
    {'name': 'Davao Region',      'code': 'XI',    'multiplier': 1.05, 'anchor': 3,
     'nx': 0.72, 'ny': 0.82, 'isle': 'M'},
    {'name': 'SOCCSKSARGEN',      'code': 'XII',   'multiplier': 1.07, 'anchor': 3,
     'nx': 0.56, 'ny': 0.87, 'isle': 'M'},
    {'name': 'BARMM',             'code': 'BARMM', 'multiplier': 1.10, 'anchor': 3,
     'nx': 0.38, 'ny': 0.84, 'isle': 'M'},
]


def derive_regional_estimates(
    base_estimate: Optional[float],
    anchor_estimates: Optional[dict] = None,
) -> dict:
    """Derive per-region price change estimates for all 17 PH regions.

    anchor_estimates maps group_id (0-3) → survivor price estimate.
    Falls back to base_estimate when an anchor is missing.
    Multiplies the anchor change by the region's logistics freight factor.
    """
    anchors = anchor_estimates or {}
    result: dict[str, Optional[float]] = {}
    for reg in ALL_REGIONS:
        anchor_est = anchors.get(reg['anchor'], base_estimate)
        if anchor_est is None:
            anchor_est = base_estimate
        result[reg['name']] = (
            round(anchor_est * reg['multiplier'], 2) if anchor_est is not None else None
        )
    return result


def _make_system_prompt(role: str, region: str, current_price: float = _FALLBACK_RETAIL_PRICE_PHP) -> str:
    price_anchor = (
        f"IMPORTANT: The current DOE-published retail gasoline price in the Philippines "
        f"is approximately ₱{current_price:.2f}/L (unleaded 95). "
        f"Your ESTIMATE must be a realistic price CHANGE from this baseline — "
        f"typical weekly adjustments are ±₱0.20 to ±₱3.00/L. "
        f"Do NOT output the absolute price; output only the signed change. "
    )
    base = (
        f"You are analyzing fuel price dynamics specifically for the {region} region "
        f"of the Philippines. {price_anchor}"
    )
    if role == 'Forecaster':
        return (
            base +
            "Project the short-term retail gasoline price CHANGE for this region "
            "based on crude oil prices, forex, and regional demand patterns. "
            "End with exactly: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'DataExtractor':
        return (
            base +
            "Extract and highlight the most relevant economic data points for this region "
            "(infrastructure, income levels, freight costs, demand patterns). "
            "Reference specific peso values from the DOE bulletin context where available. "
            "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'Synthesizer':
        return (
            base +
            "Integrate all data and prior estimates into a coherent regional price view. "
            "Resolve contradictions between other agents' estimates. "
            "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'Critic':
        return (
            base +
            "Challenge the reasoning of other agents in your group. Identify flaws, "
            "unsupported claims, and biases. Give your own estimate, then rate each "
            "agent's reasoning quality using this exact format on separate lines: "
            "SCORE: <agent_name>: X  (1–10, no /10 suffix). "
            "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    if role == 'ConfidenceScorer':
        return (
            base +
            "Evaluate confidence in each agent's price estimate based on evidence "
            "quality and internal consistency. Give your own estimate, then assign "
            "confidence using this exact format on separate lines: "
            "CONFIDENCE: <agent_name>: 0.XX  (0.0–1.0). "
            "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
    return base + "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"


def build_swarm_agents(current_price: float = _FALLBACK_RETAIL_PRICE_PHP) -> list[SwarmAgent]:
    """Build all 20 SwarmAgents (4 groups × 5 agents = 1 per role per group)."""
    agents: list[SwarmAgent] = []
    for group_id, region in enumerate(REGIONS):
        for role in _ROLE_ORDER:
            agents.append(SwarmAgent(
                name=f"{region} {role}",
                role=role,
                tier=_ROLE_TIERS[role],
                group_id=group_id,
                region_name=region,
                system_prompt=_make_system_prompt(role, region, current_price),
                rag_sources=_ROLE_RAG[role],
            ))
    return agents


# ── Scoring utilities ─────────────────────────────────────────────────────────

def _parse_scores(text: str, agent_names: list[str]) -> dict[str, float]:
    """Parse 'SCORE: <name>: X' lines. Missing agents default to 0.5."""
    result = {}
    for name in agent_names:
        m = re.search(rf'SCORE:\s*{re.escape(name)}:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
        result[name] = min(float(m.group(1)), 10.0) / 10.0 if m else 0.5
    return result


def _parse_confidence(text: str, agent_names: list[str]) -> dict[str, float]:
    """Parse 'CONFIDENCE: <name>: 0.XX' lines. Missing agents default to 0.5."""
    result = {}
    for name in agent_names:
        m = re.search(
            rf'CONFIDENCE:\s*{re.escape(name)}:\s*(0?\.\d+|1\.0+)',
            text, re.IGNORECASE,
        )
        result[name] = float(m.group(1)) if m else 0.5
    return result


def compute_combined_score(
    response: AgentResponse,
    critic_score: float,
    confidence: float,
    group_estimates: list[float],
) -> float:
    """
    combined = 0.4 × critic_score + 0.6 × (confidence × (1 − deviation_normalized))
    Returns 0.0 if response.price_estimate is None.
    """
    if response.price_estimate is None:
        return 0.0
    if len(group_estimates) < 2:
        deviation_norm = 0.0
    else:
        est_range = max(group_estimates) - min(group_estimates)
        median = sorted(group_estimates)[len(group_estimates) // 2]
        deviation_norm = (abs(response.price_estimate - median) / est_range
                          if est_range > 0 else 0.0)
    return 0.4 * critic_score + 0.6 * (confidence * (1.0 - deviation_norm))


def eliminate_bottom_n(
    agents: list[SwarmAgent], n: int
) -> tuple[list[SwarmAgent], list[SwarmAgent]]:
    """Sort by combined_score ascending; remove bottom n. Returns (survivors, eliminated)."""
    sorted_agents = sorted(agents, key=lambda a: a.combined_score)
    eliminated = sorted_agents[:n]
    survivors = sorted_agents[n:]
    for e in eliminated:
        e.is_alive = False
    return survivors, eliminated


# ── GroupArena ────────────────────────────────────────────────────────────────

# Rounds: (round_number, agents_to_eliminate_this_round)
# 5 agents → eliminate 2 → 3 left → eliminate 2 → 1 winner
_BRACKET = [(1, 2), (2, 2)]


class GroupArena:
    def __init__(
        self,
        group_id: int,
        agents: list[SwarmAgent],
        rag: RagEngine,
        scenario: dict,
        on_event: Optional[Callable] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        ml_baseline: str = '',
    ):
        self._group_id = group_id
        self._agents = agents          # 5 SwarmAgents, all is_alive=True
        self._rag = rag
        self._scenario = scenario
        self._on_event = on_event      # callable(event_type, *args)
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._history: list[AgentResponse] = []

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil price {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP rate {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand index {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _calibration_rule(self) -> str:
        anchor_text = self._ml_baseline or 'the ML anchor if supplied by the prompt'
        return (
            "\nCALIBRATION RULE:\n"
            f"- Treat {anchor_text} as the center of gravity for the forecast.\n"
            "- Your estimate should normally stay within +/-P1.00/L of the ML anchor.\n"
            "- You may leave that band only if you cite a specific DATA BRIEF figure "
            "or peer argument explaining why.\n"
            "- Do not output absolute pump prices. Output only the next price CHANGE.\n"
            f"- Any estimate outside +/-P{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L is invalid.\n"
        )

    def _reconciliation_rule(self) -> str:
        estimates = [
            r.price_estimate for r in self._history
            if _is_realistic_fuel_change(r.price_estimate)
        ]
        if not estimates:
            return ''
        median = statistics.median(estimates)
        low = min(estimates)
        high = max(estimates)
        return (
            "\nRECONCILIATION RULE:\n"
            f"- Prior valid estimate range: {low:+.2f} to {high:+.2f} P/L; "
            f"group median: {median:+.2f} P/L.\n"
            "- If your estimate differs from the group median by more than P1.00/L, "
            "revise toward the median or explicitly cite the reason for keeping the disagreement.\n"
            "- Prefer a calibrated consensus over a dramatic outlier unless the data clearly supports it.\n"
        )

    def _build_prompt(
        self,
        agent: SwarmAgent,
        round_num: int,
        round_responses: list[AgentResponse],
    ) -> list[dict]:
        scenario_text = self._scenario_text()
        chunks = self._rag.query(scenario_text, top_k=3, sources=agent.rag_sources)
        rag_text = '\n'.join(
            f"[{c['source']}] {c['text'][:200]}" for c in chunks
        ) or 'No context.'
        prior_rounds = '\n'.join(
            f"{r.agent_name} (Round {r.round_num}): {r.statement[:300]}"
            for r in self._history
        )
        this_round = '\n'.join(
            f"{r.agent_name}: {r.statement[:300]}"
            for r in round_responses
        )
        user_parts = [
            self._brief_block(),
            scenario_text,
            f"\nContext:\n{rag_text}",
            self._calibration_rule(),
        ]
        if self._ml_baseline:
            user_parts.append(f"\nML ANCHOR: {self._ml_baseline}\n"
                              "(Use this as a calibration anchor — debate around it, "
                              "not away from it without strong evidence. Estimates more than "
                              f"±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L are invalid absolute-price parses.)")
        if prior_rounds:
            user_parts.append(f"\nPrevious rounds:\n{prior_rounds}")
            user_parts.append(self._reconciliation_rule())
        if this_round:
            user_parts.append(f"\nThis round so far:\n{this_round}")
        user_parts.append(
            "\nYou MUST cite specific data from the DATA BRIEF when available. "
            "Give your analysis and end with BOTH lines:\n"
            "CAUSAL CHAIN: [scenario shock] -> [market effect] -> [retail mechanism] -> [consumer impact]\n"
            "ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
        )
        return [
            {'role': 'system', 'content': agent.system_prompt},
            {'role': 'user', 'content': ''.join(user_parts)},
        ]

    def _call_agent(self, agent: SwarmAgent, messages: list[dict]) -> AgentResponse:
        if self._on_event:
            self._on_event('agent_typing', self._group_id, agent.name)
        full_text = ''
        for token in llm.stream(messages, tier=agent.tier, max_tokens=_AGENT_MAX_TOKENS):
            full_text += token
        if self._on_event:
            self._on_event('agent_done_typing', self._group_id, agent.name)
        thinking, statement = _parse_think(full_text)
        return AgentResponse(
            agent_name=agent.name,
            round_num=0,
            thinking=thinking,
            statement=statement,
            price_estimate=_extract_fuel_change(statement),
        )

    def run(self) -> GroupSurvivor:
        alive = sorted(self._agents, key=lambda a: _ROLE_ORDER.index(a.role))

        for round_num, n_eliminate in _BRACKET:
            if round_num == 1:
                # Round 1: sequential so each agent can read peers' responses in order.
                # Critic and ConfidenceScorer react to Forecaster/Synthesizer/Extractor
                # — the sequential context is what makes the debate meaningful.
                round_responses: list[AgentResponse] = []
                for agent in alive:
                    messages = self._build_prompt(agent, round_num, round_responses)
                    resp = self._call_agent(agent, messages)
                    resp = AgentResponse(agent.name, round_num, resp.thinking,
                                         resp.statement, resp.price_estimate)
                    round_responses.append(resp)
                self._history.extend(round_responses)
            else:
                # Round 2+: agents already debated in Round 1; run in parallel.
                # Each agent sees the full Round 1 history via self._history.
                def _call_one(agent: SwarmAgent, rn: int = round_num) -> AgentResponse:
                    msgs = self._build_prompt(agent, rn, [])
                    resp = self._call_agent(agent, msgs)
                    return AgentResponse(agent.name, rn, resp.thinking,
                                         resp.statement, resp.price_estimate)

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(alive)) as pool:
                    futs = {pool.submit(_call_one, a): a for a in alive}
                    name_to_resp: dict[str, AgentResponse] = {}
                    for fut in concurrent.futures.as_completed(futs):
                        name_to_resp[futs[fut].name] = fut.result()

                round_responses = [name_to_resp[a.name] for a in alive]
                self._history.extend(round_responses)

            # Collect scores from Critic responses
            critic_responses = [
                r for r, a in zip(round_responses, alive) if a.role == 'Critic'
            ]
            alive_names = [a.name for a in alive]
            merged_critic_text = ' '.join(r.statement for r in critic_responses)
            critic_scores = _parse_scores(merged_critic_text, alive_names)

            # Collect confidence from ConfidenceScorer responses
            conf_responses = [
                r for r, a in zip(round_responses, alive) if a.role == 'ConfidenceScorer'
            ]
            merged_conf_text = ' '.join(r.statement for r in conf_responses)
            conf_scores = _parse_confidence(merged_conf_text, alive_names)

            # Collect group estimates for deviation calc
            group_estimates = [
                r.price_estimate for r in round_responses if r.price_estimate is not None
            ]

            # Assign combined scores
            for agent, resp in zip(alive, round_responses):
                agent.combined_score = compute_combined_score(
                    resp,
                    critic_score=critic_scores.get(agent.name, 0.5),
                    confidence=conf_scores.get(agent.name, 0.5),
                    group_estimates=group_estimates,
                )

            alive, eliminated = eliminate_bottom_n(alive, n=n_eliminate)

            for e in eliminated:
                if self._on_event:
                    self._on_event('eliminated', self._group_id, e.name,
                                   e.combined_score, round_num)

            if self._on_event:
                self._on_event('group_round_done', self._group_id, round_num,
                               list(round_responses))

        winner = alive[0]
        winner_resp = next(
            r for r in reversed(self._history) if r.agent_name == winner.name
        )
        survivor = GroupSurvivor(
            group_id=self._group_id,
            region_name=winner.region_name,
            response=winner_resp,
            combined_score=winner.combined_score,
            agent_role=winner.role,
            # Record the concrete model, not the tier: this is provenance for
            # the report, and 'fast' resolves differently per provider/config.
            agent_model=llm.describe_model(winner.tier),
        )
        if self._on_event:
            self._on_event('survivor', self._group_id, survivor)
        return survivor


# ── RegionalJudge ─────────────────────────────────────────────────────────────

class RegionalJudge:
    def __init__(
        self,
        judge_id: int,
        survivors: tuple[GroupSurvivor, GroupSurvivor],
        rag: RagEngine,
        scenario: dict,
        data_brief: Optional['LiveDataBrief'] = None,
    ):
        self._judge_id = judge_id
        self._s1, self._s2 = survivors
        self._rag = rag
        self._scenario = scenario
        self._data_brief = data_brief

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )

    def _defense_prompt(
        self, defender: GroupSurvivor, opponent: GroupSurvivor
    ) -> list[dict]:
        return [
            {'role': 'system', 'content': (
                f"You are a regional economic analyst representing the {defender.region_name} "
                "region. Defend your price estimate against your opponent's critique. "
                "Cite DATA BRIEF figures when available. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "Apply the project calibration policy: prefer estimates close to the group median unless a cited figure justifies disagreement. "
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{self._scenario_text()}\n\n"
                f"Your previous estimate: {defender.response.statement[:400]}\n\n"
                f"Opponent ({opponent.region_name}) argues: {opponent.response.statement[:400]}\n\n"
                "Defend your position or update your estimate based on their critique."
            )},
        ]

    def _synthesis_prompt(
        self, defense1: str, defense2: str
    ) -> list[dict]:
        return [
            {'role': 'system', 'content': (
                "You are a regional judge synthesizing two regional estimates into a "
                "single consensus. Weigh both defenses, resolve differences, and produce "
                "a final regional verdict. "
                "Cite DATA BRIEF figures when available. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "Apply the project reconciliation policy: prefer the calibrated midpoint unless a cited figure justifies a regional exception. "
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{self._scenario_text()}\n\n"
                f"{self._s1.region_name} defense: {defense1[:500]}\n\n"
                f"{self._s2.region_name} defense: {defense2[:500]}\n\n"
                "Produce the final regional consensus estimate."
            )},
        ]

    def _call(self, messages: list[dict], tier: str = _JUDGE_TIER) -> str:
        full = ''.join(llm.stream(messages, tier=tier, max_tokens=_JUDGE_MAX_TOKENS))
        _, statement = _parse_think(full)
        return statement

    def run(self) -> RegionalVerdict:
        def1 = self._call(self._defense_prompt(self._s1, self._s2))
        def2 = self._call(self._defense_prompt(self._s2, self._s1))
        synthesis = self._call(self._synthesis_prompt(def1, def2))
        estimate = _extract_fuel_change(synthesis)
        confidence = 0.75 if estimate is not None else 0.3
        return RegionalVerdict(
            judge_id=self._judge_id,
            region_pair=(self._s1.region_name, self._s2.region_name),
            estimate=estimate,
            confidence=confidence,
            reasoning=synthesis,
            survivor_names=(self._s1.response.agent_name, self._s2.response.agent_name),
        )


# ── MasterJudge ───────────────────────────────────────────────────────────────

_HIGH_WEIGHT_REGIONS = {'NCR'}


class MasterJudge:
    def __init__(
        self,
        verdicts: list[RegionalVerdict],
        rag: RagEngine,
        scenario: dict,
        survivors: Optional[list[GroupSurvivor]] = None,
        data_brief: Optional['LiveDataBrief'] = None,
    ):
        self._verdicts = verdicts
        self._rag = rag
        self._scenario = scenario
        self._survivors = survivors or []
        self._data_brief = data_brief

    def _brief_block(self) -> str:
        if self._data_brief is None:
            return ''
        try:
            return self._data_brief.as_prompt_block(self._scenario) + '\n\n'
        except Exception:
            return ''

    def _build_prompt(self) -> list[dict]:
        s = self._scenario
        scenario_text = (
            f"Current PH retail gasoline baseline: ₱{s.get('current_price', _FALLBACK_RETAIL_PRICE_PHP):.2f}/L. "
            f"AUTHORITATIVE SCENARIO SHOCK: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, demand {s.get('demand_index', 72):.0f}. "
            "Treat DATA BRIEF market history as calibration context, not as a replacement for this scenario."
        )
        verdicts_text = '\n\n'.join(
            f"[{'HIGH WEIGHT — ' if any(r in _HIGH_WEIGHT_REGIONS for r in v.region_pair) else ''}"
            f"{' & '.join(v.region_pair)}] "
            f"Estimate: {f'+₱{v.estimate:.2f}/L' if v.estimate is not None else 'N/A'} "
            f"(confidence {v.confidence:.2f})\n{v.reasoning[:400]}"
            for v in self._verdicts
        )
        return [
            {'role': 'system', 'content': (
                "You are the Master Judge synthesizing 2 regional Philippine fuel price "
                "estimates into a single national verdict. Give special weight to the "
                "NCR region as it represents the majority of fuel consumption. "
                "Cite DATA BRIEF figures when available. "
                "Identify any dissenting regions. "
                f"Ignore estimates outside ±{_MAX_REALISTIC_FUEL_CHANGE:.0f}/L as invalid absolute-price parses. "
                "Apply the project confidence policy: prefer calibrated, reconciled estimates over unreconciled outliers. "
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._brief_block()}{scenario_text}\n\nRegional verdicts:\n{verdicts_text}"
            )},
        ]

    def run(self) -> MasterVerdict:
        full = ''.join(
            llm.stream(self._build_prompt(), tier=_JUDGE_TIER,
                       max_tokens=_MASTER_MAX_TOKENS)
        )
        _, statement = _parse_think(full)
        final_estimate = _extract_fuel_change(statement)

        # Collect all estimates: group survivors + regional judges + master final
        all_estimates: list[float] = [
            s.response.price_estimate
            for s in self._survivors
            if _is_realistic_fuel_change(s.response.price_estimate)
        ]
        all_estimates += [v.estimate for v in self._verdicts if _is_realistic_fuel_change(v.estimate)]
        if _is_realistic_fuel_change(final_estimate):
            all_estimates.append(final_estimate)

        confidence_pct = _robust_confidence_pct(all_estimates, final_estimate)

        dissenting = [
            ' & '.join(v.region_pair)
            for v in self._verdicts
            if _is_realistic_fuel_change(v.estimate) and _is_realistic_fuel_change(final_estimate)
            and abs(v.estimate - final_estimate) > 0.50
        ]

        # Build per-region estimates using group survivor anchors
        anchor_estimates = {
            s.group_id: s.response.price_estimate
            for s in self._survivors
            if _is_realistic_fuel_change(s.response.price_estimate)
        }
        regional_estimates = derive_regional_estimates(final_estimate, anchor_estimates)

        return MasterVerdict(
            final_estimate=final_estimate,
            confidence_pct=confidence_pct,
            dissenting_regions=dissenting,
            reasoning=statement,
            regional_verdicts=self._verdicts,
            regional_estimates=regional_estimates,
        )


# ── SwarmOrchestrator ─────────────────────────────────────────────────────────

class SwarmOrchestrator:
    def __init__(
        self,
        rag: RagEngine,
        scenario: dict,
        parallel_n: int = 4,
        on_event: Optional[Callable] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        ml_baseline: str = '',
        evolved_agents: Optional[list] = None,
    ):
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._on_event = on_event
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents

    def run(self) -> MasterVerdict:
        live_price = fetch_live_retail_price()
        self._scenario = {**self._scenario, 'current_price': live_price}
        if self._evolved_agents is not None:
            all_agents = self._evolved_agents
        else:
            all_agents = build_swarm_agents(live_price)
        sem = threading.Semaphore(self._parallel_n)
        # Derived from the agents actually built, not a hardcoded 4: the group
        # count follows REGIONS, so a hardcoded literal silently drops any
        # extra region and leaves a None survivor if one is removed.
        n_groups = len({a.group_id for a in all_agents})
        survivors: list[Optional[GroupSurvivor]] = [None] * n_groups
        errors: list[str] = []
        lock = threading.Lock()
        all_arena_responses: list = []

        def run_group(group_id: int):
            with sem:
                group_agents = [a for a in all_agents if a.group_id == group_id]
                arena = GroupArena(
                    group_id=group_id,
                    agents=group_agents,
                    rag=self._rag,
                    scenario=self._scenario,
                    on_event=self._on_event,
                    data_brief=self._data_brief,
                    ml_baseline=self._ml_baseline,
                )
                try:
                    s = arena.run()
                    with lock:
                        survivors[group_id] = s
                        all_arena_responses.extend(arena._history)
                except Exception as e:
                    with lock:
                        errors.append(f"Group {group_id}: {e}")

        threads = [threading.Thread(target=run_group, args=(i,))
                   for i in range(n_groups)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Group errors: {'; '.join(errors)}")

        # Phase 2: regional judges (sequential)
        regional_verdicts: list[RegionalVerdict] = []
        for judge_id, (i, j) in enumerate(REGION_PAIRS):
            # A pair can reference a group that does not exist when REGIONS is
            # trimmed (e.g. during an ablation) — skip rather than IndexError.
            if i >= n_groups or j >= n_groups:
                continue
            s1, s2 = survivors[i], survivors[j]
            if s1 is None or s2 is None:
                continue
            judge = RegionalJudge(
                judge_id=judge_id,
                survivors=(s1, s2),
                rag=self._rag,
                scenario=self._scenario,
                data_brief=self._data_brief,
            )
            verdict = judge.run()
            regional_verdicts.append(verdict)
            if self._on_event:
                self._on_event('regional_done', judge_id, verdict)

        # Phase 3: master judge
        valid_survivors = [s for s in survivors if s is not None]
        master = MasterJudge(
            verdicts=regional_verdicts,
            rag=self._rag,
            scenario=self._scenario,
            survivors=valid_survivors,
            data_brief=self._data_brief,
        )
        mv = master.run()
        mv.all_responses = all_arena_responses
        return mv


# ── SwarmThread ───────────────────────────────────────────────────────────────

class SwarmThread(QThread):
    group_round_done  = pyqtSignal(int, int, object)
    group_eliminated  = pyqtSignal(int, str, float, int)
    group_survivor    = pyqtSignal(int, object)
    agent_typing      = pyqtSignal(int, str)   # group_id, agent_name
    agent_done_typing = pyqtSignal(int, str)   # group_id, agent_name
    regional_done     = pyqtSignal(int, object)
    swarm_complete    = pyqtSignal(object)
    error_occurred    = pyqtSignal(str)

    def __init__(self, rag: RagEngine, scenario: dict, parallel_n: int = 4,
                 data_brief: Optional['LiveDataBrief'] = None,
                 ml_baseline: str = '', evolved_agents=None, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents

    def run(self):
        # PyQt6 routes signals emitted from non-QThread Python threads through
        # Qt's queued connection mechanism automatically — cross-thread emit is safe.
        def on_event(event_type, *args):
            if event_type == 'eliminated':
                self.group_eliminated.emit(*args)
            elif event_type == 'survivor':
                self.group_survivor.emit(*args)
            elif event_type == 'regional_done':
                self.regional_done.emit(*args)
            elif event_type == 'group_round_done':
                self.group_round_done.emit(*args)
            elif event_type == 'agent_typing':
                self.agent_typing.emit(*args)
            elif event_type == 'agent_done_typing':
                self.agent_done_typing.emit(*args)

        orch = SwarmOrchestrator(
            rag=self._rag,
            scenario=self._scenario,
            parallel_n=self._parallel_n,
            on_event=on_event,
            data_brief=self._data_brief,
            ml_baseline=self._ml_baseline,
            evolved_agents=self._evolved_agents,
        )
        try:
            mv = orch.run()
            self.swarm_complete.emit(mv)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
