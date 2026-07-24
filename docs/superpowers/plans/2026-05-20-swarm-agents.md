# Hierarchical Swarm Agent System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 15-agent flat debate with a 3-tier hierarchical swarm — 20 regional groups × 10 agents with elimination rounds → 10 regional judges → 1 master judge — producing a `MasterVerdict` that feeds Stage 4/5.

**Architecture:** `GroupArena` runs a 3-round bracket (10→5→2→1) using a Critic+ConfidenceScorer composite score. `SwarmOrchestrator` runs up to N groups concurrently via `threading.Semaphore`, then runs regional judges and master judge sequentially. `SwarmThread(QThread)` wraps the orchestrator and emits signals for live UI updates. A new `Stage3SwarmPanel` shows the 3-tier canvas; `main_window.py` selects between old and swarm panels based on the "Swarm Mode" checkbox in Stage 2.

**Tech Stack:** Python 3.10, PyQt6, ollama (deepseek-r1:8b / qwen2.5:7b / llama3.2:3b / mistral:7b / phi4:14b), `threading`, `queue`, pytest, `unittest.mock`

**Read the spec before starting:** `docs/superpowers/specs/2026-05-20-swarm-agents-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ph_economic_ai/engine/swarm.py` | **Create** | All swarm logic: dataclasses, factory, scoring, GroupArena, RegionalJudge, MasterJudge, SwarmOrchestrator, SwarmThread |
| `ph_economic_ai/tests/test_swarm.py` | **Create** | Unit tests for scoring, bracket, arena, judges (ollama mocked) |
| `ph_economic_ai/ui/stage3_swarm_canvas.py` | **Create** | 3-tier swarm canvas panel (outer group clusters, middle regional ring, centre master) |
| `ph_economic_ai/ui/stage2_setup.py` | **Modify** | Add "Swarm Mode" checkbox + "Parallel Groups" slider; update `run_requested` signal |
| `ph_economic_ai/ui/stage4_report.py` | **Modify** | Add `populate_swarm(master_verdict, regressor, df, cv_rmse, scenario)` method |
| `ph_economic_ai/ui/stage5_interact.py` | **Modify** | Add `set_swarm_context(master_verdict, scenario)` |
| `ph_economic_ai/ui/main_window.py` | **Modify** | Wire swarm flow: show `Stage3SwarmPanel`, handle `swarm_complete`, feed Stage 4/5 |

---

## Task 1: Swarm Dataclasses + REGIONS Constant

**Files:**
- Create: `ph_economic_ai/engine/swarm.py`
- Test: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write the failing test**

```python
# ph_economic_ai/tests/test_swarm.py
import pytest
from ph_economic_ai.engine.swarm import (
    SwarmAgent, GroupSurvivor, RegionalVerdict, MasterVerdict, REGIONS
)


def test_regions_has_20_entries():
    assert len(REGIONS) == 20


def test_swarm_agent_defaults():
    agent = SwarmAgent(
        name='NCR Forecaster-1', role='Forecaster',
        model='deepseek-r1:8b', group_id=0, region_name='NCR',
        system_prompt='You are...', rag_sources=['YahooFinanceCrude'],
    )
    assert agent.is_alive is True
    assert agent.combined_score == 0.0


def test_regional_verdict_fields():
    rv = RegionalVerdict(
        judge_id=0, region_pair=('NCR', 'CAR'),
        estimate=1.5, confidence=0.8,
        reasoning='test', survivor_names=('NCR Forecaster-1', 'CAR Synthesizer-1'),
    )
    assert rv.estimate == 1.5


def test_master_verdict_fields():
    mv = MasterVerdict(
        final_estimate=2.0, confidence_pct=80,
        dissenting_regions=['BARMM'], reasoning='test', regional_verdicts=[],
    )
    assert mv.confidence_pct == 80
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest ph_economic_ai/tests/test_swarm.py -v
```
Expected: `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Create `ph_economic_ai/engine/swarm.py` with dataclasses + REGIONS**

```python
from __future__ import annotations

import re
import threading
import queue
from dataclasses import dataclass, field
from typing import Callable, Optional

import ollama
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import AgentResponse, _parse_think, _extract_price


# ── Region list ───────────────────────────────────────────────────────────────
REGIONS: list[str] = [
    'NCR', 'CAR', 'Region I — Ilocos', 'Region II — Cagayan Valley',
    'Region III — Central Luzon', 'Region IV-A — CALABARZON',
    'Region IV-B — MIMAROPA', 'Region V — Bicol',
    'Region VI — Western Visayas', 'Region VII — Central Visayas',
    'Region VIII — Eastern Visayas', 'Region IX — Zamboanga Peninsula',
    'Region X — Northern Mindanao', 'Region XI — Davao Region',
    'Region XII — SOCCSKSARGEN', 'Region XIII — Caraga', 'BARMM',
    'Mega Manila Corridor', 'Clark Economic Zone',
    'Cebu Metropolitan Economic Zone',
]

# Pairs: groups (0,1), (2,3), ... (18,19) → 10 regional judges
REGION_PAIRS: list[tuple[int, int]] = [(i, i + 1) for i in range(0, 20, 2)]

# ── Model assignments ─────────────────────────────────────────────────────────
_ROLE_MODELS: dict[str, str] = {
    'Forecaster':        'deepseek-r1:8b',
    'DataExtractor':     'qwen2.5:7b',
    'Synthesizer':       'llama3.2:3b',
    'Critic':            'mistral:7b',
    'ConfidenceScorer':  'phi4:14b',
}
_JUDGE_MODEL = 'deepseek-r1:8b'

# Role processing order within a round (Critic and ConfidenceScorer last so they
# can score agents they've already seen)
_ROLE_ORDER = ['Forecaster', 'DataExtractor', 'Synthesizer', 'Critic', 'ConfidenceScorer']


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class SwarmAgent:
    name: str
    role: str            # 'Forecaster' | 'DataExtractor' | 'Synthesizer' | 'Critic' | 'ConfidenceScorer'
    model: str
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
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest ph_economic_ai/tests/test_swarm.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add swarm dataclasses, REGIONS constant, test scaffold"
```

---

## Task 2: Agent Factory

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py` (add `build_swarm_agents` + system prompt templates)
- Modify: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write the failing test**

```python
# Add to ph_economic_ai/tests/test_swarm.py
from ph_economic_ai.engine.swarm import build_swarm_agents


def test_build_swarm_agents_count():
    agents = build_swarm_agents()
    assert len(agents) == 200  # 20 groups × 10 agents


def test_build_swarm_agents_group_composition():
    agents = build_swarm_agents()
    group_0 = [a for a in agents if a.group_id == 0]
    assert len(group_0) == 10
    roles = [a.role for a in group_0]
    assert roles.count('Forecaster') == 2
    assert roles.count('DataExtractor') == 2
    assert roles.count('Synthesizer') == 2
    assert roles.count('Critic') == 2
    assert roles.count('ConfidenceScorer') == 2


def test_build_swarm_agents_models():
    agents = build_swarm_agents()
    forecasters = [a for a in agents if a.role == 'Forecaster']
    assert all(a.model == 'deepseek-r1:8b' for a in forecasters)
    critics = [a for a in agents if a.role == 'Critic']
    assert all(a.model == 'mistral:7b' for a in critics)


def test_build_swarm_agents_names_unique():
    agents = build_swarm_agents()
    names = [a.name for a in agents]
    assert len(names) == len(set(names))
```

- [ ] **Step 2: Run to verify failure**

```
pytest ph_economic_ai/tests/test_swarm.py::test_build_swarm_agents_count -v
```
Expected: `ImportError: cannot import name 'build_swarm_agents'`

- [ ] **Step 3: Add `build_swarm_agents` + system prompt templates to `swarm.py`**

Add after the dataclasses section:

```python
# ── RAG source assignments per role ───────────────────────────────────────────
_ROLE_RAG: dict[str, list[str]] = {
    'Forecaster':       ['YahooFinanceCrude', 'YahooFinanceForex'],
    'DataExtractor':    ['YahooFinanceCrude', 'ManilaBulletin'],
    'Synthesizer':      ['neda_2024_2026', 'YahooFinanceForex'],
    'Critic':           ['BusinessWorld', 'ManilaBulletin'],
    'ConfidenceScorer': ['neda_2024_2026', 'BusinessWorld'],
}


def _make_system_prompt(role: str, region: str) -> str:
    base = f"You are analyzing fuel price dynamics specifically for the {region} region of the Philippines. "
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


def build_swarm_agents() -> list[SwarmAgent]:
    """Build all 200 SwarmAgents (20 groups × 10 agents = 2 per role per group)."""
    agents: list[SwarmAgent] = []
    for group_id, region in enumerate(REGIONS):
        for role in _ROLE_ORDER:
            model = _ROLE_MODELS[role]
            for idx in range(1, 3):  # agent 1 and 2 of each role
                name = f"{region} {role}-{idx}"
                agents.append(SwarmAgent(
                    name=name,
                    role=role,
                    model=model,
                    group_id=group_id,
                    region_name=region,
                    system_prompt=_make_system_prompt(role, region),
                    rag_sources=_ROLE_RAG[role],
                ))
    return agents
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_swarm.py -k "build_swarm" -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add build_swarm_agents factory with per-role prompts and model assignments"
```

---

## Task 3: Scoring Utilities

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py`
- Modify: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to ph_economic_ai/tests/test_swarm.py
from ph_economic_ai.engine.swarm import (
    _parse_scores, _parse_confidence, compute_combined_score, eliminate_bottom_n
)
from ph_economic_ai.engine.debate import AgentResponse


def _make_response(name, estimate, round_num=1):
    return AgentResponse(
        agent_name=name, round_num=round_num,
        thinking='', statement=f'Analysis. ESTIMATE: +₱{estimate:.2f}/L',
        price_estimate=float(estimate),
    )


def test_parse_scores_extracts_lines():
    text = "Good work.\nSCORE: Alice: 8\nSCORE: Bob: 4\nSCORE: Charlie: 10"
    scores = _parse_scores(text, ['Alice', 'Bob', 'Charlie'])
    assert scores == {'Alice': 0.8, 'Bob': 0.4, 'Charlie': 1.0}


def test_parse_scores_missing_agent_defaults_to_half():
    scores = _parse_scores("SCORE: Alice: 7", ['Alice', 'Bob'])
    assert scores['Bob'] == 0.5


def test_parse_confidence_extracts_lines():
    text = "CONFIDENCE: Alice: 0.85\nCONFIDENCE: Bob: 0.40"
    confs = _parse_confidence(text, ['Alice', 'Bob'])
    assert confs == {'Alice': 0.85, 'Bob': 0.40}


def test_parse_confidence_missing_defaults_to_half():
    confs = _parse_confidence("", ['Alice'])
    assert confs['Alice'] == 0.5


def test_compute_combined_score_none_estimate_is_zero():
    resp = AgentResponse('x', 1, '', 'no estimate here', None)
    score = compute_combined_score(resp, critic_score=0.9, confidence=0.9,
                                   group_estimates=[1.0, 2.0, 3.0])
    assert score == 0.0


def test_compute_combined_score_at_median_gives_high_score():
    resp = _make_response('x', 2.0)
    score = compute_combined_score(resp, critic_score=0.8, confidence=0.9,
                                   group_estimates=[1.0, 2.0, 3.0])
    # deviation_normalized = 0.0 → confidence component = 0.9
    assert score == pytest.approx(0.4 * 0.8 + 0.6 * 0.9, rel=1e-3)


def test_eliminate_bottom_n_removes_lowest_scorers():
    agents = [
        SwarmAgent('A', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.9),
        SwarmAgent('B', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.1),
        SwarmAgent('C', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.5),
    ]
    survivors, eliminated = eliminate_bottom_n(agents, n=1)
    assert len(survivors) == 2
    assert eliminated[0].name == 'B'
```

- [ ] **Step 2: Run to verify failure**

```
pytest ph_economic_ai/tests/test_swarm.py -k "parse_scores or parse_confidence or combined_score or eliminate" -v
```
Expected: `ImportError` for new names.

- [ ] **Step 3: Add scoring utilities to `swarm.py`**

Add after `build_swarm_agents`:

```python
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
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_swarm.py -k "parse_scores or parse_confidence or combined_score or eliminate" -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add swarm scoring utilities — parse_scores, parse_confidence, compute_combined_score, eliminate_bottom_n"
```

---

## Task 4: GroupArena

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py`
- Modify: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to ph_economic_ai/tests/test_swarm.py
from unittest.mock import patch, MagicMock
from ph_economic_ai.engine.swarm import GroupArena, build_swarm_agents


def _stream(text: str):
    """Returns a one-chunk ollama stream."""
    return [{'message': {'content': text}}]


def _make_rag():
    rag = MagicMock()
    rag.query.return_value = []
    return rag


SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}


def _build_arena(group_id=0):
    all_agents = build_swarm_agents()
    group_agents = [a for a in all_agents if a.group_id == group_id]
    return GroupArena(
        group_id=group_id,
        agents=group_agents,
        rag=_make_rag(),
        scenario=SCENARIO,
    )


def test_group_arena_run_returns_one_survivor():
    arena = _build_arena()
    critic_text = (
        "\n".join(f"SCORE: {a.name}: 7" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    conf_text = (
        "\n".join(f"CONFIDENCE: {a.name}: 0.70" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    normal_text = "Analysis here. ESTIMATE: +₱1.50/L"

    def fake_chat(model, messages, stream, **kwargs):
        if 'mistral' in model:
            return _stream(critic_text)
        if 'phi4' in model:
            return _stream(conf_text)
        return _stream(normal_text)

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        survivor = arena.run()

    assert isinstance(survivor, GroupSurvivor)
    assert survivor.group_id == 0


def test_group_arena_elimination_events_fired():
    arena = _build_arena()
    events = []

    def on_event(event_type, *args):
        events.append((event_type, *args))

    arena._on_event = on_event

    critic_text = (
        "\n".join(f"SCORE: {a.name}: 5" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    conf_text = (
        "\n".join(f"CONFIDENCE: {a.name}: 0.50" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    normal_text = "ESTIMATE: +₱2.00/L"

    def fake_chat(model, messages, stream, **kwargs):
        if 'mistral' in model:
            return _stream(critic_text)
        if 'phi4' in model:
            return _stream(conf_text)
        return _stream(normal_text)

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        arena.run()

    eliminated_events = [e for e in events if e[0] == 'eliminated']
    assert len(eliminated_events) == 9  # 5 + 3 + 1 total eliminations
```

- [ ] **Step 2: Run to verify failure**

```
pytest ph_economic_ai/tests/test_swarm.py -k "group_arena" -v
```
Expected: `ImportError` — `GroupArena` not defined.

- [ ] **Step 3: Add GroupArena to `swarm.py`**

Add after the scoring utilities:

```python
# ── Ollama helper ─────────────────────────────────────────────────────────────

def _ollama_extras(model: str) -> dict:
    """think=False only for deepseek models; other models ignore or error on it."""
    return {'think': False} if 'deepseek' in model else {}


# ── GroupArena ────────────────────────────────────────────────────────────────

# Rounds: (round_number, agents_to_eliminate_this_round)
_BRACKET = [(1, 5), (2, 3), (3, 1)]


class GroupArena:
    def __init__(
        self,
        group_id: int,
        agents: list[SwarmAgent],
        rag: RagEngine,
        scenario: dict,
        on_event: Optional[Callable] = None,
    ):
        self._group_id = group_id
        self._agents = agents          # 10 SwarmAgents, all is_alive=True
        self._rag = rag
        self._scenario = scenario
        self._on_event = on_event      # callable(event_type, *args)
        self._history: list[AgentResponse] = []

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Scenario: oil price {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP rate {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand index {s.get('demand_index', 72):.0f}."
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
        user_parts = [scenario_text, f"\nContext:\n{rag_text}"]
        if prior_rounds:
            user_parts.append(f"\nPrevious rounds:\n{prior_rounds}")
        if this_round:
            user_parts.append(f"\nThis round so far:\n{this_round}")
        user_parts.append("\nGive your analysis and end with ESTIMATE: +₱X.XX/L or -₱X.XX/L")
        return [
            {'role': 'system', 'content': agent.system_prompt},
            {'role': 'user', 'content': ''.join(user_parts)},
        ]

    def _call_agent(self, agent: SwarmAgent, messages: list[dict]) -> AgentResponse:
        full_text = ''
        stream = ollama.chat(
            model=agent.model,
            messages=messages,
            stream=True,
            **_ollama_extras(agent.model),
        )
        for chunk in stream:
            full_text += chunk['message']['content']
        thinking, statement = _parse_think(full_text)
        return AgentResponse(
            agent_name=agent.name,
            round_num=0,   # set by caller
            thinking=thinking,
            statement=statement,
            price_estimate=_extract_price(statement),
        )

    def run(self) -> 'GroupSurvivor':
        alive = sorted(self._agents, key=lambda a: _ROLE_ORDER.index(a.role))

        for round_num, n_eliminate in _BRACKET:
            round_responses: list[AgentResponse] = []

            for agent in alive:
                messages = self._build_prompt(agent, round_num, round_responses)
                resp = self._call_agent(agent, messages)
                resp = AgentResponse(agent.name, round_num, resp.thinking,
                                     resp.statement, resp.price_estimate)
                round_responses.append(resp)
                self._history.append(resp)

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
            agent_model=winner.model,
        )
        if self._on_event:
            self._on_event('survivor', self._group_id, survivor)
        return survivor
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_swarm.py -k "group_arena" -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add GroupArena with 3-round elimination bracket"
```

---

## Task 5: RegionalJudge + MasterJudge

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py`
- Modify: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write failing tests**

```python
# Add to ph_economic_ai/tests/test_swarm.py
from ph_economic_ai.engine.swarm import RegionalJudge, MasterJudge


def _make_survivor(group_id, region, estimate_val):
    resp = AgentResponse(
        agent_name=f'{region} Forecaster-1', round_num=3,
        thinking='', statement=f'ESTIMATE: +₱{estimate_val:.2f}/L',
        price_estimate=float(estimate_val),
    )
    return GroupSurvivor(
        group_id=group_id, region_name=region, response=resp,
        combined_score=0.8, agent_role='Forecaster', agent_model='deepseek-r1:8b',
    )


def test_regional_judge_returns_verdict():
    s1 = _make_survivor(0, 'NCR', 1.5)
    s2 = _make_survivor(1, 'CAR', 2.0)
    judge = RegionalJudge(judge_id=0, survivors=(s1, s2), rag=_make_rag(),
                          scenario=SCENARIO)

    with patch('ph_economic_ai.engine.swarm.ollama.chat',
               return_value=_stream('Good analysis. ESTIMATE: +₱1.75/L')):
        verdict = judge.run()

    assert isinstance(verdict, RegionalVerdict)
    assert verdict.judge_id == 0
    assert verdict.estimate == pytest.approx(1.75)
    assert verdict.region_pair == ('NCR', 'CAR')


def test_master_judge_returns_master_verdict():
    verdicts = [
        RegionalVerdict(i, (f'R{2*i}', f'R{2*i+1}'), 1.5 + i * 0.1,
                        0.8, 'ok', (f'a{i}', f'b{i}'))
        for i in range(10)
    ]
    master = MasterJudge(verdicts=verdicts, rag=_make_rag(), scenario=SCENARIO)

    with patch('ph_economic_ai.engine.swarm.ollama.chat',
               return_value=_stream(
                   'Final analysis. ESTIMATE: +₱1.80/L\n'
                   'Dissenting: Region IX — Zamboanga Peninsula'
               )):
        mv = master.run()

    assert isinstance(mv, MasterVerdict)
    assert mv.final_estimate == pytest.approx(1.80)
    assert mv.confidence_pct >= 0
```

- [ ] **Step 2: Run to verify failure**

```
pytest ph_economic_ai/tests/test_swarm.py -k "regional_judge or master_judge" -v
```
Expected: `ImportError`.

- [ ] **Step 3: Add RegionalJudge + MasterJudge to `swarm.py`**

```python
# ── RegionalJudge ─────────────────────────────────────────────────────────────

class RegionalJudge:
    def __init__(
        self,
        judge_id: int,
        survivors: tuple[GroupSurvivor, GroupSurvivor],
        rag: RagEngine,
        scenario: dict,
    ):
        self._judge_id = judge_id
        self._s1, self._s2 = survivors
        self._rag = rag
        self._scenario = scenario

    def _scenario_text(self) -> str:
        s = self._scenario
        return (
            f"Scenario: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, "
            f"demand {s.get('demand_index', 72):.0f}."
        )

    def _defense_prompt(
        self, defender: GroupSurvivor, opponent: GroupSurvivor
    ) -> list[dict]:
        return [
            {'role': 'system', 'content': (
                f"You are a regional economic analyst representing the {defender.region_name} "
                "region. Defend your price estimate against your opponent's critique. "
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._scenario_text()}\n\n"
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
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': (
                f"{self._scenario_text()}\n\n"
                f"{self._s1.region_name} defense: {defense1[:500]}\n\n"
                f"{self._s2.region_name} defense: {defense2[:500]}\n\n"
                "Produce the final regional consensus estimate."
            )},
        ]

    def _call(self, messages: list[dict], model: str = _JUDGE_MODEL) -> str:
        full = ''
        for chunk in ollama.chat(model=model, messages=messages,
                                 stream=True, **_ollama_extras(model)):
            full += chunk['message']['content']
        _, statement = _parse_think(full)
        return statement

    def run(self) -> RegionalVerdict:
        def1 = self._call(self._defense_prompt(self._s1, self._s2))
        def2 = self._call(self._defense_prompt(self._s2, self._s1))
        synthesis = self._call(self._synthesis_prompt(def1, def2))
        estimate = _extract_price(synthesis)
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

_HIGH_WEIGHT_REGIONS = {'NCR', 'Mega Manila Corridor'}


class MasterJudge:
    def __init__(self, verdicts: list[RegionalVerdict], rag: RagEngine, scenario: dict):
        self._verdicts = verdicts
        self._rag = rag
        self._scenario = scenario

    def _build_prompt(self) -> list[dict]:
        s = self._scenario
        scenario_text = (
            f"Scenario: oil {s.get('oil_pct', 0):+.1f}%, "
            f"USD/PHP {s.get('usd_pct', 0):+.1f}%, "
            f"BSP {s.get('bsp_rate', 6.5):.2f}%, demand {s.get('demand_index', 72):.0f}."
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
                "You are the Master Judge synthesizing 10 regional Philippine fuel price "
                "estimates into a single national verdict. Give special weight to the "
                "NCR and Mega Manila Corridor regions as they represent the majority of "
                "fuel consumption. Identify any dissenting regions. "
                "End with: ESTIMATE: +₱X.XX/L or ESTIMATE: -₱X.XX/L"
            )},
            {'role': 'user', 'content': f"{scenario_text}\n\nRegional verdicts:\n{verdicts_text}"},
        ]

    def run(self) -> MasterVerdict:
        full = ''
        model = _JUDGE_MODEL
        for chunk in ollama.chat(model=model, messages=self._build_prompt(),
                                 stream=True, **_ollama_extras(model)):
            full += chunk['message']['content']
        _, statement = _parse_think(full)
        final_estimate = _extract_price(statement)

        valid = [v.estimate for v in self._verdicts if v.estimate is not None]
        if valid and final_estimate is not None:
            within = sum(1 for e in valid if abs(e - final_estimate) <= 0.30)
            confidence_pct = int(within / len(valid) * 100)
        else:
            confidence_pct = 0

        dissenting = [
            ' & '.join(v.region_pair)
            for v in self._verdicts
            if v.estimate is not None and final_estimate is not None
            and abs(v.estimate - final_estimate) > 0.50
        ]

        return MasterVerdict(
            final_estimate=final_estimate,
            confidence_pct=confidence_pct,
            dissenting_regions=dissenting,
            reasoning=statement,
            regional_verdicts=self._verdicts,
        )
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_swarm.py -k "regional_judge or master_judge" -v
```
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add RegionalJudge and MasterJudge with debate + synthesis flow"
```

---

## Task 6: SwarmOrchestrator + SwarmThread

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py`
- Modify: `ph_economic_ai/tests/test_swarm.py`

- [ ] **Step 1: Write failing test**

```python
# Add to ph_economic_ai/tests/test_swarm.py
from ph_economic_ai.engine.swarm import SwarmOrchestrator


def test_swarm_orchestrator_returns_master_verdict():
    def fake_chat(model, messages, stream, **kwargs):
        if 'mistral' in model:
            all_agents = build_swarm_agents()
            names = [a.name for a in all_agents[:10]]
            text = '\n'.join(f'SCORE: {n}: 7' for n in names) + '\nESTIMATE: +₱1.50/L'
            return _stream(text)
        if 'phi4' in model:
            all_agents = build_swarm_agents()
            names = [a.name for a in all_agents[:10]]
            text = '\n'.join(f'CONFIDENCE: {n}: 0.70' for n in names) + '\nESTIMATE: +₱1.50/L'
            return _stream(text)
        return _stream('ESTIMATE: +₱1.50/L')

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        orch = SwarmOrchestrator(rag=_make_rag(), scenario=SCENARIO, parallel_n=2)
        mv = orch.run()

    assert isinstance(mv, MasterVerdict)
    assert mv.final_estimate is not None
```

- [ ] **Step 2: Run to verify failure**

```
pytest ph_economic_ai/tests/test_swarm.py::test_swarm_orchestrator_returns_master_verdict -v
```
Expected: `ImportError`.

- [ ] **Step 3: Add SwarmOrchestrator + SwarmThread to `swarm.py`**

```python
# ── SwarmOrchestrator ─────────────────────────────────────────────────────────

class SwarmOrchestrator:
    def __init__(
        self,
        rag: RagEngine,
        scenario: dict,
        parallel_n: int = 4,
        on_event: Optional[Callable] = None,
    ):
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._on_event = on_event

    def run(self) -> MasterVerdict:
        all_agents = build_swarm_agents()
        sem = threading.Semaphore(self._parallel_n)
        survivors: list[Optional[GroupSurvivor]] = [None] * 20
        errors: list[str] = []
        lock = threading.Lock()

        def run_group(group_id: int):
            with sem:
                group_agents = [a for a in all_agents if a.group_id == group_id]
                arena = GroupArena(
                    group_id=group_id,
                    agents=group_agents,
                    rag=self._rag,
                    scenario=self._scenario,
                    on_event=self._on_event,
                )
                try:
                    s = arena.run()
                    with lock:
                        survivors[group_id] = s
                except Exception as e:
                    with lock:
                        errors.append(f"Group {group_id}: {e}")

        threads = [threading.Thread(target=run_group, args=(i,), daemon=True)
                   for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            raise RuntimeError(f"Group errors: {'; '.join(errors)}")

        # Phase 2: regional judges (sequential)
        regional_verdicts: list[RegionalVerdict] = []
        for judge_id, (i, j) in enumerate(REGION_PAIRS):
            s1, s2 = survivors[i], survivors[j]
            if s1 is None or s2 is None:
                continue
            judge = RegionalJudge(
                judge_id=judge_id,
                survivors=(s1, s2),
                rag=self._rag,
                scenario=self._scenario,
            )
            verdict = judge.run()
            regional_verdicts.append(verdict)
            if self._on_event:
                self._on_event('regional_done', judge_id, verdict)

        # Phase 3: master judge
        master = MasterJudge(
            verdicts=regional_verdicts,
            rag=self._rag,
            scenario=self._scenario,
        )
        return master.run()


# ── SwarmThread ───────────────────────────────────────────────────────────────

class SwarmThread(QThread):
    group_round_done  = pyqtSignal(int, int, object)   # group_id, round_num, survivors_list
    group_eliminated  = pyqtSignal(int, str, float, int)  # group_id, name, score, round_num
    group_survivor    = pyqtSignal(int, object)           # group_id, GroupSurvivor
    regional_done     = pyqtSignal(int, object)           # judge_id, RegionalVerdict
    swarm_complete    = pyqtSignal(object)                # MasterVerdict
    error_occurred    = pyqtSignal(str)

    def __init__(self, rag: RagEngine, scenario: dict, parallel_n: int = 4, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n

    def run(self):
        def on_event(event_type, *args):
            if event_type == 'eliminated':
                self.group_eliminated.emit(*args)
            elif event_type == 'survivor':
                self.group_survivor.emit(*args)
            elif event_type == 'regional_done':
                self.regional_done.emit(*args)

        orch = SwarmOrchestrator(
            rag=self._rag,
            scenario=self._scenario,
            parallel_n=self._parallel_n,
            on_event=on_event,
        )
        try:
            mv = orch.run()
            self.swarm_complete.emit(mv)
        except Exception as e:
            self.error_occurred.emit(f"{type(e).__name__}: {e}")
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_swarm.py::test_swarm_orchestrator_returns_master_verdict -v
```
Expected: PASS (may take a moment with 20 threads even mocked).

- [ ] **Step 5: Run the full test suite**

```
pytest ph_economic_ai/tests/test_swarm.py -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add SwarmOrchestrator (semaphore-batched) and SwarmThread (Qt signals)"
```

---

## Task 7: Stage 2 UI — Swarm Controls

**Files:**
- Modify: `ph_economic_ai/ui/stage2_setup.py`

The current `run_requested` signal emits `(Scenario, list[Agent])`. Change it to emit `(Scenario, list[Agent], bool, int)` adding swarm_mode and parallel_n.

- [ ] **Step 1: Update the `run_requested` signal signature**

In `stage2_setup.py`, find:
```python
class Stage2SetupPanel(QWidget):
    run_requested = pyqtSignal(object, list)  # Scenario, list[Agent]
```
Replace with:
```python
class Stage2SetupPanel(QWidget):
    run_requested = pyqtSignal(object, list, bool, int)  # Scenario, list[Agent], swarm_mode, parallel_n
```

- [ ] **Step 2: Add swarm controls to `_build`**

In `stage2_setup.py`, find the section after `root.addLayout(pills_row)` (around line 157). Add the following directly after it:

```python
        # ── Swarm controls ────────────────────────────────────────────────────
        swarm_row = QHBoxLayout()
        swarm_row.setSpacing(12)

        self._swarm_checkbox = QPushButton('⬛  Swarm Mode')
        self._swarm_checkbox.setCheckable(True)
        self._swarm_checkbox.setChecked(True)
        self._swarm_checkbox.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:8px;'
            'padding:6px 14px;font-size:10px;font-weight:600;border:none;}'
            'QPushButton:checked{background:#6366F1;}'
            'QPushButton:hover{background:#374151;}'
        )
        self._swarm_checkbox.toggled.connect(self._on_swarm_toggled)

        parallel_lbl = QLabel('Parallel Groups')
        parallel_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        self._parallel_slider = QSlider(Qt.Orientation.Horizontal)
        self._parallel_slider.setRange(1, 8)
        self._parallel_slider.setValue(4)
        self._parallel_slider.setFixedWidth(120)
        self._parallel_slider.setStyleSheet(
            'QSlider::groove:horizontal{height:3px;background:#EAECF0;border-radius:2px;}'
            'QSlider::handle:horizontal{width:10px;height:10px;margin:-4px 0;'
            'border-radius:5px;background:#1C1E26;}'
            'QSlider::sub-page:horizontal{background:#1C1E26;border-radius:2px;}'
        )
        self._parallel_val_lbl = QLabel('4')
        self._parallel_val_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        self._parallel_slider.valueChanged.connect(
            lambda v: self._parallel_val_lbl.setText(str(v))
        )

        swarm_row.addWidget(self._swarm_checkbox)
        swarm_row.addSpacing(16)
        swarm_row.addWidget(parallel_lbl)
        swarm_row.addWidget(self._parallel_slider)
        swarm_row.addWidget(self._parallel_val_lbl)
        swarm_row.addStretch()
        root.addLayout(swarm_row)
```

Add import at the top of the file (already present: `QSlider`; add `QPushButton` if not already imported):
```python
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QSlider, QDialog, QLineEdit, QDialogButtonBox,
)
```

- [ ] **Step 3: Add `_on_swarm_toggled` and update `_on_run`**

Find the existing `_on_run` method (which calls `self.run_requested.emit(scenario, agents)`) and update it. Also add the toggle handler. Search for the `_on_run` method:

```python
# Find this pattern and replace:
    def _on_run(self):
        scenario = Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )
        self.run_requested.emit(scenario, self._agents)
```

Replace with:

```python
    def _on_swarm_toggled(self, checked: bool):
        self._parallel_slider.setEnabled(checked)
        self._parallel_val_lbl.setEnabled(checked)

    def _on_run(self):
        scenario = Scenario(
            oil_pct=self._pills[0].value,
            usd_pct=self._pills[1].value,
            bsp_rate=self._pills[2].value,
            demand_index=self._pills[3].value,
        )
        swarm_mode = self._swarm_checkbox.isChecked()
        parallel_n = self._parallel_slider.value()
        self.run_requested.emit(scenario, self._agents, swarm_mode, parallel_n)
```

- [ ] **Step 4: Verify the app still imports**

```
python -c "from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/stage2_setup.py
git commit -m "feat: add swarm mode checkbox and parallel groups slider to Stage 2"
```

---

## Task 8: Stage 3 Swarm Canvas Panel

**Files:**
- Create: `ph_economic_ai/ui/stage3_swarm_canvas.py`

This panel has the same external interface as `Stage3CanvasPanel` but shows the 3-tier swarm layout. It emits `swarm_complete(MasterVerdict)` instead of `simulation_complete`.

- [ ] **Step 1: Create the file**

```python
# ph_economic_ai/ui/stage3_swarm_canvas.py
import math
from collections import defaultdict

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsView, QGraphicsScene, QGraphicsObject,
    QSizePolicy, QScrollArea, QTextEdit, QGraphicsEllipseItem,
    QGraphicsTextItem,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.swarm import (
    SwarmThread, GroupSurvivor, RegionalVerdict, MasterVerdict, REGIONS, REGION_PAIRS
)

_BG         = '#F0F2F7'
_GROUP_CLR  = '#6366F1'
_JUDGE_CLR  = '#F59E0B'
_MASTER_CLR = '#10B981'
_DEAD_CLR   = '#CCCCCC'

_CANVAS_W = 900
_CANVAS_H = 900
_CX = _CANVAS_W / 2
_CY = _CANVAS_H / 2

_OUTER_R  = 360   # group cluster centres
_MIDDLE_R = 200   # regional judge centres
_AGENT_R  = 6     # small agent dot radius
_GROUP_R  = 30    # group cluster bounding radius
_JUDGE_R  = 18    # regional judge node radius
_MASTER_R = 28    # master judge radius


class Stage3SwarmPanel(QWidget):
    swarm_complete = pyqtSignal(object)   # MasterVerdict

    def __init__(self, rag: RagEngine, regressor, df, cv_rmse: float, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._regressor = regressor
        self._df = df
        self._cv_rmse = cv_rmse
        self._thread: SwarmThread | None = None
        self._scenario: dict = {}
        self._master_verdict: MasterVerdict | None = None
        # Track per-group alive agents (name → node item)
        self._agent_nodes: dict[int, dict[str, QGraphicsEllipseItem]] = defaultdict(dict)
        self._build()

    # ── Public interface ──────────────────────────────────────────────────────

    def start_swarm(self, scenario: dict, parallel_n: int = 4):
        self._scenario = scenario
        self._master_verdict = None
        self._reset_canvas()
        self._phase_lbl.setText('Phase 1 — Group Elimination · starting…')
        self._log.clear()
        self._thread = SwarmThread(self._rag, scenario, parallel_n, parent=self)
        self._thread.group_eliminated.connect(self._on_eliminated)
        self._thread.group_survivor.connect(self._on_survivor)
        self._thread.regional_done.connect(self._on_regional_done)
        self._thread.swarm_complete.connect(self._on_swarm_complete)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

    def verdict(self) -> MasterVerdict | None:
        return self._master_verdict

    def scenario(self) -> dict:
        return self._scenario

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Phase banner
        banner = QFrame()
        banner.setStyleSheet('background:#1C1E26;')
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(20, 8, 20, 8)
        self._phase_lbl = QLabel('Stage 3 — Swarm Debate')
        self._phase_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#FFFFFF;')
        banner_layout.addWidget(self._phase_lbl)
        banner_layout.addStretch()
        root.addWidget(banner)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, stretch=1)

        # Canvas
        self._scene = QGraphicsScene(0, 0, _CANVAS_W, _CANVAS_H)
        self._scene.setBackgroundBrush(QBrush(QColor(_BG)))
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setFrameShape(QFrame.Shape.NoFrame)
        body.addWidget(self._view, stretch=2)

        # Right panel: elimination log
        right = QFrame()
        right.setFixedWidth(280)
        right.setStyleSheet('background:#FFFFFF;border-left:1px solid #EAECF0;')
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(8)
        log_lbl = QLabel('ELIMINATION LOG')
        log_lbl.setStyleSheet('font-size:8px;font-weight:700;color:#9EA3AE;letter-spacing:0.7px;')
        right_layout.addWidget(log_lbl)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            'QTextEdit{background:#F7F8FA;border:1px solid #EAECF0;'
            'border-radius:8px;font-size:9px;color:#374151;}'
        )
        right_layout.addWidget(self._log, stretch=1)

        self._verdict_lbl = QLabel('')
        self._verdict_lbl.setWordWrap(True)
        self._verdict_lbl.setStyleSheet(
            'font-size:11px;font-weight:700;color:#10B981;'
            'background:#F0FDF4;border:1px solid #BBF7D0;'
            'border-radius:8px;padding:8px;'
        )
        self._verdict_lbl.hide()
        right_layout.addWidget(self._verdict_lbl)
        body.addWidget(right)

        self._draw_static_canvas()

    def _draw_static_canvas(self):
        """Draw the static 3-tier skeleton."""
        self._scene.clear()
        self._agent_nodes.clear()

        # Master node
        self._master_node = self._add_ellipse(
            _CX, _CY, _MASTER_R, _MASTER_CLR, opacity=0.4
        )
        t = QGraphicsTextItem('MASTER')
        t.setDefaultTextColor(QColor('#10B981'))
        t.setFont(QFont('Arial', 7, QFont.Weight.Bold))
        t.setPos(_CX - 20, _CY - 8)
        self._scene.addItem(t)

        # Regional judge nodes + group clusters
        self._judge_nodes: list[QGraphicsEllipseItem] = []
        for judge_id, (gi, gj) in enumerate(REGION_PAIRS):
            angle = math.radians(judge_id * 36 - 90)
            jx = _CX + _MIDDLE_R * math.cos(angle)
            jy = _CY + _MIDDLE_R * math.sin(angle)
            node = self._add_ellipse(jx, jy, _JUDGE_R, _JUDGE_CLR, opacity=0.3)
            self._judge_nodes.append(node)

            # Draw 2 group clusters around this judge
            for offset, group_id in enumerate([gi, gj]):
                cluster_angle = math.radians(judge_id * 36 + (offset * 2 - 1) * 18 - 90)
                cx = _CX + _OUTER_R * math.cos(cluster_angle)
                cy = _CY + _OUTER_R * math.sin(cluster_angle)
                self._draw_group_cluster(group_id, cx, cy)

    def _draw_group_cluster(self, group_id: int, cx: float, cy: float):
        """Draw 10 agent dots in a mini-ring around (cx, cy)."""
        region = REGIONS[group_id]
        # Mini label
        lbl = QGraphicsTextItem(region[:10])
        lbl.setDefaultTextColor(QColor('#9EA3AE'))
        lbl.setFont(QFont('Arial', 5))
        lbl.setPos(cx - 25, cy + _GROUP_R + 2)
        self._scene.addItem(lbl)

        colors = ['#6366F1', '#0EA5E9', '#F59E0B', '#10B981', '#EF4444',
                  '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#84CC16']
        for i in range(10):
            angle = math.radians(i * 36)
            ax = cx + (_GROUP_R - 8) * math.cos(angle)
            ay = cy + (_GROUP_R - 8) * math.sin(angle)
            node = self._add_ellipse(ax, ay, _AGENT_R, colors[i], opacity=1.0)
            # name placeholder — actual SwarmAgent name unknown at draw time; index by position
            self._agent_nodes[group_id][i] = node

    def _add_ellipse(self, cx, cy, r, color_hex, opacity=1.0) -> QGraphicsEllipseItem:
        item = self._scene.addEllipse(
            cx - r, cy - r, r * 2, r * 2,
            QPen(Qt.PenStyle.NoPen),
            QBrush(QColor(color_hex)),
        )
        item.setOpacity(opacity)
        return item

    def _reset_canvas(self):
        self._draw_static_canvas()

    # ── Signal handlers ───────────────────────────────────────────────────────

    def _on_eliminated(self, group_id: int, agent_name: str, score: float, round_num: int):
        self._log.append(
            f'<span style="color:#EF4444">✗</span> '
            f'<b>R{round_num}</b> {agent_name[:30]} '
            f'<span style="color:#9EA3AE">(score {score:.2f})</span>'
        )
        # Grey out one agent dot in the group cluster (first alive dot)
        nodes = self._agent_nodes.get(group_id, {})
        for idx, item in nodes.items():
            if item.opacity() > 0.5:
                item.setBrush(QBrush(QColor(_DEAD_CLR)))
                item.setOpacity(0.4)
                break

    def _on_survivor(self, group_id: int, survivor: GroupSurvivor):
        self._log.append(
            f'<span style="color:#10B981">✓</span> '
            f'<b>Survivor</b> {survivor.response.agent_name[:30]} '
            f'→ Regional Judge {group_id // 2}'
        )
        # Light up corresponding judge node
        judge_id = group_id // 2
        if judge_id < len(self._judge_nodes):
            self._judge_nodes[judge_id].setOpacity(0.8)
        # Update phase label
        survivors_so_far = self._log.toPlainText().count('Survivor')
        self._phase_lbl.setText(
            f'Phase 1 — Group Elimination · {survivors_so_far}/20 groups complete'
        )

    def _on_regional_done(self, judge_id: int, verdict: RegionalVerdict):
        est_str = f'+₱{verdict.estimate:.2f}/L' if verdict.estimate is not None else 'N/A'
        self._log.append(
            f'<span style="color:#F59E0B">⚖</span> '
            f'<b>Judge {judge_id}</b> {" & ".join(verdict.region_pair)} → {est_str}'
        )
        if judge_id < len(self._judge_nodes):
            self._judge_nodes[judge_id].setBrush(QBrush(QColor(_JUDGE_CLR)))
            self._judge_nodes[judge_id].setOpacity(1.0)
        self._phase_lbl.setText(f'Phase 2 — Regional Judges · judge {judge_id + 1}/10 done')

    def _on_swarm_complete(self, master_verdict: MasterVerdict):
        self._master_verdict = master_verdict
        self._master_node.setBrush(QBrush(QColor(_MASTER_CLR)))
        self._master_node.setOpacity(1.0)
        self._phase_lbl.setText('Phase 3 — Complete ✓')
        est_str = (f'+₱{master_verdict.final_estimate:.2f}/L'
                   if master_verdict.final_estimate is not None else 'N/A')
        self._verdict_lbl.setText(
            f'Master Verdict: {est_str}\n'
            f'Confidence: {master_verdict.confidence_pct}%\n'
            f'Dissenting: {", ".join(master_verdict.dissenting_regions) or "none"}'
        )
        self._verdict_lbl.show()
        self.swarm_complete.emit(master_verdict)

    def _on_error(self, msg: str):
        self._phase_lbl.setText(f'Error: {msg[:80]}')
        self._log.append(f'<span style="color:#EF4444">ERROR: {msg}</span>')
```

- [ ] **Step 2: Verify it imports**

```
python -c "from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/ui/stage3_swarm_canvas.py
git commit -m "feat: add Stage3SwarmPanel with 3-tier canvas, elimination log, phase banner"
```

---

## Task 9: Stage 4 Swarm Adapter

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py`

Add a `populate_swarm` method that accepts a `MasterVerdict` and reuses the existing chart rendering with swarm data.

- [ ] **Step 1: Read the current `populate` method signature**

Open `ph_economic_ai/ui/stage4_report.py` and locate `def populate(...)` (line 75). Note that it already accepts `responses, consensus, regressor, df, cv_rmse, scenario`.

- [ ] **Step 2: Add imports + `populate_swarm` to `stage4_report.py`**

At the top of `stage4_report.py`, add the import:
```python
from ph_economic_ai.engine.swarm import MasterVerdict
```

After the existing `populate` method, add:

```python
    def populate_swarm(self, master_verdict: MasterVerdict,
                       regressor, df, cv_rmse: float, scenario: dict):
        """Populate the report from a MasterVerdict (swarm mode)."""
        # Build a legacy-compatible consensus dict for existing chart code
        estimates = [
            v.estimate for v in master_verdict.regional_verdicts
            if v.estimate is not None
        ]
        consensus = {
            'weighted_avg': master_verdict.final_estimate,
            'low': min(estimates) if estimates else None,
            'high': max(estimates) if estimates else None,
            'confidence_pct': master_verdict.confidence_pct,
            'verdicts': [
                {
                    'agent': ' & '.join(v.region_pair),
                    'estimate': v.estimate,
                    'statement': v.reasoning,
                }
                for v in master_verdict.regional_verdicts
            ],
        }
        # Build synthetic AgentResponse list for existing per-agent charts
        from ph_economic_ai.engine.debate import AgentResponse
        responses = [
            AgentResponse(
                agent_name=' & '.join(v.region_pair),
                round_num=1,
                thinking='',
                statement=v.reasoning,
                price_estimate=v.estimate,
            )
            for v in master_verdict.regional_verdicts
        ]
        self.populate(responses, consensus, regressor, df, cv_rmse, scenario)
```

- [ ] **Step 3: Verify it imports**

```
python -c "from ph_economic_ai.ui.stage4_report import Stage4ReportPanel; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py
git commit -m "feat: add Stage4ReportPanel.populate_swarm for MasterVerdict compatibility"
```

---

## Task 10: Stage 5 Swarm Context + main_window.py Wiring

**Files:**
- Modify: `ph_economic_ai/ui/stage5_interact.py`
- Modify: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Add `set_swarm_context` to `stage5_interact.py`**

At the top of `stage5_interact.py`, add:
```python
from ph_economic_ai.engine.swarm import MasterVerdict
```

After `def set_debate_engine(self, engine: DebateEngine):` (line 200), add:

```python
    def set_swarm_context(self, master_verdict: MasterVerdict, scenario: dict):
        """Set context from a swarm run. Disables per-agent ask; shows summary."""
        self._last_scenario = scenario
        # Build a flat list of responses from regional verdicts for display
        from ph_economic_ai.engine.debate import AgentResponse
        responses = [
            AgentResponse(
                agent_name=' & '.join(v.region_pair),
                round_num=1,
                thinking='',
                statement=v.reasoning,
                price_estimate=v.estimate,
            )
            for v in master_verdict.regional_verdicts
        ]
        self.update_context(responses, scenario)
```

- [ ] **Step 2: Update `main_window.py`**

Replace the entire `main_window.py` with the updated version that handles both modes:

```python
from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import DEFAULT_AGENTS
from ph_economic_ai.engine.swarm import MasterVerdict
from ph_economic_ai.ui.sidebar import SidebarWidget
from ph_economic_ai.ui.stage1_rag import Stage1RagPanel
from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel
from ph_economic_ai.ui.stage3_canvas import Stage3CanvasPanel
from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
from ph_economic_ai.ui.stage5_interact import Stage5InteractPanel


class SimMainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._cv_rmse = cv_rmse
        self._rag = RagEngine()
        self._agents = list(DEFAULT_AGENTS)
        self._swarm_mode = False

        self.setWindowTitle('PH Economic Pressure Simulation Engine')
        self.setMinimumSize(1200, 720)
        self.setStyleSheet('background:#F7F8FA;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget()
        self._sidebar.stage_changed.connect(self._on_stage_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._stage1 = Stage1RagPanel(self._rag)
        self._stage2 = Stage2SetupPanel(self._agents)
        self._stage3_classic = Stage3CanvasPanel(self._rag, self._agents,
                                                  self._regressor, self._df, self._cv_rmse)
        self._stage3_swarm = Stage3SwarmPanel(self._rag, self._regressor,
                                               self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel()
        self._stage5 = Stage5InteractPanel(self._rag, self._agents,
                                            self._regressor, self._df, self._cv_rmse)

        # Stack indices: 0=stage1, 1=stage2, 2=stage3_classic, 3=stage3_swarm, 4=stage4, 5=stage5
        for w in (self._stage1, self._stage2, self._stage3_classic,
                  self._stage3_swarm, self._stage4, self._stage5):
            self._stack.addWidget(w)

        # Sidebar maps indices 0-4 → stack indices 0,1,2-or-3,4,5
        self._stage3_classic.simulation_complete.connect(self._on_classic_complete)
        self._stage3_swarm.swarm_complete.connect(self._on_swarm_complete)
        self._stage5.rerun_requested.connect(self._stage3_classic.start_simulation)
        self._stage2.run_requested.connect(self._on_run_requested)

        corpus_path = Path(__file__).parent.parent / 'assets' / 'corpus' / 'neda_2024_2026.txt'
        if corpus_path.exists():
            self._rag.add_text('neda_2024_2026', corpus_path.read_text(encoding='utf-8'))

    def _on_stage_changed(self, idx: int):
        # Sidebar emits 0-4; map idx=2 to whichever stage3 is active
        if idx == 2:
            self._stack.setCurrentIndex(3 if self._swarm_mode else 2)
        elif idx >= 3:
            self._stack.setCurrentIndex(idx + 1)   # shift past stage3_swarm slot
        else:
            self._stack.setCurrentIndex(idx)

    def _on_run_requested(self, scenario, agents, swarm_mode: bool, parallel_n: int):
        self._swarm_mode = swarm_mode
        self._sidebar.set_active(2)
        if swarm_mode:
            self._stack.setCurrentIndex(3)
            self._stage3_swarm.start_swarm(scenario.to_dict(), parallel_n)
        else:
            self._stack.setCurrentIndex(2)
            self._stage3_classic.start_simulation(scenario, agents)

    def _on_classic_complete(self, responses):
        consensus = self._stage3_classic.engine.consensus()
        self._stage4.populate(responses, consensus, self._regressor,
                               self._df, self._cv_rmse,
                               self._stage3_classic.scenario())
        self._stage5.update_context(responses, self._stage3_classic.scenario())
        self._stage5.set_debate_engine(self._stage3_classic.engine)
        self._sidebar.unlock_stages([2, 3, 4])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(4)

    def _on_swarm_complete(self, master_verdict: MasterVerdict):
        scenario = self._stage3_swarm.scenario()
        self._stage4.populate_swarm(master_verdict, self._regressor,
                                     self._df, self._cv_rmse, scenario)
        self._stage5.set_swarm_context(master_verdict, scenario)
        self._sidebar.unlock_stages([2, 3, 4])
        self._sidebar.set_active(3)
        self._stack.setCurrentIndex(4)
```

- [ ] **Step 3: Verify the app imports cleanly**

```
python -c "from ph_economic_ai.ui.main_window import SimMainWindow; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run all tests**

```
pytest ph_economic_ai/tests/ -v
```
Expected: all existing + new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/stage5_interact.py ph_economic_ai/ui/main_window.py
git commit -m "feat: wire swarm mode end-to-end — Stage2→SwarmThread→Stage3Swarm→Stage4/5"
```

---

## Self-Review Checklist

- [x] **Spec §2** (20 groups × 10 agents, 5 models) — covered in Task 2 `build_swarm_agents`
- [x] **Spec §3** (3-round bracket, combined score formula) — covered in Tasks 3 + 4
- [x] **Spec §4** (semaphore concurrency, Qt signals) — covered in Task 6
- [x] **Spec §5** (RegionalJudge defense+synthesis, MasterJudge NCR weighting) — covered in Task 5
- [x] **Spec §6** (data structures) — covered in Task 1
- [x] **Spec §7** (3-tier canvas, elimination log, phase banner, parallel slider) — covered in Tasks 7 + 8
- [x] **Spec §8** (backward compat, swarm mode toggle) — covered in Tasks 7 + 10
- [x] **Type consistency** — `GroupSurvivor` defined in Task 1 and used in Task 4; `RegionalVerdict` defined in Task 1 and used in Task 5; `SwarmThread` signals match `Stage3SwarmPanel` handlers
- [x] **No placeholders** — every implementation step has exact code
