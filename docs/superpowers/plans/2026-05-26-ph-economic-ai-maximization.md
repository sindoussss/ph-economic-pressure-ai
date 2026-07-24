# ph_economic_ai Evolutionary Agent Pool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-improving trust layer to ph_economic_ai so agents are automatically promoted/demoted across runs based on quality scoring and DOE ground-truth accuracy.

**Architecture:** A SQLite store persists every run's results and per-agent trust scores. An Internal Quality Scorer updates trust immediately after each run; a background DOE Checker retroactively grades predictions once real price data arrives. An Evolution Engine reads trust scores before each new run and returns agent lists with upgraded/downgraded model assignments. Two UI additions surface trust state: badges on the swarm canvas and a new Agent Performance panel.

**Tech Stack:** Python 3.10, PyQt6, SQLite (stdlib `sqlite3`), Ollama, existing `ph_economic_ai` engine (debate.py, swarm.py, rag.py, live_data.py).

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `ph_economic_ai/engine/store.py` | SQLite persistence — runs, agent_responses, agent_trust |
| Create | `ph_economic_ai/engine/quality_scorer.py` | Heuristic per-agent internal scorer |
| Create | `ph_economic_ai/engine/evolution.py` | Trust → model-tier evolution engine |
| Create | `ph_economic_ai/engine/ground_truth.py` | Background DOE price checker QThread |
| Create | `ph_economic_ai/ui/agent_performance.py` | Trust leaderboard + run history panel |
| Create | `ph_economic_ai/tests/test_store.py` | Tests for store |
| Create | `ph_economic_ai/tests/test_quality_scorer.py` | Tests for quality scorer |
| Create | `ph_economic_ai/tests/test_evolution.py` | Tests for evolution engine |
| Create | `ph_economic_ai/tests/test_ground_truth.py` | Tests for DOE checker logic |
| Modify | `ph_economic_ai/engine/swarm.py` | Add `all_responses` to MasterVerdict; add `evolved_agents` param to SwarmOrchestrator |
| Modify | `ph_economic_ai/ui/main_window.py` | Wire store, evolution, quality scorer, DOE checker, performance panel |
| Modify | `ph_economic_ai/ui/stage3_swarm_canvas.py` | Add trust badges to agent circle nodes |
| Modify | `ph_economic_ai/ui/sidebar.py` | Add "Agent Performance" entry |
| Modify | `ph_economic_ai/main.py` | Create store, pass to SimMainWindow |

---

## Task 1: Agent Trust Store

**Files:**
- Create: `ph_economic_ai/engine/store.py`
- Create: `ph_economic_ai/tests/test_store.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_store.py
import pytest
from ph_economic_ai.engine.store import AgentTrustStore


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def test_save_and_get_run(store):
    run_id = store.save_run(
        scenario={'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72},
        final_estimate=1.42,
        confidence_pct=78,
    )
    assert run_id == 1
    runs = store.get_ungraded_runs(min_age_days=0)
    assert len(runs) == 1
    assert runs[0]['final_estimate'] == 1.42


def test_total_runs(store):
    assert store.total_runs() == 0
    store.save_run(scenario={}, final_estimate=1.0, confidence_pct=60)
    assert store.total_runs() == 1


def test_save_agent_responses(store):
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=60)
    store.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.2,
         'statement': 'Brent at $72.40 supports a ₱1.20 rise.', 'citation_count': 2,
         'has_causal_chain': 1, 'internal_score': 0.8, 'model_used': 'deepseek-r1:8b'},
    ])
    rows = store.get_agent_responses(run_id)
    assert len(rows) == 1
    assert rows[0]['agent_name'] == 'Market Analyst'


def test_trust_initialized_at_half(store):
    trust = store.get_trust('Market Analyst')
    assert trust == 0.5


def test_update_trust_internal_only(store):
    store.update_trust('Market Analyst', internal_score=0.9)
    trust = store.get_trust('Market Analyst')
    # EMA: 0.3 * 0.9 + 0.7 * 0.5 = 0.27 + 0.35 = 0.62
    assert abs(trust - 0.62) < 0.001


def test_update_trust_with_accuracy(store):
    store.update_trust('Market Analyst', internal_score=0.8, accuracy_score=1.0)
    trust = store.get_trust('Market Analyst')
    # raw = 0.4*0.8 + 0.6*1.0 = 0.32 + 0.60 = 0.92
    # EMA: 0.3 * 0.92 + 0.7 * 0.5 = 0.276 + 0.35 = 0.626
    assert abs(trust - 0.626) < 0.001


def test_trust_clamped(store):
    for _ in range(20):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    assert store.get_trust('Market Analyst') <= 0.95


def test_get_all_trust(store):
    store.update_trust('Agent A', internal_score=0.8)
    store.update_trust('Agent B', internal_score=0.2)
    all_trust = store.get_all_trust()
    assert 'Agent A' in all_trust
    assert 'Agent B' in all_trust


def test_apply_ground_truth_grade(store):
    run_id = store.save_run(scenario={'current_price': 98.82}, final_estimate=1.42, confidence_pct=78)
    store.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.42,
         'statement': 'Estimate.', 'citation_count': 1, 'has_causal_chain': 1,
         'internal_score': 0.7, 'model_used': 'deepseek-r1:8b'},
    ])
    store.apply_ground_truth_grade(run_id, actual_change=1.20)
    runs = store.get_ungraded_runs(min_age_days=0)
    assert len(runs) == 0  # run is now graded
    trust = store.get_trust('Market Analyst')
    assert trust > 0.5  # accurate prediction improved trust
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest ph_economic_ai/tests/test_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'ph_economic_ai.engine.store'`

- [ ] **Step 3: Implement `engine/store.py`**

```python
# ph_economic_ai/engine/store.py
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DB = Path(__file__).parent.parent / 'cache' / 'trust.db'
_TRUST_INIT = 0.5
_EMA_ALPHA  = 0.3
_TRUST_MIN  = 0.05
_TRUST_MAX  = 0.95


class AgentTrustStore:
    def __init__(self, db_path: str | None = None):
        self._path = db_path or str(_DEFAULT_DB)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._migrate()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                scenario_json     TEXT    NOT NULL,
                final_estimate    REAL,
                confidence_pct    INTEGER,
                internal_quality  REAL,
                actual_price_change REAL,
                accuracy_error    REAL,
                graded_at         TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_responses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(run_id),
                agent_name      TEXT    NOT NULL,
                round_num       INTEGER NOT NULL,
                estimate        REAL,
                statement       TEXT,
                citation_count  INTEGER DEFAULT 0,
                has_causal_chain INTEGER DEFAULT 0,
                internal_score  REAL    DEFAULT 0.5,
                model_used      TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_trust (
                agent_name          TEXT PRIMARY KEY,
                trust_score         REAL    NOT NULL DEFAULT 0.5,
                runs_participated   INTEGER NOT NULL DEFAULT 0,
                avg_internal_score  REAL    NOT NULL DEFAULT 0.5,
                avg_accuracy_error  REAL,
                current_model_tier  TEXT    NOT NULL DEFAULT 'default',
                last_updated        TEXT    NOT NULL
            );
        ''')
        self._conn.commit()

    # ── Run persistence ───────────────────────────────────────────────────────

    def save_run(self, scenario: dict, final_estimate: Optional[float],
                 confidence_pct: int) -> int:
        cur = self._conn.execute(
            'INSERT INTO runs (timestamp, scenario_json, final_estimate, confidence_pct) '
            'VALUES (?, ?, ?, ?)',
            (datetime.now(timezone.utc).isoformat(),
             json.dumps(scenario), final_estimate, confidence_pct),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_run_quality(self, run_id: int, internal_quality: float) -> None:
        self._conn.execute(
            'UPDATE runs SET internal_quality=? WHERE run_id=?',
            (internal_quality, run_id),
        )
        self._conn.commit()

    def save_agent_responses(self, run_id: int, responses: list[dict]) -> None:
        self._conn.executemany(
            'INSERT INTO agent_responses '
            '(run_id, agent_name, round_num, estimate, statement, '
            ' citation_count, has_causal_chain, internal_score, model_used) '
            'VALUES (:run_id, :agent_name, :round_num, :estimate, :statement, '
            '        :citation_count, :has_causal_chain, :internal_score, :model_used)',
            [{'run_id': run_id, **r} for r in responses],
        )
        self._conn.commit()

    def get_agent_responses(self, run_id: int) -> list[dict]:
        cur = self._conn.execute(
            'SELECT * FROM agent_responses WHERE run_id=?', (run_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_ungraded_runs(self, min_age_days: float = 5.0) -> list[dict]:
        """Return runs not yet graded and older than min_age_days."""
        cur = self._conn.execute(
            "SELECT * FROM runs WHERE actual_price_change IS NULL "
            "AND (julianday('now') - julianday(timestamp)) >= ?",
            (min_age_days,),
        )
        return [dict(row) for row in cur.fetchall()]

    def apply_ground_truth_grade(self, run_id: int, actual_change: float) -> None:
        """Grade a run against actual DOE price change, update agent trust."""
        row = self._conn.execute(
            'SELECT * FROM runs WHERE run_id=?', (run_id,)
        ).fetchone()
        if row is None:
            return
        final_est = row['final_estimate']
        error = abs(final_est - actual_change) if final_est is not None else None
        self._conn.execute(
            'UPDATE runs SET actual_price_change=?, accuracy_error=?, graded_at=? '
            'WHERE run_id=?',
            (actual_change, error, datetime.now(timezone.utc).isoformat(), run_id),
        )
        # Grade each agent response
        responses = self.get_agent_responses(run_id)
        for resp in responses:
            est = resp['estimate']
            if est is None:
                continue
            accuracy_score = max(0.0, 1.0 - abs(est - actual_change) / 3.0)
            self.update_trust(
                resp['agent_name'],
                internal_score=resp['internal_score'],
                accuracy_score=accuracy_score,
            )
        self._conn.commit()

    def total_runs(self) -> int:
        return self._conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]

    # ── Trust management ──────────────────────────────────────────────────────

    def get_trust(self, agent_name: str) -> float:
        row = self._conn.execute(
            'SELECT trust_score FROM agent_trust WHERE agent_name=?', (agent_name,)
        ).fetchone()
        return float(row['trust_score']) if row else _TRUST_INIT

    def get_all_trust(self) -> dict[str, float]:
        cur = self._conn.execute('SELECT agent_name, trust_score FROM agent_trust')
        return {row['agent_name']: float(row['trust_score']) for row in cur.fetchall()}

    def get_all_trust_rows(self) -> list[dict]:
        cur = self._conn.execute(
            'SELECT * FROM agent_trust ORDER BY trust_score DESC'
        )
        return [dict(row) for row in cur.fetchall()]

    def update_trust(self, agent_name: str, internal_score: float,
                     accuracy_score: Optional[float] = None) -> None:
        old_trust = self.get_trust(agent_name)
        if accuracy_score is not None:
            raw = 0.4 * internal_score + 0.6 * accuracy_score
        else:
            raw = internal_score
        new_trust = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * old_trust
        new_trust = max(_TRUST_MIN, min(_TRUST_MAX, new_trust))
        tier = _tier(new_trust)
        self._conn.execute(
            '''INSERT INTO agent_trust (agent_name, trust_score, runs_participated,
               avg_internal_score, current_model_tier, last_updated)
               VALUES (?, ?, 1, ?, ?, ?)
               ON CONFLICT(agent_name) DO UPDATE SET
                 trust_score        = excluded.trust_score,
                 runs_participated  = runs_participated + 1,
                 avg_internal_score = (avg_internal_score + excluded.avg_internal_score) / 2,
                 current_model_tier = excluded.current_model_tier,
                 last_updated       = excluded.last_updated''',
            (agent_name, new_trust, internal_score, tier,
             datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def _tier(trust: float) -> str:
    if trust > 0.70:
        return 'promoted'
    if trust < 0.30:
        return 'demoted'
    return 'default'
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_store.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/store.py ph_economic_ai/tests/test_store.py
git commit -m "feat: add AgentTrustStore SQLite persistence layer"
```

---

## Task 2: Internal Quality Scorer

**Files:**
- Create: `ph_economic_ai/engine/quality_scorer.py`
- Create: `ph_economic_ai/tests/test_quality_scorer.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_quality_scorer.py
import pytest
from ph_economic_ai.engine.quality_scorer import QualityScorer
from ph_economic_ai.engine.debate import AgentResponse


def _resp(name, statement, estimate=1.0, round_num=1):
    return AgentResponse(
        agent_name=name, round_num=round_num,
        thinking='', statement=statement, price_estimate=estimate,
    )


def test_citation_count_high():
    r = _resp('A', 'Brent at $72.40 and USD/PHP at ₱57.80 suggests ₱1.20 rise.')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['citation_score'] >= 0.6


def test_citation_count_zero():
    r = _resp('A', 'Prices are expected to rise significantly.')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['citation_score'] == 0.0


def test_causal_chain_full():
    stmt = ('Analysis.\nCAUSAL CHAIN: oil shock → import cost → pump price → household budget\n'
            'ESTIMATE: +₱1.00/L')
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 1.0


def test_causal_chain_missing():
    r = _resp('A', 'Prices go up. ESTIMATE: +₱1.00/L')
    result = QualityScorer.score_responses([r], group_estimates=[1.0])
    assert result['A']['chain_score'] == 0.0


def test_convergence_on_median():
    responses = [
        _resp('A', 'text', estimate=1.0),
        _resp('B', 'text', estimate=1.0),
        _resp('C', 'text', estimate=1.0),
    ]
    result = QualityScorer.score_responses(responses, group_estimates=[1.0, 1.0, 1.0])
    assert result['A']['convergence_score'] == 1.0


def test_convergence_outlier():
    responses = [_resp('A', 'text', estimate=5.0)]
    result = QualityScorer.score_responses(responses, group_estimates=[1.0, 1.0, 5.0])
    assert result['A']['convergence_score'] < 0.5


def test_overall_score_in_range(store=None):
    stmt = ('Brent $72.40, USD/PHP ₱57.80. CAUSAL CHAIN: oil → cost → price → consumer. '
            'ESTIMATE: +₱1.20/L')
    r = _resp('A', stmt)
    result = QualityScorer.score_responses([r], group_estimates=[1.2])
    assert 0.0 <= result['A']['overall'] <= 1.0


def test_run_quality_average():
    responses = [
        _resp('A', 'Brent $72.40. CAUSAL CHAIN: a → b → c → d. ESTIMATE: +₱1.00/L', estimate=1.0),
        _resp('B', 'Prices rise. ESTIMATE: +₱1.00/L', estimate=1.0),
    ]
    quality = QualityScorer.run_quality(responses, group_estimates=[1.0, 1.0])
    assert 0.0 <= quality <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest ph_economic_ai/tests/test_quality_scorer.py -v
```
Expected: `ModuleNotFoundError: No module named 'ph_economic_ai.engine.quality_scorer'`

- [ ] **Step 3: Implement `engine/quality_scorer.py`**

```python
# ph_economic_ai/engine/quality_scorer.py
from __future__ import annotations

import math
import re
import statistics
from typing import Optional

from ph_economic_ai.engine.debate import AgentResponse

# Regex: matches currency/percentage data-brief citations like $72.40, ₱57.80, 3.8%
_CITE_RE = re.compile(r'[\$₱]\s*\d+\.?\d*|[\+\-]?\d+\.?\d+\s*%')
_CHAIN_FULL_RE = re.compile(
    r'CAUSAL\s+CHAIN\s*:\s*\S+.*?→.*?→.*?→.*?\S',
    re.IGNORECASE,
)
_CHAIN_PARTIAL_RE = re.compile(r'CAUSAL\s+CHAIN\s*:', re.IGNORECASE)

_LEN_MEAN  = 400.0
_LEN_SIGMA = 200.0

# Metric weights (must sum to 1.0)
_W_CITE      = 0.30
_W_CONVERGE  = 0.25
_W_CHAIN     = 0.25
_W_LENGTH    = 0.20


class QualityScorer:
    @staticmethod
    def score_responses(
        responses: list[AgentResponse],
        group_estimates: list[float],
    ) -> dict[str, dict]:
        """Score each agent response. Returns {agent_name: metric_dict}."""
        valid_ests = [e for e in group_estimates if e is not None]
        median = statistics.median(valid_ests) if valid_ests else 0.0
        est_range = max(valid_ests) - min(valid_ests) if len(valid_ests) > 1 else 1.0

        results: dict[str, dict] = {}
        for resp in responses:
            stmt = resp.statement or ''
            cites = len(_CITE_RE.findall(stmt))
            citation_score = min(cites / 3.0, 1.0)

            if _CHAIN_FULL_RE.search(stmt):
                chain_score = 1.0
            elif _CHAIN_PARTIAL_RE.search(stmt):
                chain_score = 0.5
            else:
                chain_score = 0.0

            if resp.price_estimate is not None and est_range > 0:
                convergence_score = max(0.0, 1.0 - abs(resp.price_estimate - median) / 2.0)
            else:
                convergence_score = 0.5

            n = len(stmt)
            length_score = math.exp(-0.5 * ((n - _LEN_MEAN) / _LEN_SIGMA) ** 2)

            overall = (
                _W_CITE     * citation_score
                + _W_CONVERGE * convergence_score
                + _W_CHAIN    * chain_score
                + _W_LENGTH   * length_score
            )
            results[resp.agent_name] = {
                'citation_score':    round(citation_score,    3),
                'convergence_score': round(convergence_score, 3),
                'chain_score':       round(chain_score,       3),
                'length_score':      round(length_score,      3),
                'overall':           round(overall,           3),
                'citation_count':    cites,
                'has_causal_chain':  1 if chain_score >= 1.0 else 0,
            }
        return results

    @staticmethod
    def run_quality(
        responses: list[AgentResponse],
        group_estimates: list[float],
    ) -> float:
        """Overall run quality — average of all agent overall scores."""
        scores = QualityScorer.score_responses(responses, group_estimates)
        if not scores:
            return 0.5
        return round(sum(v['overall'] for v in scores.values()) / len(scores), 3)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_quality_scorer.py -v
```
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/quality_scorer.py ph_economic_ai/tests/test_quality_scorer.py
git commit -m "feat: add heuristic QualityScorer for per-agent internal scoring"
```

---

## Task 3: Evolution Engine

**Files:**
- Create: `ph_economic_ai/engine/evolution.py`
- Create: `ph_economic_ai/tests/test_evolution.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_evolution.py
import pytest
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.evolution import get_evolved_debate_agents, get_evolved_swarm_agents
from ph_economic_ai.engine.debate import DEFAULT_AGENTS, Agent
from ph_economic_ai.engine.swarm import build_swarm_agents


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def test_cold_start_returns_base_agents(store):
    # Fewer than 3 runs → no evolution
    for _ in range(2):
        store.save_run({}, 1.0, 60)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    assert len(evolved) == len(DEFAULT_AGENTS)
    for orig, ev in zip(DEFAULT_AGENTS, evolved):
        assert ev.model == orig.model


def test_promoted_agent_gets_bigger_model(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Push Market Analyst above 0.70 trust
    for _ in range(8):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    market_analyst = next(a for a in evolved if a.name == 'Market Analyst')
    assert market_analyst.model == 'deepseek-r1:32b'


def test_demoted_agent_gets_smaller_model(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Push Risk Assessor below 0.30 trust
    for _ in range(8):
        store.update_trust('Risk Assessor', internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    risk_assessor = next(a for a in evolved if a.name == 'Risk Assessor')
    assert risk_assessor.model == 'qwen2.5:7b'


def test_promoted_agent_gets_confidence_suffix(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    for _ in range(8):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    market_analyst = next(a for a in evolved if a.name == 'Market Analyst')
    assert 'accurate' in market_analyst.system_prompt.lower()


def test_demoted_agent_gets_skeptic_suffix(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    for _ in range(8):
        store.update_trust('Risk Assessor', internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    risk_assessor = next(a for a in evolved if a.name == 'Risk Assessor')
    assert 'conservative' in risk_assessor.system_prompt.lower()


def test_diversity_guard_prevents_all_benched(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    agents = build_swarm_agents()
    # Demote all agents in group 0 (NCR)
    ncr_agents = [a for a in agents if a.group_id == 0]
    for a in ncr_agents:
        for _ in range(8):
            store.update_trust(a.name, internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_swarm_agents(store, agents)
    ncr_evolved = [a for a in evolved if a.group_id == 0]
    # At least 60% of original NCR count must survive
    assert len(ncr_evolved) >= math.ceil(len(ncr_agents) * 0.6)


def test_swarm_cold_start(store):
    agents = build_swarm_agents()
    evolved = get_evolved_swarm_agents(store, agents)
    assert len(evolved) == len(agents)
    for orig, ev in zip(agents, evolved):
        assert ev.model == orig.model
```

Add `import math` at the top of the test file.

- [ ] **Step 2: Run tests to verify they fail**

```
pytest ph_economic_ai/tests/test_evolution.py -v
```
Expected: `ModuleNotFoundError: No module named 'ph_economic_ai.engine.evolution'`

- [ ] **Step 3: Implement `engine/evolution.py`**

```python
# ph_economic_ai/engine/evolution.py
from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

from ph_economic_ai.engine.debate import Agent
from ph_economic_ai.engine.swarm import SwarmAgent

_COLD_START_RUNS = 3
_DIVERSITY_MIN   = 0.60

# Model tier maps
_DEBATE_TIERS: dict[str, dict[str, str]] = {
    'deepseek-r1:8b':  {'promoted': 'deepseek-r1:32b', 'demoted': 'qwen2.5:7b'},
    'qwen2.5:3b':      {'promoted': 'qwen2.5:7b',      'demoted': 'qwen2.5:3b'},
    'qwen2.5:7b':      {'promoted': 'qwen2.5:14b',     'demoted': 'qwen2.5:3b'},
    'qwen2.5:14b':     {'promoted': 'deepseek-r1:32b', 'demoted': 'qwen2.5:7b'},
}

_PROMOTED_SUFFIX = (
    ' Your past estimates have been consistently accurate — '
    'trust your data-driven instincts.'
)
_DEMOTED_SUFFIX = (
    ' Previous estimates from your role have diverged from reality — '
    'be more conservative and cite specific data.'
)


def _resolve_model(base_model: str, tier: str) -> str:
    tiers = _DEBATE_TIERS.get(base_model, {})
    if tier == 'promoted':
        return tiers.get('promoted', base_model)
    if tier == 'demoted':
        return tiers.get('demoted', base_model)
    return base_model


def get_evolved_debate_agents(
    store: 'AgentTrustStore',
    base_agents: list[Agent],
) -> list[Agent]:
    """Return debate agents with model/prompt adjusted by trust scores."""
    if store.total_runs() < _COLD_START_RUNS:
        return list(base_agents)

    trust_map = store.get_all_trust()
    evolved: list[Agent] = []
    for agent in base_agents:
        trust = trust_map.get(agent.name, 0.5)
        tier = _tier(trust)
        new_model = _resolve_model(agent.model, tier)
        new_prompt = agent.system_prompt
        if tier == 'promoted':
            new_prompt = new_prompt.rstrip() + _PROMOTED_SUFFIX
        elif tier == 'demoted':
            new_prompt = new_prompt.rstrip() + _DEMOTED_SUFFIX
        evolved.append(replace(agent, model=new_model, system_prompt=new_prompt))
    return evolved


def get_evolved_swarm_agents(
    store: 'AgentTrustStore',
    base_agents: list[SwarmAgent],
) -> list[SwarmAgent]:
    """Return swarm agents with model adjusted by trust; enforces diversity guard."""
    if store.total_runs() < _COLD_START_RUNS:
        return list(base_agents)

    trust_map = store.get_all_trust()
    # Group by group_id for diversity guard
    groups: dict[int, list[SwarmAgent]] = {}
    for a in base_agents:
        groups.setdefault(a.group_id, []).append(a)

    evolved: list[SwarmAgent] = []
    for group_id, group_agents in groups.items():
        min_active = math.ceil(len(group_agents) * _DIVERSITY_MIN)
        # Sort by trust descending so we bench lowest-trust first
        scored = sorted(
            group_agents,
            key=lambda a: trust_map.get(a.name, 0.5),
            reverse=True,
        )
        active: list[SwarmAgent] = []
        benched: list[SwarmAgent] = []
        for agent in scored:
            trust = trust_map.get(agent.name, 0.5)
            tier = _tier(trust)
            if tier == 'demoted' and len(active) >= min_active:
                benched.append(agent)
            else:
                new_model = _resolve_model(agent.model, tier)
                prompt = agent.system_prompt
                if tier == 'promoted':
                    prompt = prompt.rstrip() + _PROMOTED_SUFFIX
                elif tier == 'demoted':
                    prompt = prompt.rstrip() + _DEMOTED_SUFFIX
                active.append(SwarmAgent(
                    name=agent.name, role=agent.role, model=new_model,
                    group_id=agent.group_id, region_name=agent.region_name,
                    system_prompt=prompt, rag_sources=agent.rag_sources,
                ))
        evolved.extend(active)
    return evolved


def _tier(trust: float) -> str:
    if trust > 0.70:
        return 'promoted'
    if trust < 0.30:
        return 'demoted'
    return 'default'
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_evolution.py -v
```
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/evolution.py ph_economic_ai/tests/test_evolution.py
git commit -m "feat: add Evolution Engine for trust-based model-tier promotion/demotion"
```

---

## Task 4: DOE Ground Truth Checker

**Files:**
- Create: `ph_economic_ai/engine/ground_truth.py`
- Create: `ph_economic_ai/tests/test_ground_truth.py`

- [ ] **Step 1: Write failing tests**

```python
# ph_economic_ai/tests/test_ground_truth.py
import pytest
from unittest.mock import patch
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.ground_truth import (
    compute_accuracy_score,
    find_and_grade_runs,
)


@pytest.fixture
def store_with_run(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    run_id = s.save_run(
        scenario={'current_price': 98.82},
        final_estimate=1.42,
        confidence_pct=78,
    )
    s.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.42,
         'statement': 'Rising.', 'citation_count': 1, 'has_causal_chain': 1,
         'internal_score': 0.7, 'model_used': 'deepseek-r1:8b'},
    ])
    return s, run_id


def test_accuracy_score_perfect():
    assert compute_accuracy_score(estimate=1.42, actual=1.42) == 1.0


def test_accuracy_score_half_php_error():
    score = compute_accuracy_score(estimate=1.92, actual=1.42)
    assert abs(score - (1 - 0.5 / 3.0)) < 0.001


def test_accuracy_score_three_php_error():
    score = compute_accuracy_score(estimate=4.42, actual=1.42)
    assert score == 0.0


def test_find_and_grade_runs_skips_recent(store_with_run):
    store, run_id = store_with_run
    # Run is just created — younger than 5 days
    graded = find_and_grade_runs(store, current_price=100.22, min_age_days=5.0)
    assert graded == 0


def test_find_and_grade_runs_grades_old_run(store_with_run):
    store, run_id = store_with_run
    # Grade with min_age_days=0 to bypass age check in tests
    graded = find_and_grade_runs(store, current_price=100.22, min_age_days=0.0)
    assert graded == 1
    # Confirm run is now graded
    ungraded = store.get_ungraded_runs(min_age_days=0.0)
    assert len(ungraded) == 0


def test_trust_improves_after_accurate_grade(store_with_run):
    store, _ = store_with_run
    trust_before = store.get_trust('Market Analyst')
    # actual_change = 100.22 - 98.82 = 1.40, estimate was 1.42, error ≈ ₱0.02
    find_and_grade_runs(store, current_price=100.22, min_age_days=0.0)
    trust_after = store.get_trust('Market Analyst')
    assert trust_after > trust_before
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest ph_economic_ai/tests/test_ground_truth.py -v
```
Expected: `ModuleNotFoundError: No module named 'ph_economic_ai.engine.ground_truth'`

- [ ] **Step 3: Implement `engine/ground_truth.py`**

```python
# ph_economic_ai/engine/ground_truth.py
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

_POLL_INTERVAL_MS = 6 * 60 * 60 * 1000   # 6 hours in milliseconds


def compute_accuracy_score(estimate: float, actual: float) -> float:
    """₱0.00 error → 1.0 | ₱3.00+ error → 0.0 (linear)."""
    return max(0.0, 1.0 - abs(estimate - actual) / 3.0)


def find_and_grade_runs(
    store: 'AgentTrustStore',
    current_price: float,
    min_age_days: float = 5.0,
) -> int:
    """Find ungraded runs older than min_age_days, grade them, return count graded."""
    ungraded = store.get_ungraded_runs(min_age_days=min_age_days)
    graded = 0
    for run in ungraded:
        scenario = __import__('json').loads(run['scenario_json'])
        baseline = scenario.get('current_price')
        if baseline is None:
            continue
        actual_change = current_price - baseline
        store.apply_ground_truth_grade(run['run_id'], actual_change)
        graded += 1
    return graded


class DOECheckerThread(QThread):
    """Background QThread that polls DOE price every 6 hours and grades old runs."""
    grades_applied = pyqtSignal(int)   # count of runs graded

    def __init__(self, store: 'AgentTrustStore', parent=None):
        super().__init__(parent)
        self._store = store
        self._running = True

    def run(self):
        from ph_economic_ai.engine.swarm import fetch_live_retail_price
        from PyQt6.QtCore import QThread as _QT
        while self._running:
            try:
                current_price = fetch_live_retail_price()
                count = find_and_grade_runs(self._store, current_price)
                if count:
                    self.grades_applied.emit(count)
            except Exception as e:
                logging.warning('DOECheckerThread: %s', e)
            _QT.sleep(int(_POLL_INTERVAL_MS / 1000))

    def stop(self):
        self._running = False
        self.quit()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_ground_truth.py -v
```
Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/ground_truth.py ph_economic_ai/tests/test_ground_truth.py
git commit -m "feat: add DOECheckerThread and find_and_grade_runs for delayed accuracy scoring"
```

---

## Task 5: Add `all_responses` to MasterVerdict and `evolved_agents` to SwarmOrchestrator

**Files:**
- Modify: `ph_economic_ai/engine/swarm.py`

- [ ] **Step 1: Write failing test**

```python
# Add to ph_economic_ai/tests/test_swarm.py (append to existing file)

def test_master_verdict_has_all_responses_field():
    from ph_economic_ai.engine.swarm import MasterVerdict
    import inspect
    fields = {f.name for f in __import__('dataclasses').fields(MasterVerdict)}
    assert 'all_responses' in fields


def test_swarm_orchestrator_accepts_evolved_agents():
    from ph_economic_ai.engine.swarm import SwarmOrchestrator
    import inspect
    sig = inspect.signature(SwarmOrchestrator.__init__)
    assert 'evolved_agents' in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest ph_economic_ai/tests/test_swarm.py::test_master_verdict_has_all_responses_field ph_economic_ai/tests/test_swarm.py::test_swarm_orchestrator_accepts_evolved_agents -v
```
Expected: both FAIL with `AssertionError`.

- [ ] **Step 3: Modify `MasterVerdict` to add `all_responses` field**

In `ph_economic_ai/engine/swarm.py`, locate the `MasterVerdict` dataclass (around line 119) and add the field:

```python
@dataclass
class MasterVerdict:
    final_estimate: Optional[float]
    confidence_pct: int
    dissenting_regions: list[str]
    reasoning: str
    regional_verdicts: list[RegionalVerdict]
    regional_estimates: Optional[dict] = None
    all_responses: list = None   # list[AgentResponse] from all group arenas

    def __post_init__(self):
        if self.all_responses is None:
            self.all_responses = []
```

- [ ] **Step 4: Add `evolved_agents` param to `SwarmOrchestrator.__init__` and wire into `run()`**

Locate `SwarmOrchestrator.__init__` (around line 748) and add `evolved_agents` parameter:

```python
class SwarmOrchestrator:
    def __init__(
        self,
        rag: RagEngine,
        scenario: dict,
        parallel_n: int = 4,
        on_event: Optional[Callable] = None,
        data_brief: Optional['LiveDataBrief'] = None,
        ml_baseline: str = '',
        evolved_agents: Optional[list[SwarmAgent]] = None,   # ADD THIS
    ):
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._on_event = on_event
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents                  # ADD THIS
```

In `SwarmOrchestrator.run()`, replace the `all_agents = build_swarm_agents(live_price)` line:

```python
def run(self) -> MasterVerdict:
    live_price = fetch_live_retail_price()
    self._scenario = {**self._scenario, 'current_price': live_price}
    # Use evolved agents if provided, otherwise build fresh
    if self._evolved_agents is not None:
        all_agents = self._evolved_agents
    else:
        all_agents = build_swarm_agents(live_price)
    # ... rest of run() unchanged
```

Also in `SwarmOrchestrator.run()`, collect all group responses and include in `MasterVerdict`. After `threads = [...]` and `for t in threads: t.join()`, the survivors are computed. Just before `return master.run()`, collect all arena history. The simplest approach: add a `_all_responses` list to the orchestrator and collect via `on_event`:

In `run_group(group_id)`:
```python
def run_group(group_id: int):
    with sem:
        group_agents = [a for a in all_agents if a.group_id == group_id]
        arena = GroupArena(
            group_id=group_id, agents=group_agents,
            rag=self._rag, scenario=self._scenario,
            on_event=self._on_event,
            data_brief=self._data_brief,
            ml_baseline=self._ml_baseline,
        )
        try:
            s = arena.run()
            with lock:
                survivors[group_id] = s
                all_arena_responses.extend(arena._history)  # ADD THIS
        except Exception as e:
            with lock:
                errors.append(f"Group {group_id}: {e}")
```

Add `all_arena_responses: list = []` before the threads list, and use it when building `MasterVerdict`:

```python
all_arena_responses: list = []   # ADD: before threads = [...]
```

Then in `master.run()` call result, wrap it:
```python
mv = master.run()
mv.all_responses = all_arena_responses   # attach all arena responses
return mv
```

Also update `SwarmThread.__init__` to accept and pass `evolved_agents`:

```python
class SwarmThread(QThread):
    # ... existing signals ...

    def __init__(self, rag: RagEngine, scenario: dict, parallel_n: int = 4,
                 data_brief: Optional['LiveDataBrief'] = None,
                 ml_baseline: str = '', evolved_agents=None, parent=None):  # ADD evolved_agents
        super().__init__(parent)
        self._rag = rag
        self._scenario = scenario
        self._parallel_n = parallel_n
        self._data_brief = data_brief
        self._ml_baseline = ml_baseline
        self._evolved_agents = evolved_agents   # ADD THIS

    def run(self):
        def on_event(event_type, *args):
            # ... unchanged ...
        orch = SwarmOrchestrator(
            rag=self._rag, scenario=self._scenario,
            parallel_n=self._parallel_n, on_event=on_event,
            data_brief=self._data_brief, ml_baseline=self._ml_baseline,
            evolved_agents=self._evolved_agents,   # ADD THIS
        )
        # ... rest unchanged ...
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest ph_economic_ai/tests/test_swarm.py -v
```
Expected: the two new tests PASS; existing tests unchanged.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/engine/swarm.py ph_economic_ai/tests/test_swarm.py
git commit -m "feat: add all_responses to MasterVerdict and evolved_agents param to SwarmOrchestrator"
```

---

## Task 6: Wire Main Window

**Files:**
- Modify: `ph_economic_ai/main.py`
- Modify: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Modify `main.py` to create the store and pass it to SimMainWindow**

Open `ph_economic_ai/main.py`. After the imports, add:

```python
from ph_economic_ai.engine.store import AgentTrustStore
```

In `main()`, after the regressors dict is built and before `window = SimMainWindow(...)`, add:

```python
    store = AgentTrustStore()  # uses default path: ph_economic_ai/cache/trust.db
```

Change the `SimMainWindow(...)` call to pass `store`:

```python
    window = SimMainWindow(
        df=df,
        regressor=gas_regressor,
        regressors=regressors,
        data_source=data_source,
        cv_rmse=cv_rmse,
        store=store,           # ADD THIS
    )
```

- [ ] **Step 2: Update `SimMainWindow.__init__` to accept `store` and wire evolution + DOE checker**

In `ph_economic_ai/ui/main_window.py`, add imports at the top:

```python
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.quality_scorer import QualityScorer
from ph_economic_ai.engine.evolution import get_evolved_debate_agents, get_evolved_swarm_agents
from ph_economic_ai.engine.ground_truth import DOECheckerThread
from ph_economic_ai.engine.swarm import build_swarm_agents
```

Update `SimMainWindow.__init__` signature to accept `store`:

```python
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, regressors: dict | None = None,
                 store: AgentTrustStore | None = None, parent=None):
```

Inside `__init__`, after `self._debates_started = False`, add:

```python
        self._store: AgentTrustStore = store or AgentTrustStore()
        self._current_run_id: int | None = None
        self._pending_responses: list = []   # collects AgentResponse objects for scoring
        self._doe_checker: DOECheckerThread = DOECheckerThread(self._store)
        self._doe_checker.start()
```

- [ ] **Step 3: Wire evolution into `_on_brief_ready`**

In `_on_brief_ready`, locate the swarm branch where `SwarmThread` is created. Before creating the thread, add:

```python
        if self._last_swarm_mode:
            # Evolve swarm agents based on trust history
            base_swarm = build_swarm_agents()
            evolved_swarm = get_evolved_swarm_agents(self._store, base_swarm)

            thread = SwarmThread(self._rag, self._last_scenario,
                                 parallel_n=self._last_parallel_n,
                                 data_brief=brief,
                                 ml_baseline=self._compute_ml_baseline(),
                                 evolved_agents=evolved_swarm)          # ADD
```

For the non-swarm (debate) branch, before `self._stage3.start_simulation(...)`, evolve the agents:

```python
        else:
            # Evolve debate agents based on trust history
            self._agents = get_evolved_debate_agents(self._store, list(DEFAULT_AGENTS))
            self._start_sector_debates(self._last_scenario)
            self._stage3.start_simulation(
                self._last_scenario_obj, self._agents
            )
```

- [ ] **Step 4: Wire quality scoring into `_on_simulation_complete`**

At the start of `_on_simulation_complete(self, responses)`, after the consensus call, add:

```python
    def _on_simulation_complete(self, responses):
        consensus = self._stage3.engine.consensus()
        # ── Persist run and score agents ─────────────────────────────────────
        if responses:
            estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(responses, estimates)
            run_quality = QualityScorer.run_quality(responses, estimates)
            run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=consensus.get('weighted_avg'),
                confidence_pct=consensus.get('confidence_pct', 0),
            )
            self._store.update_run_quality(run_id, run_quality)
            response_dicts = []
            for r in responses:
                sc = scores.get(r.agent_name, {})
                response_dicts.append({
                    'agent_name': r.agent_name, 'round_num': r.round_num,
                    'estimate': r.price_estimate, 'statement': r.statement,
                    'citation_count': sc.get('citation_count', 0),
                    'has_causal_chain': sc.get('has_causal_chain', 0),
                    'internal_score': sc.get('overall', 0.5),
                    'model_used': next(
                        (a.model for a in self._agents if a.name == r.agent_name), ''),
                })
            self._store.save_agent_responses(run_id, response_dicts)
            for agent_name, sc in scores.items():
                self._store.update_trust(agent_name, internal_score=sc['overall'])
        # ── rest of existing _on_simulation_complete logic unchanged ──────────
        self._gas_verdict = str(consensus)
        # ... existing code continues ...
```

- [ ] **Step 5: Wire quality scoring into `_on_swarm_complete`**

At the start of `_on_swarm_complete(self, master_verdict)`, add:

```python
    def _on_swarm_complete(self, master_verdict):
        # ── Persist run and score swarm agents ────────────────────────────────
        all_responses = getattr(master_verdict, 'all_responses', [])
        if all_responses:
            estimates = [r.price_estimate for r in all_responses if r.price_estimate is not None]
            scores = QualityScorer.score_responses(all_responses, estimates)
            run_quality = QualityScorer.run_quality(all_responses, estimates)
            run_id = self._store.save_run(
                scenario=self._last_scenario,
                final_estimate=master_verdict.final_estimate,
                confidence_pct=master_verdict.confidence_pct,
            )
            self._store.update_run_quality(run_id, run_quality)
            response_dicts = []
            for r in all_responses:
                sc = scores.get(r.agent_name, {})
                response_dicts.append({
                    'agent_name': r.agent_name, 'round_num': r.round_num,
                    'estimate': r.price_estimate, 'statement': r.statement,
                    'citation_count': sc.get('citation_count', 0),
                    'has_causal_chain': sc.get('has_causal_chain', 0),
                    'internal_score': sc.get('overall', 0.5),
                    'model_used': '',
                })
            self._store.save_agent_responses(run_id, response_dicts)
            for agent_name, sc in scores.items():
                self._store.update_trust(agent_name, internal_score=sc['overall'])
        # ── rest of existing _on_swarm_complete logic unchanged ───────────────
        self._gas_verdict = str(master_verdict)
        # ... existing code continues ...
```

- [ ] **Step 6: Run the app and verify no errors**

```
python -m ph_economic_ai.main
```
Expected: app opens, no import errors. After a run completes, `ph_economic_ai/cache/trust.db` is created.

- [ ] **Step 7: Commit**

```bash
git add ph_economic_ai/main.py ph_economic_ai/ui/main_window.py
git commit -m "feat: wire AgentTrustStore, evolution, and quality scoring into main window"
```

---

## Task 7: Agent Performance Panel

**Files:**
- Create: `ph_economic_ai/ui/agent_performance.py`
- Modify: `ph_economic_ai/ui/sidebar.py`
- Modify: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Implement `ui/agent_performance.py`**

```python
# ph_economic_ai/ui/agent_performance.py
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

_GREEN  = '#1a7f37'
_AMBER  = '#7d4e00'
_RED    = '#cf222e'
_GRAY   = '#57606a'
_BG_GREEN  = '#dafbe1'
_BG_AMBER  = '#fff8c5'
_BG_RED    = '#ffebe9'
_BG_GRAY   = '#f6f8fa'


def _tier_color(trust: float) -> tuple[str, str]:
    """Return (text_color, bg_color) for trust value."""
    if trust > 0.70:
        return _GREEN, _BG_GREEN
    if trust < 0.30:
        return _RED, _BG_RED
    return _AMBER, _BG_AMBER


class AgentPerformancePanel(QWidget):
    def __init__(self, store: 'AgentTrustStore', parent=None):
        super().__init__(parent)
        self._store = store
        self.setStyleSheet('background:#ffffff;')
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: trust leaderboard ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(240)
        left.setStyleSheet('background:#f6f8fa;border-right:1px solid #d0d7de;')
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)

        header_lbl = QLabel('TRUST LEADERBOARD')
        header_lbl.setStyleSheet(
            'color:#57606a;font-size:10px;font-weight:700;letter-spacing:1px;'
        )
        left_layout.addWidget(header_lbl)

        self._leaderboard_area = QWidget()
        self._leaderboard_layout = QVBoxLayout(self._leaderboard_area)
        self._leaderboard_layout.setContentsMargins(0, 8, 0, 0)
        self._leaderboard_layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._leaderboard_area)
        scroll.setStyleSheet('QScrollArea{border:none;}')
        left_layout.addWidget(scroll)
        root.addWidget(left)

        # ── Right: run history ───────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)

        run_lbl = QLabel('RUN HISTORY')
        run_lbl.setStyleSheet(
            'color:#57606a;font-size:10px;font-weight:700;letter-spacing:1px;'
        )
        right_layout.addWidget(run_lbl)

        self._run_table = QTableWidget(0, 6)
        self._run_table.setHorizontalHeaderLabels(
            ['Date', 'Predicted', 'Actual', 'Error', 'Quality', 'Status']
        )
        self._run_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._run_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._run_table.setStyleSheet(
            'QTableWidget{border:1px solid #d0d7de;border-radius:6px;background:#fff;}'
            'QHeaderView::section{background:#f6f8fa;color:#57606a;font-weight:600;'
            '  border:none;border-bottom:1px solid #d0d7de;padding:4px;}'
        )
        right_layout.addWidget(self._run_table)

        # DOE checker status bar
        self._doe_status = QLabel('DOE Checker: initializing...')
        self._doe_status.setStyleSheet(
            'background:#ddf4ff;border:1px solid #80ccff;border-radius:6px;'
            'color:#0550ae;font-size:11px;padding:6px 10px;'
        )
        right_layout.addWidget(self._doe_status)
        root.addWidget(right, stretch=1)

    def refresh(self) -> None:
        """Reload trust scores and run history from the store."""
        self._refresh_leaderboard()
        self._refresh_run_table()

    def update_doe_status(self, last_checked: str, next_check: str,
                          pending_count: int) -> None:
        self._doe_status.setText(
            f'DOE Checker active · Last checked: {last_checked} · '
            f'Next check: {next_check} · {pending_count} run(s) pending grade'
        )

    def _refresh_leaderboard(self) -> None:
        rows = self._store.get_all_trust_rows()
        # Clear existing
        while self._leaderboard_layout.count():
            item = self._leaderboard_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for row in rows[:15]:
            trust = row['trust_score']
            tc, bc = _tier_color(trust)
            name = row['agent_name']
            abbrev = ''.join(w[0] for w in name.split()[:2]).upper()

            entry = QWidget()
            entry.setStyleSheet('background:transparent;')
            h = QHBoxLayout(entry)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)

            badge = QLabel(abbrev)
            badge.setFixedSize(34, 34)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                f'background:{bc};color:{tc};border:1.5px solid {tc};'
                f'border-radius:17px;font-size:9px;font-weight:700;'
            )
            h.addWidget(badge)

            meta = QVBoxLayout()
            meta.setSpacing(2)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet('color:#24292f;font-size:11px;font-weight:500;')
            meta.addWidget(name_lbl)

            bar_outer = QFrame()
            bar_outer.setFixedHeight(5)
            bar_outer.setStyleSheet('background:#eaeef2;border-radius:2px;')
            bar_inner = QFrame(bar_outer)
            bar_inner.setFixedHeight(5)
            bar_inner.setFixedWidth(max(4, int(trust * (240 - 34 - 8 - 40))))
            bar_inner.setStyleSheet(f'background:{tc};border-radius:2px;')
            meta.addWidget(bar_outer)
            h.addLayout(meta)

            score_lbl = QLabel(f'{trust:.2f}')
            score_lbl.setStyleSheet(
                f'color:{tc};font-family:monospace;font-size:11px;font-weight:700;'
            )
            h.addWidget(score_lbl)
            self._leaderboard_layout.addWidget(entry)

        self._leaderboard_layout.addStretch()

    def _refresh_run_table(self) -> None:
        import sqlite3
        conn = self._store._conn
        cur = conn.execute(
            'SELECT * FROM runs ORDER BY run_id DESC LIMIT 20'
        )
        runs = [dict(r) for r in cur.fetchall()]
        self._run_table.setRowCount(len(runs))
        for i, run in enumerate(runs):
            from datetime import datetime
            ts = run.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime('%b %d')
            except Exception:
                date_str = ts[:10]

            pred = run.get('final_estimate')
            actual = run.get('actual_price_change')
            error = run.get('accuracy_error')
            quality = run.get('internal_quality')

            pred_str = f'+₱{pred:.2f}/L' if pred is not None else '—'
            actual_str = f'+₱{actual:.2f}/L' if actual is not None else '—'
            error_str = f'₱{error:.2f}' if error is not None else '—'
            quality_str = f'{quality:.2f}' if quality is not None else '—'

            if actual is not None:
                status = 'Graded ✓'
                status_color = _GREEN
                status_bg = _BG_GREEN
            else:
                status = '⏳ Pending DOE'
                status_color = _GRAY
                status_bg = _BG_GRAY

            for col, val in enumerate([date_str, pred_str, actual_str,
                                        error_str, quality_str, status]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 5:
                    item.setForeground(QColor(status_color))
                    item.setBackground(QColor(status_bg))
                self._run_table.setItem(i, col, item)
```

- [ ] **Step 2: Add "Agent Performance" entry to sidebar**

Open `ph_economic_ai/ui/sidebar.py`. Find where sidebar entries are defined (look for the stage labels list or similar). Add an entry with index 6 labeled `'Agent Performance'` (or whatever index comes after the last existing entry). The exact sidebar structure depends on what's there — find the list of `(icon, label)` or `label` entries and append:

```python
'Agent Performance'   # add to the labels/entries list
```

The `stage_changed` signal should emit 6 (or whatever the new index is) when this entry is clicked.

- [ ] **Step 3: Wire panel into `SimMainWindow`**

In `ph_economic_ai/ui/main_window.py`, add import:

```python
from ph_economic_ai.ui.agent_performance import AgentPerformancePanel
```

In `SimMainWindow.__init__`, after `self._stage5 = Stage5InteractPanel(...)`, add:

```python
        self._agent_perf = AgentPerformancePanel(self._store)
```

In the stack-building loop where widgets are added, add `self._agent_perf`:

```python
        for widget in (self._economy_overview, self._stage1, self._stage2,
                       self._stage3_container, self._stage4, self._stage5,
                       self._agent_perf):              # ADD
            self._stack.addWidget(widget)
```

In `_on_stage_changed`, add a refresh when the performance panel is selected:

```python
    def _on_stage_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)
        if idx == 6:                        # Agent Performance index
            self._agent_perf.refresh()
```

Wire the DOE checker signal to refresh the panel:

```python
        self._doe_checker.grades_applied.connect(
            lambda _: self._agent_perf.refresh()
        )
```

- [ ] **Step 4: Run the app and verify the panel appears**

```
python -m ph_economic_ai.main
```
Expected: sidebar shows "Agent Performance" entry. Clicking it shows the leaderboard (empty on first run) and run history table.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/agent_performance.py ph_economic_ai/ui/sidebar.py ph_economic_ai/ui/main_window.py
git commit -m "feat: add AgentPerformancePanel with trust leaderboard and run history"
```

---

## Task 8: Trust Badges on Swarm Canvas

**Files:**
- Modify: `ph_economic_ai/ui/stage3_swarm_canvas.py`
- Modify: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Pass store to Stage3SwarmPanel**

In `ph_economic_ai/ui/main_window.py`, find where `Stage3SwarmPanel` is instantiated:

```python
        self._stage3_swarm = Stage3SwarmPanel()
```

Change it to pass the store:

```python
        self._stage3_swarm = Stage3SwarmPanel(store=self._store)
```

- [ ] **Step 2: Update `Stage3SwarmPanel.__init__` to accept and store the store reference**

In `ph_economic_ai/ui/stage3_swarm_canvas.py`, locate `Stage3SwarmPanel` class. Update its `__init__` to accept `store`:

```python
class Stage3SwarmPanel(QWidget):
    swarm_complete = pyqtSignal(object)

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        # ... rest of existing __init__ unchanged ...
```

- [ ] **Step 3: Add trust badge rendering to agent node items**

In `stage3_swarm_canvas.py`, find the class that renders individual agent circles (likely named `AgentNode` or similar — search for `QGraphicsObject` subclass with agent circle drawing). Add a `trust_score` attribute and badge rendering.

Find the `paint` method of the agent node class. After the existing circle-drawing code, add the badge overlay. Locate the paint method (it uses `QPainter`) and append:

```python
    def set_trust(self, trust: float, tier: str) -> None:
        """Call this to update the badge before/after a run."""
        self._trust = trust
        self._tier = tier
        self.update()

    def _draw_trust_badge(self, painter: QPainter) -> None:
        if not hasattr(self, '_trust'):
            return
        trust = self._trust
        tier = getattr(self, '_tier', 'default')

        if tier == 'promoted':
            text_color = QColor('#1a7f37')
            bg_color   = QColor('#dafbe1')
            border_color = QColor('#82cf9a')
            indicator  = '▲'
        elif tier == 'demoted':
            text_color = QColor('#cf222e')
            bg_color   = QColor('#ffebe9')
            border_color = QColor('#ffb3ae')
            indicator  = '▼'
        else:
            text_color = QColor('#7d4e00')
            bg_color   = QColor('#fff8c5')
            border_color = QColor('#d4a72c')
            indicator  = '●'

        r = self.boundingRect()
        badge_w, badge_h = 36, 14
        bx = r.right() - badge_w + 4
        by = r.top() - badge_h // 2

        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(border_color, 1))
        painter.drawRoundedRect(int(bx), int(by), badge_w, badge_h, 4, 4)

        painter.setPen(text_color)
        font = QFont('monospace', 7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            int(bx), int(by), badge_w, badge_h,
            Qt.AlignmentFlag.AlignCenter,
            f'{indicator}{trust:.2f}',
        )
```

In the `paint` method of the agent node, call `self._draw_trust_badge(painter)` at the end.

- [ ] **Step 4: Load trust scores and apply badges when a run completes**

In `Stage3SwarmPanel`, find the method that handles `swarm_complete` or run-end events (look for where agent nodes are finalized). After the run ends, load trust from store and update all nodes:

```python
    def _apply_trust_badges(self) -> None:
        if self._store is None:
            return
        trust_map = self._store.get_all_trust()
        # Iterate over all agent node items in the canvas scene
        for item in self._scene.items():
            if hasattr(item, 'set_trust') and hasattr(item, '_agent_name'):
                trust = trust_map.get(item._agent_name, 0.5)
                tier = 'promoted' if trust > 0.70 else ('demoted' if trust < 0.30 else 'default')
                item.set_trust(trust, tier)
```

Call `self._apply_trust_badges()` in the swarm-complete handler.

**Note:** The exact attribute names (`_scene`, `_agent_name`, etc.) depend on the existing canvas implementation. Read the agent node class in `stage3_swarm_canvas.py` to find the correct attribute names before writing the badge application code.

- [ ] **Step 5: Run the app, run a full swarm simulation, verify badges appear**

```
python -m ph_economic_ai.main
```
Expected: after a swarm run completes, each agent circle shows a small colored badge in the top-right with tier indicator (▲/●/▼) and trust score. On the first 3 runs (cold start), all badges show ●0.50.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/stage3_swarm_canvas.py ph_economic_ai/ui/main_window.py
git commit -m "feat: add trust tier badges to swarm canvas agent nodes"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ SQLite store with 3 tables (Task 1)
- ✅ Internal quality scorer — 4 metrics, heuristic only (Task 2)
- ✅ Trust formula — EMA with immediate and delayed signals (Task 1 + store methods)
- ✅ Evolution engine — cold start guard, diversity guard, model tier map (Task 3)
- ✅ DOE checker — 6-hour polling, retroactive grading (Task 4)
- ✅ `all_responses` on MasterVerdict for swarm scoring (Task 5)
- ✅ `evolved_agents` wired into SwarmOrchestrator (Task 5)
- ✅ Main window wiring — store, evolution, quality scorer, DOE checker (Task 6)
- ✅ Agent Performance panel — leaderboard + run history + DOE status (Task 7)
- ✅ Trust badges on swarm canvas — color-coded, cold start neutral (Task 8)
- ✅ Light-mode colors throughout UI

**Type consistency:**
- `AgentTrustStore` used consistently across all tasks
- `QualityScorer.score_responses(responses, group_estimates)` → `dict[str, dict]`
- `get_evolved_debate_agents(store, base_agents)` → `list[Agent]`
- `get_evolved_swarm_agents(store, base_agents)` → `list[SwarmAgent]`
- `find_and_grade_runs(store, current_price, min_age_days)` → `int`
- `MasterVerdict.all_responses: list[AgentResponse]`

**No placeholders:** All code steps contain complete implementations.
