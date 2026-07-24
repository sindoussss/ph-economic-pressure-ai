# Hierarchical Swarm Agent System — ph_economic_ai

**Date:** 2026-05-20  
**Status:** Approved  
**Scope:** Replace the flat 15-agent debate engine with a 3-tier hierarchical swarm of 211 agents (200 group + 10 regional + 1 master) with elimination rounds.

---

## 1. Overview

The system upgrades `ph_economic_ai` from a flat sequential debate (15 agents, 3 rounds) to a hierarchical swarm:

```
Tier 1: 20 groups × 10 agents = 200 base agents  (parallel batched, elimination rounds)
Tier 2: 10 regional judges                         (sequential, debate + synthesize)
Tier 3: 1 master judge                             (single synthesis call)
```

Total worst-case Ollama calls: ~371 (340 group: 200 round-1 + 100 round-2 + 40 round-3; 30 regional: 3 calls × 10 judges; 1 master).

---

## 2. Group Composition (Tier 1)

### 2.1 The 20 Groups — Philippine Regions

| # | Group Name |
|---|---|
| 1 | NCR (National Capital Region) |
| 2 | CAR (Cordillera Administrative Region) |
| 3 | Region I — Ilocos |
| 4 | Region II — Cagayan Valley |
| 5 | Region III — Central Luzon |
| 6 | Region IV-A — CALABARZON |
| 7 | Region IV-B — MIMAROPA |
| 8 | Region V — Bicol |
| 9 | Region VI — Western Visayas |
| 10 | Region VII — Central Visayas |
| 11 | Region VIII — Eastern Visayas |
| 12 | Region IX — Zamboanga Peninsula |
| 13 | Region X — Northern Mindanao |
| 14 | Region XI — Davao Region |
| 15 | Region XII — SOCCSKSARGEN |
| 16 | Region XIII — Caraga |
| 17 | BARMM |
| 18 | Mega Manila Corridor (NCR + III + IV-A urban belt) |
| 19 | Clark Economic Zone |
| 20 | Cebu Metropolitan Economic Zone |

### 2.2 Agent Roles per Group (10 agents = 2 per model)

| Role | Model (Ollama tag) | Count | Purpose |
|---|---|---|---|
| Forecaster | `deepseek-r1:8b` | 2 | Regional price trajectory projection |
| Data Extractor | `qwen2.5:7b` | 2 | Pulls region-specific economic indicators |
| Synthesizer | `llama3.2:3b` | 2 | Integrates multi-factor view |
| Critic | `mistral:7b` | 2 | Challenges and stress-tests estimates |
| Confidence Scorer | `phi4:14b` | 2 | Calibrates uncertainty on each estimate |

Each agent has a system prompt that combines its role description with its group's regional context (e.g., "You are a Forecaster analyzing fuel price dynamics specifically for the Davao Region...").

---

## 3. Elimination Mechanics

### 3.1 Combined Score Formula

After each round within a group, every surviving agent receives a combined score:

```
combined_score = 0.4 × critic_score_normalized
              + 0.6 × (confidence × (1 − deviation_from_median_normalized))
```

- **critic_score_normalized**: The Mistral Critic agent rates each agent's statement 1–10; divided by 10.
- **confidence**: The Phi Confidence Scorer outputs 0.0–1.0 for each estimate.
- **deviation_from_median_normalized**: Absolute distance from group median price estimate, normalized to 0–1 across the group's min–max range. An agent exactly at the median scores 1.0 here.
- **Special case**: If an agent's `price_estimate` is `None` (parse failure), `combined_score = 0.0` — always eliminated first.

### 3.2 3-Round Bracket

| State | Agents Alive | Eliminated This Round |
|---|---|---|
| Start | 10 | — |
| After Round 1 | 5 | 5 (bottom half by combined score) |
| After Round 2 | 2 | 3 (bottom 3 of remaining 5) |
| After Round 3 | 1 | 1 (lower scorer of final duel) |

The single survivor carries its final estimate, full reasoning, and combined score to the regional judge.

---

## 4. Orchestration & Concurrency

### 4.1 New File: `ph_economic_ai/engine/swarm.py`

Contains:
- `GroupArena` — runs one group's 3 elimination rounds
- `RegionalJudge` — debates 2 group survivors, synthesizes verdict
- `MasterJudge` — synthesizes 10 regional verdicts into final estimate
- `SwarmOrchestrator` — top-level coordinator
- `SwarmThread(QThread)` — off-thread runner, emits Qt signals

### 4.2 Concurrency Model

- `threading.Semaphore(N)` caps concurrent group workers (default `N=4`)
- `N` is configurable via a new "Parallel Groups" slider in Stage 2 setup (range 1–8)
- Each `GroupArena` runs in its own `threading.Thread`
- Regional judges and master judge run sequentially (low call count, no need for parallelism)

### 4.3 Execution Phases

```
Phase 1 — Group Elimination (parallel batches):
  All 20 GroupArenas fire with semaphore(N) concurrency
  Each arena runs 3 rounds internally → emits 1 survivor
  Duration: ceil(20 / N) × (3 rounds × avg_call_time)

Phase 2 — Regional Debate (sequential):
  10 RegionalJudges run one after another
  Each: 2 survivor prompts + 1 synthesis call = 3 Ollama calls

Phase 3 — Master Synthesis (single):
  1 MasterJudge call with all 10 regional verdicts
```

### 4.4 Qt Signal Chain (SwarmThread)

| Signal | Args | When emitted |
|---|---|---|
| `group_round_done` | `(group_id: int, round_num: int, survivors: list[AgentResponse])` | After each group round |
| `group_eliminated` | `(group_id: int, agent_name: str, score: float, round_num: int)` | When an agent is eliminated |
| `group_survivor` | `(group_id: int, survivor: AgentResponse)` | When a group produces its final survivor |
| `regional_done` | `(judge_id: int, verdict: RegionalVerdict)` | When a regional judge finishes |
| `swarm_complete` | `(master_verdict: MasterVerdict)` | When master judge finishes |
| `error_occurred` | `(str)` | On any unhandled exception |

---

## 5. Regional & Master Judge Tier

### 5.1 RegionalJudge Flow

Input: 2 `AgentResponse` survivors from a paired group set. Groups are paired by index: (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16), (17,18), (19,20) — roughly geographic neighbors in the region list above.

1. **Defense round**: Each survivor is prompted to defend its estimate against the other's critique (2 Ollama calls, model: `deepseek-r1:8b`).
2. **Synthesis call**: Regional judge reviews both defenses and issues a regional consensus estimate (1 Ollama call).

Output: `RegionalVerdict(judge_id, region_pair, estimate, confidence, reasoning, survivor_names)`

### 5.2 MasterJudge Flow

Input: 10 `RegionalVerdict` objects.

1. Single prompt containing all 10 verdicts (estimates + reasoning summaries).
2. NCR and Mega Manila corridor verdicts are weighted slightly higher (they represent majority fuel consumption volume) — expressed in the system prompt, not as numeric multipliers.
3. Output: `MasterVerdict(final_estimate, confidence_pct, dissenting_regions, reasoning)`

Model: `deepseek-r1:8b`

`MasterVerdict` replaces the existing `consensus()` output and feeds Stage 4 report unchanged.

---

## 6. Data Structures

```python
@dataclass
class SwarmAgent:
    name: str
    role: str           # 'Forecaster' | 'DataExtractor' | 'Synthesizer' | 'Critic' | 'ConfidenceScorer'
    model: str
    group_id: int
    region_name: str
    system_prompt: str
    rag_sources: list[str]
    is_alive: bool = True
    combined_score: float = 0.0

@dataclass
class RegionalVerdict:
    judge_id: int
    region_pair: tuple[str, str]   # names of the 2 groups
    estimate: float | None
    confidence: float
    reasoning: str
    survivor_names: tuple[str, str]

@dataclass
class MasterVerdict:
    final_estimate: float | None
    confidence_pct: int
    dissenting_regions: list[str]
    reasoning: str
    regional_verdicts: list[RegionalVerdict]
```

---

## 7. UI Changes (Stage 3 Canvas)

### 7.1 Canvas Layout

Replace the current flat ring layout with a 3-tier radial hierarchy:

- **Outer zone**: 20 group clusters arranged in a circle. Each cluster is a mini-ring of 10 small agent dots around a group label.
- **Middle ring**: 10 regional judge nodes. Light up (pulse) when their 2 survivors arrive.
- **Centre node**: Master judge. Pulses when all 10 regional verdicts are in.

### 7.2 Elimination Animation

- Eliminated agents: fade to grey (#CCCCCC), shrink radius by 50% over 300ms.
- Survivors: glow briefly in gold before an animated travel-dot shoots from the group cluster to the paired regional judge node.

### 7.3 New Panels

- **Elimination Log** (right panel, replaces mini-agent section): streams each elimination event — agent name, round, combined score, reason.
- **Phase Banner** (top of canvas): shows current phase ("Phase 1 — Group Elimination · 14/20 groups complete").

### 7.4 Stage 2 Setup Change

Add one slider to the existing scenario pill row:

```
Parallel Groups   [1 ──────●── 8]   default: 4
```

Controls the semaphore batch size passed to `SwarmOrchestrator`.

---

## 8. Backward Compatibility

- `DebateEngine`, `DebateThread`, `DEFAULT_AGENTS` remain untouched.
- Stage 3 canvas gets a toggle in Stage 2: **"Swarm Mode"** checkbox (default ON). When OFF, the old 15-agent flat debate runs as before.
- Stage 4 report and Stage 5 interact panels consume `MasterVerdict` when swarm mode is ON, `consensus()` dict when OFF — a thin adapter handles the interface difference.

---

## 9. Files Changed / Created

| File | Change |
|---|---|
| `ph_economic_ai/engine/swarm.py` | **New** — SwarmAgent, GroupArena, RegionalJudge, MasterJudge, SwarmOrchestrator, SwarmThread, data classes |
| `ph_economic_ai/engine/debate.py` | Unchanged |
| `ph_economic_ai/ui/stage2_setup.py` | Add "Parallel Groups" slider + "Swarm Mode" checkbox |
| `ph_economic_ai/ui/stage3_canvas.py` | New 3-tier canvas layout, elimination animations, phase banner, elimination log |
| `ph_economic_ai/ui/stage4_report.py` | Accept MasterVerdict or legacy consensus dict |
| `ph_economic_ai/ui/stage5_interact.py` | Accept MasterVerdict for follow-up questions |
| `ph_economic_ai/tests/test_swarm.py` | **New** — unit tests for scoring, elimination bracket, orchestrator |

---

## 10. Out of Scope

- Fine-tuning or swapping models mid-run
- Persistent agent memory across sessions
- Network-distributed Ollama (all calls to localhost)
- More than 3 tiers
