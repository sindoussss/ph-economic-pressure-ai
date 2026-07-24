# ph_economic_ai — Evolutionary Agent Pool Design

**Date:** 2026-05-26  
**Approach:** B — Evolutionary Agent Pool  
**Goal:** Maximum single-run output quality + self-improving loop across runs  
**Model strategy:** Stay local (Ollama) — upgrade model tiers, no external API

---

## 1. Overview

The system adds a persistent trust layer on top of the existing Debate + Swarm engines. After every run, two independent scoring signals update per-agent trust scores stored in SQLite. Before each new run, an Evolution Engine reads those trust scores and promotes high-trust agents to stronger model tiers while demoting poor performers. The loop closes: better agents produce better runs, which produce better scores, which evolve the agent pool further.

No changes to the existing 5-stage UI flow. Two additions only: trust badges on the swarm canvas, and a new "Agent Performance" sidebar panel.

---

## 2. Architecture

Five new components wrap the existing engine without modifying its core logic:

| Component | File | Role |
|---|---|---|
| Agent Trust Store | `engine/store.py` | SQLite persistence for runs, responses, and trust scores |
| Internal Quality Scorer | `engine/quality_scorer.py` | Scores each agent immediately after every run (no LLM needed) |
| Background DOE Checker | `engine/ground_truth.py` | Polls fuelprice.ph weekly, retroactively grades stored predictions |
| Evolution Engine | `engine/evolution.py` | Reads trust scores before each run, returns evolved agent list |
| Agent Performance Panel | `ui/agent_performance.py` | Light-mode UI showing trust leaderboard + run history |

Modified files: `engine/debate.py`, `engine/swarm.py` (model-tier logic), `main.py`, `ui/main_window.py` (wire evolution + panel).

**Data flow:**
```
[Agent Trust Store] + [Upgraded Models] + [Evolution Engine]
          ↓ selects evolved agent pool
   [Existing Debate / Swarm Engine]
          ↓ produces consensus + agent responses
   [Internal Quality Scorer] ──→ instant trust delta (weight: 0.4)
   [Background DOE Checker]  ──→ delayed accuracy delta (weight: 0.6)
          ↓ both signals
   [Agent Trust Store updated]
          ↓ loop: Evolution Engine reads for next run
```

---

## 3. Data Model (SQLite — `engine/store.py`)

### `runs` table
| Column | Type | Notes |
|---|---|---|
| run_id | INTEGER PK | auto-increment |
| timestamp | TEXT | ISO-8601 UTC |
| scenario_json | TEXT | oil_pct, usd_pct, bsp_rate, demand_index |
| final_estimate | REAL | ₱/L change (Master Judge output) |
| confidence_pct | INTEGER | 0–100 |
| internal_quality | REAL | 0.0–1.0, set immediately after run |
| actual_price_change | REAL NULL | filled by DOE Checker when available |
| accuracy_error | REAL NULL | \|predicted − actual\| |
| graded_at | TEXT NULL | timestamp when DOE grade was applied |

### `agent_responses` table
| Column | Type | Notes |
|---|---|---|
| id | INTEGER PK | |
| run_id | INTEGER FK→runs | |
| agent_name | TEXT | |
| round_num | INTEGER | |
| estimate | REAL NULL | parsed ₱/L or % change |
| statement | TEXT | full response text |
| citation_count | INTEGER | DATA BRIEF number mentions found |
| has_causal_chain | INTEGER | 0 or 1 |
| internal_score | REAL | quality scorer output 0.0–1.0 |
| model_used | TEXT | which Ollama model ran this agent |

### `agent_trust` table
| Column | Type | Notes |
|---|---|---|
| agent_name | TEXT PK | |
| trust_score | REAL | 0.0–1.0, initialized at 0.5 |
| runs_participated | INTEGER | |
| avg_internal_score | REAL | rolling average |
| avg_accuracy_error | REAL NULL | ₱ error vs DOE actual; NULL until first grade |
| current_model_tier | TEXT | 'promoted' / 'default' / 'demoted' |
| last_updated | TEXT | ISO-8601 UTC |

---

## 4. Trust Score Formula

```python
# After every run (immediate signal only, before DOE grade):
raw_update = internal_score          # 0.0–1.0 from quality scorer
trust_new  = 0.3 * raw_update + 0.7 * trust_old

# When DOE grade arrives (replaces the internal-only update for that run):
accuracy_score = max(0, 1 - abs(estimate - actual_change) / 3.0)
# ₱0.00 error → 1.0 | ₱0.50 error → 0.83 | ₱1.50 error → 0.50 | ₱3.00+ error → 0.0
raw_update = 0.4 * internal_score + 0.6 * accuracy_score
trust_new  = 0.3 * raw_update + 0.7 * trust_old

# EMA smoothing (α=0.3) applied in both cases — prevents wild swings
trust = clamp(trust_new, 0.05, 0.95)
```

**Tier thresholds:**
- `trust > 0.70` → **promoted** (bigger model, confidence anchor in prompt)
- `trust 0.30–0.70` → **default** (original model, no prompt change)
- `trust < 0.30` → **demoted** (lighter model, skeptic framing, Round 2 only)

---

## 5. Model Tier Map

| Role | Default (current) | Promoted | Demoted |
|---|---|---|---|
| Debate main agents | deepseek-r1:8b | deepseek-r1:32b | qwen2.5:7b |
| Swarm Forecaster/Critic | qwen2.5:7b | qwen2.5:14b | qwen2.5:3b |
| Swarm mini agents | qwen2.5:3b | qwen2.5:7b | benched |
| Regional / Master Judge | qwen2.5:14b | deepseek-r1:32b | qwen2.5:7b |

---

## 6. Internal Quality Scorer (`engine/quality_scorer.py`)

Runs synchronously after every run. Pure heuristics — no LLM calls.

| Metric | Weight | Scoring logic |
|---|---|---|
| Evidence citation rate | 30% | `min(citation_count / 3.0, 1.0)` — counts ₱/$/% values cited from DATA BRIEF |
| Estimate convergence | 25% | `max(0, 1 − abs(estimate − median) / 2.0)` — proximity to group median |
| Causal chain completeness | 25% | `1.0` full chain / `0.5` partial / `0.0` missing |
| Reasoning length & structure | 20% | Gaussian centered at 400 chars (σ=200) — penalizes too short or padding-heavy responses |

`internal_score = weighted_sum(metrics)` stored per agent in `agent_responses`.

---

## 7. Background DOE Checker (`engine/ground_truth.py`)

Runs as a QThread, polling every 6 hours while the app is open.

1. Query SQLite for runs where `actual_price_change IS NULL` and `timestamp` is >5 days old
2. Call existing `fetch_live_retail_price()` from `swarm.py` to get current NCR pump price
3. Compute `actual_change = current_price − run.scenario['current_price']`
4. For each agent response in the run: `accuracy_score = max(0, 1 − abs(estimate − actual_change) / 3.0)`
5. Apply trust delta (weight 0.6) to `agent_trust` table via EMA formula
6. Emit `grades_applied` signal → UI refreshes trust badges without restart

**Constraint:** Only runs when the app is open. No background daemon. Skips if no ungraded runs exist. Reuses existing fuelprice.ph scraper — no new dependencies.

---

## 8. Evolution Engine (`engine/evolution.py`)

`get_evolved_agents(store, base_agents) → list[Agent]`

1. Load `agent_trust` table from SQLite
2. Determine tier per agent (promoted / default / demoted)
3. Return a modified copy of each agent with `model` and `system_prompt` updated:
   - **Promoted prompt suffix:** *"Your past estimates have been consistently accurate — trust your data-driven instincts."*
   - **Demoted prompt suffix:** *"Previous estimates from your role have diverged from reality — be more conservative and cite specific data."*
4. Benched agents (demoted mini-validators) excluded from returned list

**Guards:**
- **Cold start:** If `store.total_runs() < 3`, return `base_agents` unchanged — no evolution until sufficient signal
- **Diversity:** At least 60% of agents per group must remain active; if demotion would bench too many, keep lowest-trust agents at demoted-tier instead of fully benching them

---

## 9. UI Changes

### Trust badges on swarm canvas (`ui/stage3_swarm_canvas.py`)
Small overlays on existing agent circles — no layout changes:
- Badge shows: tier indicator (▲/●/▼) + trust score (monospace)
- Color coding: green (`#1a7f37`/`#dafbe1`) / amber (`#7d4e00`/`#fff8c5`) / red (`#cf222e`/`#ffebe9`) / gray for cold start
- Demoted agents get dashed circle border
- All colors light-mode compatible — warm white backgrounds, dark colored text

### Agent Performance panel (`ui/agent_performance.py` + new sidebar entry)
Split layout:
- **Left (230px):** Trust leaderboard — agent avatar circles, name, horizontal trust bar, score value
- **Right:** Run history table — date, predicted, actual, error, quality score, status badge
- **Footer:** DOE Checker status row (active dot, last-checked time, next check, pending count)
- Status badges: Graded ✓ (green) / Graded ✓ warn (amber) / ⏳ Pending DOE (gray) / Cold start (gray)

---

## 10. What Is Not Changing

- The 5-stage UI flow (RAG → Setup → Swarm Canvas → Report → Interact) — unchanged
- The debate engine logic, swarm group elimination bracket, regional/master judge — unchanged
- The RAG engine sources, chunking, embedding — unchanged
- The food/electricity agent pools — evolution applies to them too (same trust store), no special handling needed
- The economy overview bento dashboard — unchanged

---

## 11. Open Questions (resolved)

- **DOE data source:** Reuse `fetch_live_retail_price()` from `swarm.py` (fuelprice.ph scraper) — no new dependency
- **Trust initialization:** 0.5 for all agents; evolution disabled for first 3 runs
- **Prompt evolution (Approach C):** Out of scope for this spec — can be layered on later
- **Claude API integration:** Out of scope — all models stay local Ollama
