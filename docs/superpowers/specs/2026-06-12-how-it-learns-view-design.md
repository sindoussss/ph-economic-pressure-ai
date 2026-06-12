# ph_economic_ai — "How it learns" view (Design)

**Date:** 2026-06-12
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Context:** Low-risk/high-return defense prep (part 2 of 2; part 1 = `docs/defense/talking-points.md`). Surface the learning Strata *already* does — honestly — as a demo-able screen. No new ML; no faked adaptation.

---

## 1. Problem & Goal

The app has real learning (within-run debate revisions; outcome-graded trust → swarm evolution) but it's invisible, so users assume either "it learns every run" (overclaim) or "it doesn't learn" (underclaim). 

**Goal:** a dedicated **Learning** tab that shows, truthfully and demo-ably: the *mechanism* (3 honest layers), this run's within-run revisions, the trust ladder, and the graded track record — each with an honest empty/sparse state so it reads well even with few graded runs.

---

## 2. Scope

### In scope
- New `ui/learning_view.py` → `LearningView(QWidget)` — a scrollable, editorial-themed page with four blocks + a `refresh(run_id=None)` method.
- `ui/main_window.py` wiring: construct `LearningView(self._store)`, add it as a stack page, add a `('Learning', unlocked)` entry to `_NAV`, and call `refresh(run_id)` on run-complete and on tab-show.

### Out of scope
- Any change to the learning *mechanism* (trust store, evolution, grading) — read-only surfacing.
- Track-record memory / SFT / faster grading (explicitly deferred per the earlier decision).
- The `kg_live.py` dead-code cleanup (separate tidy-up).

### Non-negotiables
- **Honest by construction:** the explainer states the model is frozen + the ~5-day grading lag; every data block has a truthful empty/sparse state; the "chain-verified" cue is real (`track_record.verify_chain()`), shown only when it actually verifies.
- **Read-only + robust:** all store/track-record reads are wrapped; missing/empty data → honest placeholder, never a crash.
- **Reuse, don't duplicate:** the trust-ladder block embeds the existing `AgentPerformancePanel` rather than re-implementing the ladder.

---

## 3. Components

### 3.1 `LearningView(QWidget)` — four blocks (top → bottom, in a `QScrollArea`)

**Block 1 — How it learns** *(always present; the demo centerpiece)*
Static editorial explainer of the three layers, mirroring `docs/defense/talking-points.md` §4:
1. *Within a run* — multi-round debate; agents see prior rounds and revise; **resets each run**.
2. *Across runs* — outcome-graded trust → evolution (benches low-trust, adjusts model tier/prompt) after a **cold-start of `_COLD_START_RUNS` (=3)** runs; trust only moves when a **real DOE pump price** grades a past forecast (~5 days).
3. *The models* — **frozen**; not trained on your runs.
No data dependency — always truthful.

**Block 2 — This run's revisions**
From `store.get_agent_responses(run_id)` (rows: `agent_name, round_num, estimate, statement, …`). Group by `agent_name`, show the estimate per `round_num` (Round 1 → Round N) so the within-run adaptation is visible (e.g. `FCST-NCR: R1 −1.2 → R2 −1.8`). Empty-state (no `run_id` yet): *"Run a simulation to see agents revise across debate rounds."*

**Block 3 — Trust ladder**
Embed an `AgentPerformancePanel(self._store)` instance (it already renders `get_all_trust_rows()` with tiers). A one-line honest caption above it: *"Agents rise/fall as real outcomes grade their past calls."* When `total_runs() < _COLD_START_RUNS` or trust is near 0.5, the caption adds: *"— near baseline; evolution activates after 3 runs."*

**Block 4 — Track record**
From `track_record.scorecard()` → `{n_matured, mae, coverage_90}` + `store.total_runs()`. Render a status line: *"`total_runs` logged · `n_matured` graded · evolution active"* (or *"activates at 3"* if cold). When `verify_chain()` is True, show a small **"✓ chain-verified"** integrity chip. Empty-state (`n_matured == 0`): *"No graded outcomes yet — grading waits ~5 days for real DOE pump prices."*

### 3.2 `main_window` wiring
- Construct `self._learning = LearningView(self._store)` (it builds its own `TrackRecord` or receives one).
- Add it to `self._stack` (new index) and append `(<idx>, 'Learning', False)` to `_NAV` so the tab renders unlocked.
- On run-complete (`_on_swarm_complete`) call `self._learning.refresh(self._current_run_id)`; also `refresh()` when the Learning tab is shown (nav handler) so it's current.

## 4. Data flow
```
nav 'Learning' clicked / run completes -> LearningView.refresh(run_id)
  block 2 <- store.get_agent_responses(run_id)        (this run's revisions)
  block 3 <- AgentPerformancePanel.refresh()          (trust ladder, get_all_trust_rows)
  block 4 <- track_record.scorecard() + store.total_runs() + verify_chain()
block 1 is static (the honest mechanism)
```

## 5. Error handling / robustness
- `refresh` wraps each block's data read in `try/except`; a failure renders that block's honest placeholder.
- `run_id is None` → block 2 empty-state. `n_matured == 0` → block 4 empty-state. `verify_chain()` False/raises → simply omit the chip (never claim verified when it isn't).
- No writes; no network (the DOE checker already runs elsewhere).

## 6. Testing
- `test_learning_view.py` (offscreen Qt):
  - builds with a fresh/empty `AgentTrustStore` → all four blocks render; blocks 2 & 4 show empty-states; no crash.
  - with a seeded store (a saved run + `save_agent_responses` two rounds + a trust row) → `refresh(run_id)` shows the per-round revision for the agent and the trust ladder is non-empty.
  - block 4: with a graded outcome, the scorecard line shows `n_matured >= 1`; chain chip appears only when `verify_chain()` is True.
- `test_main_window` smoke: the `Learning` nav entry exists and `setCurrentIndex` to its page builds without error.
- Full suite green.

## 7. Deliverables (definition of done)
1. `ui/learning_view.py` — `LearningView` with the four blocks + `refresh(run_id=None)`, honest empty-states, read-only + guarded.
2. `main_window` exposes it as the **Learning** tab and refreshes it on completion + tab-show.
3. Tests per §6; full suite green; honest by construction.

## 8. Why it matters
It makes the *real* learning legible and demo-able — exactly the honest "how it learns" story from the defense doc, on screen — without inventing any new adaptation. You can show it live, it never overclaims, and it stays sharply separate from the validated benchmark.
