# "How it learns" View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A demo-able **Learning** tab that surfaces Strata's real learning honestly — the 3-layer mechanism, this run's within-run revisions, the trust ladder, and a store-derived graded track record — each with a truthful empty state.

**Architecture:** New `ui/learning_view.py` (`LearningView(store)`) with four blocks + `refresh(run_id)`, added to `main_window` as a stack page + a `_TopNavBar._ITEMS` entry, refreshed on completion and tab-show. Read-only; reuses `AgentPerformancePanel` for the ladder; derives the track record from the store (the `TrackRecord` jsonl is not wired to live runs).

**Tech Stack:** Python 3.10, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-12-how-it-learns-view-design.md`.

**Confirmed anchors:**
- `engine/store.py` `AgentTrustStore(db_path=None)`: `save_run(scenario, final_estimate, confidence_pct)->run_id`; `save_agent_responses(run_id, [ {agent_name,round_num,estimate,statement,citation_count,has_causal_chain,internal_score,model_used} ])`; `get_agent_responses(run_id)->rows`; `get_recent_runs(limit=20)->run dicts incl. actual_price_change/accuracy_error`; `total_runs()->int`; `apply_ground_truth_grade(run_id, actual_change)` (sets `actual_price_change`,`accuracy_error`); `get_all_trust_rows()`.
- `engine/evolution.py`: `_COLD_START_RUNS = 3`.
- `ui/agent_performance.py`: `AgentPerformancePanel(store)` is a `QWidget` with `.refresh()` rendering `get_all_trust_rows()`.
- `ui/theme.py`: helpers `card(title)->(QFrame, QVBoxLayout)`, `muted(text,size=9,color=MUTED,upper=False)->QLabel`, `hairline()->QFrame`, tokens `SURFACE,CARD,INK,MUTED,FAINT,HAIRLINE,MONO,SERIF`. **Verify exact signatures in theme.py before use; adapt if different.**
- `ui/main_window.py`: `_TopNavBar._ITEMS` (line ~53) = list of `(stack_idx, label, locked)`; stack pages added (line ~255) in order `landing(0),overview(1),stage3(2),stage4(3),agent_perf(4),accuracy(5)`; `_on_stage_changed(idx)` (~283); `_on_swarm_complete` sets/【uses `self._current_run_id`】.

**Conventions:** offscreen Qt tests in `ph_economic_ai/tests/`. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`. Never add `self.show()`; tests use `not widget.isHidden()`.

**Task 0 (branch):** Continue on `feature/how-it-learns-view` (already checked out; holds the spec). Confirm `git branch --show-current`.

---

## Task 1: `ui/learning_view.py` — the LearningView

**Files:** Create `ph_economic_ai/ui/learning_view.py`; Test `ph_economic_ai/tests/test_learning_view.py`

- [ ] **Step 1: Write the failing test** — create `ph_economic_ai/tests/test_learning_view.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication
from ph_economic_ai.engine.store import AgentTrustStore


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_empty_states(app, tmp_path):
    from ph_economic_ai.ui.learning_view import LearningView
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    v = LearningView(store)
    assert 'Run a simulation' in v._revisions_lbl.text()         # block 2 empty
    assert '0 runs logged' in v._track_lbl.text()                # block 4 status
    assert 'grading waits' in v._track_lbl.text()                # block 4 empty


def test_revisions_and_grade(app, tmp_path):
    from ph_economic_ai.ui.learning_view import LearningView
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    rid = store.save_run(scenario={'x': 1}, final_estimate=-1.8, confidence_pct=77)
    base = dict(statement='s', citation_count=1, has_causal_chain=0,
                internal_score=0.5, model_used='m')
    store.save_agent_responses(rid, [
        {'agent_name': 'FCST-NCR', 'round_num': 1, 'estimate': -1.2, **base},
        {'agent_name': 'FCST-NCR', 'round_num': 2, 'estimate': -1.8, **base},
    ])
    v = LearningView(store)
    v.refresh(rid)
    txt = v._revisions_lbl.text()
    assert 'FCST-NCR' in txt and 'R1' in txt and 'R2' in txt      # within-run revision
    store.apply_ground_truth_grade(rid, actual_change=-1.5)       # grade it
    v.refresh(rid)
    assert '1 graded' in v._track_lbl.text()                      # store-derived scorecard
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_learning_view.py -v`).

- [ ] **Step 3: Implement** — create `ph_economic_ai/ui/learning_view.py`:
```python
"""The 'How it learns' view — surfaces Strata's real, honest learning (read-only)."""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QScrollArea, QLabel, QFrame

from ph_economic_ai.ui import theme as _t
from ph_economic_ai.ui.agent_performance import AgentPerformancePanel
from ph_economic_ai.engine.evolution import _COLD_START_RUNS

_LAYERS = [
    ("Within a run — it adapts",
     "Agents debate over several rounds; each sees the previous rounds and revises. "
     "This resets every run."),
    ("Across runs — only on real outcomes",
     f"A background checker grades past forecasts against the real DOE pump price "
     f"(~5 days later); trust updates and the swarm evolves after a cold-start of "
     f"{_COLD_START_RUNS} runs. Same-day reruns change nothing."),
    ("The models — frozen",
     "The LLMs are not trained on your runs. What adapts is which agents are trusted, "
     "not the agents themselves."),
]


class LearningView(QWidget):
    def __init__(self, store, parent=None):
        super().__init__(parent)
        self._store = store
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget(); body.setStyleSheet(f'background:{_t.SURFACE};')
        col = QVBoxLayout(body)
        col.setContentsMargins(28, 24, 28, 24); col.setSpacing(16)
        scroll.setWidget(body)

        # Block 1 — How it learns (static, always truthful)
        c1, l1 = _t.card('How it learns')
        for head, desc in _LAYERS:
            h = QLabel(head)
            h.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:11px;font-weight:700;')
            l1.addWidget(h)
            d = QLabel(desc); d.setWordWrap(True)
            d.setStyleSheet(f'color:{_t.MUTED};font-family:{_t.MONO};font-size:10px;')
            l1.addWidget(d)
            l1.addWidget(_t.hairline())
        col.addWidget(c1)

        # Block 2 — This run's revisions
        c2, l2 = _t.card("This run's revisions")
        self._revisions_lbl = QLabel(); self._revisions_lbl.setWordWrap(True)
        self._revisions_lbl.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:10px;')
        l2.addWidget(self._revisions_lbl)
        col.addWidget(c2)

        # Block 3 — Trust ladder (reuse AgentPerformancePanel)
        c3, l3 = _t.card('Trust ladder')
        cap = QLabel('Agents rise and fall as real outcomes grade their past calls.')
        cap.setWordWrap(True)
        cap.setStyleSheet(f'color:{_t.MUTED};font-family:{_t.MONO};font-size:9px;')
        l3.addWidget(cap)
        self._perf = AgentPerformancePanel(self._store)
        l3.addWidget(self._perf)
        col.addWidget(c3)

        # Block 4 — Track record (store-derived)
        c4, l4 = _t.card('Track record')
        self._track_lbl = QLabel(); self._track_lbl.setWordWrap(True)
        self._track_lbl.setStyleSheet(f'color:{_t.INK};font-family:{_t.MONO};font-size:10px;')
        l4.addWidget(self._track_lbl)
        col.addWidget(c4)

        col.addStretch(1)
        self.refresh(None)

    def refresh(self, run_id=None):
        rows = []
        try:
            if run_id is not None:
                rows = self._store.get_agent_responses(run_id)
        except Exception:
            rows = []
        self._revisions_lbl.setText(self._format_revisions(rows))
        try:
            self._perf.refresh()
        except Exception:
            pass
        self._track_lbl.setText(self._format_track())

    @staticmethod
    def _fmt_est(v):
        return f'{v:+.2f}' if isinstance(v, (int, float)) else '—'

    def _format_revisions(self, rows):
        if not rows:
            return "Run a simulation to see agents revise across debate rounds."
        by_agent: dict = {}
        for r in rows:
            by_agent.setdefault(r.get('agent_name', '?'), []).append(
                (r.get('round_num', 0), r.get('estimate')))
        lines = []
        for name in sorted(by_agent):
            seq = sorted(by_agent[name], key=lambda t: t[0])
            chain = '  ->  '.join(f"R{rn} {self._fmt_est(est)}" for rn, est in seq)
            lines.append(f'{name:<22} {chain}')
        return '\n'.join(lines)

    def _format_track(self):
        try:
            logged = self._store.total_runs()
            recent = self._store.get_recent_runs(limit=200)
            graded = [r for r in recent if r.get('actual_price_change') is not None]
            n_graded = len(graded)
            errs = [r['accuracy_error'] for r in graded if r.get('accuracy_error') is not None]
            mae = (sum(errs) / len(errs)) if errs else None
        except Exception:
            logged, n_graded, mae = 0, 0, None
        evo = 'active' if logged >= _COLD_START_RUNS else f'activates at {_COLD_START_RUNS}'
        out = [f'{logged} runs logged   .   {n_graded} graded   .   evolution {evo}']
        if n_graded == 0:
            out.append('No graded outcomes yet - grading waits ~5 days for real DOE pump prices.')
        elif mae is not None:
            out.append(f'mean abs error: ₱{mae:.2f}/L over {n_graded} graded runs')
        return '\n'.join(out)
```
NOTE: confirm `theme.card`/`muted`/`hairline` signatures in `ui/theme.py`; if `card` returns something different, adapt (the only hard requirement is a title + a vertical layout to add widgets to). Keep the public attrs `_revisions_lbl` and `_track_lbl` (the tests use them).

- [ ] **Step 4: Run → passes** (`python -m pytest ph_economic_ai/tests/test_learning_view.py -v`). Also `python -c "import ph_economic_ai.ui.learning_view; print('import OK')"`.

- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/learning_view.py ph_economic_ai/tests/test_learning_view.py && git commit -m "feat(ui): 'How it learns' view (mechanism + revisions + trust ladder + track record)"`

---

## Task 2: Wire the Learning tab into `main_window`

**Files:** Modify `ph_economic_ai/ui/main_window.py`; Test `ph_economic_ai/tests/test_main_window.py` (append)

- [ ] **Step 1: Append the smoke test**
```python
def test_learning_tab_present_and_refreshes(window):
    from ph_economic_ai.ui.main_window import _TopNavBar
    from ph_economic_ai.ui.learning_view import LearningView
    labels = [lbl for _idx, lbl, _lk in _TopNavBar._ITEMS]
    assert 'Learning' in labels                                  # nav tab exists
    # its stack page is a LearningView and is reachable
    learn_idx = next(i for i, lbl, _ in _TopNavBar._ITEMS if lbl == 'Learning')
    window._stack.setCurrentIndex(learn_idx)
    assert isinstance(window._stack.widget(learn_idx), LearningView)
    window._learning.refresh(None)                              # no crash, empty-states
    assert '0 runs logged' in window._learning._track_lbl.text() or 'runs logged' in window._learning._track_lbl.text()
```
(If the `window` fixture builds with a seeded store, adjust the final assertion to just `'runs logged' in …`.)

- [ ] **Step 2: Run → fails** (no Learning tab / `_learning`).

- [ ] **Step 3: Implement** — in `main_window.py`:
  (a) import: `from ph_economic_ai.ui.learning_view import LearningView`.
  (b) In `_TopNavBar._ITEMS`, append `(6, 'Learning', False)` (the new stack index, unlocked).
  (c) Where panels are constructed (near `self._agent_perf = AgentPerformancePanel(self._store)`), add `self._learning = LearningView(self._store)`.
  (d) In the stack `addWidget` tuple (currently `landing_scroll, self._economy_overview, self._stage3_container, self._stage4, self._agent_perf, self._accuracy_view`), append `self._learning` as the LAST item → it becomes index 6. Update the `# Stack order:` comment.
  (e) In `_on_stage_changed(idx)`, add:
  ```python
          if idx == 6:
              self._learning.refresh(getattr(self, '_current_run_id', None))
  ```
  (f) In `_on_swarm_complete`, after the run is graded/saved (where `self._current_run_id` is set), add:
  ```python
          try:
              self._learning.refresh(self._current_run_id)
          except Exception:
              pass
  ```

- [ ] **Step 4: Run → passes** (`python -m pytest ph_economic_ai/tests/test_main_window.py -v`). Then `python -c "import ph_economic_ai.ui.main_window; print('import OK')"`.

- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/main_window.py ph_economic_ai/tests/test_main_window.py && git commit -m "feat(ui): add the Learning tab + refresh on completion/tab-show"`

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): a **Learning** tab appears in the top nav; it shows the 3-layer explainer, this run's per-round revisions after a run, the trust ladder, and a store-derived track-record status with honest empty-states.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 Block 1 (explainer) / Block 2 (revisions via `get_agent_responses`) / Block 3 (embedded `AgentPerformancePanel`) / Block 4 (store-derived `total_runs`+`get_recent_runs`, the corrected source) → Task 1; §3.2 wiring (construct, stack page, `_ITEMS` tab, refresh on complete + `_on_stage_changed`) → Task 2; §5 robustness (per-block try/except, honest empty-states) → Task 1 `refresh`/`_format_*`; §6 testing → both tasks.
**Placeholder scan:** none — full `LearningView` code, both tests, and the six concrete main_window edits. The one soft spot (theme helper signatures) is flagged with a concrete fallback and the only hard contract (`_revisions_lbl`/`_track_lbl`) is stated.
**Type consistency:** `LearningView(store)` + `refresh(run_id=None)` used identically in Task 2. Block 4 reads `r['actual_price_change']`/`r['accuracy_error']` — the exact fields `apply_ground_truth_grade` writes. `_COLD_START_RUNS` imported from `evolution`. Stack index 6 is consistent across the `_ITEMS` entry (c/b), the addWidget append (d), and `_on_stage_changed` (e). No `TrackRecord`/`verify_chain` anywhere (the corrected, honest source).
