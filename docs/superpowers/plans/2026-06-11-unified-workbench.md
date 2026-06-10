# Unified Report/Interact Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge Report + Interact into one workbench: the report stays in `Stage4ReportPanel`'s left column; its right column becomes an `Outputs ⇄ Interact` toggle that embeds the existing `Stage5InteractPanel`. The separate Interact stack page + nav item are removed.

**Architecture:** `Stage4ReportPanel` is already a two-column report (`self._left` = report doc, `self._right` = outputs). Wrap the right column in a `QStackedWidget` whose page 0 is the existing Outputs and page 1 is an injected `Stage5InteractPanel`, with a two-button toggle. `main_window` constructs the interact panel, passes it into the report panel, drops the standalone Interact page, and fixes the one shifted index (AgentPerf 5→4). No compute/store change; all existing public methods on both panels are preserved.

**Tech Stack:** Python 3.10, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-unified-workbench-design.md`. (Implementation refinement: rather than a brand-new `workbench.py`, evolve the already-two-column `Stage4ReportPanel` in place — same outcome, far less risk.)

**Prereqs (on `master`), exact anchors:**
- `ui/stage4_report.py`: `__init__(self, parent=None)`; `_build` builds `top_row = QHBoxLayout()`, `self._left = QVBoxLayout()`, `self._right = QVBoxLayout()`, then `top_row.addLayout(self._left, stretch=1)` and `top_row.addLayout(self._right, stretch=1)`. Public methods `populate`, `populate_swarm`, `set_regional_estimates`, `set_chain`, `set_bsp_alert`, `set_policy_recos`, `set_sector_forecasts` all target `self._left`/`self._right`. Imports include `QStackedWidget`? (check; add if missing) and from `PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QPushButton, QFileDialog`.
- `ui/stage5_interact.py`: `Stage5InteractPanel(rag, agents, regressor, df, cv_rmse)`; public `set_swarm_context`, `update_gas_verdict`, `update_food_verdict`, `update_elec_verdict`, `update_context`, `set_debate_engine`, signal `rerun_requested`.
- `ui/main_window.py`:
  - `_TopNavBar._ITEMS` (lines 53–58) = `[(0,'Home',False),(2,'Simulation',True),(3,'Report',True),(4,'Interact',True)]`; `_TopNavBar.__init__` sets `self._locked = {2, 3, 4}` (line 69).
  - Construction (224–226): `self._stage4 = Stage4ReportPanel()` then `self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor, self._df, self._cv_rmse)`.
  - Stack tuple (254–256): `(landing_scroll, self._economy_overview, self._stage3_container, self._stage4, self._stage5, self._agent_perf, self._accuracy_view)` → indices 0..6.
  - `view_performance_requested` (272–273): `lambda: (self._sidebar.set_active(5), self._stack.setCurrentIndex(5))`.
  - `unlock_stages([2, 3, 4])` at lines 578 and 631.
  - All other `setCurrentIndex`/`set_active` use indices ≤3 (unchanged). `self._accuracy_view` (Methodology, index 6) has **no** navigation reference → it simply shifts to 5; nothing to update.
- Tests: `tests/test_stage4_sector.py` and `tests/test_main_window.py` (offscreen Qt, `app` fixture). Existing tests construct `Stage4ReportPanel()` with no args — backward compatibility required.

**Conventions:** Tests in `ph_economic_ai/tests/`. **Git hygiene:** staging clean; commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/unified-workbench
```

---

## Task 1: Right-pane Outputs/Interact toggle in `Stage4ReportPanel`

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py`
- Test: `ph_economic_ai/tests/test_stage4_workbench.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_stage4_workbench.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_right_pane_has_outputs_and_interact(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    interact = QWidget()
    panel = Stage4ReportPanel(interact_panel=interact)
    assert panel._right_stack.count() == 2
    assert panel._right_stack.widget(1) is interact
    panel._set_right_pane(1)
    assert panel._right_stack.currentIndex() == 1
    panel._set_right_pane(0)
    assert panel._right_stack.currentIndex() == 0


def test_no_interact_panel_outputs_only(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()                 # backward compatible (no interact)
    assert panel._right_stack.count() == 1
    assert panel._right_stack.currentIndex() == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_stage4_workbench.py -v`
Expected: FAIL — `Stage4ReportPanel()` has no `interact_panel` kwarg / no `_right_stack`.

- [ ] **Step 3: Add the `interact_panel` arg + right-pane toggle**

In `ph_economic_ai/ui/stage4_report.py`:
(a) Ensure imports include `QStackedWidget` (add to the `from PyQt6.QtWidgets import (...)` line if absent) and `from PyQt6.QtCore import Qt` (add if absent).
(b) Change the constructor:
```python
    def __init__(self, interact_panel=None, parent=None):
        super().__init__(parent)
        self._interact = interact_panel
        self._responses: list = []
        self._consensus: dict = {}
        self._build()
```
(Keep any other existing `__init__` body lines that followed `super().__init__`; just add the `self._interact` line and the param.)
(c) In `_build`, replace:
```python
        top_row.addLayout(self._left, stretch=1)
        top_row.addLayout(self._right, stretch=1)
```
with:
```python
        top_row.addLayout(self._left, stretch=1)
        top_row.addWidget(self._build_right_pane(), stretch=1)
```
(d) Add these two methods to the class (e.g. just after `_build`):
```python
    def _build_right_pane(self) -> QWidget:
        container = QWidget()
        cv = QVBoxLayout(container)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)

        toggle = QHBoxLayout()
        self._btn_outputs = QPushButton('Outputs')
        self._btn_interact = QPushButton('Interact')
        for b in (self._btn_outputs, self._btn_interact):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            toggle.addWidget(b)
        toggle.addStretch()
        cv.addLayout(toggle)

        self._right_stack = QStackedWidget()
        outputs_page = QWidget()
        outputs_page.setLayout(self._right)          # the existing Outputs column
        self._right_stack.addWidget(outputs_page)    # index 0
        if self._interact is not None:
            self._right_stack.addWidget(self._interact)  # index 1
        cv.addWidget(self._right_stack, stretch=1)

        self._btn_outputs.clicked.connect(lambda: self._set_right_pane(0))
        self._btn_interact.clicked.connect(lambda: self._set_right_pane(1))
        self._btn_interact.setVisible(self._interact is not None)
        self._set_right_pane(0)
        return container

    def _set_right_pane(self, idx: int):
        if idx == 1 and self._interact is None:
            return
        self._right_stack.setCurrentIndex(idx)
        for b, on in ((self._btn_outputs, idx == 0), (self._btn_interact, idx == 1)):
            b.setStyleSheet(
                'QPushButton{border:none;border-radius:6px;padding:4px 12px;'
                'font-family:Consolas,monospace;font-size:10px;'
                + ('background:#1C1E26;color:#FFFFFF;' if on
                   else 'background:transparent;color:#9EA3AE;') + '}'
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_stage4_workbench.py ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: PASS (new + the existing stage4 sector tests, which construct `Stage4ReportPanel()` with no args, still work).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_workbench.py
git commit -m "feat(ui): Report panel right column toggles Outputs/Interact (embeds interact panel)"
```

---

## Task 2: Wire the workbench in `main_window` (embed interact, drop the page + nav)

**Files:**
- Modify: `ph_economic_ai/ui/main_window.py`
- Test: window smoke

- [ ] **Step 1: Construct interact first, inject into the report panel**

Replace (lines ~224–226):
```python
        self._stage4 = Stage4ReportPanel()
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)
```
with:
```python
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel(interact_panel=self._stage5)
```

- [ ] **Step 2: Drop the standalone Interact page from the stack**

In the stack tuple (lines ~254–256), remove `self._stage5`:
```python
        # Stack order: 0=Home(scroll), 1=Overview, 2=Simulation, 3=Report(workbench),
        #              4=AgentPerf, 5=Methodology & Accuracy
        for widget in (landing_scroll, self._economy_overview,
                       self._stage3_container, self._stage4,
                       self._agent_perf, self._accuracy_view):
            self._stack.addWidget(widget)
```
(`self._stage5` is no longer a stack page — it now lives inside `self._stage4`'s right pane. All `self._stage5.*` calls elsewhere are unchanged.)

- [ ] **Step 3: Fix the one shifted index (AgentPerf 5→4)**

Change `view_performance_requested` (lines ~272–273):
```python
        self._landing.view_performance_requested.connect(
            lambda: (self._sidebar.set_active(4), self._stack.setCurrentIndex(4)))
```

- [ ] **Step 4: Remove the Interact nav item + unlock list**

In `_TopNavBar._ITEMS` (lines 53–58), remove the Interact row:
```python
    _ITEMS: list[tuple[int, str, bool]] = [
        (0, 'Home',        False),
        (2, 'Simulation',  True),
        (3, 'Report',      True),
    ]
```
In `_TopNavBar.__init__`, change `self._locked = {2, 3, 4}` to `self._locked = {2, 3}`.
Change both `self._sidebar.unlock_stages([2, 3, 4])` (lines ~578, ~631) to `self._sidebar.unlock_stages([2, 3])`.

- [ ] **Step 5: Import + window smoke**

Run: `python -c "import ph_economic_ai.ui.main_window; print('import OK')"` → `import OK`
Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass — `test_main_window_has_swarm_panel`, `test_on_run_requested_accepts_4_args` (navigates to stack index 2), and `test_stage5_has_set_swarm_context` (still constructs `window._stage5`).

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/main_window.py
git commit -m "feat(ui): merge Interact into the Report workbench; drop the separate page + nav item"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (adds passing tests; no regressions).

- [ ] **Manual visual check (GUI session)**

Run a simulation → the Report screen shows the report on the left and a right pane with an `Outputs / Interact` toggle; toggling reveals the chart outputs and the Ask-an-Agent chat respectively. The top nav no longer has a separate "Interact" item; "Report" is the workbench.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 right pane toggle Outputs/Interact embedding the interact panel → Task 1. §3.2 reuse the already-two-column report (no rewrite) → Task 1 (refines the spec's "new file" to in-place evolution; noted in header). §3.3 main_window injects interact + drops the page + sidebar/nav merge + index fix → Task 2. §5 migration safety (all public methods preserved; main_window keeps `self._stage5` ref so its calls are unchanged; backward-compatible `interact_panel=None`) → Tasks 1, 2. §6 testing (toggle 2-page/1-page; window smoke incl. existing stage5 method assertion) → Tasks 1, 2.

**Placeholder scan:** none — every step has complete code/exact strings.

**Type consistency:** `Stage4ReportPanel(interact_panel=None, parent=None)` defined in Task 1, called as `Stage4ReportPanel(interact_panel=self._stage5)` in Task 2. `self._right_stack`/`_set_right_pane`/`_build_right_pane` defined and used within Task 1; tests reference `_right_stack`. `self._right` (existing QVBoxLayout) is re-parented into `outputs_page` via `setLayout` — valid since it was previously only added to `top_row`; that `top_row.addLayout(self._right…)` line is replaced (Task 1c) so the layout isn't double-parented. `self._stage5` still constructed in main_window (Task 2 Step 1) so every existing `self._stage5.*` call and the `rerun_requested` connection remain valid. Index references: only AgentPerf (5→4) is actively navigated and is updated; Methodology shifts 6→5 with no code reference; Report stays 3; Simulation stays 2.
```
