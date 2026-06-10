# Three-Sector Forecast Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the gas/food/electricity forecasts a run already computes in one honest "Next-month sector forecast" card on the Report, each with direction + correct unit + an "exploratory — not validated" label, and relabel the fuel-only history strip.

**Architecture:** A pure formatter (`ui/sector_forecast.py`) turns the three numeric estimates into display rows; `stage4_report` renders them in a persistent card; `main_window` captures the three estimates a run already produces and pushes them. Clarity-only — no store change, no recompute.

**Tech Stack:** Python 3.10, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-three-sector-forecast-design.md`.

**Prereqs (on `master`):**
- `ui/stage4_report.py::Stage4ReportPanel._build` creates `root = QVBoxLayout(self)`; adds a top bar then `self._bsp_banner` to `root`; the left/right columns are rebuilt on `populate_swarm`, but `root` and items added directly to it persist. `QLabel`, `QWidget`, `QVBoxLayout`, `QFrame` are imported.
- `ui/main_window.py`: `_on_swarm_complete(self, master_verdict)` — `master_verdict.final_estimate` is the gas ₱/L estimate; sets `self._gas_verdict`. `_on_food_complete(self, responses)` — computes `avg = c.get('weighted_avg')` (food %, inside `if responses and self._food_engine:`). `_on_elec_complete(self, responses)` — computes `avg = c.get('weighted_avg')` (electricity ₱/kWh). `__init__` initialises `self._gas_verdict = ''` etc. (~line 314); a new run resets `self._gas_verdict=''`/`_food_verdict=''`/`_elec_verdict=''` (~line 314–316). `self._stage4` is the Stage4ReportPanel.
- `ui/landing.py::_build_recent_strip` (~line 413) renders a heading with the text "RECENT WORK".

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top; PyQt tests use `os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')`. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/three-sector-forecast
```

---

## File Structure
**Create:** `ui/sector_forecast.py`; tests `test_sector_forecast.py`, `test_stage4_sector.py`.
**Modify:** `ui/stage4_report.py` (persistent card + `set_sector_forecasts`), `ui/main_window.py` (capture + push), `ui/landing.py` (relabel).

---

## Task 1: Pure sector-forecast formatter

**Files:**
- Create: `ph_economic_ai/ui/sector_forecast.py`
- Test: `ph_economic_ai/tests/test_sector_forecast.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_sector_forecast.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.ui.sector_forecast import sector_forecast_rows


def test_values_units_directions():
    rows = sector_forecast_rows(-2.40, 0.50, 0.05)
    by = {r['key']: r for r in rows}
    assert by['gas']['direction'] == 'down'
    assert '-2.40' in by['gas']['value_str'] and '/L' in by['gas']['value_str']
    assert by['food']['direction'] == 'up'
    assert '+0.50' in by['food']['value_str'] and '%' in by['food']['value_str']
    assert by['elec']['direction'] == 'up'
    assert '+0.0500' in by['elec']['value_str'] and 'kWh' in by['elec']['value_str']


def test_none_and_flat():
    rows = sector_forecast_rows(None, 0.0, None)
    by = {r['key']: r for r in rows}
    assert by['gas']['value_str'] == '—' and by['gas']['direction'] == 'na'
    assert by['food']['direction'] == 'flat'
    assert by['elec']['value_str'] == '—' and by['elec']['direction'] == 'na'


def test_always_three_rows_in_order():
    assert [r['key'] for r in sector_forecast_rows()] == ['gas', 'food', 'elec']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_sector_forecast.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.ui.sector_forecast`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/ui/sector_forecast.py`:

```python
"""Pure formatting of the three sector forecasts for display (no PyQt).

A run produces a next-month estimate for each sector in its own unit:
gas/fuel ₱/L, food %, electricity ₱/kWh. This turns the raw numbers into
display rows; any sector may be None (debate unavailable) -> shown as an em dash.
"""
from typing import Optional

# (key, label, value format string)
_SECTORS = [
    ('gas',  'Gas / fuel',  '{:+.2f} ₱/L'),
    ('food', 'Food',        '{:+.2f} %'),
    ('elec', 'Electricity', '{:+.4f} ₱/kWh'),
]


def _direction(v: Optional[float]) -> str:
    if v is None:
        return 'na'
    if v > 0:
        return 'up'
    if v < 0:
        return 'down'
    return 'flat'


def sector_forecast_rows(gas: Optional[float] = None, food: Optional[float] = None,
                         elec: Optional[float] = None) -> list:
    """Return one display row per sector (always gas, food, elec in that order):
    {key, label, value, value_str, direction}. direction in up/down/flat/na."""
    vals = {'gas': gas, 'food': food, 'elec': elec}
    rows = []
    for key, label, fmt in _SECTORS:
        v = vals[key]
        rows.append({
            'key': key,
            'label': label,
            'value': v,
            'value_str': fmt.format(v) if v is not None else '—',
            'direction': _direction(v),
        })
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_sector_forecast.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/sector_forecast.py ph_economic_ai/tests/test_sector_forecast.py
git commit -m "feat(ui): pure sector-forecast row formatter (gas/food/electricity)"
```

---

## Task 2: Render the card in the Report

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py`
- Test: `ph_economic_ai/tests/test_stage4_sector.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_stage4_sector.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_set_sector_forecasts_renders_card(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-2.40, 0.50, 0.05)
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'SECTOR FORECAST' in texts.upper()
    assert 'Gas / fuel' in texts and '/L' in texts
    assert 'Food' in texts and '%' in texts
    assert 'Electricity' in texts and 'kWh' in texts
    assert 'exploratory' in texts.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: FAIL — `AttributeError: 'Stage4ReportPanel' object has no attribute 'set_sector_forecasts'`

- [ ] **Step 3: Add the persistent holder + method**

In `ph_economic_ai/ui/stage4_report.py`, in `_build`, immediately after `root.addWidget(self._bsp_banner)`, add:
```python
        # Persistent three-sector forecast card (populated via set_sector_forecasts)
        self._sector_holder = QWidget()
        self._sector_holder.setStyleSheet('background:#FFFFFF;border-bottom:1px solid #EAECF0;')
        self._sector_holder_layout = QVBoxLayout(self._sector_holder)
        self._sector_holder_layout.setContentsMargins(20, 8, 20, 8)
        self._sector_holder_layout.setSpacing(2)
        self._sector_holder.setVisible(False)
        root.addWidget(self._sector_holder)
```
Then add the method (anywhere in the class, e.g. after `_build`):
```python
    def set_sector_forecasts(self, gas=None, food=None, elec=None):
        """Render the gas/food/electricity next-month forecasts as a card."""
        from ph_economic_ai.ui.sector_forecast import sector_forecast_rows
        try:
            while self._sector_holder_layout.count():
                it = self._sector_holder_layout.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.deleteLater()
            title = QLabel('NEXT-MONTH SECTOR FORECAST')
            title.setStyleSheet('font-size:10px;font-weight:700;letter-spacing:1px;'
                                'color:#6B7280;')
            self._sector_holder_layout.addWidget(title)
            sub = QLabel('exploratory — not validated')
            sub.setStyleSheet('font-size:9px;color:#9EA3AE;')
            self._sector_holder_layout.addWidget(sub)
            arrows = {'up': '▲', 'down': '▼', 'flat': '■', 'na': '·'}
            # up = price rising = bad (red); down = falling = good (green)
            colors = {'up': '#EF4444', 'down': '#16A34A', 'flat': '#6B7280', 'na': '#9EA3AE'}
            for r in sector_forecast_rows(gas, food, elec):
                lbl = QLabel(f"{r['label']}:  {arrows[r['direction']]}  {r['value_str']}")
                lbl.setStyleSheet(f"font-size:12px;font-weight:600;color:{colors[r['direction']]};")
                self._sector_holder_layout.addWidget(lbl)
            self._sector_holder.setVisible(True)
        except Exception:
            pass
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_sector.py
git commit -m "feat(ui): three-sector forecast card on the Report (set_sector_forecasts)"
```

---

## Task 3: Capture the three estimates in main_window and push them

**Files:**
- Modify: `ph_economic_ai/ui/main_window.py`
- Test: window smoke

- [ ] **Step 1: Initialise the three estimates**

In `ph_economic_ai/ui/main_window.py::__init__`, near the existing `self._gas_verdict = ''` initialisation (~line 314), add:
```python
        self._gas_estimate = None
        self._food_estimate = None
        self._elec_estimate = None
```
And wherever a new run resets the verdicts (the block setting `self._gas_verdict = ''`, `self._food_verdict = ''`, `self._elec_verdict = ''`), add alongside:
```python
        self._gas_estimate = None
        self._food_estimate = None
        self._elec_estimate = None
```

- [ ] **Step 2: Add the push helper**

Add a method to the class (e.g. near `_on_food_complete`):
```python
    def _push_sector_forecasts(self):
        try:
            self._stage4.set_sector_forecasts(
                self._gas_estimate, self._food_estimate, self._elec_estimate)
        except Exception:
            pass
```

- [ ] **Step 3: Capture gas + push in `_on_swarm_complete`**

In `_on_swarm_complete`, after `self._gas_verdict = str(master_verdict)` (or after `populate_swarm`), add:
```python
        self._gas_estimate = getattr(master_verdict, 'final_estimate', None)
        self._push_sector_forecasts()
```

- [ ] **Step 4: Capture food + electricity + push**

In `_on_food_complete`, inside the `if responses and self._food_engine:` block right after `avg = c.get('weighted_avg')`, add `self._food_estimate = avg`; and at the end of the method (after `self._stage5.update_food_verdict(...)`) add `self._push_sector_forecasts()`.

In `_on_elec_complete`, inside its `if responses and self._elec_engine:` block right after `avg = c.get('weighted_avg')`, add `self._elec_estimate = avg`; and at the end of the method (after `self._stage5.update_elec_verdict(...)`) add `self._push_sector_forecasts()`.

- [ ] **Step 5: Window smoke + import check**

Run: `python -c "import ph_economic_ai.ui.main_window; import ph_economic_ai.ui.stage4_report; print('import OK')"`
Expected: `import OK`
Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass (suite is currently fully green).

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/main_window.py
git commit -m "feat(ui): push gas/food/electricity estimates to the Report sector card"
```

---

## Task 4: Relabel the landing fuel-history strip

**Files:**
- Modify: `ph_economic_ai/ui/landing.py`

- [ ] **Step 1: Relabel**

In `ph_economic_ai/ui/landing.py` (`_build_recent_strip`, ~line 413), find the heading label with text `'RECENT WORK'` and change it to `'RECENT FUEL FORECASTS'`. Change only the displayed string; leave everything else.

- [ ] **Step 2: Verify + smoke**

Run: `python -c "import ph_economic_ai.ui.landing; s=open('ph_economic_ai/ui/landing.py',encoding='utf-8').read(); print('relabeled:', 'RECENT FUEL FORECASTS' in s and 'RECENT WORK' not in s)"`
Expected: `relabeled: True`
Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass (no new failures).

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/ui/landing.py
git commit -m "feat(ui): relabel landing strip 'RECENT FUEL FORECASTS' (clarify fuel-only)"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (the suite is currently fully green; this adds passing tests).

- [ ] **Manual visual check (optional, GUI session)**

Run a simulation, open Report: a "NEXT-MONTH SECTOR FORECAST · exploratory — not validated" card shows Gas (₱/L), Food (%), Electricity (₱/kWh) each with an up/down indicator; the landing strip reads "RECENT FUEL FORECASTS".

---

## Self-Review (completed by plan author)

**Spec coverage:** §4.1 pure formatter → Task 1. §4.2 `set_sector_forecasts` card → Task 2. §4.3 main_window capture + push → Task 3. §4.4 landing relabel → Task 4. §6 error handling (None → "—"/na in formatter; card + push wrapped in try/except) → Tasks 1–3. §7 testing (formatter units/None/flat; card render; window smoke) → Tasks 1–3.

**Placeholder scan:** none — all code steps contain complete code; no TBD/vague items.

**Type consistency:** `sector_forecast_rows(gas=None, food=None, elec=None) -> list[dict{key,label,value,value_str,direction}]` defined in Task 1, consumed in Task 2's `set_sector_forecasts`. `set_sector_forecasts(self, gas=None, food=None, elec=None)` defined in Task 2, called via `_push_sector_forecasts` in Task 3. `self._gas_estimate/_food_estimate/_elec_estimate` initialised (Task 3 Step 1) before use (Steps 3–4). Persistent `self._sector_holder`/`self._sector_holder_layout` created in `_build` (Task 2 Step 3) before `set_sector_forecasts` uses them. Direction keys (up/down/flat/na) match between formatter (Task 1) and the arrows/colors dicts (Task 2).
```
