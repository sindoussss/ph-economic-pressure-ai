# Landing "Latest Forecast" Combined Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the latest run's full three-sector forecast (gas/food/electricity) on the landing, combined into one card with the existing fuel-history strip.

**Architecture:** Persist food + electricity estimates per run in the store (idempotent migration + an update method); `main_window` updates the saved run when those debates finish; the landing's recent-strip card gains a "LATEST FORECAST" 3-sector row (reusing the tested `sector_forecast_rows`) above the fuel history.

**Tech Stack:** Python 3.10, sqlite3, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-landing-latest-forecast-card-design.md`.

**Prereqs (on `master`):**
- `engine/store.py::AgentTrustStore`: `_migrate` runs `cur.executescript('CREATE TABLE IF NOT EXISTS runs (...)')` then `self._conn.commit()`; `row_factory = sqlite3.Row`. `save_run(scenario, final_estimate, confidence_pct) -> int` (returns run_id). `update_run_quality(run_id, internal_quality)` is the update pattern. `get_recent_runs(limit=20)` does `SELECT * ... ORDER BY run_id DESC` → list[dict]. `from typing import Optional` already imported.
- `ui/main_window.py`: `self._current_run_id` (set on `save_run`), `self._store`, `self._food_estimate`/`self._elec_estimate` (set in `_on_food_complete`/`_on_elec_complete`, which already call `self._push_sector_forecasts()` at their end).
- `ui/landing.py`: `LandingPanel(store=None, parent=None)`; `_build_recent_strip()` builds a `wrap` QFrame with `wrap.setFixedHeight(124)` and a centered column `cl`, a `head = QLabel('RECENT FUEL FORECASTS')`, and `self._runs_row` (QHBoxLayout) with run tiles. `_refresh_recent_runs()` fetches `runs = self._store.get_recent_runs(limit=4)`, clears/builds `self._runs_row`. Palette constants `UP`, `DOWN`, `TEXT_3` and widgets `QWidget`/`QVBoxLayout`/`QHBoxLayout`/`QLabel` are available in the module.
- `ui/sector_forecast.py::sector_forecast_rows(gas, food, elec) -> list[{key,label,value,value_str,direction}]` (direction up/down/flat/na).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top; PyQt tests set `os.environ.setdefault('QT_QPA_PLATFORM','offscreen')`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` first.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/landing-latest-forecast
```

---

## File Structure
**Create:** tests `test_store_sectors.py`, `test_landing_latest.py`.
**Modify:** `engine/store.py` (migration + `update_run_sectors`), `ui/main_window.py` (update call), `ui/landing.py` (combined card).

---

## Task 1: Store — persist food/electricity per run

**Files:**
- Modify: `ph_economic_ai/engine/store.py`
- Test: `ph_economic_ai/tests/test_store_sectors.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_store_sectors.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.store import AgentTrustStore


def test_update_and_read_sectors(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't.db'))
    rid = s.save_run({'x': 1}, final_estimate=-2.40, confidence_pct=54)
    s.update_run_sectors(rid, 0.50, 0.05)
    run = s.get_recent_runs(1)[0]
    assert run['final_estimate'] == -2.40
    assert run['food_estimate'] == 0.50
    assert run['electricity_estimate'] == 0.05


def test_unset_sectors_are_none(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't2.db'))
    s.save_run({'x': 1}, final_estimate=1.0, confidence_pct=50)
    run = s.get_recent_runs(1)[0]
    assert run['food_estimate'] is None
    assert run['electricity_estimate'] is None


def test_migration_idempotent(tmp_path):
    s = AgentTrustStore(str(tmp_path / 't3.db'))
    s._migrate()   # second call must not raise (columns already present)
    rid = s.save_run({'x': 1}, final_estimate=1.0, confidence_pct=50)
    s.update_run_sectors(rid, 0.1, 0.2)
    assert s.get_recent_runs(1)[0]['food_estimate'] == 0.1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_store_sectors.py -v`
Expected: FAIL — `AttributeError: 'AgentTrustStore' object has no attribute 'update_run_sectors'` (and/or no such column).

- [ ] **Step 3: Implement**

In `ph_economic_ai/engine/store.py`, in `_migrate`, immediately AFTER `self._conn.commit()` (the line ending the `executescript`), add an idempotent column migration:
```python
        # Sector estimates added after initial release — add to existing DBs.
        existing = {r['name'] for r in cur.execute('PRAGMA table_info(runs)').fetchall()}
        for col in ('food_estimate', 'electricity_estimate'):
            if col not in existing:
                cur.execute(f'ALTER TABLE runs ADD COLUMN {col} REAL')
        self._conn.commit()
```
Then add the update method (next to `update_run_quality`):
```python
    def update_run_sectors(self, run_id: int, food_estimate: Optional[float],
                           electricity_estimate: Optional[float]) -> None:
        with self._lock:
            self._conn.execute(
                'UPDATE runs SET food_estimate=?, electricity_estimate=? WHERE run_id=?',
                (food_estimate, electricity_estimate, run_id),
            )
            self._conn.commit()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_store_sectors.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/store.py ph_economic_ai/tests/test_store_sectors.py
git commit -m "feat(store): persist food/electricity estimates per run (+ update_run_sectors)"
```

---

## Task 2: main_window — update the saved run with food/electricity

**Files:**
- Modify: `ph_economic_ai/ui/main_window.py`
- Test: window smoke

- [ ] **Step 1: Add the update in both sector handlers**

In `ph_economic_ai/ui/main_window.py`, at the END of `_on_food_complete` (after the existing `self._push_sector_forecasts()` line you added earlier), add:
```python
        if self._store is not None and self._current_run_id is not None:
            try:
                self._store.update_run_sectors(
                    self._current_run_id, self._food_estimate, self._elec_estimate)
            except Exception:
                pass
```
Add the identical block at the END of `_on_elec_complete` (after its `self._push_sector_forecasts()`).

- [ ] **Step 2: Import + window smoke**

Run: `python -c "import ph_economic_ai.ui.main_window; print('import OK')"`
Expected: `import OK`
Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass (suite is fully green).

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/ui/main_window.py
git commit -m "feat(ui): persist food/electricity to the saved run when debates complete"
```

---

## Task 3: Landing — combined card (latest 3-sector forecast + fuel history)

**Files:**
- Modify: `ph_economic_ai/ui/landing.py`
- Test: `ph_economic_ai/tests/test_landing_latest.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_landing_latest.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


class _FakeStore:
    def __init__(self, runs):
        self._runs = runs

    def get_recent_runs(self, limit=20):
        return self._runs[:limit]


def test_landing_latest_shows_three_sectors(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [{'run_id': 4, 'timestamp': '2026-06-10T00:00:00+00:00',
             'final_estimate': -2.40, 'confidence_pct': 54,
             'food_estimate': 0.50, 'electricity_estimate': 0.05,
             'actual_price_change': None}]
    panel = LandingPanel(store=_FakeStore(runs))
    panel._refresh_recent_runs()
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in texts
    assert 'Gas / fuel' in texts and 'Food' in texts and 'Electricity' in texts
    assert '/L' in texts and '%' in texts and 'kWh' in texts


def test_landing_empty_store_no_crash(app):
    from ph_economic_ai.ui.landing import LandingPanel
    panel = LandingPanel(store=_FakeStore([]))
    panel._refresh_recent_runs()  # must not raise
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'No simulations on record yet.' in texts
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_landing_latest.py -v`
Expected: FAIL — `'LATEST FORECAST'` not found (no latest-forecast block yet).

- [ ] **Step 3: Add the latest-forecast row to the card**

In `_build_recent_strip`:
(a) Replace `wrap.setFixedHeight(124)` with:
```python
        wrap.setMinimumHeight(190)
```
(b) Immediately BEFORE the existing `head = QLabel('RECENT FUEL FORECASTS')` line, insert:
```python
        latest_head = QLabel('LATEST FORECAST  ·  exploratory')
        latest_head.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:2px;'
        )
        cl.addWidget(latest_head)
        self._latest_row = QHBoxLayout()
        self._latest_row.setSpacing(28)
        cl.addLayout(self._latest_row)
```

- [ ] **Step 4: Add the sector-tile builder**

Add this method to `LandingPanel` (e.g. just before `_build_run_tile`):
```python
    def _build_sector_tile(self, r: dict) -> QWidget:
        tile = QWidget()
        tile.setStyleSheet('background:transparent;')
        tile.setMaximumWidth(180)
        v = QVBoxLayout(tile)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        name = QLabel(r['label'])
        name.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1px;'
        )
        v.addWidget(name)
        arrows = {'up': '▲', 'down': '▼', 'flat': '■', 'na': '·'}
        color = {'up': UP, 'down': DOWN, 'flat': TEXT_3, 'na': TEXT_3}[r['direction']]
        val = QLabel(f"{arrows[r['direction']]}  {r['value_str']}")
        val.setStyleSheet(f'font-size:15px;font-weight:700;color:{color};')
        v.addWidget(val)
        return tile
```

- [ ] **Step 5: Populate the latest row in `_refresh_recent_runs`**

In `_refresh_recent_runs`, immediately AFTER the `runs = ...` fetch (the try/except that sets `runs`) and BEFORE the existing `# Clear existing` of `self._runs_row`, insert:
```python
        # Latest 3-sector forecast (top of the card)
        while self._latest_row.count():
            it = self._latest_row.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if runs:
            from ph_economic_ai.ui.sector_forecast import sector_forecast_rows
            latest = runs[0]
            for r in sector_forecast_rows(
                gas=latest.get('final_estimate'),
                food=latest.get('food_estimate'),
                elec=latest.get('electricity_estimate'),
            ):
                self._latest_row.addWidget(self._build_sector_tile(r))
        self._latest_row.addStretch()
```
(The existing fuel-history logic below is unchanged.)

- [ ] **Step 6: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_landing_latest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Window smoke**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass (no new failures).

- [ ] **Step 8: Commit**

```bash
git add ph_economic_ai/ui/landing.py ph_economic_ai/tests/test_landing_latest.py
git commit -m "feat(ui): landing card shows latest 3-sector forecast above fuel history"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (adds passing tests; suite currently fully green).

- [ ] **Manual visual check (optional, GUI session)**

Run a simulation, return to Home: the recent-forecasts card now shows a "LATEST FORECAST · exploratory" row with Gas/Food/Electricity (each ↑/↓ + unit), above the "RECENT FUEL FORECASTS" history tiles.

---

## Self-Review (completed by plan author)

**Spec coverage:** §4.1 store migration + `update_run_sectors` → Task 1. §4.2 main_window update → Task 2. §4.3 landing combined card (latest row + tiles + refresh) → Task 3. §6 error handling (None → "—" via formatter; store/run-id guard; idempotent migration) → Tasks 1–3. §7 testing (store round-trip/idempotent/none; landing render + empty; window smoke) → Tasks 1–3.

**Placeholder scan:** none — all code steps contain complete code.

**Type consistency:** `update_run_sectors(run_id, food_estimate, electricity_estimate)` defined in Task 1; called with `(self._current_run_id, self._food_estimate, self._elec_estimate)` in Task 2. Column names `food_estimate`/`electricity_estimate` match between migration (Task 1), the UPDATE (Task 1), and the landing read `latest.get('food_estimate')`/`latest.get('electricity_estimate')` (Task 3). `sector_forecast_rows(gas, food, elec)` consumed in Task 3 with the run's `final_estimate`/`food_estimate`/`electricity_estimate`. `self._latest_row` created in `_build_recent_strip` (Task 3 Step 3) before `_refresh_recent_runs` uses it (Step 5); `_build_sector_tile` (Step 4) used in Step 5. Palette `UP`/`DOWN`/`TEXT_3` reused from the existing module.
```
