# Real 3-Sector Trajectory Charts (SP2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Recent sector trajectories" small-multiples panel to the report — gas (₱/L, from `df`), food + electricity (CPI MoM %, from committed PSA gold) — with a forecast marker where the unit aligns (gas + food); electricity history-only.

**Architecture:** All in `ui/stage4_report.py`. A persistent `_trajectory_holder` built by `set_sector_forecasts` (which already fires when estimates arrive). The report self-serves history: `df['gas_price']` (stored from `populate`/`populate_swarm`) + `benchmark.psa_cpi.load_food_mom/load_electricity_mom` (read-only committed CSVs). No `main_window`/swarm changes. Everything wrapped so missing data never crashes the report.

**Tech Stack:** Python 3.10, PyQt6, matplotlib, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-11-sector-trajectories-design.md`.

**Confirmed anchors (`ui/stage4_report.py`):**
- `_build`: body uses a centred `inner`/`body_layout`; `body_layout.addLayout(top_row)` at line 99, then `self._reco_widget`/`self._map_widget` added (lines 102/106). `Figure` + `FigureCanvasQTAgg`, `QWidget`, `QVBoxLayout`, `QLabel` are imported in this file.
- `set_sector_forecasts(self, gas=None, food=None, elec=None)` (~line 163) builds the magnitude-bar card into `self._sector_holder_layout`, all inside a `try/except`.
- `populate(self, responses, consensus, regressor, df, cv_rmse, scenario)` (~line 190) and `populate_swarm(self, master_verdict, regressor, df, cv_rmse, scenario)` (~line 209) both receive `df`.
- `benchmark/psa_cpi.py`: `load_food_mom()` / `load_electricity_mom()` → `pd.Series` (MoM %, index 'YYYY-MM') reading committed CSVs (`psa_food_cpi_monthly.csv`, `psa_electricity_cpi_monthly.csv` — both present). No network.

**Conventions:** Tests in `ph_economic_ai/tests/`, offscreen Qt. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/sector-trajectories`

---

## Task 1: Persistent `_trajectory_holder` + store `df`

**Files:** Modify `ph_economic_ai/ui/stage4_report.py`. Verification: import + construct.

- [ ] **Step 1:** In `__init__` (where other attrs are initialised, before `self._build()`), add `self._df = None`.

- [ ] **Step 2:** In `populate` and `populate_swarm`, add `self._df = df` near the top of each method body (e.g. right after the docstring / first statement).

- [ ] **Step 3:** In `_build`, immediately after `body_layout.addLayout(top_row)` (line ~99) and before the policy-reco widget, insert:
```python
        # Recent sector trajectories (small multiples) — populated by set_sector_forecasts
        self._trajectory_holder = QWidget()
        self._trajectory_holder_layout = QVBoxLayout(self._trajectory_holder)
        self._trajectory_holder_layout.setContentsMargins(0, 0, 0, 0)
        self._trajectory_holder_layout.setSpacing(6)
        self._trajectory_holder.setVisible(False)
        body_layout.addWidget(self._trajectory_holder)
```

- [ ] **Step 4: Verify construct**

Run: `python -c "import os; os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); import sys; from PyQt6.QtWidgets import QApplication; a=QApplication.instance() or QApplication(sys.argv); from ph_economic_ai.ui.stage4_report import Stage4ReportPanel; p=Stage4ReportPanel(); print('holder', hasattr(p,'_trajectory_holder'), 'df', p._df)"`
Expected: `holder True df None`.

- [ ] **Step 5: Commit**
```bash
git add ph_economic_ai/ui/stage4_report.py
git commit -m "feat(ui): persistent sector-trajectory holder + store df in report"
```

---

## Task 2: Build the 3 trajectory mini-charts

**Files:** Modify `ph_economic_ai/ui/stage4_report.py`; Test `ph_economic_ai/tests/test_stage4_trajectories.py`

- [ ] **Step 1: Failing test** — create `ph_economic_ai/tests/test_stage4_trajectories.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_trajectories_built_with_markers(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p._df = pd.DataFrame({'gas_price': [58., 59., 60., 61., 60.5, 60.]})
    p.set_sector_forecasts(-1.8, -2.6, 0.18)
    assert p._trajectory_holder.isVisible()
    canvases = p._trajectory_holder.findChildren(FigureCanvasQTAgg)
    assert len(canvases) >= 1                       # >=1 (3 when gold present)
    texts = ' || '.join(l.text() for l in p._trajectory_holder.findChildren(QLabel))
    assert 'kWh' in texts                            # electricity ₱/kWh note (gold present)


def test_trajectories_graceful_without_df(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p._df = None
    p.set_sector_forecasts(-1.8, -2.6, 0.18)        # must not raise
    # the magnitude-bar card still renders
    assert p._sector_holder.isVisible()
```
(The second assertion in the first test assumes the committed gold CSVs exist — they do; if a runner lacks them the electricity chart simply won't build, so keep `>= 1` on canvases as the hard assertion and treat the kWh assertion as gold-present.)

- [ ] **Step 2: Run → fails** (`set_sector_forecasts` doesn't build trajectories yet): `python -m pytest ph_economic_ai/tests/test_stage4_trajectories.py -v`.

- [ ] **Step 3: Implement** — in `stage4_report.py`:

(a) At the END of `set_sector_forecasts` (just before the method's closing `except Exception: pass` — i.e. inside the `try`, after the bar-card loop and `self._sector_holder.setVisible(True)`), add:
```python
            self._build_sector_trajectories(gas, food, elec)
```

(b) Add these two methods to the class (e.g. right after `set_sector_forecasts`):
```python
    def _add_trajectory_chart(self, title: str, hist: list, forecast, color: str) -> bool:
        try:
            n = len(hist)
            fig = Figure(figsize=(4.6, 1.3), facecolor='#FBFBFA')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#FBFBFA')
            ax.plot(range(n), hist, color=color, linewidth=1.6)
            if forecast is not None:
                ax.plot([n - 1, n], [hist[-1], forecast], color=color,
                        linewidth=1.4, linestyle=':')
                ax.plot(n, forecast, 'o', color=color, markersize=5)
            for _sp in ('top', 'right'):
                ax.spines[_sp].set_visible(False)
            for _sp in ('left', 'bottom'):
                ax.spines[_sp].set_color('#E5E7EB')
            ax.grid(axis='y', color='#EEEEEE', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6, colors='#9AA1AC')
            ax.set_title(title, fontsize=8, color='#6B7280', loc='left', pad=2)
            fig.tight_layout(pad=0.6)
            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(120)
            self._trajectory_holder_layout.addWidget(canvas)
            return True
        except Exception:
            return False

    def _build_sector_trajectories(self, gas, food, elec):
        """Small-multiples of recent real history: gas ₱/L (df) + forecast marker,
        food/elec CPI MoM % (PSA gold); marker only where the unit aligns."""
        try:
            while self._trajectory_holder_layout.count():
                it = self._trajectory_holder_layout.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.deleteLater()
            title = QLabel('RECENT SECTOR TRAJECTORIES')
            title.setStyleSheet('font-size:10px;font-weight:700;letter-spacing:1px;'
                                'color:#6B7280;')
            self._trajectory_holder_layout.addWidget(title)

            built = 0
            df = getattr(self, '_df', None)
            if df is not None and 'gas_price' in getattr(df, 'columns', []):
                hist = df['gas_price'].dropna().tail(12).tolist()
                if hist:
                    fc = (hist[-1] + gas) if gas is not None else None
                    if self._add_trajectory_chart('GAS · ₱/L', hist, fc, '#1C1E26'):
                        built += 1
            try:
                from ph_economic_ai.benchmark.psa_cpi import load_food_mom
                fh = load_food_mom().dropna().tail(12).tolist()
                if fh and self._add_trajectory_chart('FOOD · CPI MoM %', fh, food, '#B45309'):
                    built += 1
            except Exception:
                pass
            try:
                from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom
                eh = load_electricity_mom().dropna().tail(12).tolist()
                if eh and self._add_trajectory_chart('ELECTRICITY · CPI MoM %', eh, None, '#15803D'):
                    note = QLabel('next-month forecast in ₱/kWh — see card above')
                    note.setStyleSheet('font-size:8px;color:#9EA3AE;font-style:italic;')
                    self._trajectory_holder_layout.addWidget(note)
                    built += 1
            except Exception:
                pass
            self._trajectory_holder.setVisible(built > 0)
        except Exception:
            pass
```

- [ ] **Step 4: Run → passes**: `python -m pytest ph_economic_ai/tests/test_stage4_trajectories.py -v` (2 passed).

- [ ] **Step 5: Stage4 regression**: `python -m pytest ph_economic_ai/tests/test_stage4_sector.py ph_economic_ai/tests/test_stage4_swarm.py -q` → pass.

- [ ] **Step 6: Commit**
```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_trajectories.py
git commit -m "feat(ui): recent sector trajectory small-multiples (gas/food/electricity)"
```

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): run a sim → below the report columns, a "RECENT SECTOR TRAJECTORIES" panel shows 3 editorial mini-charts: gas ₱/L with a dotted forecast marker, food CPI MoM % with a marker, electricity CPI MoM % history-only + the ₱/kWh note.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 store `df` → Task 1 Step 2; §3.2 persistent `_trajectory_holder` → Task 1 Step 3; §3.3 build in `set_sector_forecasts` (gas df + marker, food gold + marker, elec gold history-only + ₱/kWh note, editorial axes, gold via psa_cpi in try/except) → Task 2; §5 testing (built-with-markers + graceful-without-df) → Task 2 Step 1. §2 non-negotiables: real data only (df + PSA gold), marker only on gas+food (elec passes `forecast=None`), graceful (nested try/except, `built>0` gate).
**Placeholder scan:** none — complete code; the test's kWh assertion is annotated as gold-present (the hard assertion is `canvases >= 1`).
**Type consistency:** `load_food_mom()/load_electricity_mom()` return `pd.Series` → `.dropna().tail(12).tolist()` → `hist: list[float]` consumed by `_add_trajectory_chart(title, hist, forecast, color)`. Gas forecast = `hist[-1] + gas` (level + the ₱/L change); food forecast = `food` (% matches MoM % history); electricity passes `None` (history-only). `Figure`/`FigureCanvasQTAgg`/`QWidget`/`QVBoxLayout`/`QLabel` already imported. `set_sector_forecasts` calls `_build_sector_trajectories` inside its existing `try`; both builders independently guarded so one failure can't blank the report.
````
