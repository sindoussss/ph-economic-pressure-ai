# CPI Inflation Nowcasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPI nowcaster that estimates Philippine monthly inflation *before PSA's release* from intra-month-observable drivers (contemporaneous oil/FX/fuel + previous print), and rigorously test whether it beats the naive last-published-inflation benchmark.

**Architecture:** A new `benchmark/nowcast.py` builds a **contemporaneous** feature frame (lag-0 drivers + lagged target) and runs it through the *existing* forecaster panel + Diebold-Mariano + conformal, with the naive baseline = last published inflation. `run.py`/`report.py`/`figures.py`/`accuracy_view.py` are extended to surface the result. No new data — reuses committed `features_monthly.csv` + `ph_cpi_monthly.csv`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, statsmodels, scipy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-ph-cpi-nowcasting-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/targets.py::{load_inflation, _features}` — `load_inflation()` → YoY inflation Series ('YYYY-MM'); `_features()` → DataFrame indexed 'YYYY-MM' with `oil_price, usd_php, gas_price, demand_index`.
- `benchmark/efficiency.py::run_panel(frame, methods, min_train, feature_cols, target_col='ron95')` → rows `{method, rmse, mae, skill_vs_rw, dm_stat, dm_p, n}`; `random_walk` row has `dm_p=None`.
- `benchmark/backtest.py::walk_forward(y, X, predict_fn, min_train)`; `benchmark/forecasters.py::make_forecaster(name)`.
- `benchmark/conformal.py::build_calibration_table(cal_residuals, y_true, y_pred, levels)`.
- `benchmark/report.py::{build_report (...has audit=), REQUIRED_KEYS, ARTIFACTS, load_report}`.
- `benchmark/run.py`, `benchmark/figures.py` (`_ensure_dir`, `FIG_DIR`, `plt`), `ui/accuracy_view.py::AccuracyView`.
- Committed data present: `data/features_monthly.csv`, `data/ph_cpi_monthly.csv`.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, import `from ph_economic_ai.X import Y`, start with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  ```
- Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing; `global_fuel_prices.xlsx` is gitignored — never stage it.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Create:** `benchmark/nowcast.py`; test `test_nowcast.py`.
**Modify:** `benchmark/report.py` (+`nowcast` key), `benchmark/figures.py` (+`plot_nowcast`), `benchmark/run.py` (run+figure+artifact), `ui/accuracy_view.py` (+panel); tests `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: Contemporaneous nowcast frame

**Files:**
- Create: `ph_economic_ai/benchmark/nowcast.py`
- Test: `ph_economic_ai/tests/test_nowcast.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_nowcast.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.nowcast import build_nowcast_frame


def _fake_sources(monkeypatch, n=40):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(0)
    infl = pd.Series(3.0 + rng.normal(0, 0.5, n), index=idx)
    feats = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': 60 + np.cumsum(rng.normal(0, 0.5, n)),
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, 'load_inflation', lambda: infl)
    monkeypatch.setattr(nc, '_features', lambda: feats)
    return idx, infl, feats


def test_frame_has_contemporaneous_drivers_and_lagged_target(monkeypatch):
    idx, infl, feats = _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_inflation', 'target']
    assert not f.isna().any().any()
    t = f.index[5]
    pos = list(idx).index(t)
    # drivers are CONTEMPORANEOUS (month t, not shifted)
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
    assert f.loc[t, 'fuel'] == pytest.approx(feats['gas_price'].iloc[pos])
    # prev_inflation is lagged target
    assert f.loc[t, 'prev_inflation'] == pytest.approx(infl.iloc[pos - 1])
    # target is contemporaneous inflation
    assert f.loc[t, 'target'] == pytest.approx(infl.iloc[pos])


def test_no_same_month_cpi_feature_leak(monkeypatch):
    """Integrity guard: the only CPI-derived columns may be the lagged prev_inflation
    and the target itself. No contemporaneous CPI feature is allowed."""
    _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    cpi_like = [c for c in f.columns if 'infl' in c.lower() or 'cpi' in c.lower()]
    assert cpi_like == ['prev_inflation']   # target is named 'target', not cpi-like
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.nowcast`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/nowcast.py`:

```python
"""CPI inflation nowcasting: estimate month-t inflation BEFORE PSA's release using
intra-month-observable drivers.

Integrity rule: only features whose month-t value is published before the CPI_t
release may enter the frame — month-t oil, FX, and fuel (all complete by end of t;
CPI_t releases ~7 days into t+1), plus the already-published prev_inflation. No
same-month CPI-derived feature is used. The walk-forward trains on past complete
(drivers, inflation) pairs only, then applies the mapping to month-t drivers to
estimate the not-yet-released inflation_t — a causal nowcast, not a forecast of
the unknowable future.
"""
import pandas as pd

from ph_economic_ai.benchmark.targets import _features, load_inflation


def build_nowcast_frame() -> pd.DataFrame:
    """Columns: oil, fx, fuel (contemporaneous, month t), prev_inflation (t-1),
    target (= inflation_t). Inner-joined on the monthly index, dropna'd."""
    infl = load_inflation()
    feats = _features()
    base = pd.DataFrame({
        'oil': feats['oil_price'],
        'fx': feats['usd_php'],
        'fuel': feats['gas_price'],
    })
    base = base.join(infl.rename('target'), how='inner').sort_index()
    base['prev_inflation'] = base['target'].shift(1)
    return base[['oil', 'fx', 'fuel', 'prev_inflation', 'target']].dropna()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): contemporaneous CPI nowcast frame (+ leakage guard test)"
```

---

## Task 2: Nowcast runner + verdict + conformal

**Files:**
- Modify: `ph_economic_ai/benchmark/nowcast.py` (append)
- Modify: `ph_economic_ai/tests/test_nowcast.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_nowcast.py`:

```python
from ph_economic_ai.benchmark.nowcast import run_nowcast


def test_run_nowcast_beats_naive_on_constructed_signal():
    # inflation strongly driven by contemporaneous fuel -> a method should beat naive
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(1)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, 90))
    target = 0.5 * (fuel - 60) + rng.normal(0, 0.05, 90)   # contemporaneous, low noise
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90),
        'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': fuel,
        'prev_inflation': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'beats_naive'
    assert res['best_method'] != 'random_walk'
    assert res['best_skill'] > 0


def test_run_nowcast_ties_naive_on_random_walk_target():
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    target = np.cumsum(rng.normal(0, 0.3, 90)) + 3.0      # random-walk inflation
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90),
        'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': 60 + rng.normal(0, 1, 90),                # noise driver
        'prev_inflation': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'no_better_than_naive'
    assert res['best_method'] == 'random_walk'


def test_run_nowcast_insufficient_data():
    idx = pd.date_range('2020-01', periods=10, freq='MS').strftime('%Y-%m')
    frame = pd.DataFrame({'oil': range(10), 'fx': range(10), 'fuel': range(10),
                          'prev_inflation': range(10), 'target': range(10)},
                         index=idx).astype(float)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'insufficient_data'
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -k run_nowcast -v`
Expected: FAIL — `ImportError: cannot import name 'run_nowcast'`

- [ ] **Step 3: Append the implementation to nowcast.py**

```python
from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.conformal import build_calibration_table
from ph_economic_ai.benchmark.efficiency import run_panel
from ph_economic_ai.benchmark.forecasters import make_forecaster

PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']
FEATURE_COLS = ['oil', 'fx', 'fuel', 'prev_inflation']
CONFORMAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def run_nowcast(min_train: int = 24, frame=None) -> dict:
    """Nowcast inflation via the panel; naive baseline = last published inflation.
    Verdict 'beats_naive' if any method significantly beats naive (dm_p<0.05,
    skill>0), else 'no_better_than_naive'."""
    if frame is None:
        frame = build_nowcast_frame()
    if len(frame) < min_train + 5:
        return {'verdict': 'insufficient_data', 'n': int(len(frame))}

    panel = run_panel(frame, PANEL_METHODS, min_train, FEATURE_COLS, target_col='target')
    beats = [r for r in panel
             if r['dm_p'] is not None and r['dm_p'] < 0.05 and r['skill_vs_rw'] > 0]
    if beats:
        best = max(beats, key=lambda r: r['skill_vs_rw'])
        verdict = 'beats_naive'
    else:
        best = next(r for r in panel if r['method'] == 'random_walk')
        verdict = 'no_better_than_naive'

    # Conformal calibration on the chosen method's out-of-sample residuals.
    y = frame['target'].to_numpy(dtype=float)
    X = frame[FEATURE_COLS].to_numpy(dtype=float)
    bt = walk_forward(y, X, make_forecaster(best['method']), min_train)
    res = bt['y_true'] - bt['y_pred']
    half = max(1, len(res) // 2)
    calib = build_calibration_table(res[:half], bt['y_true'][half:], bt['y_pred'][half:],
                                    CONFORMAL_LEVELS) if len(res) > 3 else []
    return {
        'verdict': verdict,
        'best_method': best['method'],
        'best_skill': best['skill_vs_rw'],
        'best_dm_p': best.get('dm_p'),
        'n': int(panel[0]['n']),
        'calibration': calib,
        'panel': panel,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): nowcast runner (panel + DM vs naive + conformal + verdict)"
```

---

## Task 3: Report key for the nowcast

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_nowcast():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        nowcast={'verdict': 'beats_naive', 'best_method': 'ridge', 'best_skill': 0.18,
                 'best_dm_p': 0.02, 'n': 70},
    )
    assert rep['nowcast']['verdict'] == 'beats_naive'
    assert 'nowcast' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_nowcast -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'nowcast'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'nowcast'` to the `REQUIRED_KEYS` tuple.
(b) Add `nowcast=None` as the LAST parameter of `build_report` (after `audit=None`).
(c) In the returned dict, add just before `'limitations'`:
```python
        'nowcast': nowcast if nowcast is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all; existing calls default `nowcast` to `{}`)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the CPI nowcast result"
```

---

## Task 4: Run the nowcast in run.py + figure

**Files:**
- Modify: `ph_economic_ai/benchmark/figures.py`
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: end-to-end run on real data

- [ ] **Step 1: Add the figure**

Append to `ph_economic_ai/benchmark/figures.py`:

```python
def plot_nowcast(dates, actual, nowcast, naive):
    """Inflation: actual vs nowcast vs naive (last published) over the backtest."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, actual, label='Actual inflation', color='black')
    ax.plot(dates, nowcast, label='Nowcast', color='tab:blue')
    ax.plot(dates, naive, label='Naive (last published)', color='tab:gray', linestyle='--')
    ax.set_ylabel('YoY inflation (%)'); ax.legend(); ax.tick_params(axis='x', rotation=45)
    ax.set_title('CPI nowcast vs actual vs naive')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'nowcast.png', dpi=120); plt.close(fig)
```

- [ ] **Step 2: Wire into run.py**

In `ph_economic_ai/benchmark/run.py`:
(a) Add imports near the other benchmark imports:
```python
from ph_economic_ai.benchmark import nowcast as nowcast_mod
from ph_economic_ai.benchmark.backtest import walk_forward as _walk_forward
from ph_economic_ai.benchmark.forecasters import make_forecaster as _make_forecaster
```
(Note: `walk_forward` may already be imported in run.py — if so, reuse it and skip the aliased import; same for `make_forecaster`. Check with `git grep -n "import walk_forward\|make_forecaster" ph_economic_ai/benchmark/run.py` and avoid a duplicate import.)
(b) After the audit block (before `rep = report.build_report(`), add:
```python
    # -- CPI nowcast (estimate inflation before official release) --
    nowcast_res = nowcast_mod.run_nowcast(MIN_TRAIN)
    if nowcast_res['verdict'] == 'insufficient_data':
        print(f"CPI nowcast: insufficient_data (n={nowcast_res.get('n', 0)})")
    else:
        print(f"CPI nowcast: {nowcast_res['verdict']} | best={nowcast_res['best_method']} "
              f"skill_vs_naive={nowcast_res['best_skill']:+.3f} DM p={nowcast_res['best_dm_p']}")
```
(c) In the `report.build_report(` call, add (dropping the heavy `panel`):
```python
        nowcast={k: v for k, v in nowcast_res.items() if k != 'panel'},
```
(d) After the existing figure calls, add:
```python
    import json as _json3
    (report.ARTIFACTS / 'nowcast_table.json').write_text(
        _json3.dumps(nowcast_res, indent=2), encoding='utf-8')
    if nowcast_res['verdict'] != 'insufficient_data':
        _nf = nowcast_mod.build_nowcast_frame()
        _y = _nf['target'].to_numpy(float)
        _X = _nf[nowcast_mod.FEATURE_COLS].to_numpy(float)
        _bt = _walk_forward(_y, _X, _make_forecaster(nowcast_res['best_method']), MIN_TRAIN)
        _nbt = _walk_forward(_y, None, _make_forecaster('random_walk'), MIN_TRAIN)
        _nd = [_nf.index[i] for i in _bt['index']]
        figures.plot_nowcast(_nd, _bt['y_true'], _bt['y_pred'], _nbt['y_pred'])
```

- [ ] **Step 3: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints the existing summaries plus a `CPI nowcast: ...` line; writes `nowcast_table.json` and (if not insufficient) `figures/nowcast.png`, no errors.

- [ ] **Step 4: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; print(load_report()['nowcast'])"`
Record the printed dict — this is the nowcast result for the spec/write-up (verdict, best_method, best_skill, best_dm_p, n).

- [ ] **Step 5: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window::test_on_run_requested_accepts_4_args`. Report counts.

- [ ] **Step 6: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/figures.py ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/nowcast_table.json ph_economic_ai/benchmark/artifacts/figures/nowcast.png
git status --short
git commit -m "feat(benchmark): run CPI nowcast + figure; report the result"
```
(If verdict was `insufficient_data`, `nowcast.png` won't exist — omit it from the `git add`.)

---

## Task 5: Surface the nowcast in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'nowcast': {'verdict': 'beats_naive', 'best_method': 'ridge', 'best_skill': 0.18,
                    'best_dm_p': 0.02, 'n': 70},
```
Add the test:
```python
def test_view_shows_nowcast(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.nowcast_summary()
    assert 'nowcast' in s.lower()
    assert 'ridge' in s
    assert 'beats' in s.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_nowcast -v`
Expected: FAIL — `AttributeError: ... has no attribute 'nowcast_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method to `AccuracyView`:
```python
    def nowcast_summary(self) -> str:
        if not self._report:
            return ''
        n = self._report.get('nowcast') or {}
        if not n or n.get('verdict') == 'insufficient_data':
            return ''
        return (f"CPI nowcast (estimate inflation before release): {n['verdict']} "
                f"— best {n['best_method']}, skill {n['best_skill']:+.2f} vs naive "
                f"(DM p={n['best_dm_p']}).")
```
(b) In `_build`, inside `if self._report is not None:`, after the audit block, add:
```python
            _nc = self.nowcast_summary()
            if _nc:
                ncl = QLabel('<b>Nowcast (present-before-release)</b><br>' + _nc)
                ncl.setWordWrap(True)
                col.addWidget(ncl)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py -v`
Expected: PASS (existing + new)

- [ ] **Step 5: Smoke-test the window**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: only the pre-existing `test_on_run_requested_accepts_4_args` may fail; nothing new.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/accuracy_view.py ph_economic_ai/tests/test_accuracy_view.py
git commit -m "feat(ui): show the CPI nowcast result in the Accuracy view"
```

---

## Task 6: Record the nowcast outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-ph-cpi-nowcasting-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 4 Step 4**

Replace the `[beats / does not beat]`/`[method]`/`[x]`/`[p]`/`[n]` placeholders in §9 with the measured nowcast result. State plainly whether the nowcast beats naive (verdict + best method + skill vs naive + DM p + n), and one sentence of interpretation tied to the nowcasting literature (Giannone-Reichlin-Small; Atkeson-Ohanian for why naive is the bar). Copy numbers from `nowcast_table.json`; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-08-ph-cpi-nowcasting-design.md
git commit -m "docs: record measured CPI nowcast result in spec section 9"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Nowcast result legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; n=load_report()['nowcast']; print(n.get('verdict'), n.get('best_method'), n.get('best_skill'))"`
Expected: prints the verdict (`beats_naive` or `no_better_than_naive`), method, and skill — the honest nowcast result, whichever way it lands.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3 definitions + integrity rule → Task 1 (frame + leakage-guard test). §4.1 build_nowcast_frame → Task 1. §4.2 run_nowcast (panel + DM + conformal + verdict) → Task 2. §4.3 integration (report/run/figure/view) → Tasks 3, 4, 5. §6 error handling (insufficient_data) → Task 2 + Task 4 wiring. §7 testing incl. leakage guard + synthetic beats/ties → Tasks 1, 2. §8 deliverables → all. §9 write-up outcome → Task 6.

**Placeholder scan:** §9's `[beats/...]` markers are filled at Task 6 from real output (empirical; cannot be known at plan time), matching the accepted Phase 1/2/Efficiency/Audit pattern. No other red-flag placeholders; every code step has complete code.

**Type consistency:** `build_nowcast_frame() -> frame[oil,fx,fuel,prev_inflation,target]` consistent across Tasks 1, 2, 4. `run_nowcast(min_train, frame=None) -> {verdict,best_method,best_skill,best_dm_p,n,calibration,panel}` consistent across Tasks 2, 4, 5. `FEATURE_COLS` defined in Task 2, reused in Task 4. `build_report(... nowcast=)` matches Task 3 (def) and Task 4 (call). `plot_nowcast(dates, actual, nowcast, naive)` consistent (Task 4). `run_panel(..., target_col='target')` uses the param added in the predictability-audit work (already merged). Report key `nowcast` consistent across Tasks 3, 4, 5.
```
