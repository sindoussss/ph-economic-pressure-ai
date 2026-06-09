# MoM Nowcast Longer-Sample Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm (or revise) the MoM CPI nowcast win on a ~3× longer feature history (Yahoo `range='max'`, ~2007→), isolated so existing committed results are untouched — re-running only the MoM nowcast + driver-only ablation.

**Architecture:** A longer-history feature CSV (`features_monthly_long.csv`) built from Yahoo `max`; a small `features=` parameter on `build_nowcast_frame`; and a `longsample` module that feeds the long features into the *existing* `run_mom_nowcast` + `run_driver_only_ablation`. Report/run/view/spec extended. The main benchmark pipeline is not changed.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, statsmodels, requests, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-ph-cpi-mom-longsample-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/nowcast.py::{build_nowcast_frame(target_loader=None, prev_col='prev_inflation'), run_mom_nowcast(min_train, baseline_pool, frame, methods), run_driver_only_ablation(min_train, frame), load_inflation_mom}`. `build_nowcast_frame` currently maps `feats['oil_price']→oil, feats['usd_php']→fx, feats['gas_price']→fuel` where `feats = _features()`.
- `benchmark/targets.py::{_features, load_inflation_mom}`; `_features()` reads `data/features_monthly.csv`.
- `benchmark/refresh_data.py::{_yahoo_monthly(ticker, rng='10y'), HERE}`; `build_features_csv` builds the 10y features (Brent `BZ=F`, USD/PHP `PHP=X`, RBOB `RB=F`, gas proxy, demand). `fetcher._compute_demand`.
- `benchmark/report.py::{build_report(... mom_driver_ablation=None), REQUIRED_KEYS, ARTIFACTS, load_report}`.
- `benchmark/run.py` (has `MIN_TRAIN`, `report`, runs MoM nowcast + driver ablation), `ui/accuracy_view.py::AccuracyView`.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing; `global_fuel_prices.xlsx` is gitignored — never stage it.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Create:** `benchmark/longsample.py`, `benchmark/data/features_monthly_long.csv` (by running Task 2); test `test_longsample.py`.
**Modify:** `benchmark/nowcast.py` (+`features=` param), `benchmark/refresh_data.py` (+`build_long_features`), `benchmark/report.py` (+`mom_longsample`), `benchmark/run.py` (run+record), `ui/accuracy_view.py` (+note); tests `test_nowcast.py`, `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: `features=` parameter on `build_nowcast_frame`

**Files:**
- Modify: `ph_economic_ai/benchmark/nowcast.py`
- Modify: `ph_economic_ai/tests/test_nowcast.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_nowcast.py`:

```python
def test_build_nowcast_frame_accepts_features_arg(monkeypatch):
    idx = pd.date_range('2010-01', periods=50, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(9)
    mom = pd.Series(rng.normal(0.3, 0.4, 50), index=idx)
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, 50)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, 50)),
        'gas_price': 45 + np.cumsum(rng.normal(0, 0.5, 50)),
        'demand_index': 70 + rng.normal(0, 2, 50),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    # _features() must NOT be used when features= is supplied:
    monkeypatch.setattr(nc, '_features', lambda: (_ for _ in ()).throw(AssertionError('used _features')))
    f = nc.build_nowcast_frame(target_loader=lambda: mom, prev_col='prev_mom', features=feats)
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_mom', 'target']
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py::test_build_nowcast_frame_accepts_features_arg -v`
Expected: FAIL — `build_nowcast_frame() got an unexpected keyword argument 'features'`

- [ ] **Step 3: Implement — edit `build_nowcast_frame` in nowcast.py**

Change the signature to add `features=None` and use it when provided. Replace the signature line and the `feats = _features()` line:
```python
def build_nowcast_frame(target_loader=None, prev_col: str = 'prev_inflation', features=None) -> pd.DataFrame:
```
and replace `feats = _features()` with:
```python
    feats = _features() if features is None else features
```
Nothing else changes (the rest already references `feats`).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS (all existing nowcast tests + the new one; the default path still calls `_features()`).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): build_nowcast_frame accepts an explicit features frame"
```

---

## Task 2: Longer-history feature builder + committed CSV

**Files:**
- Modify: `ph_economic_ai/benchmark/refresh_data.py`
- Create (by running): `ph_economic_ai/benchmark/data/features_monthly_long.csv`

(No unit test — this is a network data-refresh function, matching how `build_features_csv`/`build_world_bank_csv` are handled.)

- [ ] **Step 1: Append `build_long_features` to refresh_data.py**

```python
LONG_FEATURES_OUT = HERE / 'data' / 'features_monthly_long.csv'


def build_long_features(rng: str = 'max') -> None:
    """Longer-history predictor matrix (default Yahoo range='max') for the MoM
    nowcast longer-sample confirmation. Same columns/derivations as
    build_features_csv, just a longer window -> data/features_monthly_long.csv."""
    from ph_economic_ai.fetcher import _compute_demand
    oil = _yahoo_monthly('BZ=F', rng)
    usd = _yahoo_monthly('PHP=X', rng)
    rbob = _yahoo_monthly('RB=F', rng)
    base = pd.concat([oil.rename('oil_price'), usd.rename('usd_php'),
                      rbob.rename('rbob')], axis=1).dropna()
    base['gas_price'] = ((base['rbob'] / 3.785 * base['usd_php']) * 1.35 + 12).round(2)
    base = base.drop(columns=['rbob']).reset_index().rename(columns={'index': 'date'})
    base['demand_index'] = _compute_demand(base['date'].tolist())
    base = base.sort_values('date')
    LONG_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(LONG_FEATURES_OUT, index=False)
    print(f'Wrote features_monthly_long.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')
```
(`_yahoo_monthly` already takes a `rng` arg; `HERE` and `import pandas as pd`/`requests` already exist in the module.)

- [ ] **Step 2: Build the long features CSV (network — Yahoo max)**

Run: `python -c "from ph_economic_ai.benchmark.refresh_data import build_long_features; build_long_features()"`
Expected: prints `Wrote features_monthly_long.csv (N rows, YYYY-MM..YYYY-MM)` with N noticeably larger than the 10y file (~150–220 rows, starting ~2007–2010). If Yahoo `max` fails for a ticker, retry once; if it still fails, STOP and report (do not fabricate).

- [ ] **Step 3: Verify it loads and overlaps CPI enough**

Run: `python -c "import pandas as pd; from ph_economic_ai.benchmark.targets import load_inflation_mom; f=pd.read_csv('ph_economic_ai/benchmark/data/features_monthly_long.csv',dtype={'date':str}).set_index('date'); m=load_inflation_mom(); ov=f.index.intersection(m.index); print('long rows',len(f),'| CPI-MoM rows',len(m),'| overlap',len(ov))"`
Expected: overlap clearly larger than ~85 (the 10y overlap) — ideally 150+.

- [ ] **Step 4: Commit (code + committed CSV)**

```bash
git add ph_economic_ai/benchmark/refresh_data.py ph_economic_ai/benchmark/data/features_monthly_long.csv
git status --short
git commit -m "feat(benchmark): long-history feature builder + committed features_monthly_long.csv"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 3: `longsample` runner

**Files:**
- Create: `ph_economic_ai/benchmark/longsample.py`
- Test: `ph_economic_ai/tests/test_longsample.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_longsample.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.longsample as ls


def test_run_mom_longsample_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2010-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(11)
    gas = 45 + np.cumsum(rng.normal(0, 0.5, n))
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    mom = pd.Series(0.4 * np.r_[0.0, np.diff(gas)] + rng.normal(0, 0.05, n), index=idx)
    # MoM target driven by contemporaneous gas change; CPI loader monkeypatched
    monkeypatch.setattr(ls, 'load_inflation_mom', lambda: mom)
    res = ls.run_mom_longsample(min_train=24, features=feats)
    assert set(res) == {'n_long', 'mom', 'driver_ablation'}
    assert res['n_long'] > 60                       # longer than the 10y backtest
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert 'driver_edge' in res['driver_ablation']
    # heavy internals dropped
    assert 'panel' not in res['mom'] and 'calibration' not in res['mom']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_longsample.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.longsample`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/longsample.py`:

```python
"""Longer-sample confirmation of the MoM CPI nowcast.

Feeds a longer feature history (features_monthly_long.csv) into the existing
MoM nowcast + driver-only ablation to test whether the result holds on ~3x the
sample. Isolated from the main pipeline.
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.nowcast import (
    build_nowcast_frame, run_driver_only_ablation, run_mom_nowcast,
)
from ph_economic_ai.benchmark.targets import load_inflation_mom

LONG_FEATURES_CSV = Path(__file__).parent / 'data' / 'features_monthly_long.csv'


def load_long_features(csv_path: Path = LONG_FEATURES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={'date': str}).set_index('date').sort_index()


def run_mom_longsample(min_train: int = 24, features=None) -> dict:
    """Run the MoM nowcast + driver-only ablation on the long feature history.
    Returns {n_long, mom, driver_ablation} with heavy internals dropped."""
    feats = load_long_features() if features is None else features
    frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom',
                                features=feats)
    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    drop = ('panel', 'calibration')
    return {
        'n_long': int(mom.get('n', len(frame))),
        'mom': {k: v for k, v in mom.items() if k not in drop},
        'driver_ablation': {k: v for k, v in abl.items() if k not in drop},
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_longsample.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/longsample.py ph_economic_ai/tests/test_longsample.py
git commit -m "feat(benchmark): longsample runner (MoM nowcast + ablation on long history)"
```

---

## Task 4: Report key for the longer-sample result

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_mom_longsample():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        mom_longsample={'n_long': 190,
                        'mom': {'verdict': 'beats_best_naive', 'best_method': 'arima',
                                'best_skill_vs_naive': 0.14, 'dm_p': 0.01},
                        'driver_ablation': {'verdict': 'no_better_than_naive',
                                            'driver_edge': False}},
    )
    assert rep['mom_longsample']['n_long'] == 190
    assert 'mom_longsample' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_mom_longsample -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'mom_longsample'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'mom_longsample'` to `REQUIRED_KEYS`.
(b) Add `mom_longsample=None` as the LAST parameter of `build_report` (after `mom_driver_ablation=None`).
(c) In the returned dict, add just before `'limitations'`:
```python
        'mom_longsample': mom_longsample if mom_longsample is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the MoM longer-sample confirmation"
```

---

## Task 5: Run the longer-sample confirmation in run.py

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: end-to-end run on real data

- [ ] **Step 1: Wire into run.py**

In `ph_economic_ai/benchmark/run.py`, after the driver-ablation block (`mom_abl = nowcast_mod.run_driver_only_ablation(MIN_TRAIN)` + its print), BEFORE the `rep = report.build_report(` call, add:
```python
    # -- MoM nowcast longer-sample confirmation (isolated long feature history) --
    try:
        from ph_economic_ai.benchmark import longsample as longsample_mod
        mom_long = longsample_mod.run_mom_longsample(MIN_TRAIN)
        _lm, _la = mom_long['mom'], mom_long['driver_ablation']
        print(f"MoM long-sample (n={mom_long['n_long']}): mom={_lm['verdict']} "
              f"best={_lm.get('best_method')} skill={_lm.get('best_skill_vs_naive')} "
              f"DM p={_lm.get('dm_p')} | driver_edge={_la.get('driver_edge')}")
    except FileNotFoundError:
        mom_long = {'verdict': 'not_run', 'reason': 'features_monthly_long.csv missing'}
        print('MoM long-sample: not_run (features_monthly_long.csv missing)')
```
In the `report.build_report(` call, add:
```python
        mom_longsample=mom_long,
```
After the existing artifact writes, add:
```python
    import json as _json6
    (report.ARTIFACTS / 'mom_longsample_table.json').write_text(
        _json6.dumps(mom_long, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints all prior summaries plus a `MoM long-sample (n=...): ...` line; writes `mom_longsample_table.json`, no errors. (The long features CSV exists from Task 2, so it should NOT report `not_run`.)

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; import json; print(json.dumps(load_report()['mom_longsample'], indent=2))"`
Record the output — `n_long`, the `mom` verdict/best_method/skill/DM p, and the `driver_ablation` verdict/driver_edge. **Compare to the n=61 result** (MoM beats_best_naive ARIMA +16.2% p=0.032; driver_edge False).

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window.py::test_on_run_requested_accepts_4_args`. Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/mom_longsample_table.json
git status --short
git commit -m "feat(benchmark): run the MoM longer-sample confirmation; report the result"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 6: Surface the longer-sample result in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'mom_longsample': {'n_long': 190,
                           'mom': {'verdict': 'beats_best_naive', 'best_method': 'arima',
                                   'best_skill_vs_naive': 0.14, 'dm_p': 0.01},
                           'driver_ablation': {'verdict': 'no_better_than_naive',
                                               'driver_edge': False}},
```
Add the test:
```python
def test_view_shows_mom_longsample(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.mom_longsample_summary()
    assert '190' in s
    assert 'arima' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_mom_longsample -v`
Expected: FAIL — `AttributeError: ... has no attribute 'mom_longsample_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method to `AccuracyView`:
```python
    def mom_longsample_summary(self) -> str:
        if not self._report:
            return ''
        L = self._report.get('mom_longsample') or {}
        if not L or L.get('verdict') == 'not_run':
            return ''
        mom = L.get('mom') or {}
        abl = L.get('driver_ablation') or {}
        return (f"Longer sample (n={L.get('n_long')}): MoM {mom.get('verdict')} "
                f"(best {mom.get('best_method')}, skill {mom.get('best_skill_vs_naive')}, "
                f"DM p={mom.get('dm_p')}); driver_edge={abl.get('driver_edge')}.")
```
(b) In `_build`, inside `if self._report is not None:`, after the driver-ablation block, add:
```python
            _ls = self.mom_longsample_summary()
            if _ls:
                lsl = QLabel('<b>MoM longer-sample confirmation</b><br>' + _ls)
                lsl.setWordWrap(True)
                col.addWidget(lsl)
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
git commit -m "feat(ui): show the MoM longer-sample confirmation in the Accuracy view"
```

---

## Task 7: Record the longer-sample outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-09-ph-cpi-mom-longsample-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 5 Step 3**

Replace the `[n_long]`/`[method]`/`[x]`/`[p]`/`[bool]`/`[r]`/`[rb]` placeholders in §9 with the measured longer-sample result, and pick the matching interpretation bullet (confirmed / driver-edge-now-significant / changed-with-regime-caveat). Explicitly state the comparison to the n=61 result. Copy numbers from `mom_longsample_table.json`; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-09-ph-cpi-mom-longsample-design.md
git commit -m "docs: record measured MoM longer-sample confirmation result"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Longer-sample result legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; L=load_report()['mom_longsample']; print('n_long', L.get('n_long'), '| mom', L['mom']['verdict'], L['mom'].get('best_method'), '| driver_edge', L['driver_ablation'].get('driver_edge'))"`
Expected: prints the long-sample n, MoM verdict + method, and driver_edge — the honest confirmation, however it lands vs n=61.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3 definitions (long features, long MoM frame, confirmation verdicts) → Tasks 2, 3. §4.1 build_long_features → Task 2. §4.2 build_nowcast_frame(features=) → Task 1. §4.3 longsample.py → Task 3. §4.4 integration (report/run/view) → Tasks 4, 5, 6. §6 error handling (not_run guard, FileNotFoundError) → Task 5. §7 testing (features= path, longsample wiring) → Tasks 1, 3. §8 deliverables → all. §9 write-up → Task 7.

**Placeholder scan:** §9's bracketed markers are filled at Task 7 from real output (empirical; accepted pattern). Task 2 has no unit test by design (network refresh function, consistent with `build_features_csv`/`build_world_bank_csv`). No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `build_nowcast_frame(target_loader=None, prev_col='prev_inflation', features=None)` — `features` added Task 1, used by `run_mom_longsample` Task 3. `_yahoo_monthly(ticker, rng)` used in Task 2 (already accepts rng). `run_mom_longsample(min_train=24, features=None) -> {n_long, mom, driver_ablation}` consistent across Tasks 3, 5, 6. `build_report(... mom_longsample=)` matches Task 4 (def) and Task 5 (call). Report key `mom_longsample` consistent across Tasks 4, 5, 6. `load_long_features`/`LONG_FEATURES_CSV` (Task 3) read the CSV written by `LONG_FEATURES_OUT` (Task 2) — same path `data/features_monthly_long.csv`. run.py `mom_long` is either the longsample dict or `{'verdict': 'not_run', ...}`; the view's `mom_longsample_summary` guards both (`not_run` → '' ; success → reads `mom`/`driver_ablation`).
```
