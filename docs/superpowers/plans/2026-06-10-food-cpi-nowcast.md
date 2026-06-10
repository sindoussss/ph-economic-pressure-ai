# MoM Food-CPI Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nowcast month-on-month PH Food-CPI inflation before the PSA release using free global agri-commodity prices + FX, and test honestly (DM vs strongest naive + driver-only ablation + preliminary-data robustness re-test) whether food inflation is nowcastable.

**Architecture:** Generalize the existing PSA PX-Web fetch to resolve any commodity group by COICOP code prefix; add a free Yahoo agri-futures predictor panel; build a food nowcast frame directly and reuse the existing `run_mom_nowcast` + `run_driver_only_ablation` (unchanged) plus the transport-style robustness re-test. Report/run/view extended; verdict reported either way.

**Tech Stack:** Python 3.10, requests, pandas, numpy, scikit-learn, statsmodels, pytest. Data: PSA OpenSTAT PX-Web, Yahoo Finance.

**Spec:** `docs/superpowers/specs/2026-06-10-food-cpi-nowcast-design.md`.

**Prereqs (on `master`):**
- `benchmark/psa_cpi.py` — `_fetch_px_table(url, first_year)` (currently hardcodes the Transport commodity id `'203'` in its query and parses `row['key']=[Geolocation, Commodity, Year, Period]`), `fetch_transport_cpi`, `load_transport_cpi`, `load_transport_mom`, `_label_to_ym`, `HERE`, `TRANSPORT_CSV`, `_PERIOD_TO_MM`, `_PSA_HEADERS`, `PSA_TRANSPORT_URL_BACKCAST`, `PSA_TRANSPORT_URL_CURRENT`. Commodity variable code is `'Commodity Description'`; values like `'1' = '01 - FOOD AND NON-ALCOHOLIC BEVERAGES'`, `'203' = '07 - TRANSPORT'`.
- `benchmark/nowcast.py` — reused UNCHANGED: `run_mom_nowcast(min_train, frame=…)` uses **all non-`target` columns** as features; `run_driver_only_ablation(min_train, frame=…)` drops `prev_mom` by name. Reserved frame column names: `prev_mom`, `target`.
- `benchmark/transport_nowcast.py::run_transport_nowcast(min_train, features, prelim_months)` — the structural template (full + robust driver ablation, `driver_edge_robust`).
- `benchmark/refresh_data.py::_yahoo_monthly(ticker, rng='10y')`, `HERE`.
- `benchmark/targets.py::cpi_to_mom`.
- `benchmark/report.py::build_report(...)` (trailing optional kwargs + `REQUIRED_KEYS` + `ARTIFACTS` + `load_report`), `benchmark/run.py` `main()`, `ui/accuracy_view.py::AccuracyView` (`transport_nowcast_summary` is the template).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing. `global_fuel_prices.xlsx` is gitignored — never stage it.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/food-cpi-nowcast
```

---

## File Structure
**Create:** `benchmark/food_nowcast.py`, `benchmark/data/psa_food_cpi_monthly.csv` (Task 2), `benchmark/data/food_features_monthly.csv` (Task 3); tests `test_food_nowcast.py` (+ additions to `test_psa_cpi.py`).
**Modify:** `benchmark/psa_cpi.py` (generalize + food loaders), `benchmark/refresh_data.py` (food features), `benchmark/report.py`, `benchmark/run.py`, `ui/accuracy_view.py`; tests `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: Generalize PSA fetch by COICOP prefix + food loaders

**Files:**
- Modify: `ph_economic_ai/benchmark/psa_cpi.py`
- Modify: `ph_economic_ai/tests/test_psa_cpi.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `ph_economic_ai/tests/test_psa_cpi.py`:

```python
from ph_economic_ai.benchmark.psa_cpi import _resolve_commodity_id, load_food_mom


def _commodity_var():
    return {
        'code': 'Commodity Description',
        'values': ['0', '1', '2', '203'],
        'valueTexts': ['0 - ALL ITEMS', '01 - FOOD AND NON-ALCOHOLIC BEVERAGES',
                       '01.1 - FOOD', '07 - TRANSPORT'],
    }


def test_resolve_commodity_id_by_coicop_prefix():
    v = _commodity_var()
    assert _resolve_commodity_id(v, '01') == '1'      # division, not '01.1 - FOOD'
    assert _resolve_commodity_id(v, '07') == '203'


def test_resolve_commodity_id_missing_raises():
    import pytest
    with pytest.raises(ValueError):
        _resolve_commodity_id(_commodity_var(), '99')


def test_load_food_mom(tmp_path):
    import pytest
    p = tmp_path / 'food.csv'
    p.write_text('date,food_cpi\n2018-01,100.0\n2018-02,102.0\n2018-03,102.0\n',
                 encoding='utf-8')
    mom = load_food_mom(p)
    assert mom['2018-02'] == pytest.approx(2.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -k "commodity or food_mom" -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_commodity_id'`.

- [ ] **Step 3: Implement in psa_cpi.py**

First read `ph_economic_ai/benchmark/psa_cpi.py` to locate `_fetch_px_table`, `fetch_transport_cpi`, and `TRANSPORT_CSV`.

(a) Add the resolver (near `_label_to_ym`):
```python
def _resolve_commodity_id(commodity_var: dict, coicop_prefix: str):
    """Return the value id whose label is the COICOP division `coicop_prefix`
    (e.g. '01' -> '01 - FOOD AND NON-ALCOHOLIC BEVERAGES', not '01.1 - FOOD')."""
    needle = f'{coicop_prefix} -'
    for vid, txt in zip(commodity_var['values'], commodity_var['valueTexts']):
        if txt.strip().startswith(needle):
            return vid
    raise ValueError(f"no commodity matching '{needle}'; "
                     f"available: {commodity_var['valueTexts'][:12]}")
```

(b) Generalize `_fetch_px_table` to take a `coicop_prefix` argument and resolve the commodity id instead of hardcoding `'203'`. Change its signature to `def _fetch_px_table(url: str, first_year: int, coicop_prefix: str) -> dict:` and, inside, after `by_code = {v['code']: v for v in meta['variables']}` (or equivalent), resolve:
```python
    commodity_id = _resolve_commodity_id(by_code['Commodity Description'], coicop_prefix)
```
and in the query body replace the hardcoded commodity selection with:
```python
        {'code': 'Commodity Description',
         'selection': {'filter': 'item', 'values': [commodity_id]}},
```

(c) Update `fetch_transport_cpi` to pass `coicop_prefix='07'` to both `_fetch_px_table(...)` calls (backcast + current). Behaviour must be unchanged.

(d) Add the food constants + fetch + loaders (place near the transport equivalents):
```python
FOOD_CSV = HERE / 'data' / 'psa_food_cpi_monthly.csv'


def fetch_food_cpi(out_csv: Path = FOOD_CSV) -> None:
    """Fetch monthly Food (COICOP 01) CPI from PSA OpenSTAT and freeze to CSV."""
    series = {}
    series.update(_fetch_px_table(PSA_TRANSPORT_URL_BACKCAST, 1994, '01'))
    series.update(_fetch_px_table(PSA_TRANSPORT_URL_CURRENT, 2018, '01'))  # current wins
    if len(series) < 100:
        raise ValueError(f'food CPI series too short ({len(series)} rows)')
    df = (pd.DataFrame(sorted(series.items()), columns=['date', 'food_cpi'])
          .sort_values('date'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Wrote psa_food_cpi_monthly.csv ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def load_food_cpi(csv_path: Path = FOOD_CSV) -> pd.Series:
    """Monthly Food CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['food_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_food_mom(csv_path: Path = FOOD_CSV) -> pd.Series:
    """Month-over-month Food inflation % from the committed gold."""
    return cpi_to_mom(load_food_cpi(csv_path))
```
NOTE: if `fetch_transport_cpi` currently calls `_fetch_px_table` WITHOUT building a merged dict the way shown above, keep its existing merge logic and only thread the `coicop_prefix='07'` argument through; mirror whatever it does for `fetch_food_cpi` with `'01'`. If `_fetch_px_table`'s real internals differ from the prereq description, adapt minimally and preserve transport behaviour.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/tests/test_psa_cpi.py
git commit -m "feat(benchmark): generalize PSA fetch by COICOP prefix; add food-CPI loaders"
```

---

## Task 2: Fetch + commit Food gold; verify transport unchanged (network)

**Files:**
- Create (by running): `ph_economic_ai/benchmark/data/psa_food_cpi_monthly.csv`

- [ ] **Step 1: Fetch the food gold (network)**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import fetch_food_cpi; fetch_food_cpi()"`
Expected: `Wrote psa_food_cpi_monthly.csv (N rows, 1994-01..YYYY-MM)`, N ≈ 350–390. If the `01 -` label is not found, read the error (lists labels) and adjust; do NOT fabricate.

- [ ] **Step 2: Sanity-check the food gold**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import load_food_cpi, load_food_mom; c=load_food_cpi(); m=load_food_mom(); print('rows',len(c),c.index[0],c.index[-1]); print('2018 mean', round(c[[i for i in c.index if i.startswith('2018')]].mean(),1)); print('mom n',len(m),'std',round(m.std(),3))"`
Expected: ~350–390 rows; **2018 mean ≈ 100** (base year — confirms the right series); sensible MoM std.

- [ ] **Step 3: Verify the transport gold is UNCHANGED after the refactor**

Run: `python -c "import pandas as pd, hashlib; before=open('ph_economic_ai/benchmark/data/psa_transport_cpi_monthly.csv','rb').read(); from ph_economic_ai.benchmark.psa_cpi import fetch_transport_cpi; import tempfile,os; from pathlib import Path; tmp=Path(tempfile.gettempdir())/'t_check.csv'; fetch_transport_cpi(tmp); after=tmp.read_bytes(); print('transport reproduces identically:', hashlib.md5(before).hexdigest()==hashlib.md5(after).hexdigest()); import pandas as pd; a=pd.read_csv('ph_economic_ai/benchmark/data/psa_transport_cpi_monthly.csv'); b=pd.read_csv(tmp); print('rows', len(a), len(b), '| equal', a.equals(b))"`
Expected: prints `transport reproduces identically: True` (or at least identical row count + `a.equals(b) True`). If NOT identical, the refactor changed transport behaviour — STOP and report.

- [ ] **Step 4: Commit (committed gold CSV only)**

```bash
git add ph_economic_ai/benchmark/data/psa_food_cpi_monthly.csv
git status --short
git commit -m "feat(benchmark): freeze PSA Food-CPI gold (PX-Web, COICOP 01)"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 3: Free agri-futures food predictor panel (network)

**Files:**
- Modify: `ph_economic_ai/benchmark/refresh_data.py`
- Create (by running): `ph_economic_ai/benchmark/data/food_features_monthly.csv`

- [ ] **Step 1: Append `build_food_features` to refresh_data.py**

```python
FOOD_FEATURES_OUT = HERE / 'data' / 'food_features_monthly.csv'


def build_food_features(rng: str = 'max') -> None:
    """Free global food-commodity predictor panel for the Food-CPI nowcast:
    Yahoo agri futures + oil + USD/PHP -> data/food_features_monthly.csv."""
    cols = {'ZR=F': 'rice', 'ZW=F': 'wheat', 'ZC=F': 'corn', 'ZS=F': 'soybean',
            'BZ=F': 'oil_price', 'PHP=X': 'usd_php'}
    parts = [_yahoo_monthly(t, rng).rename(name) for t, name in cols.items()]
    base = pd.concat(parts, axis=1).dropna().reset_index().rename(columns={'index': 'date'})
    base = base.sort_values('date')
    FOOD_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(FOOD_FEATURES_OUT, index=False)
    print(f'Wrote food_features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')
```
(`_yahoo_monthly` returns a Series indexed `YYYY-MM`; `HERE` and `pd` already exist in the module.)

- [ ] **Step 2: Build the panel (network)**

Run: `python -c "from ph_economic_ai.benchmark.refresh_data import build_food_features; build_food_features()"`
Expected: `Wrote food_features_monthly.csv (N rows, ~2000-1x..YYYY-MM)`, N ≈ 280–310 (agri futures start ~2000). If a ticker fails, retry once; if persistent, STOP and report.

- [ ] **Step 3: Verify overlap with the food gold**

Run: `python -c "import pandas as pd; from ph_economic_ai.benchmark.psa_cpi import load_food_mom; f=pd.read_csv('ph_economic_ai/benchmark/data/food_features_monthly.csv',dtype={'date':str}).set_index('date'); m=load_food_mom(); ov=f.index.intersection(m.index); print('feat rows',len(f),'| food MoM rows',len(m),'| overlap',len(ov),'=> backtest n ~',len(ov)-24); print('cols', list(f.columns))"`
Expected: overlap ≈ 280+; columns `['rice','wheat','corn','soybean','oil_price','usd_php']`.

- [ ] **Step 4: Commit (code + committed CSV)**

```bash
git add ph_economic_ai/benchmark/refresh_data.py ph_economic_ai/benchmark/data/food_features_monthly.csv
git status --short
git commit -m "feat(benchmark): free agri-futures food predictor panel (committed CSV)"
```
Confirm xlsx NOT staged.

---

## Task 4: Food nowcast runner (with robustness)

**Files:**
- Create: `ph_economic_ai/benchmark/food_nowcast.py`
- Test: `ph_economic_ai/tests/test_food_nowcast.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_food_nowcast.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.food_nowcast as fn


def test_run_food_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2005-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(5)
    rice = 12 + np.cumsum(rng.normal(0, 0.2, n))
    feats = pd.DataFrame({
        'rice': rice,
        'wheat': 500 + np.cumsum(rng.normal(0, 5, n)),
        'corn': 400 + np.cumsum(rng.normal(0, 4, n)),
        'soybean': 1000 + np.cumsum(rng.normal(0, 8, n)),
        'oil_price': 60 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 50 + np.cumsum(rng.normal(0, 0.1, n)),
    }, index=idx)
    mom = pd.Series(0.4 * np.r_[0.0, np.diff(rice)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(fn, 'load_food_mom', lambda: mom)
    res = fn.run_food_nowcast(min_train=24, features=feats, prelim_months=6)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']
    assert 'panel' not in res['mom']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_food_nowcast.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.food_nowcast`.

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/food_nowcast.py`:

```python
"""MoM Food-CPI nowcast — global food-commodity pass-through, tested honestly.

Builds a food nowcast frame from free agri-futures + oil + FX predictors and the
PSA Food-CPI target, then reuses the existing MoM nowcast runners (unchanged) with
the same preliminary-data robustness re-test as the Transport nowcast.
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation, run_mom_nowcast
from ph_economic_ai.benchmark.psa_cpi import load_food_mom

FOOD_FEATURES_CSV = Path(__file__).parent / 'data' / 'food_features_monthly.csv'


def load_food_features(csv_path: Path = FOOD_FEATURES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={'date': str}).set_index('date').sort_index()


def _build_food_frame(features: pd.DataFrame) -> pd.DataFrame:
    tgt = load_food_mom()
    base = pd.DataFrame({
        'rice': features['rice'], 'wheat': features['wheat'], 'corn': features['corn'],
        'soybean': features['soybean'], 'oil': features['oil_price'],
        'fx': features['usd_php'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base['prev_mom'] = base['target'].shift(1)
    cols = ['rice', 'wheat', 'corn', 'soybean', 'oil', 'fx', 'prev_mom', 'target']
    return base[cols].dropna()


def run_food_nowcast(min_train: int = 24, features=None, prelim_months: int = 6) -> dict:
    """Food-MoM nowcast + driver-only ablation + trailing-preliminary robustness
    re-test. `driver_edge_robust` is the canonical verdict."""
    feats = load_food_features() if features is None else features
    frame = _build_food_frame(feats)
    drop = ('panel', 'calibration')

    def _slim(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in drop}

    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    robust_frame = (frame.iloc[:-prelim_months]
                    if prelim_months and len(frame) > prelim_months + min_train
                    else frame)
    r_abl = run_driver_only_ablation(min_train, frame=robust_frame)

    return {
        'n': int(mom.get('n', len(frame))),
        'mom': _slim(mom),
        'driver_ablation': _slim(abl),
        'driver_edge': bool(abl.get('driver_edge', False)),
        'robust': {
            'prelim_months_dropped': int(prelim_months),
            'n': int(r_abl.get('n', len(robust_frame))),
            'driver_ablation': _slim(r_abl),
            'driver_edge': bool(r_abl.get('driver_edge', False)),
        },
        'driver_edge_robust': bool(r_abl.get('driver_edge', False)),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_food_nowcast.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/food_nowcast.py ph_economic_ai/tests/test_food_nowcast.py
git commit -m "feat(benchmark): food-CPI nowcast runner (agri drivers + robustness re-test)"
```

---

## Task 5: Report key

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_food_nowcast():
    rep = build_report(
        date_range=('2000-11', '2025-05'), n_months=280,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 280},
        data_hash='abc123',
        food_nowcast={'n': 270, 'driver_edge': False, 'driver_edge_robust': False,
                      'mom': {'verdict': 'no_better_than_naive'},
                      'driver_ablation': {'verdict': 'no_better_than_naive',
                                          'driver_edge': False},
                      'robust': {'prelim_months_dropped': 6, 'n': 264,
                                 'driver_edge': False,
                                 'driver_ablation': {'verdict': 'no_better_than_naive'}}},
    )
    assert rep['food_nowcast']['n'] == 270
    assert 'food_nowcast' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_food_nowcast -v`
Expected: FAIL — unexpected keyword argument `food_nowcast`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`: (a) append `'food_nowcast'` to `REQUIRED_KEYS`; (b) add `food_nowcast=None` as the LAST parameter of `build_report`; (c) in the returned dict, just before `'limitations'`:
```python
        'food_nowcast': food_nowcast if food_nowcast is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the food-CPI nowcast result"
```

---

## Task 6: Run the food nowcast in run.py (real result)

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`

- [ ] **Step 1: Wire into run.py**

In `main()`, after the transport-nowcast block and BEFORE `rep = report.build_report(`, add:
```python
    # -- MoM Food-CPI nowcast (global food-commodity pass-through) --
    try:
        from ph_economic_ai.benchmark import food_nowcast as food_mod
        food_res = food_mod.run_food_nowcast(MIN_TRAIN)
        _fm = food_res['mom']
        print(f"Food nowcast (n={food_res['n']}): mom={_fm['verdict']} "
              f"best={_fm.get('best_method')} skill={_fm.get('best_skill_vs_naive')} "
              f"DM p={_fm.get('dm_p')} | driver_edge_robust={food_res['driver_edge_robust']}")
    except FileNotFoundError:
        food_res = {'verdict': 'not_run', 'reason': 'food gold/features missing'}
        print('Food nowcast: not_run (gold or features CSV missing)')
```
In the `report.build_report(` call add:
```python
        food_nowcast=food_res,
```
After the existing artifact writes add:
```python
    import json as _json8
    (report.ARTIFACTS / 'food_nowcast_table.json').write_text(
        _json8.dumps(food_res, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Run end-to-end (a few minutes)**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints a `Food nowcast (n=...): ...` line (n large, NOT not_run); writes `food_nowcast_table.json`; no errors. Fix only run.py wiring on error; if a real bug surfaces in food_nowcast/nowcast, STOP and report.

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; import json; print(json.dumps(load_report()['food_nowcast'], indent=2))"`
Record `n`, the `mom` dict, `driver_edge` (full) and `driver_edge_robust` (canonical). **This is the answer** — note whether food shows a robust edge or is efficient. Report either way.

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (suite is currently fully green). Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/food_nowcast_table.json
git status --short
git commit -m "feat(benchmark): run the food-CPI nowcast; report the result"
```
Confirm xlsx NOT staged.

---

## Task 7: Surface in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'food_nowcast': {'n': 270, 'driver_edge': False, 'driver_edge_robust': False,
                         'mom': {'verdict': 'no_better_than_naive'},
                         'driver_ablation': {'verdict': 'no_better_than_naive',
                                             'driver_edge': False},
                         'robust': {'prelim_months_dropped': 6, 'n': 264,
                                    'driver_edge': False,
                                    'driver_ablation': {'verdict': 'no_better_than_naive'}}},
```
Add the test:
```python
def test_view_shows_food_nowcast(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.food_nowcast_summary()
    assert '270' in s
    assert 'driver_edge_robust=False' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_food_nowcast -v`
Expected: FAIL — no attribute `food_nowcast_summary`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add the method (next to `transport_nowcast_summary`):
```python
    def food_nowcast_summary(self) -> str:
        if not self._report:
            return ''
        F = self._report.get('food_nowcast') or {}
        if not F or F.get('verdict') == 'not_run':
            return ''
        robust = bool(F.get('driver_edge_robust'))
        rob = F.get('robust') or {}
        verdict = ('robust food-commodity edge — significant' if robust
                   else 'efficient — no robust food-commodity edge')
        caveat = ''
        if F.get('driver_edge') and not robust:
            caveat = (f" (full-sample driver_edge=True is an artifact of "
                      f"{rob.get('prelim_months_dropped')} preliminary recent months; "
                      f"dropping them → not significant)")
        return (f"Food-CPI nowcast (n={F.get('n')}): {verdict}; "
                f"driver_edge_robust={robust}{caveat}.")
```
(b) In `_build`, inside `if self._report is not None:`, after the transport-nowcast block, add:
```python
            _fn = self.food_nowcast_summary()
            if _fn:
                fnl = QLabel('<b>Food-CPI nowcast (food commodities→inflation)</b><br>' + _fn)
                fnl.setWordWrap(True)
                col.addWidget(fnl)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py -v`
Expected: PASS (existing + new)

- [ ] **Step 5: Window smoke**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/accuracy_view.py ph_economic_ai/tests/test_accuracy_view.py
git commit -m "feat(ui): show the food-CPI nowcast in the Accuracy view"
```

---

## Task 8: Record the outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-food-cpi-nowcast-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 6 Step 3**

Replace the `[n]`/`[m]`/`[x]`/`[p]`/`[bool]`/`[k]`/`[nr]` markers with the measured values from `food_nowcast_table.json`, and select the matching interpretation bullet (robust True → genuine edge; efficient/not robust → honest negative). Copy numbers verbatim; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-10-food-cpi-nowcast-design.md
git commit -m "docs: record measured food-CPI nowcast result"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass.

- [ ] **Result legible from report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; F=load_report()['food_nowcast']; print('n', F.get('n'), '| mom', F['mom']['verdict'], '| food driver_edge_robust', F.get('driver_edge_robust'))"`
Expected: prints the food nowcast verdict + whether the food-commodity edge is robust — the honest answer either way.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 generalized gold fetch → Tasks 1–2; §3.2 agri predictor panel → Task 3; §4.1 psa_cpi generalization + food loaders → Task 1; §4.2 build_food_features → Task 3; §4.3 food_nowcast runner + robustness → Task 4; §4.4 report/run/view → Tasks 5–7; §6 error handling (not_run guard, label-resolution error, reserved names) → Tasks 1, 4, 6; §7 testing → Tasks 1, 4, 5, 7; §8 deliverables incl. transport-unchanged check → Task 2 Step 3; §9 write-up → Task 8.

**Placeholder scan:** §9 markers filled at Task 8 from real output (empirical). Tasks 2–3 have no unit tests by design (network refresh). No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `_resolve_commodity_id(commodity_var, coicop_prefix)` (Task 1) used by generalized `_fetch_px_table` and by `fetch_food_cpi`/`fetch_transport_cpi`. `load_food_mom()` (Task 1) reused in Task 4 (`fn.load_food_mom` monkeypatch + `_build_food_frame`). `run_food_nowcast(min_train=24, features=None, prelim_months=6) -> {n, mom, driver_ablation, driver_edge, robust, driver_edge_robust}` consistent across Tasks 4, 6, 7. Frame columns `['rice','wheat','corn','soybean','oil','fx','prev_mom','target']` — only `prev_mom`/`target` are reserved by the runners (verified against `nowcast.py`: `run_mom_nowcast` uses all non-`target` cols; `run_driver_only_ablation` drops `prev_mom`). `build_report(... food_nowcast=)` matches Task 5 (def) and Task 6 (call); report key `food_nowcast` consistent across Tasks 5–7. The view guards both the `not_run` dict and the full result.
```
