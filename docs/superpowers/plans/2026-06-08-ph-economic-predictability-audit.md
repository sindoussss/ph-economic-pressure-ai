# Predictability Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the benchmark into a predictability audit across three Philippine economic series (fuel RON95, USD/PHP FX, CPI inflation), each scored through the existing forecaster panel + Diebold-Mariano test, producing an efficient/predictable verdict per target.

**Architecture:** A `Target` registry (`targets.py`) abstracts each series as `{name, load_gold, build_frame, has_mechanism}`; a generic `build_target_frame` (in `features.py`) makes lagged feature frames; `audit.py` runs each target through the existing `efficiency.run_panel` (generalized with a `target_col` param) and assigns a verdict. New FX + CPI gold CSVs are fetched once via `refresh_data.py` and committed, same reproducible pattern as the World Bank fuel workbook.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, statsmodels, scipy, matplotlib, pytest. Builds on the committed Phase 1/2/Efficiency benchmark.

**Spec:** `docs/superpowers/specs/2026-06-08-ph-economic-predictability-audit-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/efficiency.py::run_panel(frame, methods, min_train, feature_cols)` → rows `{method,rmse,mae,skill_vs_rw,dm_stat,dm_p,n}`; currently reads `frame['ron95']`.
- `benchmark/forecasters.py` (7 methods), `benchmark/significance.py`, `benchmark/conformal.py`, `benchmark/backtest.py::walk_forward`.
- `benchmark/features.py::build_feature_frame`, `VARIANTS`.
- `benchmark/ground_truth.py::load_world_bank_ron95`.
- `benchmark/report.py::{build_report (has efficiency=, passthrough=), REQUIRED_KEYS, ARTIFACTS, load_report}`.
- `benchmark/run.py`, `benchmark/figures.py`, `benchmark/refresh_data.py`, `ui/accuracy_view.py`.
- Committed data: `data/world_bank_ron95.csv`, `data/features_monthly.csv` (cols `date,oil_price,usd_php,gas_price,demand_index`).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, import `from ph_economic_ai.X import Y`, start with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  ```
- Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging area clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing. `global_fuel_prices.xlsx` is gitignored — never stage it.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure

**Create:** `benchmark/targets.py`, `benchmark/audit.py`; tests `test_targets.py`, `test_audit.py`.
**Modify:** `benchmark/features.py` (+`build_target_frame`), `benchmark/efficiency.py` (+`target_col`), `benchmark/report.py` (+`audit` key), `benchmark/refresh_data.py` (+FX/CPI CSVs), `benchmark/run.py`, `benchmark/figures.py` (+`plot_audit_verdicts`), `ui/accuracy_view.py` (+audit panel); tests `test_features.py`, `test_efficiency.py`, `test_report.py`, `test_accuracy_view.py`; spec §8.

---

## Task 1: Generic lagged-feature builder

**Files:**
- Modify: `ph_economic_ai/benchmark/features.py` (append)
- Modify: `ph_economic_ai/tests/test_features.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_features.py`:

```python
from ph_economic_ai.benchmark.features import build_target_frame


def test_build_target_frame_lags_and_target():
    idx = pd.date_range('2018-01', periods=30, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    tgt = pd.Series(50 + np.cumsum(rng.normal(0, 1, 30)), index=idx)
    drivers = pd.DataFrame({'oil': 70 + np.cumsum(rng.normal(0, 1, 30)),
                            'fx': 55 + np.cumsum(rng.normal(0, 0.1, 30))}, index=idx)
    f = build_target_frame(tgt, drivers, 'fx', ['oil', 'fx'])
    assert 'target' in f.columns
    assert 'prev_fx' in f.columns
    for c in ('oil_lag1', 'oil_ma3', 'fx_lag1', 'fx_ma3'):
        assert c in f.columns
    assert not f.isna().any().any()
    # prev_fx at row t equals target at t-1 (causal)
    t = f.index[4]
    pos = list(tgt.index).index(t)
    assert f.loc[t, 'prev_fx'] == pytest.approx(tgt.iloc[pos - 1])
    # oil_lag1 equals raw oil at t-1
    assert f.loc[t, 'oil_lag1'] == pytest.approx(drivers['oil'].iloc[pos - 1])


def test_build_target_frame_inner_join_on_dates():
    tgt = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                    index=['2020-01', '2020-02', '2020-03', '2020-04', '2020-05'])
    drivers = pd.DataFrame({'oil': [10.0, 11.0, 12.0]},
                           index=['2020-01', '2020-02', '2020-03'])
    f = build_target_frame(tgt, drivers, 'x', ['oil'])
    # only shared dates survive the join; after lag/ma + dropna, <= 1 row
    assert len(f) <= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_features.py -k build_target_frame -v`
Expected: FAIL — `ImportError: cannot import name 'build_target_frame'`

- [ ] **Step 3: Append the implementation to features.py**

```python
def build_target_frame(target_series, driver_df, target_name: str, drivers: list):
    """Generic 1-month-ahead frame for any target. Produces prev_<target_name>,
    plus lag-1 and 3-month-MA of each driver, and a 'target' column. All features
    are lagged (known at t-1). Inner-joins target and drivers on the date index,
    dropna'd to common support."""
    base = pd.DataFrame({'__t__': target_series})
    joined = base.join(driver_df[list(drivers)], how='inner').sort_index()
    f = pd.DataFrame(index=joined.index)
    f[f'prev_{target_name}'] = joined['__t__'].shift(1)
    for d in drivers:
        f[f'{d}_lag1'] = joined[d].shift(1)
        f[f'{d}_ma3'] = joined[d].shift(1).rolling(3).mean()
    f['target'] = joined['__t__']
    return f.dropna()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_features.py -v`
Expected: PASS (existing feature tests + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/features.py ph_economic_ai/tests/test_features.py
git commit -m "feat(benchmark): generic build_target_frame for the predictability audit"
```

---

## Task 2: Generalize run_panel with a target column

**Files:**
- Modify: `ph_economic_ai/benchmark/efficiency.py`
- Modify: `ph_economic_ai/tests/test_efficiency.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_efficiency.py`:

```python
def test_run_panel_accepts_custom_target_col():
    idx = pd.date_range('2017-01', periods=70, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(5)
    y = 50 + np.cumsum(rng.normal(0, 0.4, 70))
    frame = pd.DataFrame({
        'prev_x': np.r_[y[0], y[:-1]],
        'drv_lag1': np.r_[0, np.diff(y)],
        'target': y,
    }, index=idx)
    rows = run_panel(frame, ['random_walk', 'ridge'], min_train=24,
                     feature_cols=['prev_x', 'drv_lag1'], target_col='target')
    assert [r['method'] for r in rows] == ['random_walk', 'ridge']
    assert rows[0]['n'] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_efficiency.py::test_run_panel_accepts_custom_target_col -v`
Expected: FAIL — `run_panel() got an unexpected keyword argument 'target_col'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/efficiency.py`, change the `run_panel` signature and the line that reads `y`:

```python
def run_panel(frame, methods, min_train: int, feature_cols, target_col: str = 'ron95') -> list:
```
and replace `y = frame['ron95'].to_numpy(dtype=float)` with:
```python
    y = frame[target_col].to_numpy(dtype=float)
```
Leave the rest unchanged (default `'ron95'` keeps all existing callers working).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_efficiency.py -v`
Expected: PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/efficiency.py ph_economic_ai/tests/test_efficiency.py
git commit -m "feat(benchmark): run_panel accepts a configurable target column"
```

---

## Task 3: Target registry

**Files:**
- Create: `ph_economic_ai/benchmark/targets.py`
- Test: `ph_economic_ai/tests/test_targets.py`

`Target` is a dataclass holding callables (so tests can build synthetic targets). Loaders read committed CSVs (real data arrives in Task 6). The inflation loader converts a CPI index to YoY %.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_targets.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.targets import (
    Target, TARGETS, load_fx, load_inflation, cpi_to_yoy,
)


def test_cpi_to_yoy_computes_year_on_year_percent():
    idx = pd.date_range('2019-01', periods=14, freq='MS').strftime('%Y-%m')
    cpi = pd.Series(100.0 * (1.03) ** (np.arange(14) / 12.0), index=idx)  # ~3%/yr
    infl = cpi_to_yoy(cpi)
    # first 12 months dropped; remaining ~3%
    assert infl.index[0] == '2020-01'
    assert infl.iloc[0] == pytest.approx(3.0, abs=0.2)


def test_load_fx_reads_csv(tmp_path):
    p = tmp_path / 'fx.csv'
    p.write_text('date,usd_php\n2020-01,50.0\n2020-02,51.0\n', encoding='utf-8')
    s = load_fx(p)
    assert list(s.index) == ['2020-01', '2020-02']
    assert s.iloc[1] == pytest.approx(51.0)


def test_load_inflation_reads_index_csv(tmp_path):
    idx = pd.date_range('2019-01', periods=14, freq='MS').strftime('%Y-%m')
    vals = 100.0 * (1.04) ** (np.arange(14) / 12.0)
    p = tmp_path / 'cpi.csv'
    p.write_text('date,cpi_index\n' + '\n'.join(f'{d},{v:.4f}' for d, v in zip(idx, vals)) + '\n',
                 encoding='utf-8')
    infl = load_inflation(p)
    assert infl.iloc[0] == pytest.approx(4.0, abs=0.2)


def test_registry_has_three_targets():
    assert set(TARGETS) == {'fuel', 'fx', 'inflation'}
    for name, t in TARGETS.items():
        assert isinstance(t, Target) and t.name == name
        assert callable(t.load_gold) and callable(t.build_frame)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_targets.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/targets.py`:

```python
"""Target registry for the predictability audit. Each Target abstracts a
Philippine economic series so the same panel + DM machinery can audit it.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ph_economic_ai.benchmark.features import build_target_frame
from ph_economic_ai.benchmark.ground_truth import load_world_bank_ron95

DATA = Path(__file__).parent / 'data'
FEATURES_CSV = DATA / 'features_monthly.csv'
FX_CSV = DATA / 'usd_php_monthly.csv'
CPI_CSV = DATA / 'ph_cpi_monthly.csv'


@dataclass
class Target:
    name: str
    load_gold: Callable[[], pd.Series]
    build_frame: Callable[[], pd.DataFrame]
    has_mechanism: bool = False


# ── Loaders ─────────────────────────────────────────────────────────────────────

def _features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV, dtype={'date': str}).set_index('date').sort_index()


def load_fx(csv_path: Path = FX_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['usd_php'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def cpi_to_yoy(cpi_index: pd.Series) -> pd.Series:
    """Convert a monthly CPI index to year-on-year inflation %, dropping first 12."""
    s = cpi_index.sort_index()
    yoy = (s / s.shift(12) - 1.0) * 100.0
    return yoy.dropna()


def load_inflation(csv_path: Path = CPI_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    cpi = pd.Series(df['cpi_index'].astype(float).values, index=df['date'].astype(str).values)
    cpi = cpi[~cpi.index.duplicated(keep='last')]
    return cpi_to_yoy(cpi)


# ── Frame builders ──────────────────────────────────────────────────────────────

def _fuel_frame() -> pd.DataFrame:
    gold = load_world_bank_ron95()
    drivers = _features()
    return build_target_frame(gold, drivers, 'fuel',
                              ['oil_price', 'usd_php', 'gas_price', 'demand_index'])


def _fx_frame() -> pd.DataFrame:
    fx = load_fx()
    feats = _features()
    drivers = pd.DataFrame({'oil': feats['oil_price']})
    drivers = drivers.join(load_inflation().rename('inflation'), how='outer')
    return build_target_frame(fx, drivers, 'fx', ['oil', 'inflation'])


def _inflation_frame() -> pd.DataFrame:
    infl = load_inflation()
    feats = _features()
    drivers = pd.DataFrame({'fuel': feats['gas_price'], 'fx': feats['usd_php']})
    return build_target_frame(infl, drivers, 'inflation', ['fuel', 'fx'])


TARGETS = {
    'fuel': Target('fuel', load_world_bank_ron95, _fuel_frame, has_mechanism=True),
    'fx': Target('fx', load_fx, _fx_frame, has_mechanism=False),
    'inflation': Target('inflation', load_inflation, _inflation_frame, has_mechanism=False),
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_targets.py -v`
Expected: PASS (4 passed). (The registry test only checks structure/callables; it does NOT call `build_frame`, which needs the real CSVs from Task 6.)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/targets.py ph_economic_ai/tests/test_targets.py
git commit -m "feat(benchmark): Target registry (fuel/fx/inflation) + loaders"
```

---

## Task 4: Audit runner + verdict logic

**Files:**
- Create: `ph_economic_ai/benchmark/audit.py`
- Test: `ph_economic_ai/tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_audit.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.targets import Target
from ph_economic_ai.benchmark.audit import verdict_from_panel, run_audit


def test_verdict_predictable_when_a_method_beats_rw():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'ridge', 'skill_vs_rw': 0.2, 'dm_p': 0.01},
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'predictable' and best['method'] == 'ridge'


def test_verdict_efficient_when_none_significantly_better():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'hgb', 'skill_vs_rw': -0.05, 'dm_p': 0.9},
        {'method': 'arima', 'skill_vs_rw': -0.2, 'dm_p': 0.01},   # significant but worse
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'efficient' and best['method'] == 'random_walk'


def _predictable_target():
    idx = pd.date_range('2016-01', periods=80, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(7)
    drv = np.cumsum(rng.normal(0, 1, 80))
    y = np.r_[0, 0.9 * np.diff(drv)] + 50          # target strongly driven by drv change
    frame = pd.DataFrame({'prev_t': np.r_[y[0], y[:-1]],
                          'drv_lag1': np.r_[0, np.diff(drv)], 'target': y}, index=idx)
    return Target('synthetic', lambda: pd.Series(y, index=idx), lambda: frame)


def test_run_audit_reports_per_target_verdict():
    reg = {'synthetic': _predictable_target()}
    rows = run_audit(['synthetic'], min_train=24, registry=reg)
    assert rows[0]['target'] == 'synthetic'
    assert rows[0]['verdict'] in ('predictable', 'efficient')
    assert 'panel' in rows[0] and rows[0]['n'] > 0


def test_run_audit_insufficient_data():
    short = Target('short', lambda: pd.Series(dtype=float),
                   lambda: pd.DataFrame({'a': [1.0, 2.0], 'target': [1.0, 2.0]},
                                        index=['2020-01', '2020-02']))
    rows = run_audit(['short'], min_train=24, registry={'short': short})
    assert rows[0]['verdict'] == 'insufficient_data'
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_audit.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/audit.py`:

```python
"""Predictability audit: run each economic target through the forecaster panel +
Diebold-Mariano test and assign an efficient/predictable verdict.
"""
from ph_economic_ai.benchmark.efficiency import run_panel

PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']


def verdict_from_panel(panel: list):
    """('predictable', best_row) if any method significantly beats random walk
    (dm_p < 0.05 and skill > 0); else ('efficient', random_walk_row)."""
    beats = [r for r in panel
             if r.get('dm_p') is not None and r['dm_p'] < 0.05 and r['skill_vs_rw'] > 0]
    if beats:
        return 'predictable', max(beats, key=lambda r: r['skill_vs_rw'])
    rw = next((r for r in panel if r['method'] == 'random_walk'), panel[0])
    return 'efficient', rw


def run_audit(target_names, min_train: int = 24, registry=None) -> list:
    """Audit each named target. registry defaults to targets.TARGETS."""
    if registry is None:
        from ph_economic_ai.benchmark.targets import TARGETS
        registry = TARGETS

    rows = []
    for name in target_names:
        target = registry[name]
        try:
            frame = target.build_frame()
        except Exception as e:                       # data missing / unreadable
            rows.append({'target': name, 'verdict': 'insufficient_data',
                         'error': str(e)[:120], 'n': 0})
            continue
        if len(frame) < min_train + 5:
            rows.append({'target': name, 'verdict': 'insufficient_data',
                         'n': int(len(frame))})
            continue
        feature_cols = [c for c in frame.columns if c != 'target']
        panel = run_panel(frame, PANEL_METHODS, min_train, feature_cols, target_col='target')
        verdict, best = verdict_from_panel(panel)
        rows.append({
            'target': name,
            'verdict': verdict,
            'best_method': best['method'],
            'best_skill': best['skill_vs_rw'],
            'best_dm_p': best.get('dm_p'),
            'n': int(panel[0]['n']),
            'panel': panel,
        })
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_audit.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/audit.py ph_economic_ai/tests/test_audit.py
git commit -m "feat(benchmark): predictability audit runner + verdict logic"
```

---

## Task 5: Report key for the audit

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_audit():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        audit=[{'target': 'fuel', 'verdict': 'efficient', 'best_method': 'random_walk',
                'best_skill': 0.0, 'best_dm_p': None, 'n': 79}],
    )
    assert rep['audit'][0]['verdict'] == 'efficient'
    assert 'audit' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_audit -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'audit'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'audit'` to `REQUIRED_KEYS`.
(b) Add `audit=None` as the last param of `build_report` (after `passthrough=None`).
(c) In the returned dict, add before `'limitations'`:
```python
        'audit': audit if audit is not None else [],
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all; existing calls default `audit` to `[]`)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the predictability-audit table"
```

---

## Task 6: Fetch + commit FX and CPI gold series

**Files:**
- Modify: `ph_economic_ai/benchmark/refresh_data.py`
- Create (by running): `data/usd_php_monthly.csv`, `data/ph_cpi_monthly.csv`

- [ ] **Step 1: Add builders to refresh_data.py**

Append to `ph_economic_ai/benchmark/refresh_data.py`:

```python
FX_OUT = HERE / 'data' / 'usd_php_monthly.csv'
CPI_OUT = HERE / 'data' / 'ph_cpi_monthly.csv'
FRED_CPI_ID = 'PHLCPIALLMINMEI'   # OECD MEI monthly CPI, Philippines (index)


def build_fx_csv() -> None:
    """USD/PHP monthly close from Yahoo -> data/usd_php_monthly.csv."""
    fx = _yahoo_monthly('PHP=X')                       # defined earlier in this module
    df = fx.rename('usd_php').reset_index()
    df.columns = ['date', 'usd_php']
    FX_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FX_OUT, index=False)
    print(f'Wrote usd_php_monthly.csv ({len(df)} rows, {df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def build_cpi_csv() -> None:
    """PH monthly CPI index from FRED -> data/ph_cpi_monthly.csv.

    If the FRED id is retired/unreachable, download manually from DBnomics:
      https://api.db.nomics.world/v22/series/IMF/IFS/M.PH.PCPI_IX?observations=1
    and save a 2-column CSV 'date,cpi_index' (date as YYYY-MM) to CPI_OUT.
    """
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_CPI_ID}'
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(r.text))
    raw.columns = ['date', 'cpi_index']
    raw['date'] = pd.to_datetime(raw['date']).dt.strftime('%Y-%m')
    raw = raw[pd.to_numeric(raw['cpi_index'], errors='coerce').notna()]
    CPI_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CPI_OUT, index=False)
    print(f'Wrote ph_cpi_monthly.csv ({len(raw)} rows, {raw["date"].iloc[0]}..{raw["date"].iloc[-1]})')
```

`import requests` is already present (used by the WB downloader); `_yahoo_monthly` and `_HEADERS` already exist in this module from the Phase-1/efficiency work — verify with `git grep -n "_yahoo_monthly\|_HEADERS" ph_economic_ai/benchmark/refresh_data.py`. If `_yahoo_monthly` is absent, copy it from the earlier features-refresh code (it fetches `interval=1mo&range=10y`).

- [ ] **Step 2: Build the FX CSV (Yahoo — works headless)**

Run: `python -c "from ph_economic_ai.benchmark.refresh_data import build_fx_csv; build_fx_csv()"`
Expected: prints `Wrote usd_php_monthly.csv (~120 rows ...)`.

- [ ] **Step 3: Build the CPI CSV (FRED)**

Run: `python -c "from ph_economic_ai.benchmark.refresh_data import build_cpi_csv; build_cpi_csv()"`
Expected: prints `Wrote ph_cpi_monthly.csv (...)`.
If FRED is unreachable/blocked or the id is retired (timeout or HTTP error): download the series manually from the DBnomics URL in the docstring, save a `date,cpi_index` CSV (date `YYYY-MM`) to `ph_economic_ai/benchmark/data/ph_cpi_monthly.csv`, and confirm it loads:
`python -c "from ph_economic_ai.benchmark.targets import load_inflation; print(load_inflation().tail())"`

- [ ] **Step 4: Verify both load and have enough history**

Run: `python -c "from ph_economic_ai.benchmark.targets import load_fx, load_inflation; print('fx', len(load_fx())); print('infl', len(load_inflation()))"`
Expected: `fx` ≥ 60, `infl` ≥ 40 (enough for min_train=24 backtests).

- [ ] **Step 5: Commit (code + committed gold CSVs)**

```bash
git add ph_economic_ai/benchmark/refresh_data.py ph_economic_ai/benchmark/data/usd_php_monthly.csv ph_economic_ai/benchmark/data/ph_cpi_monthly.csv
git status --short
git commit -m "feat(benchmark): fetch + commit FX (Yahoo) and CPI (FRED) gold series"
```

---

## Task 7: Run the audit in run.py + figure

**Files:**
- Modify: `ph_economic_ai/benchmark/figures.py`
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: end-to-end run on real data

- [ ] **Step 1: Add the figure**

Append to `ph_economic_ai/benchmark/figures.py`:

```python
def plot_audit_verdicts(rows):
    """Per-target best-skill bar, colored green=predictable / gray=efficient."""
    _ensure_dir()
    rows = [r for r in rows if r.get('verdict') in ('efficient', 'predictable')]
    names = [r['target'] for r in rows]
    skills = [r.get('best_skill', 0.0) for r in rows]
    colors = ['tab:green' if r['verdict'] == 'predictable' else 'tab:gray' for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, skills, color=colors)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Best skill vs random walk')
    ax.set_title('Predictability audit (green = predictable)')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'audit_verdicts.png', dpi=120); plt.close(fig)
```

- [ ] **Step 2: Wire into run.py**

In `ph_economic_ai/benchmark/run.py`:
(a) Add import near other benchmark imports:
```python
from ph_economic_ai.benchmark import audit as audit_mod
```
(b) After the efficiency-panel/pass-through block (before `rep = report.build_report(`), add:
```python
    # -- Cross-target predictability audit --
    audit_rows = audit_mod.run_audit(['fuel', 'fx', 'inflation'], MIN_TRAIN)
    print('Predictability audit:')
    for a in audit_rows:
        if a['verdict'] == 'insufficient_data':
            print(f"  {a['target']:<10} insufficient_data (n={a.get('n', 0)})")
        else:
            print(f"  {a['target']:<10} {a['verdict']:<12} best={a['best_method']} "
                  f"skill={a['best_skill']:+.3f}")
```
(c) In the `report.build_report(` call, add the kwarg:
```python
        audit=[{k: v for k, v in a.items() if k != 'panel'} for a in audit_rows],
```
(the per-target `panel` is dropped from the headline report to keep it compact; the full panels go to `audit_table.json` next.)
(d) After the figure calls, add:
```python
    import json as _json2
    (report.ARTIFACTS / 'audit_table.json').write_text(
        _json2.dumps(audit_rows, indent=2), encoding='utf-8')
    figures.plot_audit_verdicts(audit_rows)
```

- [ ] **Step 3: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints the ablation table, efficiency panel, pass-through, AND the audit (fuel/fx/inflation verdicts); writes `audit_table.json` + `audit_verdicts.png` without error. If a target shows `insufficient_data`, check Task 6's CSVs loaded (`load_fx`/`load_inflation` lengths).

- [ ] **Step 4: Record the real verdicts (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; [print(a['target'], a['verdict'], a.get('best_method'), a.get('best_skill'), a.get('best_dm_p')) for a in load_report()['audit']]"`
Record the output — this is the audit result for the spec/write-up.

- [ ] **Step 5: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window::test_on_run_requested_accepts_4_args`.

- [ ] **Step 6: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/figures.py ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/audit_table.json ph_economic_ai/benchmark/artifacts/figures/audit_verdicts.png
git status --short
git commit -m "feat(benchmark): run the cross-target predictability audit + figure"
```

---

## Task 8: Surface the audit in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'audit': [
            {'target': 'fuel', 'verdict': 'efficient', 'best_method': 'random_walk', 'best_skill': 0.0, 'best_dm_p': None, 'n': 79},
            {'target': 'inflation', 'verdict': 'predictable', 'best_method': 'ridge', 'best_skill': 0.22, 'best_dm_p': 0.01, 'n': 60},
        ],
```
Add the test:
```python
def test_view_shows_audit(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    a = view.audit_summary()
    assert 'fuel' in a and 'inflation' in a
    assert 'efficient' in a and 'predictable' in a
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_audit -v`
Expected: FAIL — `AttributeError: ... has no attribute 'audit_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method:
```python
    def audit_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('audit') or []
        lines = []
        for r in rows:
            if r.get('verdict') == 'insufficient_data':
                lines.append(f"{r['target']}: insufficient data")
            else:
                lines.append(f"{r['target']}: {r['verdict']} "
                             f"(best {r['best_method']}, skill {r['best_skill']:+.2f})")
        return '\n'.join(lines)
```
(b) In `_build`, inside `if self._report is not None:`, after the mechanism block, add:
```python
            if self._report.get('audit'):
                aud = QLabel('<b>Predictability audit (PH economy)</b><br>'
                             + self.audit_summary().replace('\n', '<br>'))
                aud.setWordWrap(True)
                col.addWidget(aud)
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
git commit -m "feat(ui): show the predictability-audit verdicts in the Accuracy view"
```

---

## Task 9: Record the audit outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-ph-economic-predictability-audit-design.md`

- [ ] **Step 1: Fill §8 with the real verdicts from Task 7 Step 4**

Replace the `[verdict]`/`[method]`/`[x]`/`[p]` placeholders in §8 with the measured per-target verdicts (fuel, fx, inflation): each target's verdict, best method, skill vs RW, and DM p. State plainly which series are efficient and which (if any) are predictable, and one sentence of interpretation tied to the literature (Meese-Rogoff for FX efficiency; Atkeson-Ohanian / persistence for inflation). Copy numbers from `audit_table.json`; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-08-ph-economic-predictability-audit-design.md
git commit -m "docs: record measured predictability-audit verdicts in spec section 8"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Audit legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; print({a['target']: a['verdict'] for a in load_report()['audit']})"`
Expected: a dict like `{'fuel': 'efficient', 'fx': ..., 'inflation': ...}` — the headline audit result, reported truthfully.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 Target abstraction → Tasks 1 (frame helper), 3 (registry). §3.2 audit runner + verdict → Task 4. §3.3 data wiring (FX/CPI) → Task 6. §3.4 integration (report/run/figure/view) → Tasks 5, 7, 8. §6 testing → Tasks 1,3,4,5,8. §7 deliverables → all. §8 write-up outcome → Task 9. `run_panel` generalization needed by the audit → Task 2.

**Placeholder scan:** §8's `[verdict]` markers are filled at Task 9 from real output (empirical; cannot be known at plan time), matching the accepted Phase 1/2/Efficiency pattern. No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `build_target_frame(target_series, driver_df, target_name, drivers)` consistent (Tasks 1, 3). `run_panel(..., target_col='ron95')` added in Task 2, used with `target_col='target'` in Task 4. `Target(name, load_gold, build_frame, has_mechanism)` consistent (Tasks 3, 4 test). Audit row keys `{target, verdict, best_method, best_skill, best_dm_p, n, panel}` consistent across Tasks 4, 5, 7, 8. `verdict_from_panel`/`run_audit(target_names, min_train, registry)` consistent (Task 4). `build_report(... audit=)` matches Task 5 (def) and Task 7 (call). `load_fx`/`load_inflation`/`cpi_to_yoy` consistent across Tasks 3, 6.
```
