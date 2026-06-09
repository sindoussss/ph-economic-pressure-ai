# Efficiency + Pass-Through Mechanism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the Phase 1/2 negative result into a defensible thesis contribution — show that *no standard forecaster beats random walk* for monthly PH RON95 (Diebold-Mariano), and *explain why* by measuring the DOE pass-through (elasticity + lag) on a random-walk driver.

**Architecture:** New headless modules in `ph_economic_ai/benchmark/`: a `forecasters.py` panel of standard methods exposed as the existing `predict_fn(X_train,y_train,x_next)` signature; a hand-coded `significance.py` Diebold-Mariano test; a `passthrough.py` OLS regression; and an `efficiency.py` runner that scores every method through the existing causal `walk_forward` and attaches a DM p-value vs random walk. `run.py`, `report.py`, and `accuracy_view.py` are extended to surface the panel + pass-through. No new data — uses committed `data/world_bank_ron95.csv` + `data/features_monthly.csv`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, scipy, **statsmodels (new)**, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-ph-economic-ai-efficiency-mechanism-design.md`.

**Prereqs (already on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/backtest.py::walk_forward(y, X, predict_fn, min_train) -> {y_true,y_pred,residuals,index}` (strictly causal; X may be None).
- `benchmark/metrics.py::{rmse, mae, skill_score}`.
- `benchmark/features.py::build_feature_frame(df)` → frame with cols incl. `prev_ron95, oil_lag1, usd_lag1, gas_lag1, gas_lag2, gas_lag3, gas_ma3, fx_ma3, gas_delta1, demand_lag1, proxy_lag1, ron95`; and `VARIANTS` (dict; `VARIANTS['passthrough_lags']['cols']` is the best Phase-2 feature set).
- `benchmark/report.py::{build_report, write_report, load_report, REQUIRED_KEYS, ARTIFACTS}` (already carries `ablation`, `selected_variant`).
- `benchmark/run.py::{main, _hgb_predict_fn, MIN_TRAIN, CONFORMAL_LEVELS}`.
- `ui/accuracy_view.py::AccuracyView` (reads the report; has `headline_text`, `ablation_summary`).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, import `from ph_economic_ai.X import Y`, start with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  ```
- Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging area is clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. Run `git status --short` before committing.
- Stay on branch `feature/accuracy-evaluation-phase1` (continues the merged Phase 1+2 line).

---

## File Structure

**Create:**
- `ph_economic_ai/benchmark/forecasters.py` — `make_forecaster(name)`, `FORECASTERS` dict
- `ph_economic_ai/benchmark/significance.py` — `diebold_mariano(loss_a, loss_b, h=1)`
- `ph_economic_ai/benchmark/passthrough.py` — `estimate_passthrough(df, cost_col, pump_col)`
- `ph_economic_ai/benchmark/efficiency.py` — `run_panel(frame, methods, min_train, feature_cols)`
- Tests: `test_forecasters.py`, `test_significance.py`, `test_passthrough.py`, `test_efficiency.py`

**Modify:**
- `ph_economic_ai/benchmark/report.py` — add `efficiency` + `passthrough` keys
- `ph_economic_ai/tests/test_report.py` — extend
- `ph_economic_ai/benchmark/figures.py` — add `plot_method_skill_bar`, `plot_passthrough`
- `ph_economic_ai/benchmark/run.py` — run panel + pass-through; write to report + figures
- `ph_economic_ai/ui/accuracy_view.py` — add `efficiency_summary`, `passthrough_summary` panels
- `ph_economic_ai/tests/test_accuracy_view.py` — extend
- `docs/superpowers/specs/2026-06-08-ph-economic-ai-efficiency-mechanism-design.md` — fill §8 with real numbers

---

## Task 1: statsmodels dependency + forecaster panel

**Files:**
- Create: `ph_economic_ai/benchmark/forecasters.py`
- Test: `ph_economic_ai/tests/test_forecasters.py`

- [ ] **Step 1: Install statsmodels**

Run: `python -m pip install statsmodels -q`
Then verify: `python -c "import statsmodels; print(statsmodels.__version__)"`
Expected: prints a version (e.g. 0.14.x).

- [ ] **Step 2: Write the failing test**

Create `ph_economic_ai/tests/test_forecasters.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.forecasters import make_forecaster, FORECASTERS


def _data(n=40):
    rng = np.random.default_rng(0)
    y = 50 + np.cumsum(rng.normal(0, 0.5, n))
    X = np.column_stack([y - 1, rng.normal(0, 1, n)])   # a lag-ish col + noise
    return X, y


def test_all_forecasters_return_finite_float():
    X, y = _data()
    xn = X[-1]
    for name in FORECASTERS:
        pred = make_forecaster(name)(X[:-1], y[:-1], xn)
        assert isinstance(pred, float) and np.isfinite(pred), name


def test_random_walk_returns_last():
    f = make_forecaster('random_walk')
    assert f(None, np.array([1.0, 2.0, 3.5]), None) == pytest.approx(3.5)


def test_drift_adds_mean_step():
    f = make_forecaster('drift')
    # y = 0,1,2,3 -> mean diff 1 -> next 4
    assert f(None, np.array([0.0, 1.0, 2.0, 3.0]), None) == pytest.approx(4.0)


def test_seasonal_naive_uses_season_lag():
    f = make_forecaster('seasonal_naive')
    y = np.arange(13, dtype=float)        # 0..12
    assert f(None, y, None) == pytest.approx(1.0)


def test_arima_falls_back_on_degenerate_series():
    f = make_forecaster('arima')
    y = np.zeros(5)                       # degenerate -> fallback to random walk
    assert f(None, y, None) == pytest.approx(0.0)
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_forecasters.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.forecasters`

- [ ] **Step 4: Implement**

Create `ph_economic_ai/benchmark/forecasters.py`:

```python
"""Standard forecasters for the efficiency panel. Each is a
predict_fn(X_train, y_train, x_next) -> float so it plugs into backtest.walk_forward.

Univariate methods ignore X. ARIMA/ETS fall back to random walk on fit failure.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def _random_walk(X_train, y_train, x_next) -> float:
    return float(y_train[-1])


def _drift(X_train, y_train, x_next) -> float:
    step = float(np.mean(np.diff(y_train))) if len(y_train) > 1 else 0.0
    return float(y_train[-1] + step)


def _seasonal_naive(X_train, y_train, x_next, season: int = 12) -> float:
    return float(y_train[-season]) if len(y_train) > season else float(y_train[-1])


def _ridge(X_train, y_train, x_next) -> float:
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    return float(model.predict(x_next.reshape(1, -1))[0])


def _hgb(X_train, y_train, x_next) -> float:
    model = HistGradientBoostingRegressor(
        random_state=42, min_samples_leaf=5, max_leaf_nodes=15).fit(X_train, y_train)
    return float(model.predict(x_next.reshape(1, -1))[0])


def _arima(X_train, y_train, x_next) -> float:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(np.asarray(y_train, dtype=float), order=(1, 1, 1)).fit()
        return float(np.asarray(fit.forecast(1)).ravel()[0])
    except Exception:
        return float(y_train[-1])


def _ets(X_train, y_train, x_next) -> float:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(np.asarray(y_train, dtype=float), trend='add').fit()
        return float(np.asarray(fit.forecast(1)).ravel()[0])
    except Exception:
        return float(y_train[-1])


FORECASTERS = {
    'random_walk':   _random_walk,
    'drift':         _drift,
    'seasonal_naive': _seasonal_naive,
    'arima':         _arima,
    'ets':           _ets,
    'ridge':         _ridge,
    'hgb':           _hgb,
}


def make_forecaster(name: str):
    return FORECASTERS[name]
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_forecasters.py -v`
Expected: PASS (5 passed). (ARIMA/ETS emit convergence warnings on small series — harmless.)

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/benchmark/forecasters.py ph_economic_ai/tests/test_forecasters.py
git commit -m "feat(benchmark): standard forecaster panel (RW/drift/seasonal/ARIMA/ETS/Ridge/HGB)"
```

---

## Task 2: Diebold-Mariano significance test

**Files:**
- Create: `ph_economic_ai/benchmark/significance.py`
- Test: `ph_economic_ai/tests/test_significance.py`

Sign convention: `diebold_mariano(loss_a, loss_b)` on per-step losses; `d = loss_a - loss_b`. Positive stat ⇒ a has higher loss (worse) than b. For the panel, `loss_a = method`, `loss_b = random_walk`: a significantly *negative* stat (p<0.05) means the method genuinely beats RW.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_significance.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.significance import diebold_mariano


def test_identical_losses_give_zero_stat_p_one():
    loss = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    r = diebold_mariano(loss, loss.copy())
    assert r['dm_stat'] == pytest.approx(0.0)
    assert r['p_value'] == pytest.approx(1.0)


def test_clearly_worse_a_gives_positive_stat_small_p():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.0, 200)
    loss_a = base + 1.0          # a uniformly worse
    loss_b = base
    r = diebold_mariano(loss_a, loss_b)
    assert r['dm_stat'] > 0
    assert r['p_value'] < 0.05


def test_antisymmetry_of_stat():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 100)
    b = rng.uniform(0, 1, 100)
    r1 = diebold_mariano(a, b)
    r2 = diebold_mariano(b, a)
    assert r1['dm_stat'] == pytest.approx(-r2['dm_stat'], rel=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_significance.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/significance.py`:

```python
"""Diebold-Mariano test of equal predictive accuracy, with the Harvey-Leybourne-
Newbold (1997) small-sample correction. Pure numpy/scipy.
"""
import numpy as np
from scipy import stats


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Compare two per-step loss series. d = loss_a - loss_b.

    Returns {'dm_stat', 'p_value'}. Positive stat => series A has higher loss
    (worse) than series B. p_value is two-sided (Student-t, df = n-1).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    d_bar = float(np.mean(d))
    # Long-run variance of d (autocovariances up to h-1; none for h=1)
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var_d = gamma0
    for k in range(1, h):
        cov_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_d += 2.0 * cov_k
    if n < 2 or var_d <= 0:
        return {'dm_stat': 0.0, 'p_value': 1.0}
    dm = d_bar / np.sqrt(var_d / n)
    # HLN small-sample correction factor
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm * corr)
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return {'dm_stat': dm_hln, 'p_value': p}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_significance.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/significance.py ph_economic_ai/tests/test_significance.py
git commit -m "feat(benchmark): Diebold-Mariano predictive-accuracy test (HLN-corrected)"
```

---

## Task 3: Pass-through regression

**Files:**
- Create: `ph_economic_ai/benchmark/passthrough.py`
- Test: `ph_economic_ai/tests/test_passthrough.py`

Regress `Δpump_t = α + β0·Δcost_t + β1·Δcost_{t-1} + ε` (HAC/Newey-West SE). `cost` = the RBOB→PHP landed proxy (`gas_price` column); `pump` = `ron95` gold. Also report lag-1 autocorrelation of `Δcost` (driver-RW check).

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_passthrough.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.passthrough import estimate_passthrough


def test_recovers_known_passthrough():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    cost = 50 + np.cumsum(rng.normal(0, 1.0, n))          # random-walk driver
    dcost = np.diff(cost, prepend=cost[0])
    pump = np.empty(n); pump[0] = 60.0
    for i in range(1, n):
        pump[i] = pump[i - 1] + 0.5 * dcost[i]            # contemporaneous beta = 0.5
    df = pd.DataFrame({'gas_price': cost, 'ron95': pump}, index=idx)
    r = estimate_passthrough(df)
    assert r['beta_total'] == pytest.approx(0.5, abs=0.05)
    assert r['r2'] > 0.95
    assert r['n'] > 100


def test_short_series_returns_none_coeffs():
    df = pd.DataFrame({'gas_price': [1.0, 2.0, 3.0], 'ron95': [1.0, 2.0, 3.0]},
                      index=['2020-01', '2020-02', '2020-03'])
    r = estimate_passthrough(df)
    assert r['beta_total'] is None
    assert r['n'] < 10
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_passthrough.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/passthrough.py`:

```python
"""Measure the DOE pass-through: how a change in landed cost (RBOB->PHP proxy)
flows into the retail pump price, and confirm the driver is itself ~random walk.
"""
import numpy as np
import pandas as pd


def estimate_passthrough(df: pd.DataFrame, cost_col: str = 'gas_price',
                         pump_col: str = 'ron95') -> dict:
    """OLS Δpump_t = α + β0·Δcost_t + β1·Δcost_{t-1} + ε with HAC (Newey-West) SE.

    Returns alpha, beta0, beta1, beta_total, r2, driver_acf1, n. Coeffs are None
    if fewer than 10 usable rows.
    """
    import statsmodels.api as sm

    d = df[[cost_col, pump_col]].dropna().sort_index()
    dpump = d[pump_col].diff()
    dcost = d[cost_col].diff()
    reg = pd.DataFrame({
        'dpump': dpump,
        'dcost': dcost,
        'dcost1': dcost.shift(1),
    }).dropna()

    if len(reg) < 10:
        return {'n': int(len(reg)), 'alpha': None, 'beta0': None, 'beta1': None,
                'beta_total': None, 'r2': None, 'driver_acf1': None}

    X = sm.add_constant(reg[['dcost', 'dcost1']])
    model = sm.OLS(reg['dpump'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    b0 = float(model.params['dcost'])
    b1 = float(model.params['dcost1'])
    driver_acf1 = float(pd.Series(dcost.dropna().to_numpy()).autocorr(lag=1))
    return {
        'n': int(len(reg)),
        'alpha': round(float(model.params['const']), 4),
        'beta0': round(b0, 4),
        'beta1': round(b1, 4),
        'beta_total': round(b0 + b1, 4),
        'r2': round(float(model.rsquared), 4),
        'driver_acf1': round(driver_acf1, 4),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_passthrough.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/passthrough.py ph_economic_ai/tests/test_passthrough.py
git commit -m "feat(benchmark): DOE pass-through regression (elasticity + lag + driver RW check)"
```

---

## Task 4: Efficiency panel runner

**Files:**
- Create: `ph_economic_ai/benchmark/efficiency.py`
- Test: `ph_economic_ai/tests/test_efficiency.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_efficiency.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.features import build_feature_frame, VARIANTS
from ph_economic_ai.benchmark.efficiency import run_panel


def _frame(n=70):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    gas = 50 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)
    return build_feature_frame(df)


def test_panel_one_row_per_method_with_keys():
    methods = ['random_walk', 'drift', 'ridge', 'hgb']
    rows = run_panel(_frame(), methods, min_train=24,
                     feature_cols=VARIANTS['passthrough_lags']['cols'])
    assert [r['method'] for r in rows] == methods
    for r in rows:
        assert set(r) >= {'method', 'rmse', 'mae', 'skill_vs_rw', 'dm_stat', 'dm_p', 'n'}


def test_random_walk_row_has_zero_skill_and_null_dm():
    rows = run_panel(_frame(), ['random_walk', 'ridge'], min_train=24,
                     feature_cols=VARIANTS['passthrough_lags']['cols'])
    rw = next(r for r in rows if r['method'] == 'random_walk')
    assert rw['skill_vs_rw'] == 0.0
    assert rw['dm_p'] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_efficiency.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/efficiency.py`:

```python
"""Run a panel of forecasters through the same causal backtest and attach a
Diebold-Mariano p-value vs random walk. The efficiency result: no method's
accuracy is statistically better than naive persistence (DM p > 0.05).
"""
from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.forecasters import make_forecaster
from ph_economic_ai.benchmark.metrics import mae, rmse, skill_score
from ph_economic_ai.benchmark.significance import diebold_mariano


def run_panel(frame, methods, min_train: int, feature_cols) -> list:
    """frame: build_feature_frame output. Returns one row per method:
    {method, rmse, mae, skill_vs_rw, dm_stat, dm_p, n}. Reference = random walk."""
    y = frame['ron95'].to_numpy(dtype=float)
    X = frame[list(feature_cols)].to_numpy(dtype=float)

    rw_bt = walk_forward(y, None, make_forecaster('random_walk'), min_train)
    rw_loss = (rw_bt['y_true'] - rw_bt['y_pred']) ** 2
    rmse_rw = rmse(rw_bt['y_true'], rw_bt['y_pred'])

    rows = []
    for m in methods:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        loss = (bt['y_true'] - bt['y_pred']) ** 2
        r = rmse(bt['y_true'], bt['y_pred'])
        if m == 'random_walk':
            dm_stat, dm_p = 0.0, None
        else:
            dm = diebold_mariano(loss, rw_loss, h=1)
            dm_stat, dm_p = round(dm['dm_stat'], 4), round(dm['p_value'], 4)
        rows.append({
            'method': m,
            'rmse': round(r, 4),
            'mae': round(mae(bt['y_true'], bt['y_pred']), 4),
            'skill_vs_rw': round(skill_score(r, rmse_rw), 4),
            'dm_stat': dm_stat,
            'dm_p': dm_p,
            'n': int(len(bt['y_true'])),
        })
    return rows
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_efficiency.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/efficiency.py ph_economic_ai/tests/test_efficiency.py
git commit -m "feat(benchmark): efficiency panel runner with DM p-values vs random walk"
```

---

## Task 5: Report keys for efficiency + passthrough

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py`

- [ ] **Step 1: Add the test**

Append to `ph_economic_ai/tests/test_report.py`:

```python
def test_report_includes_efficiency_and_passthrough():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        efficiency=[{'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None, 'rmse': 1.9, 'mae': 1.5, 'dm_stat': 0.0, 'n': 79},
                    {'method': 'hgb', 'skill_vs_rw': -0.18, 'dm_p': 0.21, 'rmse': 2.2, 'mae': 1.8, 'dm_stat': 1.2, 'n': 79}],
        passthrough={'beta_total': 0.83, 'beta0': 0.6, 'beta1': 0.23, 'r2': 0.74, 'driver_acf1': 0.03, 'n': 96, 'alpha': 0.1},
    )
    assert len(rep['efficiency']) == 2
    assert rep['passthrough']['beta_total'] == 0.83
    assert 'efficiency' in REQUIRED_KEYS and 'passthrough' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_efficiency_and_passthrough -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'efficiency'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'efficiency'` and `'passthrough'` to the `REQUIRED_KEYS` tuple.
(b) Add two params to `build_report` (after `selected_variant=None`): `efficiency=None, passthrough=None`. In the returned dict, add (just before `'limitations'`):

```python
        'efficiency': efficiency if efficiency is not None else [],
        'passthrough': passthrough if passthrough is not None else {},
```

The full new signature line:
```python
def build_report(date_range, n_months, model_metrics, baseline_metrics, skill,
                 calibration, proxy, data_hash, ablation=None, selected_variant=None,
                 efficiency=None, passthrough=None) -> dict:
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all — existing tests still pass because new args default to None/[]/{}).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries efficiency panel + pass-through stats"
```

---

## Task 6: Wire panel + pass-through into run.py (+ figures)

**Files:**
- Modify: `ph_economic_ai/benchmark/figures.py`
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: manual end-to-end run on real data

- [ ] **Step 1: Add figure functions**

Append to `ph_economic_ai/benchmark/figures.py`:

```python
def plot_method_skill_bar(rows):
    """Bar of skill-vs-random-walk per method; red where DM p<0.05 (sig. different)."""
    _ensure_dir()
    rows = sorted(rows, key=lambda r: r['skill_vs_rw'])
    names = [r['method'] for r in rows]
    skills = [r['skill_vs_rw'] for r in rows]
    colors = ['tab:red' if (r['dm_p'] is not None and r['dm_p'] < 0.05) else 'tab:gray'
              for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, skills, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Skill vs random walk (>0 beats naive)')
    ax.set_title('Forecaster panel — none beats random walk')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'method_skill_bar.png', dpi=120); plt.close(fig)


def plot_passthrough(cost_delta, pump_delta, beta_total):
    """Scatter of Δpump vs Δcost with the fitted pass-through slope."""
    _ensure_dir()
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cost_delta, pump_delta, s=12, alpha=0.6)
    xs = np.array([min(cost_delta), max(cost_delta)])
    ax.plot(xs, beta_total * xs, color='black', linewidth=1,
            label=f'pass-through β={beta_total:.2f}')
    ax.set_xlabel('Δ landed cost (₱/L)'); ax.set_ylabel('Δ pump price (₱/L)')
    ax.legend(); ax.set_title('DOE pass-through')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'passthrough.png', dpi=120); plt.close(fig)
```

- [ ] **Step 2: Wire into run.py**

In `ph_economic_ai/benchmark/run.py`:
(a) Add imports near the other benchmark imports:
```python
from ph_economic_ai.benchmark import efficiency as efficiency_mod
from ph_economic_ai.benchmark import passthrough as passthrough_mod
```
(b) After the ablation block (right after `selected = winner['name']` and its print loop, before the "Re-run the winning variant" comment), add:
```python
    # ── Efficiency panel + pass-through mechanism ────────────────────────────
    panel_methods = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']
    efficiency_rows = efficiency_mod.run_panel(
        frame, panel_methods, MIN_TRAIN, VARIANTS['passthrough_lags']['cols'])
    passthrough_stats = passthrough_mod.estimate_passthrough(df)
    print('Efficiency panel (skill vs RW | DM p):')
    for r in sorted(efficiency_rows, key=lambda x: -x['skill_vs_rw']):
        p = 'n/a' if r['dm_p'] is None else f"{r['dm_p']:.3f}"
        print(f"  {r['method']:<14} skill={r['skill_vs_rw']:+.3f}  DM p={p}")
    print(f"Pass-through: beta_total={passthrough_stats['beta_total']} "
          f"R2={passthrough_stats['r2']} driver_acf1={passthrough_stats['driver_acf1']}")
```
(c) In the `report.build_report(` call, add:
```python
        efficiency=efficiency_rows, passthrough=passthrough_stats,
```
(d) After the existing figure calls, add the two new figures:
```python
    figures.plot_method_skill_bar(efficiency_rows)
    _dc = df['gas_price'].diff().dropna()
    _dp = df['ron95'].diff().reindex(_dc.index)
    _mask = _dp.notna()
    if passthrough_stats['beta_total'] is not None:
        figures.plot_passthrough(_dc[_mask].to_numpy(), _dp[_mask].to_numpy(),
                                 passthrough_stats['beta_total'])
```

- [ ] **Step 3: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints the ablation table, the efficiency panel (one line per method with DM p), the pass-through line, and writes all artifacts without error.

- [ ] **Step 4: Record the real numbers**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; r=load_report(); print('PASSTHROUGH', r['passthrough']); [print(x['method'], x['skill_vs_rw'], x['dm_p']) for x in r['efficiency']]"`
Record the output verbatim — these are the efficiency + mechanism results for the spec/write-up.

- [ ] **Step 5: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window::test_on_run_requested_accepts_4_args`.

- [ ] **Step 6: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/figures.py ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/backtest_predictions.csv ph_economic_ai/benchmark/artifacts/figures/method_skill_bar.png ph_economic_ai/benchmark/artifacts/figures/passthrough.png ph_economic_ai/benchmark/artifacts/figures/baseline_bars.png ph_economic_ai/benchmark/artifacts/figures/pred_vs_actual.png ph_economic_ai/benchmark/artifacts/figures/proxy_scatter.png ph_economic_ai/benchmark/artifacts/ablation_table.json
git status --short
git commit -m "feat(benchmark): run efficiency panel + pass-through; emit figures + report"
```

---

## Task 7: Surface efficiency + pass-through in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'efficiency': [
            {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None, 'rmse': 4.0, 'mae': 3.1, 'dm_stat': 0.0, 'n': 79},
            {'method': 'hgb', 'skill_vs_rw': -0.18, 'dm_p': 0.21, 'rmse': 4.7, 'mae': 3.5, 'dm_stat': 1.2, 'n': 79},
        ],
        'passthrough': {'beta_total': 0.83, 'beta0': 0.6, 'beta1': 0.23, 'r2': 0.74, 'driver_acf1': 0.03, 'n': 96, 'alpha': 0.1},
```
Add this test:
```python
def test_view_shows_efficiency_and_passthrough(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    eff = view.efficiency_summary()
    pt = view.passthrough_summary()
    assert 'random_walk' in eff and 'hgb' in eff
    assert 'pass-through' in pt.lower() or 'β' in pt or 'beta' in pt.lower()
    assert '0.83' in pt
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_efficiency_and_passthrough -v`
Expected: FAIL — `AttributeError: ... has no attribute 'efficiency_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add two methods to `AccuracyView`:
```python
    def efficiency_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('efficiency') or []
        lines = []
        for r in sorted(rows, key=lambda x: -x['skill_vs_rw']):
            p = 'n/a' if r.get('dm_p') is None else f"p={r['dm_p']:.2f}"
            lines.append(f"{r['method']}: skill {r['skill_vs_rw']:+.2f} vs RW ({p})")
        return '\n'.join(lines)

    def passthrough_summary(self) -> str:
        if not self._report:
            return ''
        p = self._report.get('passthrough') or {}
        if not p or p.get('beta_total') is None:
            return ''
        return (f"DOE pass-through: total β={p['beta_total']:.2f} "
                f"(contemporaneous {p['beta0']:.2f}, lag-1 {p['beta1']:.2f}), "
                f"R²={p['r2']:.2f}; driver Δ-autocorrelation={p['driver_acf1']:.2f} "
                f"(≈0 ⇒ random-walk input).")
```
(b) In `_build`, inside `if self._report is not None:`, after the ablation widget block, add:
```python
            if self._report.get('efficiency'):
                eff = QLabel('<b>Forecaster panel — efficiency (Phase: contribution)</b><br>'
                             + self.efficiency_summary().replace('\n', '<br>'))
                eff.setWordWrap(True)
                col.addWidget(eff)
            pt = self.passthrough_summary()
            if pt:
                ptl = QLabel('<b>Mechanism</b><br>' + pt)
                ptl.setWordWrap(True)
                col.addWidget(ptl)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py -v`
Expected: PASS (4 passed — 3 existing + new)

- [ ] **Step 5: Smoke-test the window**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: only the pre-existing `test_on_run_requested_accepts_4_args` may fail; nothing new.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/accuracy_view.py ph_economic_ai/tests/test_accuracy_view.py
git commit -m "feat(ui): show efficiency panel + pass-through mechanism in Accuracy view"
```

---

## Task 8: Record the measured contribution in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-ph-economic-ai-efficiency-mechanism-design.md`

- [ ] **Step 1: Fill §8 with the real numbers from Task 6 Step 4**

Replace the `<measured>` placeholders in §8 ("The contribution, stated for the eventual write-up") with the actual values from the run: the list of methods with their skill vs RW and DM p-values, the measured `beta_total` / `beta0` / `beta1` / `r2`, and `driver_acf1`. State plainly whether the efficiency claim holds (all DM p > 0.05 ⇒ no method significantly beats RW) and one sentence connecting the measured pass-through to the efficiency result. Do not invent numbers — copy them from `accuracy_report.json`.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-08-ph-economic-ai-efficiency-mechanism-design.md
git commit -m "docs: record measured efficiency + pass-through results in spec section 8"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Efficiency claim is legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; r=load_report(); beats=[x['method'] for x in r['efficiency'] if x['dm_p'] is not None and x['dm_p']<0.05 and x['skill_vs_rw']>0]; print('methods significantly beating RW:', beats or 'NONE')"`
Expected: prints `NONE` if the efficiency claim holds (the honest expected outcome), or names any method that genuinely beats RW — either way, reported truthfully.

---

## Self-Review (completed by plan author)

**Spec coverage:**
- §3.1 forecaster panel → Task 1. §3.2 Diebold-Mariano → Task 2. §3.3 pass-through regression → Task 3. §3.4 panel runner → Task 4. Report integration (§3 data flow) → Task 5. run.py + figures (§4) → Task 6. Accuracy view (§7 deliverable 4) → Task 7. Honest written outcome (§7 deliverable 3, §8) → Task 8. Testing (§6) → tests in Tasks 1-5, 7. New statsmodels dep (§2) → Task 1 Step 1.
- **Deviation from spec (noted):** the spec listed `structural_passthrough` inside the efficiency panel. The plan EXCLUDES it from the classical panel because structural reconstruction needs the variant machinery (already evaluated in Phase-2 `ablation_table.json`, where it scored worst). The panel covers the classical methods (RW/drift/seasonal/ARIMA/ETS/Ridge/HGB); the write-up (Task 8) should cite the structural result from the Phase-2 ablation alongside the panel. This keeps the panel runner uniform and is a faithful, clearly-documented simplification.

**Placeholder scan:** none in code steps. §8's `<measured>` is intentionally filled at Task 8 from real output (empirical; cannot be known at plan time) — same pattern used and accepted in Phase 1/2.

**Type consistency:** forecaster `predict_fn(X_train, y_train, x_next) -> float` matches `walk_forward`'s expected callable across Tasks 1, 4. `diebold_mariano(loss_a, loss_b, h=1) -> {dm_stat, p_value}` consistent (Tasks 2, 4). Efficiency row keys `{method, rmse, mae, skill_vs_rw, dm_stat, dm_p, n}` consistent across Tasks 4, 5, 6, 7. `estimate_passthrough(df) -> {n, alpha, beta0, beta1, beta_total, r2, driver_acf1}` consistent across Tasks 3, 5, 6, 7. `build_report(... efficiency=, passthrough=)` matches between Task 5 (def) and Task 6 (call). Figure fns `plot_method_skill_bar(rows)`, `plot_passthrough(cost_delta, pump_delta, beta_total)` consistent (Task 6).
```
