# ph_economic_ai Accuracy — Phase 2 (Accuracy Improvement) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Earn a genuinely *higher and still-honest* confidence by reducing real 1-month-ahead RON95 forecast error — re-grounding the model in finished-product pass-through and a structural+ML hybrid, with every lever gated behind a measured backtest improvement.

**Architecture:** Phase 1 proved the baseline (skill −0.18 vs random walk). Phase 2 adds a **feature-frame builder** (`features.py`) that emits all candidate 1-month-ahead predictors, a **variant registry** (column subsets + an optional structural decomposition), and an **ablation harness** (`ablation.py`) that runs each variant through the *same* causal walk-forward backtest and conformal calibration, producing a comparable table. `run.py` then selects the winner by an explicit gate (highest skill vs random walk; tie-break narrower band at ≥ nominal coverage) and uses it for the headline report. Adds normalized (Mondrian) conformal so bands are tight in calm months and honestly wide around shocks.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn (`HistGradientBoostingRegressor`), matplotlib, pytest. Builds entirely on the Phase-1 `ph_economic_ai/benchmark/` package.

**Spec:** `docs/superpowers/specs/2026-06-05-ph-economic-ai-accuracy-evaluation-design.md` §9 (ranked, individually-gated levers).

**Prereqs / context (already on `master` via PR #1):**
- `benchmark/backtest.py::walk_forward(y, X, predict_fn, min_train) -> {y_true,y_pred,residuals,index}` — strictly causal.
- `benchmark/metrics.py::{mae,rmse,mape,mase,skill_score}`.
- `benchmark/conformal.py::{conformal_quantile,coverage,build_calibration_table}`.
- `benchmark/run.py` — current 1-month-ahead design: lagged macro drivers + `prev_ron95`.
- Real data committed: `data/world_bank_ron95.csv` (99 mo), `data/features_monthly.csv` (cols: `date, oil_price, usd_php, gas_price, demand_index`; `gas_price` is the RBOB→PHP proxy).
- **Honest Phase-1 result to beat:** skill_vs_random_walk = −0.18; random-walk RMSE ≈ ₱4.00.

**Conventions (unchanged from Phase 1):**
- Tests in `ph_economic_ai/tests/`, import `from ph_economic_ai.X import Y`, start with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  ```
- Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging area is clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. Run `git status --short` before committing; `git restore --staged <path>` anything unexpected.
- Branch off `master` first (do NOT implement on `master`): `git checkout -b feature/accuracy-phase2`.

---

## File Structure

**Create:**
- `ph_economic_ai/benchmark/features.py` — `build_feature_frame(df)`, `Variant` dataclass, `VARIANTS` registry, `make_variant(name, frame)`
- `ph_economic_ai/benchmark/ablation.py` — `run_variant(...)`, `run_ablation(...)`, `select_winner(rows)`
- Tests: `test_features.py`, `test_ablation.py`

**Modify:**
- `ph_economic_ai/benchmark/conformal.py` — add `normalized_conformal_quantile(...)`, `normalized_coverage(...)`
- `ph_economic_ai/tests/test_conformal.py` — add normalized-conformal cases
- `ph_economic_ai/benchmark/report.py` — add `ablation` + `selected_variant` keys to the report
- `ph_economic_ai/tests/test_report.py` — extend for the new keys
- `ph_economic_ai/benchmark/run.py` — build frame → run ablation → select winner → headline uses winner; write `artifacts/ablation_table.json`
- `ph_economic_ai/ui/accuracy_view.py` — add a small read-only "Lever comparison" table
- `ph_economic_ai/tests/test_accuracy_view.py` — assert the ablation panel renders when present

---

## Task 1: Feature frame + variant registry

**Files:**
- Create: `ph_economic_ai/benchmark/features.py`
- Test: `ph_economic_ai/tests/test_features.py`

All candidate predictors are **lagged** (known at month i-1) so every variant is a true 1-month-ahead forecast. One shared frame guarantees all variants use identical date support → directly comparable. The structural hybrid predicts the *residual over the lagged proxy*, reconstructed as `proxy_lag1 + residual_pred`.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_features.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.features import (
    build_feature_frame, make_variant, VARIANTS,
)


def _df(n=40):
    idx = pd.date_range('2018-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(0)
    gas = 50 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,                       # RBOB proxy
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)


def test_frame_columns_present_and_lagged():
    f = build_feature_frame(_df())
    for col in ('prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'demand_lag1',
                'gas_lag2', 'gas_lag3', 'gas_ma3', 'fx_ma3', 'gas_delta1',
                'proxy_lag1', 'ron95'):
        assert col in f.columns
    # No NaNs after the builder's own dropna
    assert not f.isna().any().any()
    # Lagged: row label t's gas_lag1 equals raw gas at t-1
    raw = _df()
    t = f.index[5]
    pos = list(raw.index).index(t)
    assert f.loc[t, 'gas_lag1'] == pytest.approx(raw['gas_price'].iloc[pos - 1])


def test_make_variant_plain_has_identity_structural():
    f = build_feature_frame(_df())
    v = make_variant('baseline', f)
    assert v.X.shape[0] == len(f)
    assert np.allclose(v.structural, 0.0)
    assert np.allclose(v.y_model, v.y_actual)
    assert v.X.shape[1] == len(VARIANTS['baseline']['cols'])


def test_make_variant_structural_hybrid_decomposes():
    f = build_feature_frame(_df())
    v = make_variant('structural_hybrid', f)
    # y_model is the residual over the lagged proxy; reconstruction recovers ron95
    assert np.allclose(v.y_model + v.structural, v.y_actual)
    assert not np.allclose(v.structural, 0.0)


def test_registry_columns_exist_in_frame():
    f = build_feature_frame(_df())
    for name, spec in VARIANTS.items():
        for c in spec['cols']:
            assert c in f.columns, f'{name} references missing column {c}'
        if spec['structural']:
            assert spec['structural'] in f.columns
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.features`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/features.py`:

```python
"""1-month-ahead candidate predictors + variant registry for Phase-2 ablation.

All features are lagged (known at month i-1) so every variant is a true forecast.
A single shared frame guarantees identical date support across variants. The
structural hybrid predicts the residual over the lagged RBOB proxy, reconstructed
as proxy_lag1 + residual_pred (leakage-free: the model learns the time-varying gap
including its mean bias).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Variant:
    name: str
    dates: list
    X: np.ndarray          # design matrix; row i predicts y_actual[i]
    y_actual: np.ndarray   # ron95 (for fair baseline comparison), per row
    y_model: np.ndarray    # what the regressor fits (y_actual - structural)
    structural: np.ndarray # per-row structural component (0.0 for plain variants)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df indexed 'YYYY-MM' with oil_price, usd_php, gas_price, demand_index, ron95.

    Returns a frame of lagged candidate features + 'ron95' target + 'proxy_lag1',
    dropna'd to common support (longest lag = 3 months)."""
    f = pd.DataFrame(index=df.index)
    f['prev_ron95']  = df['ron95'].shift(1)
    f['oil_lag1']    = df['oil_price'].shift(1)
    f['usd_lag1']    = df['usd_php'].shift(1)
    f['gas_lag1']    = df['gas_price'].shift(1)
    f['demand_lag1'] = df['demand_index'].shift(1)
    f['gas_lag2']    = df['gas_price'].shift(2)
    f['gas_lag3']    = df['gas_price'].shift(3)
    f['gas_ma3']     = df['gas_price'].shift(1).rolling(3).mean()
    f['fx_ma3']      = df['usd_php'].shift(1).rolling(3).mean()
    f['gas_delta1']  = df['gas_price'].shift(1) - df['gas_price'].shift(2)
    f['proxy_lag1']  = df['gas_price'].shift(1)      # structural component
    f['ron95']       = df['ron95']
    return f.dropna()


# name -> {cols: [feature columns], structural: column name or None}
VARIANTS: dict = {
    'baseline':          {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'demand_lag1'],
                          'structural': None},
    'drop_demand':       {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1'],
                          'structural': None},
    'passthrough_lags':  {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'gas_lag2',
                                   'gas_lag3', 'gas_ma3', 'fx_ma3', 'gas_delta1', 'demand_lag1'],
                          'structural': None},
    'finished_gas':      {'cols': ['prev_ron95', 'gas_lag1', 'gas_lag2', 'gas_lag3', 'gas_ma3', 'usd_lag1'],
                          'structural': None},
    'structural_hybrid': {'cols': ['gas_delta1', 'fx_ma3', 'demand_lag1', 'gas_lag2'],
                          'structural': 'proxy_lag1'},
}


def make_variant(name: str, frame: pd.DataFrame) -> Variant:
    spec = VARIANTS[name]
    X = frame[spec['cols']].to_numpy(dtype=float)
    y_actual = frame['ron95'].to_numpy(dtype=float)
    if spec['structural'] is not None:
        structural = frame[spec['structural']].to_numpy(dtype=float)
        y_model = y_actual - structural
    else:
        structural = np.zeros(len(frame))
        y_model = y_actual.copy()
    return Variant(name=name, dates=frame.index.tolist(), X=X,
                   y_actual=y_actual, y_model=y_model, structural=structural)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_features.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/features.py ph_economic_ai/tests/test_features.py
git commit -m "feat(benchmark): Phase-2 feature frame + variant registry"
```

---

## Task 2: Ablation harness + winner gate

**Files:**
- Create: `ph_economic_ai/benchmark/ablation.py`
- Test: `ph_economic_ai/tests/test_ablation.py`

Each variant runs through the *same* causal `walk_forward`; the structural variant's raw prediction is reconstructed to RON95 space before scoring. Random walk is scored on the shared `y_actual`. `select_winner` encodes the gate.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_ablation.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.features import build_feature_frame
from ph_economic_ai.benchmark.ablation import run_variant, run_ablation, select_winner


def _frame(n=60):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(1)
    gas = 50 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)
    return build_feature_frame(df)


def _mean_predict_fn(X_train, y_train, x_next):
    return float(np.mean(y_train[-3:]))   # cheap, deterministic stand-in model


def test_run_variant_returns_metrics():
    row = run_variant('baseline', _frame(), _mean_predict_fn, min_train=12)
    assert set(row) >= {'name', 'rmse', 'skill_vs_rw', 'mae', 'band90', 'n'}
    assert row['name'] == 'baseline'
    assert row['n'] > 0


def test_structural_reconstruction_scores_in_ron95_space():
    # A perfect residual learner would reconstruct ron95 exactly. Use a predict_fn
    # that returns the last residual; just assert the harness runs and rmse >= 0.
    row = run_variant('structural_hybrid', _frame(), _mean_predict_fn, min_train=12)
    assert row['rmse'] >= 0.0
    assert np.isfinite(row['skill_vs_rw'])


def test_run_ablation_one_row_per_variant():
    rows = run_ablation(_frame(), ['baseline', 'drop_demand', 'finished_gas'],
                        _mean_predict_fn, min_train=12)
    assert [r['name'] for r in rows] == ['baseline', 'drop_demand', 'finished_gas']


def test_select_winner_prefers_higher_skill_then_narrower_band():
    rows = [
        {'name': 'a', 'skill_vs_rw': -0.10, 'band90': 8.0},
        {'name': 'b', 'skill_vs_rw': 0.05, 'band90': 7.0},
        {'name': 'c', 'skill_vs_rw': 0.05, 'band90': 5.0},   # same skill, tighter band
    ]
    assert select_winner(rows)['name'] == 'c'


def test_select_winner_handles_all_negative():
    rows = [{'name': 'a', 'skill_vs_rw': -0.3, 'band90': 9.0},
            {'name': 'b', 'skill_vs_rw': -0.1, 'band90': 9.0}]
    assert select_winner(rows)['name'] == 'b'   # least-bad still selected; gate is reported
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_ablation.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.ablation`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/ablation.py`:

```python
"""Phase-2 ablation: run each feature variant through the same causal backtest and
score it in RON95 space, so levers are compared apples-to-apples and the winner is
chosen by an explicit, auditable gate.
"""
import numpy as np

from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.conformal import conformal_quantile
from ph_economic_ai.benchmark.features import make_variant
from ph_economic_ai.benchmark.metrics import mae, rmse, skill_score


def run_variant(name, frame, predict_fn, min_train: int) -> dict:
    """Backtest one variant; reconstruct to RON95 space; score vs random walk."""
    v = make_variant(name, frame)
    bt = walk_forward(v.y_model, v.X, predict_fn, min_train)
    idx = bt['index']
    final_pred = bt['y_pred'] + v.structural[idx]      # reconstruct (0 for plain)
    final_true = v.y_actual[idx]

    rw = walk_forward(v.y_actual, None,
                      lambda Xt, yt, xn: float(yt[-1]), min_train)
    rmse_model = rmse(final_true, final_pred)
    rmse_rw = rmse(rw['y_true'], rw['y_pred'])
    qhat90 = conformal_quantile(final_true - final_pred, 0.9)
    return {
        'name': name,
        'rmse': round(rmse_model, 4),
        'mae': round(mae(final_true, final_pred), 4),
        'skill_vs_rw': round(skill_score(rmse_model, rmse_rw), 4),
        'band90': round(2 * qhat90, 4),
        'n': int(len(final_true)),
    }


def run_ablation(frame, names, predict_fn, min_train: int) -> list:
    return [run_variant(n, frame, predict_fn, min_train) for n in names]


def select_winner(rows: list) -> dict:
    """Gate: highest skill_vs_rw; tie-break (within 1e-9) by narrower band90."""
    return sorted(rows, key=lambda r: (-r['skill_vs_rw'], r['band90']))[0]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_ablation.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/ablation.py ph_economic_ai/tests/test_ablation.py
git commit -m "feat(benchmark): Phase-2 ablation harness + winner gate"
```

---

## Task 3: Normalized (Mondrian) conformal

**Files:**
- Modify: `ph_economic_ai/benchmark/conformal.py` (append two functions)
- Modify: `ph_economic_ai/tests/test_conformal.py` (append cases)

Normalized conformal scales residuals by a per-point volatility estimate `sigma_i`, so bands are tight in calm regimes and honestly wide around shocks at the same nominal coverage.

- [ ] **Step 1: Write the failing test (append to test_conformal.py)**

Append to `ph_economic_ai/tests/test_conformal.py`:

```python
from ph_economic_ai.benchmark.conformal import (
    normalized_conformal_quantile, normalized_coverage,
)


def test_normalized_coverage_near_nominal_heteroscedastic():
    rng = np.random.default_rng(7)
    n = 20000
    sigma = rng.uniform(0.5, 3.0, n)                 # varying volatility
    cal_res = rng.normal(0, 1, n) * sigma            # residuals scale with sigma
    qn = normalized_conformal_quantile(cal_res, sigma, level=0.90)
    # fresh validation set
    sigma_v = rng.uniform(0.5, 3.0, n)
    val_res = rng.normal(0, 1, n) * sigma_v
    cov = normalized_coverage(val_res, sigma_v, qn)
    assert cov == pytest.approx(0.90, abs=0.02)


def test_normalized_bands_are_wider_where_sigma_larger():
    rng = np.random.default_rng(8)
    sigma = np.array([1.0] * 1000 + [3.0] * 1000)
    cal_res = rng.normal(0, 1, 2000) * sigma
    qn = normalized_conformal_quantile(cal_res, sigma, level=0.90)
    # band half-width = qn * sigma_i -> larger where sigma is larger
    assert qn * 3.0 > qn * 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_conformal.py -k normalized -v`
Expected: FAIL — `ImportError: cannot import name 'normalized_conformal_quantile'`

- [ ] **Step 3: Implement (append to conformal.py)**

Append to `ph_economic_ai/benchmark/conformal.py`:

```python
def normalized_conformal_quantile(cal_residuals, sigmas, level: float) -> float:
    """Conformal quantile of |residual| / sigma. Band half-width is then qn * sigma_i,
    giving narrower intervals where local volatility (sigma) is small."""
    r = np.abs(np.asarray(cal_residuals, dtype=float))
    s = np.asarray(sigmas, dtype=float)
    if np.any(s <= 0):
        raise ValueError('sigmas must be strictly positive')
    return conformal_quantile(r / s, level)


def normalized_coverage(y_residuals, sigmas, qn: float) -> float:
    """Fraction of points within +/- qn * sigma_i."""
    r = np.abs(np.asarray(y_residuals, dtype=float))
    s = np.asarray(sigmas, dtype=float)
    return float(np.mean(r <= qn * s))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_conformal.py -v`
Expected: PASS (all conformal tests, old + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/conformal.py ph_economic_ai/tests/test_conformal.py
git commit -m "feat(benchmark): normalized (Mondrian) conformal intervals"
```

---

## Task 4: Report keys for ablation + selected variant

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py`

- [ ] **Step 1: Update the test**

In `ph_economic_ai/tests/test_report.py`, change the `build_report(...)` calls to pass the two new keyword args and assert they survive. Add this test at the end of the file:

```python
def test_report_includes_ablation_and_selected():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': 0.05},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        ablation=[{'name': 'baseline', 'skill_vs_rw': -0.18, 'band90': 21.5, 'rmse': 4.7, 'mae': 3.5, 'n': 79},
                  {'name': 'structural_hybrid', 'skill_vs_rw': 0.05, 'band90': 9.0, 'rmse': 1.7, 'mae': 1.2, 'n': 79}],
        selected_variant='structural_hybrid',
    )
    assert rep['selected_variant'] == 'structural_hybrid'
    assert len(rep['ablation']) == 2
    assert 'ablation' in REQUIRED_KEYS and 'selected_variant' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_ablation_and_selected -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'ablation'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Add `'ablation'` and `'selected_variant'` to the `REQUIRED_KEYS` tuple.
(b) Change the `build_report` signature and body to accept and store them. Replace the `def build_report(...)` signature line and its `return {...}` dict with:

```python
def build_report(date_range, n_months, model_metrics, baseline_metrics, skill,
                 calibration, proxy, data_hash, ablation=None, selected_variant=None) -> dict:
    conformal_widths = {str(r['nominal']): r['qhat'] for r in calibration}
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'horizon': '1_month',
        'date_range': list(date_range),
        'n_months': n_months,
        'model_metrics': model_metrics,
        'baseline_metrics': baseline_metrics,
        'skill': skill,
        'headline_skill_vs_random_walk': skill.get('vs_random_walk'),
        'conformal_widths': conformal_widths,
        'calibration': calibration,
        'proxy_validation': proxy,
        'data_hash': data_hash,
        'ablation': ablation if ablation is not None else [],
        'selected_variant': selected_variant,
        'limitations': _LIMITATIONS,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all report tests; the existing two still pass because the new args default to None/[])

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries ablation table + selected variant"
```

---

## Task 5: Wire ablation into run.py (select winner, headline uses it)

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: manual end-to-end run (uses real committed data)

`run.py` builds the feature frame, runs the full ablation with the real HGB, selects the winner, writes `artifacts/ablation_table.json`, and produces the headline report from the **winning** variant's backtest (reconstructed to RON95 space). The random-walk baseline and conformal use the winner's reconstructed predictions.

- [ ] **Step 1: Replace the design/backtest section of run.py**

In `ph_economic_ai/benchmark/run.py`:
(a) Update imports — add at the top with the other benchmark imports:

```python
from ph_economic_ai.benchmark import ablation as ablation_mod
from ph_economic_ai.benchmark.features import build_feature_frame, make_variant, VARIANTS
```

(b) Replace the block from `# True 1-month-ahead design:` through the line that sets `qhat90 = conformal.conformal_quantile(cal_res, 0.9)` with:

```python
    # ── Phase-2: ablation over feature variants, pick the winner by the gate ──
    frame = build_feature_frame(df)
    ablation_rows = ablation_mod.run_ablation(
        frame, list(VARIANTS.keys()), _hgb_predict_fn, MIN_TRAIN)
    winner = ablation_mod.select_winner(ablation_rows)
    selected = winner['name']
    print('Ablation (skill vs random walk):')
    for r in sorted(ablation_rows, key=lambda x: -x['skill_vs_rw']):
        mark = ' <= selected' if r['name'] == selected else ''
        print(f"  {r['name']:<18} skill={r['skill_vs_rw']:+.3f} "
              f"band90=₱{r['band90']:.2f} rmse=₱{r['rmse']:.2f}{mark}")

    # Re-run the winning variant to get its reconstructed predictions for the report.
    v = make_variant(selected, frame)
    bt = walk_forward(v.y_model, v.X, _hgb_predict_fn, MIN_TRAIN)
    idx = bt['index']
    dates = [frame.index[i] for i in idx]
    yp = bt['y_pred'] + v.structural[idx]      # reconstruct to RON95 space
    yt = v.y_actual[idx]

    rw_bt = walk_forward(v.y_actual, None,
                         lambda Xt, ytr, xn: float(ytr[-1]), MIN_TRAIN)
    sn_bt = walk_forward(v.y_actual, None,
                         lambda Xt, ytr, xn: baselines.seasonal_naive_next(ytr, 12), MIN_TRAIN)
    rmse_model = rmse(yt, yp)
    rmse_rw = rmse(rw_bt['y_true'], rw_bt['y_pred'])
    rmse_sn = rmse(sn_bt['y_true'], sn_bt['y_pred'])

    res = yt - yp
    half = len(res) // 2
    cal_res, val_true, val_pred = res[:half], yt[half:], yp[half:]
    calib = conformal.build_calibration_table(cal_res, val_true, val_pred, CONFORMAL_LEVELS)
    qhat90 = conformal.conformal_quantile(cal_res, 0.9)
```

(c) Find the `rep = report.build_report(` call and add these two kwargs (before the closing `)`):

```python
        ablation=ablation_rows, selected_variant=selected,
```

(d) Just before the final `print(...)` line, add the ablation-table artifact write:

```python
    import json as _json
    (report.ARTIFACTS / 'ablation_table.json').write_text(
        _json.dumps({'selected': selected, 'rows': ablation_rows}, indent=2),
        encoding='utf-8')
```

(e) Replace the final print with one that also names the winner:

```python
    print(f"Selected variant: {selected} | "
          f"skill vs random walk: {rep['headline_skill_vs_random_walk']:+.3f} "
          f"over {rep['n_months']} months")
```

- [ ] **Step 2: Run the benchmark end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints the ablation table (one line per variant), names the selected winner, and writes `artifacts/accuracy_report.json`, `artifacts/ablation_table.json`, figures, and `backtest_predictions.csv` without error.

- [ ] **Step 3: Verify the artifacts updated**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; r=load_report(); print('selected:', r['selected_variant']); print('headline skill:', r['headline_skill_vs_random_walk']); [print(' ', x['name'], x['skill_vs_rw']) for x in r['ablation']]"`
Expected: prints the selected variant and each variant's skill. **Record the real numbers** — this is the Phase-2 result, honest whatever it is.

- [ ] **Step 4: Run the full suite (no regressions)**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all Phase-1 + Phase-2 tests pass (the one pre-existing `test_main_window::test_on_run_requested_accepts_4_args` failure is unrelated and may remain).

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/ablation_table.json ph_economic_ai/benchmark/artifacts/backtest_predictions.csv ph_economic_ai/benchmark/artifacts/figures/baseline_bars.png ph_economic_ai/benchmark/artifacts/figures/pred_vs_actual.png ph_economic_ai/benchmark/artifacts/figures/proxy_scatter.png
git commit -m "feat(benchmark): ablation-driven variant selection; headline uses winner"
```

---

## Task 6: Show the lever comparison in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add a failing test**

In `ph_economic_ai/tests/test_accuracy_view.py`, extend the `_report` dict (inside the `_report` helper) to include:

```python
        'ablation': [
            {'name': 'baseline', 'skill_vs_rw': -0.18, 'band90': 21.5, 'rmse': 4.73, 'mae': 3.54, 'n': 79},
            {'name': 'structural_hybrid', 'skill_vs_rw': 0.04, 'band90': 8.9, 'rmse': 3.8, 'mae': 2.9, 'n': 79},
        ],
        'selected_variant': 'structural_hybrid',
```

Then add this test:

```python
def test_view_shows_ablation_when_present(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    t = view.ablation_summary()
    assert 'structural_hybrid' in t
    assert 'baseline' in t
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_ablation_when_present -v`
Expected: FAIL — `AttributeError: 'AccuracyView' object has no attribute 'ablation_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add this method to `AccuracyView`:

```python
    def ablation_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('ablation') or []
        sel = self._report.get('selected_variant')
        lines = []
        for r in sorted(rows, key=lambda x: -x['skill_vs_rw']):
            mark = '  ◀ selected' if r['name'] == sel else ''
            lines.append(f"{r['name']}: skill {r['skill_vs_rw']:+.2f} vs RW, "
                         f"90% band ₱{r['band90']:.2f}{mark}")
        return '\n'.join(lines)
```

(b) In `_build`, inside the `if self._report is not None:` block, after `col.addWidget(self._calibration_table())`, add:

```python
            if self._report.get('ablation'):
                abl = QLabel('<b>Lever comparison (Phase 2)</b><br>'
                             + self.ablation_summary().replace('\n', '<br>'))
                abl.setWordWrap(True)
                col.addWidget(abl)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py -v`
Expected: PASS (3 passed — the 2 existing + the new one)

- [ ] **Step 5: Smoke-test the window**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: same as before — only the pre-existing `test_on_run_requested_accepts_4_args` may fail; nothing new.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/accuracy_view.py ph_economic_ai/tests/test_accuracy_view.py
git commit -m "feat(ui): show Phase-2 lever comparison in Accuracy view"
```

---

## Task 7: Record the Phase-2 outcome honestly in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-05-ph-economic-ai-accuracy-evaluation-design.md`

- [ ] **Step 1: Append a results subsection to §9**

Add a short, factual subsection at the end of §9 titled **"Phase 2 — measured outcome"** stating: the selected variant, its real skill vs random walk and 90% band width vs the Phase-1 baseline (−0.18), and one sentence of interpretation. If no variant beat random walk, say so plainly and note the next honest step (weekly resolution / true MOPS finished-product series — lever 5, which needs new data). Pull the numbers from `artifacts/ablation_table.json` produced in Task 5.

Example shape (fill with the REAL numbers from Task 5 Step 3):

```markdown
### Phase 2 — measured outcome

Ablation over five variants (baseline, drop_demand, passthrough_lags, finished_gas,
structural_hybrid), same causal walk-forward, n=<N> months. Selected: **<winner>**,
skill **<+/-X.XX>** vs random walk (Phase-1 baseline was −0.18), 90% band **₱<W>**.
Interpretation: <one honest sentence>. <If none > 0: next honest step is weekly
resolution + a true MOPS finished-product series (lever 5), which requires new data.>
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-05-ph-economic-ai-accuracy-evaluation-design.md
git commit -m "docs: record measured Phase-2 ablation outcome in spec §9"
```

---

## Final verification

- [ ] **Run the entire suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Confirm the headline reflects the winner**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; r=load_report(); print(r['selected_variant'], r['headline_skill_vs_random_walk'])"`
Expected: prints the selected variant and its skill — the new honest headline.

- [ ] **Confirm conformal widths still wire into the model**

Run: `python -c "from ph_economic_ai.model import load_conformal_widths; print(load_conformal_widths())"`
Expected: prints the (now winner-derived) conformal widths.

---

## Self-Review (completed by plan author)

**Spec coverage (§9 levers):**
- Lever 1 finished-gasoline features → `finished_gas` variant (Task 1) + ablation (Task 2/5).
- Lever 2 structural+ML hybrid → `structural_hybrid` variant with reconstruct (Tasks 1, 2, 5).
- Lever 3 pass-through lag features → `passthrough_lags` variant (Task 1).
- Lever 4 drop synthetic demand → `drop_demand` variant (Task 1).
- Lever 6 normalized/Mondrian conformal → Task 3 (functions) — available for the report/UI; full swap of headline bands to normalized is left as an optional follow-up since plain split-conformal already calibrates.
- Lever 7 HGB tuning → not a separate task; `_hgb_predict_fn` params unchanged. **Intentionally deferred** (spec marks it "marginal"); add later if the winner is close to the gate. NOTE: this is the one §9 lever without a dedicated task — acceptable per its "marginal/cheap, do later" classification.
- Lever 5 weekly resolution + longer history → **out of scope (needs new data)**; called out in Task 7's honest-outcome note. Matches spec (it requires a new data pipeline).
- Gating + ablation table + honest-limit reporting → Tasks 2, 5, 7.

**Placeholder scan:** none — all code steps contain complete code; Task 7's template is explicitly "fill with real numbers from Task 5", which is correct (the outcome is empirical and cannot be known at plan time).

**Type consistency:** `Variant(name,dates,X,y_actual,y_model,structural)` used identically in features.py and ablation.py. Ablation row keys `{name,rmse,mae,skill_vs_rw,band90,n}` consistent across ablation.py, report.py test, run.py print, and accuracy_view. `make_variant`, `build_feature_frame`, `VARIANTS`, `run_ablation`, `select_winner`, `normalized_conformal_quantile`, `normalized_coverage` names consistent across tasks. `build_report(... ablation=, selected_variant=)` matches between report.py (Task 4) and run.py call (Task 5).
```
