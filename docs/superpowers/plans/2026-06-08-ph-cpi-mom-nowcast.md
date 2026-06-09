# Month-over-Month CPI Nowcasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nowcast Philippine month-over-month CPI inflation from intra-month-observable drivers and test — honestly — whether it beats the *strongest* simple baseline (best of random-walk / seasonal-naive / drift) via Diebold-Mariano.

**Architecture:** Adds a MoM inflation target, generalizes the existing contemporaneous nowcast frame to any target, and adds a `run_mom_nowcast` that backtests the full forecaster panel and assigns a verdict through a pure `mom_verdict` helper which runs DM **against the best simple baseline** (not random walk). Report/run/figure/view extended. No new data — reuses committed `ph_cpi_monthly.csv` + `features_monthly.csv`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, statsmodels, scipy, matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-ph-cpi-mom-nowcast-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/targets.py::{load_inflation, cpi_to_yoy, _features, CPI_CSV}`.
- `benchmark/nowcast.py::{build_nowcast_frame, run_nowcast, PANEL_METHODS, CONFORMAL_LEVELS}` (current `build_nowcast_frame()` takes no args, uses `load_inflation`/`_features`; columns `oil,fx,fuel,prev_inflation,target`).
- `benchmark/backtest.py::walk_forward`; `forecasters.py::make_forecaster`; `metrics.py::rmse`; `significance.py::diebold_mariano(loss_a, loss_b, h=1) -> {dm_stat, p_value}`; `conformal.py::build_calibration_table`.
- `benchmark/report.py::{build_report (...nowcast=None), REQUIRED_KEYS, ARTIFACTS, load_report}`.
- `benchmark/run.py` (imports `nowcast as nowcast_mod`, `walk_forward`, `make_forecaster`, `figures`, `MIN_TRAIN`), `figures.py::plot_nowcast`, `ui/accuracy_view.py::AccuracyView`.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, import `from ph_economic_ai.X import Y`, start with the path shim. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing; `global_fuel_prices.xlsx` is gitignored — never stage it.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Modify:** `benchmark/targets.py` (+`cpi_to_mom`, `load_inflation_mom`), `benchmark/nowcast.py` (generalize frame; +`mom_verdict`, `run_mom_nowcast`), `benchmark/report.py` (+`nowcast_mom`), `benchmark/figures.py` (reuse `plot_nowcast`), `benchmark/run.py` (run+figure), `ui/accuracy_view.py` (+panel); tests `test_targets.py`, `test_nowcast.py`, `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: MoM inflation target

**Files:**
- Modify: `ph_economic_ai/benchmark/targets.py`
- Modify: `ph_economic_ai/tests/test_targets.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_targets.py`:

```python
from ph_economic_ai.benchmark.targets import cpi_to_mom, load_inflation_mom


def test_cpi_to_mom_one_percent_per_month():
    idx = pd.date_range('2020-01', periods=6, freq='MS').strftime('%Y-%m')
    cpi = pd.Series(100.0 * (1.01) ** np.arange(6), index=idx)   # +1%/month
    mom = cpi_to_mom(cpi)
    assert mom.iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert len(mom) == 5   # first month dropped


def test_load_inflation_mom_reads_index_csv(tmp_path):
    idx = pd.date_range('2020-01', periods=4, freq='MS').strftime('%Y-%m')
    vals = [100.0, 101.0, 102.01, 103.0301]   # +1%/mo
    p = tmp_path / 'cpi.csv'
    p.write_text('date,cpi_index\n' + '\n'.join(f'{d},{v}' for d, v in zip(idx, vals)) + '\n',
                 encoding='utf-8')
    mom = load_inflation_mom(p)
    assert mom.iloc[0] == pytest.approx(1.0, abs=0.01)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_targets.py -k mom -v`
Expected: FAIL — `ImportError: cannot import name 'cpi_to_mom'`

- [ ] **Step 3: Implement (append to targets.py)**

Append to `ph_economic_ai/benchmark/targets.py`:

```python
def cpi_to_mom(cpi_index: pd.Series) -> pd.Series:
    """Convert a monthly CPI index to month-over-month inflation %, dropping the
    first (undefined) month."""
    s = cpi_index.sort_index()
    mom = (s / s.shift(1) - 1.0) * 100.0
    return mom.dropna()


def load_inflation_mom(csv_path: Path = CPI_CSV) -> pd.Series:
    """Load the committed CPI index and return month-over-month inflation %."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    cpi = pd.Series(df['cpi_index'].astype(float).values, index=df['date'].astype(str).values)
    cpi = cpi[~cpi.index.duplicated(keep='last')]
    return cpi_to_mom(cpi)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_targets.py -v`
Expected: PASS (existing target tests + 2 new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/targets.py ph_economic_ai/tests/test_targets.py
git commit -m "feat(benchmark): month-over-month CPI inflation target"
```

---

## Task 2: Generalize the nowcast frame to any target

**Files:**
- Modify: `ph_economic_ai/benchmark/nowcast.py` (the `build_nowcast_frame` function)
- Modify: `ph_economic_ai/tests/test_nowcast.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_nowcast.py`:

```python
def test_build_nowcast_frame_mom_variant(monkeypatch):
    idx = pd.date_range('2017-01', periods=40, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(4)
    mom = pd.Series(rng.normal(0.3, 0.4, 40), index=idx)
    feats = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, 40)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, 40)),
        'gas_price': 60 + np.cumsum(rng.normal(0, 0.5, 40)),
        'demand_index': 70 + rng.normal(0, 2, 40),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, '_features', lambda: feats)
    f = nc.build_nowcast_frame(target_loader=lambda: mom, prev_col='prev_mom')
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_mom', 'target']
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'prev_mom'] == pytest.approx(mom.iloc[pos - 1])
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
    # leakage guard: only prev_mom is target-derived
    cpi_like = [c for c in f.columns if 'mom' in c.lower() or 'infl' in c.lower() or 'cpi' in c.lower()]
    assert cpi_like == ['prev_mom']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py::test_build_nowcast_frame_mom_variant -v`
Expected: FAIL — `build_nowcast_frame() got an unexpected keyword argument 'target_loader'`

- [ ] **Step 3: Implement — replace `build_nowcast_frame` in nowcast.py**

Replace the existing `build_nowcast_frame` function (lines under its `def`) with:

```python
def build_nowcast_frame(target_loader=None, prev_col: str = 'prev_inflation') -> pd.DataFrame:
    """Contemporaneous nowcast frame: oil/fx/fuel (month t) + <prev_col> (t-1) +
    target. target_loader defaults to load_inflation (resolved at call time so tests
    can monkeypatch the module-level loader). For MoM, pass
    target_loader=load_inflation_mom and prev_col='prev_mom'."""
    loader = target_loader if target_loader is not None else load_inflation
    tgt = loader()
    feats = _features()
    base = pd.DataFrame({
        'oil': feats['oil_price'],
        'fx': feats['usd_php'],
        'fuel': feats['gas_price'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base[prev_col] = base['target'].shift(1)
    return base[['oil', 'fx', 'fuel', prev_col, 'target']].dropna()
```

(The default `target_loader=None` → `load_inflation` resolved at call time preserves the existing YoY behavior AND the existing test's monkeypatch of `nc.load_inflation`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS — all existing nowcast tests (YoY frame + run_nowcast) still pass, plus the new MoM-frame test.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): generalize nowcast frame to any target (YoY/MoM)"
```

---

## Task 3: `mom_verdict` + `run_mom_nowcast` (DM vs best simple baseline)

**Files:**
- Modify: `ph_economic_ai/benchmark/nowcast.py` (append)
- Modify: `ph_economic_ai/tests/test_nowcast.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `ph_economic_ai/tests/test_nowcast.py`:

```python
from ph_economic_ai.benchmark.nowcast import mom_verdict, run_mom_nowcast


def test_mom_verdict_beats_best_naive():
    rng = np.random.default_rng(0)
    n = 300
    base_loss = rng.uniform(0.8, 1.2, n)          # seasonal_naive: best baseline
    rmse = {'random_walk': 1.5, 'seasonal_naive': 1.0, 'drift': 1.3,
            'arima': 1.2, 'ets': 1.1, 'ridge': 0.7, 'hgb': 0.9}
    loss = {'random_walk': base_loss + 1.0, 'seasonal_naive': base_loss,
            'drift': base_loss + 0.6, 'arima': base_loss + 0.4, 'ets': base_loss + 0.2,
            'ridge': base_loss - 0.4, 'hgb': base_loss - 0.1}      # ridge clearly best
    v = mom_verdict(rmse, loss)
    assert v['verdict'] == 'beats_best_naive'
    assert v['best_method'] == 'ridge'
    assert v['best_naive'] == 'seasonal_naive'


def test_mom_verdict_hollow_win_guard():
    # ridge beats random_walk but NOT the stronger seasonal_naive -> not a real win
    rng = np.random.default_rng(1)
    n = 300
    seas = rng.uniform(0.4, 0.6, n)               # seasonal_naive: lowest loss
    rmse = {'random_walk': 1.5, 'seasonal_naive': 0.5, 'drift': 1.2,
            'arima': 1.1, 'ets': 1.0, 'ridge': 0.9, 'hgb': 1.0}    # ridge > seasonal rmse
    loss = {'random_walk': seas + 1.0, 'seasonal_naive': seas, 'drift': seas + 0.7,
            'arima': seas + 0.6, 'ets': seas + 0.5, 'ridge': seas + 0.4, 'hgb': seas + 0.5}
    v = mom_verdict(rmse, loss)
    assert v['verdict'] == 'no_better_than_naive'
    assert v['best_method'] == 'seasonal_naive'   # falls back to the best baseline


def test_run_mom_nowcast_beats_on_constructed_signal():
    idx = pd.date_range('2016-01', periods=110, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, 110))
    dfuel = np.r_[0.0, np.diff(fuel)]
    target = 0.6 * dfuel + rng.normal(0, 0.05, 110)   # MoM driven by contemporaneous fuel change
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 110),
        'fx': 55 + rng.normal(0, 0.1, 110),
        'fuel': fuel,
        'prev_mom': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_mom_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'beats_best_naive'
    assert res['best_method'] in ('ridge', 'hgb', 'arima', 'ets')


def test_run_mom_nowcast_insufficient_data():
    idx = pd.date_range('2020-01', periods=12, freq='MS').strftime('%Y-%m')
    frame = pd.DataFrame({'oil': range(12), 'fx': range(12), 'fuel': range(12),
                          'prev_mom': range(12), 'target': range(12)},
                         index=idx).astype(float)
    res = run_mom_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'insufficient_data'
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -k mom_verdict -v`
Expected: FAIL — `ImportError: cannot import name 'mom_verdict'`

- [ ] **Step 3: Append the implementation to nowcast.py**

```python
from ph_economic_ai.benchmark.metrics import rmse as _rmse
from ph_economic_ai.benchmark.significance import diebold_mariano
from ph_economic_ai.benchmark.targets import load_inflation_mom

BASELINE_POOL = ('random_walk', 'seasonal_naive', 'drift')


def mom_verdict(rmse_by_method: dict, loss_by_method: dict,
                baseline_pool=BASELINE_POOL) -> dict:
    """Verdict for the MoM nowcast: a model 'beats_best_naive' only if it has lower
    RMSE than the BEST simple baseline AND a significant Diebold-Mariano edge over
    it (p<0.05, lower loss). Otherwise 'no_better_than_naive'. Pure function."""
    pool = [m for m in baseline_pool if m in rmse_by_method] or ['random_walk']
    best_naive = min(pool, key=lambda m: rmse_by_method[m])
    base_rmse = rmse_by_method[best_naive]
    base_loss = loss_by_method[best_naive]

    winners = []
    for m in rmse_by_method:
        if m in baseline_pool or rmse_by_method[m] >= base_rmse:
            continue
        dm = diebold_mariano(loss_by_method[m], base_loss)
        if dm['p_value'] < 0.05 and dm['dm_stat'] < 0:      # m has lower loss, significant
            winners.append(m)

    if winners:
        best_method = min(winners, key=lambda m: rmse_by_method[m])
        dm_p = diebold_mariano(loss_by_method[best_method], base_loss)['p_value']
        return {'verdict': 'beats_best_naive', 'best_method': best_method,
                'best_naive': best_naive,
                'best_skill_vs_naive': round(1 - rmse_by_method[best_method] / base_rmse, 4),
                'dm_p': round(dm_p, 4)}
    return {'verdict': 'no_better_than_naive', 'best_method': best_naive,
            'best_naive': best_naive, 'best_skill_vs_naive': 0.0, 'dm_p': None}


def run_mom_nowcast(min_train: int = 24, baseline_pool=BASELINE_POOL, frame=None) -> dict:
    """Nowcast MoM inflation; verdict via DM against the best simple baseline."""
    if frame is None:
        frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    if len(frame) < min_train + 5:
        return {'verdict': 'insufficient_data', 'n': int(len(frame))}

    feature_cols = [c for c in frame.columns if c != 'target']
    y = frame['target'].to_numpy(dtype=float)
    X = frame[feature_cols].to_numpy(dtype=float)

    rmse_by, loss_by, n_pred = {}, {}, 0
    for m in PANEL_METHODS:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        loss_by[m] = (bt['y_true'] - bt['y_pred']) ** 2
        rmse_by[m] = _rmse(bt['y_true'], bt['y_pred'])
        n_pred = len(bt['y_true'])

    v = mom_verdict(rmse_by, loss_by, baseline_pool)

    bt = walk_forward(y, X, make_forecaster(v['best_method']), min_train)
    res = bt['y_true'] - bt['y_pred']
    half = max(1, len(res) // 2)
    calib = build_calibration_table(res[:half], bt['y_true'][half:], bt['y_pred'][half:],
                                    CONFORMAL_LEVELS) if len(res) > 3 else []
    return {**v, 'n': int(n_pred), 'calibration': calib,
            'rmse_by_method': {k: round(val, 4) for k, val in rmse_by.items()}}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS (all — existing + 4 new). If `test_run_mom_nowcast_beats_on_constructed_signal` is flaky, re-run once; the signal (0.6·Δfuel, noise 0.05) is strong. Do NOT weaken assertions or logic; if it genuinely fails, STOP and report.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): MoM nowcast runner + DM-vs-best-baseline verdict"
```

---

## Task 4: Report key for the MoM nowcast

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_nowcast_mom():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        nowcast_mom={'verdict': 'beats_best_naive', 'best_method': 'ridge',
                     'best_naive': 'seasonal_naive', 'best_skill_vs_naive': 0.15,
                     'dm_p': 0.03, 'n': 70},
    )
    assert rep['nowcast_mom']['verdict'] == 'beats_best_naive'
    assert 'nowcast_mom' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_nowcast_mom -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'nowcast_mom'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'nowcast_mom'` to the `REQUIRED_KEYS` tuple.
(b) Add `nowcast_mom=None` as the LAST parameter of `build_report` (after `nowcast=None`).
(c) In the returned dict, add just before `'limitations'`:
```python
        'nowcast_mom': nowcast_mom if nowcast_mom is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the MoM nowcast result"
```

---

## Task 5: Run the MoM nowcast in run.py + figure

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: end-to-end run on real data

- [ ] **Step 1: Wire into run.py**

In `ph_economic_ai/benchmark/run.py` (it already imports `nowcast as nowcast_mod`, `walk_forward`, `make_forecaster`, `figures`, and defines `MIN_TRAIN`):
(a) After the existing YoY-nowcast block (after the `nowcast_res = nowcast_mod.run_nowcast(...)` section, before `rep = report.build_report(`), add:
```python
    # -- Month-over-month CPI nowcast (vs best simple baseline) --
    mom_res = nowcast_mod.run_mom_nowcast(MIN_TRAIN)
    if mom_res['verdict'] == 'insufficient_data':
        print(f"MoM CPI nowcast: insufficient_data (n={mom_res.get('n', 0)})")
    else:
        print(f"MoM CPI nowcast: {mom_res['verdict']} | best={mom_res['best_method']} "
              f"vs {mom_res['best_naive']} | skill={mom_res['best_skill_vs_naive']:+.3f} "
              f"DM p={mom_res['dm_p']}")
```
(b) In the `report.build_report(` call, add:
```python
        nowcast_mom={k: v for k, v in mom_res.items() if k != 'calibration'},
```
(c) After the existing figure calls, add:
```python
    import json as _json4
    (report.ARTIFACTS / 'nowcast_mom_table.json').write_text(
        _json4.dumps(mom_res, indent=2), encoding='utf-8')
    if mom_res['verdict'] != 'insufficient_data':
        _mf = nowcast_mod.build_nowcast_frame(
            target_loader=nowcast_mod.load_inflation_mom, prev_col='prev_mom')
        _mcols = [c for c in _mf.columns if c != 'target']
        _my = _mf['target'].to_numpy(float); _mX = _mf[_mcols].to_numpy(float)
        _mbt = walk_forward(_my, _mX, make_forecaster(mom_res['best_method']), MIN_TRAIN)
        _mnbt = walk_forward(_my, _mX, make_forecaster(mom_res['best_naive']), MIN_TRAIN)
        _md = [_mf.index[i] for i in _mbt['index']]
        figures.plot_nowcast(_md, _mbt['y_true'], _mbt['y_pred'], _mnbt['y_pred'])
        import os as _os
        _os.replace(figures.FIG_DIR / 'nowcast.png', figures.FIG_DIR / 'nowcast_mom.png')
```
(Note: `plot_nowcast` writes `nowcast.png`; we rename to `nowcast_mom.png` so it doesn't clobber the YoY figure. `nowcast_mod.load_inflation_mom` is importable because nowcast.py imports it at module scope in Task 3.)

- [ ] **Step 2: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints all prior summaries plus a `MoM CPI nowcast: ...` line; writes `nowcast_mom_table.json` and (if not insufficient) `figures/nowcast_mom.png`, no errors.

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; print(load_report()['nowcast_mom'])"`
Record the printed dict — verdict, best_method, best_naive, best_skill_vs_naive, dm_p, n. **This is the answer to whether MoM beats the strong baseline.**

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window::test_on_run_requested_accepts_4_args`. Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/nowcast_mom_table.json
git add ph_economic_ai/benchmark/artifacts/figures/nowcast_mom.png 2>/dev/null || true
git status --short
git commit -m "feat(benchmark): run MoM CPI nowcast + figure; report the result"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 6: Surface the MoM nowcast in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'nowcast_mom': {'verdict': 'beats_best_naive', 'best_method': 'ridge',
                        'best_naive': 'seasonal_naive', 'best_skill_vs_naive': 0.15,
                        'dm_p': 0.03, 'n': 70},
```
Add the test:
```python
def test_view_shows_nowcast_mom(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.nowcast_mom_summary()
    assert 'mom' in s.lower() or 'month-over-month' in s.lower()
    assert 'ridge' in s
    assert 'seasonal_naive' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_nowcast_mom -v`
Expected: FAIL — `AttributeError: ... has no attribute 'nowcast_mom_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method to `AccuracyView`:
```python
    def nowcast_mom_summary(self) -> str:
        if not self._report:
            return ''
        m = self._report.get('nowcast_mom') or {}
        if not m or m.get('verdict') == 'insufficient_data':
            return ''
        return (f"MoM inflation nowcast: {m['verdict']} — best {m['best_method']} "
                f"vs strongest baseline {m['best_naive']}, skill "
                f"{m['best_skill_vs_naive']:+.2f} (DM p={m['dm_p']}).")
```
(b) In `_build`, inside `if self._report is not None:`, after the nowcast block, add:
```python
            _mc = self.nowcast_mom_summary()
            if _mc:
                mcl = QLabel('<b>MoM nowcast (vs strongest baseline)</b><br>' + _mc)
                mcl.setWordWrap(True)
                col.addWidget(mcl)
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
git commit -m "feat(ui): show the MoM CPI nowcast result in the Accuracy view"
```

---

## Task 7: Record the MoM outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-ph-cpi-mom-nowcast-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 5 Step 3**

Replace the `[beats / does not beat]`/`[baseline]`/`[method]`/`[x]`/`[p]`/`[n]`/`[result]` placeholders in §9 with the measured MoM result. State plainly whether the nowcast beats the **best simple baseline** (verdict + best method + which baseline was the bar + skill vs that baseline + DM p + n), and one sentence of interpretation. Copy numbers from `nowcast_mom_table.json`; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-08-ph-cpi-mom-nowcast-design.md
git commit -m "docs: record measured MoM nowcast result in spec section 9"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **MoM result legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; m=load_report()['nowcast_mom']; print(m.get('verdict'), m.get('best_method'), 'vs', m.get('best_naive'), m.get('best_skill_vs_naive'))"`
Expected: prints the verdict (`beats_best_naive` or `no_better_than_naive`), the winning method, the baseline it had to beat, and the skill — the honest MoM result, whichever way it lands.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3 definitions (MoM target, baseline pool, verdict) → Tasks 1, 3. §4.1 cpi_to_mom/load_inflation_mom → Task 1. §4.2 generalized frame → Task 2. §4.3 mom_verdict → Task 3. §4.4 run_mom_nowcast → Task 3. §4.5 integration (report/run/figure/view) → Tasks 4, 5, 6. §6 error handling (insufficient_data, empty pool guard) → Task 3. §7 testing incl. hollow-win guard + synthetic beats/insufficient → Tasks 1, 2, 3. §8 deliverables → all. §9 write-up → Task 7.

**Placeholder scan:** §9's bracketed markers are filled at Task 7 from real output (empirical; cannot be known at plan time) — accepted pattern. No other red-flag placeholders; every code step has complete code.

**Type consistency:** `cpi_to_mom`/`load_inflation_mom` (Task 1) used by `run_mom_nowcast` (Task 3) and run.py (Task 5). `build_nowcast_frame(target_loader=None, prev_col='prev_inflation')` (Task 2) called with `(load_inflation_mom, 'prev_mom')` in Tasks 3, 5 — consistent. `mom_verdict(rmse_by_method, loss_by_method, baseline_pool) -> {verdict, best_method, best_naive, best_skill_vs_naive, dm_p}` consistent across Tasks 3, 4, 6. `run_mom_nowcast(min_train, baseline_pool, frame=None)` consistent (Tasks 3, 5). `build_report(... nowcast_mom=)` matches Task 4 (def) and Task 5 (call). Report key `nowcast_mom` consistent across Tasks 4, 5, 6. `figures.plot_nowcast(dates, actual, nowcast, naive)` reused (Task 5), output renamed to `nowcast_mom.png` to avoid clobbering the YoY figure.
```
