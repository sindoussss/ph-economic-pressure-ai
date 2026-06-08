# MoM Nowcast Driver-Only Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate the pure within-month driver edge of the MoM CPI nowcast — drop the own-lag (`prev_mom`), restrict candidates to driver regressors {ridge, hgb}, and test (Diebold-Mariano, vs the best simple baseline) whether the contemporaneous oil/FX/fuel alone beat persistence.

**Architecture:** Minimal additive change to `benchmark/nowcast.py`: a `methods=` parameter on `run_mom_nowcast` (default unchanged) plus a `run_driver_only_ablation` wrapper that drops `prev_mom` and runs only {random_walk, seasonal_naive, drift, ridge, hgb}, so the existing `mom_verdict` naturally treats {ridge, hgb} as the only candidates. Report/run/view/spec extended. No new data — reuses committed `ph_cpi_monthly.csv` + `features_monthly.csv`.

**Tech Stack:** Python 3.10, numpy, pandas, scikit-learn, statsmodels, scipy, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-ph-cpi-mom-driver-ablation-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/nowcast.py::{build_nowcast_frame, run_mom_nowcast, mom_verdict, PANEL_METHODS, BASELINE_POOL, CONFORMAL_LEVELS, load_inflation_mom}`. Current `run_mom_nowcast(min_train=24, baseline_pool=BASELINE_POOL, frame=None)` loops over `PANEL_METHODS`, builds `rmse_by`/`loss_by`, calls `mom_verdict`, returns `{verdict, best_method, best_naive, best_skill_vs_naive, dm_p, n, calibration, rmse_by_method}`.
- `mom_verdict(rmse_by_method, loss_by_method, baseline_pool=BASELINE_POOL)` treats any method NOT in `baseline_pool` as a candidate.
- `benchmark/report.py::build_report(..., nowcast_mom=None)`, `REQUIRED_KEYS`, `ARTIFACTS`.
- `benchmark/run.py` (runs the MoM nowcast via `nowcast_mod.run_mom_nowcast`, has `MIN_TRAIN`, `report`, imports `nowcast as nowcast_mod`).
- `ui/accuracy_view.py::AccuracyView` (has `nowcast_mom_summary`).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`. Suite: `python -m pytest ph_economic_ai/tests/ -q`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing; `global_fuel_prices.xlsx` is gitignored — never stage it.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Modify:** `benchmark/nowcast.py` (+`methods=` param, +`run_driver_only_ablation`), `benchmark/report.py` (+`mom_driver_ablation`), `benchmark/run.py` (run+record), `ui/accuracy_view.py` (+note); tests `test_nowcast.py`, `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: `methods=` param + `run_driver_only_ablation`

**Files:**
- Modify: `ph_economic_ai/benchmark/nowcast.py`
- Modify: `ph_economic_ai/tests/test_nowcast.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `ph_economic_ai/tests/test_nowcast.py`:

```python
from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation


def test_run_mom_nowcast_respects_methods_param():
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    target = rng.normal(0.3, 0.4, 90)
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90), 'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': 60 + rng.normal(0, 1, 90), 'prev_mom': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_mom_nowcast(min_train=24, frame=frame, methods=['random_walk', 'ridge'])
    assert set(res['rmse_by_method']) == {'random_walk', 'ridge'}


def _driver_signal_frame(n=110, seed=5):
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(seed)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, n))
    target = 0.5 * fuel + rng.normal(0, 0.02, n)      # target tracks contemporaneous FUEL LEVEL
    return pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, n), 'fx': 55 + rng.normal(0, 0.1, n),
        'fuel': fuel, 'prev_mom': np.r_[target[0], target[:-1]], 'target': target,
    }, index=idx)


def test_driver_ablation_detects_driver_edge():
    res = run_driver_only_ablation(min_train=24, frame=_driver_signal_frame())
    assert res['driver_edge'] is True
    assert res['verdict'] == 'beats_best_naive'
    assert res['best_method'] in ('ridge', 'hgb')
    # the ablation ran exactly the baseline pool + driver regressors (no ARIMA/ETS)
    assert set(res['rmse_by_method']) == {'random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb'}


def test_driver_ablation_absent_when_pure_ar_noise_drivers():
    n = 130
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(6)
    target = np.empty(n); target[0] = 0.0
    for i in range(1, n):
        target[i] = 0.7 * target[i - 1] + rng.normal(0, 0.3)   # persistent AR(1)
    frame = pd.DataFrame({
        'oil': rng.normal(0, 1, n), 'fx': rng.normal(0, 1, n), 'fuel': rng.normal(0, 1, n),  # noise
        'prev_mom': np.r_[target[0], target[:-1]], 'target': target,
    }, index=idx)
    res = run_driver_only_ablation(min_train=24, frame=frame)
    assert res['driver_edge'] is False
    assert res['verdict'] == 'no_better_than_naive'


def test_driver_ablation_handles_frame_without_prev_mom():
    f = _driver_signal_frame().drop(columns=['prev_mom'])
    res = run_driver_only_ablation(min_train=24, frame=f)   # must not raise
    assert 'verdict' in res
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -k "driver_ablation or methods_param" -v`
Expected: FAIL — `ImportError: cannot import name 'run_driver_only_ablation'` (and the methods-param test errors on the unexpected kwarg).

- [ ] **Step 3: Implement — modify `run_mom_nowcast` and append `run_driver_only_ablation`**

In `ph_economic_ai/benchmark/nowcast.py`:
(a) Change the `run_mom_nowcast` signature to add `methods=None`, and replace the loop header `for m in PANEL_METHODS:` with a resolved list. Specifically, change the signature line:
```python
def run_mom_nowcast(min_train: int = 24, baseline_pool=BASELINE_POOL, frame=None,
                    methods=None) -> dict:
```
and immediately after the `insufficient_data` guard (before the `feature_cols = ...` line) add:
```python
    methods = list(PANEL_METHODS) if methods is None else list(methods)
```
and change the loop `for m in PANEL_METHODS:` to `for m in methods:`. Nothing else in the function changes (it already derives `feature_cols` from the frame and uses `mom_verdict`).

(b) Append the wrapper at the END of `nowcast.py`:
```python
def run_driver_only_ablation(min_train: int = 24, frame=None) -> dict:
    """Isolate the pure within-month driver edge: drop the own-lag (prev_mom) and
    let only driver regressors {ridge, hgb} compete against the simple baselines.
    Adds a boolean 'driver_edge' (True iff a driver regressor beats the best
    simple baseline, DM-significant)."""
    if frame is None:
        frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    driver_frame = frame.drop(columns=['prev_mom'], errors='ignore')
    res = run_mom_nowcast(min_train, frame=driver_frame,
                          methods=['random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb'])
    res['driver_edge'] = (res.get('verdict') == 'beats_best_naive')
    return res
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_nowcast.py -v`
Expected: PASS (all — existing MoM/YoY tests + 4 new). If `test_driver_ablation_detects_driver_edge` is flaky, re-run once (signal is strong: target = 0.5·fuel, noise 0.02). Do NOT weaken assertions/logic; if it genuinely fails, STOP and report.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/nowcast.py ph_economic_ai/tests/test_nowcast.py
git commit -m "feat(benchmark): driver-only MoM ablation (methods= param + run_driver_only_ablation)"
```

---

## Task 2: Report key for the ablation

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_mom_driver_ablation():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        mom_driver_ablation={'verdict': 'no_better_than_naive', 'driver_edge': False,
                             'best_method': 'random_walk', 'best_naive': 'random_walk',
                             'best_skill_vs_naive': 0.0, 'dm_p': None, 'n': 61},
    )
    assert rep['mom_driver_ablation']['driver_edge'] is False
    assert 'mom_driver_ablation' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_mom_driver_ablation -v`
Expected: FAIL — `build_report() got an unexpected keyword argument 'mom_driver_ablation'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'mom_driver_ablation'` to the `REQUIRED_KEYS` tuple.
(b) Add `mom_driver_ablation=None` as the LAST parameter of `build_report` (after `nowcast_mom=None`).
(c) In the returned dict, add just before `'limitations'`:
```python
        'mom_driver_ablation': mom_driver_ablation if mom_driver_ablation is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the MoM driver-only ablation result"
```

---

## Task 3: Run the ablation in run.py

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`
- Test: end-to-end run on real data

- [ ] **Step 1: Wire into run.py**

In `ph_economic_ai/benchmark/run.py`, after the MoM-nowcast block (`mom_res = nowcast_mod.run_mom_nowcast(MIN_TRAIN)` and its print), BEFORE the `rep = report.build_report(` call, add:
```python
    # -- Driver-only ablation of the MoM nowcast (isolate the within-month edge) --
    mom_abl = nowcast_mod.run_driver_only_ablation(MIN_TRAIN)
    if mom_abl['verdict'] == 'insufficient_data':
        print(f"MoM driver-only ablation: insufficient_data (n={mom_abl.get('n', 0)})")
    else:
        print(f"MoM driver-only ablation: driver_edge={mom_abl['driver_edge']} | "
              f"best={mom_abl['best_method']} vs {mom_abl['best_naive']} | "
              f"skill={mom_abl['best_skill_vs_naive']:+.3f} DM p={mom_abl['dm_p']}")
```
In the `report.build_report(` call, add:
```python
        mom_driver_ablation={k: v for k, v in mom_abl.items() if k != 'calibration'},
```
After the existing artifact writes, add:
```python
    import json as _json5
    (report.ARTIFACTS / 'mom_driver_ablation_table.json').write_text(
        _json5.dumps(mom_abl, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints all prior summaries plus a `MoM driver-only ablation: ...` line; writes `mom_driver_ablation_table.json`, no errors.

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; print(load_report()['mom_driver_ablation'])"`
Record the printed dict — `driver_edge`, verdict, best_method, best_naive, best_skill_vs_naive, dm_p, n, rmse_by_method. **This is the confirmation result: do the drivers alone beat naive?**

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window.py::test_on_run_requested_accepts_4_args`. Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/mom_driver_ablation_table.json
git status --short
git commit -m "feat(benchmark): run the MoM driver-only ablation; report the result"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 4: Surface the ablation in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'mom_driver_ablation': {'verdict': 'no_better_than_naive', 'driver_edge': False,
                                'best_method': 'random_walk', 'best_naive': 'random_walk',
                                'best_skill_vs_naive': 0.0, 'dm_p': None, 'n': 61},
```
Add the test:
```python
def test_view_shows_mom_driver_ablation(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.mom_driver_ablation_summary()
    assert 'driver' in s.lower()
    assert 'random_walk' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_mom_driver_ablation -v`
Expected: FAIL — `AttributeError: ... has no attribute 'mom_driver_ablation_summary'`

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method to `AccuracyView`:
```python
    def mom_driver_ablation_summary(self) -> str:
        if not self._report:
            return ''
        a = self._report.get('mom_driver_ablation') or {}
        if not a or a.get('verdict') == 'insufficient_data':
            return ''
        edge = 'CONFIRMED' if a.get('driver_edge') else 'absent'
        return (f"Driver-only ablation (no own-lag): within-month driver edge {edge} "
                f"— best {a['best_method']} vs {a['best_naive']}, "
                f"skill {a['best_skill_vs_naive']:+.2f} (DM p={a['dm_p']}).")
```
(b) In `_build`, inside `if self._report is not None:`, after the MoM nowcast block (the one using `nowcast_mom_summary`), add:
```python
            _ab = self.mom_driver_ablation_summary()
            if _ab:
                abl = QLabel('<b>MoM driver-only ablation</b><br>' + _ab)
                abl.setWordWrap(True)
                col.addWidget(abl)
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
git commit -m "feat(ui): show the MoM driver-only ablation result in the Accuracy view"
```

---

## Task 5: Record the ablation outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-09-ph-cpi-mom-driver-ablation-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 3 Step 3**

Replace the `[confirms / does not confirm]`/`[method]`/`[x]`/`[p]`/`[n]` placeholders in §9 with the measured ablation result, and pick the matching interpretation bullet (confirmed vs absent). Also add a one-sentence cross-reference into the MoM spec's §9 (`docs/superpowers/specs/2026-06-08-ph-cpi-mom-nowcast-design.md`) stating whether the driver edge was confirmed. Copy numbers from `mom_driver_ablation_table.json`; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-09-ph-cpi-mom-driver-ablation-design.md docs/superpowers/specs/2026-06-08-ph-cpi-mom-nowcast-design.md
git commit -m "docs: record measured MoM driver-only ablation result"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Ablation result legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; a=load_report()['mom_driver_ablation']; print('driver_edge', a.get('driver_edge'), '| best', a.get('best_method'), 'vs', a.get('best_naive'), '| skill', a.get('best_skill_vs_naive'))"`
Expected: prints `driver_edge True/False` + the winning method + baseline + skill — the honest confirmation result.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3 definitions (ablation feature set, candidate methods, verdict, interpretation label) → Task 1. §4.1 `methods=` param → Task 1. §4.2 `run_driver_only_ablation` + `driver_edge` → Task 1. §4.3 integration (report/run/view) → Tasks 2, 3, 4. §6 error handling (insufficient_data, `errors='ignore'`) → Task 1 (`drop(..., errors='ignore')`) + tests. §7 testing (restricted methods, driver-edge present/absent, no-prev_mom frame) → Task 1. §8 deliverables → all. §9 write-up → Task 5.

**Placeholder scan:** §9's bracketed markers are filled at Task 5 from real output (empirical; accepted pattern). No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `run_mom_nowcast(min_train, baseline_pool, frame, methods=None)` — `methods` added in Task 1, used by `run_driver_only_ablation` (Task 1) and unchanged for existing callers (default None → PANEL_METHODS). `run_driver_only_ablation(min_train, frame=None) -> {...standard MoM dict..., 'driver_edge': bool}` consistent across Tasks 1, 3, 4. `build_report(... mom_driver_ablation=)` matches Task 2 (def) and Task 3 (call). Report key `mom_driver_ablation` consistent across Tasks 2, 3, 4. `mom_verdict` reused unchanged — restricting `methods` to exclude arima/ets makes `{ridge, hgb}` the only candidates, which is the intended isolation (no change to `mom_verdict` needed). Note: `test_driver_ablation_detects_driver_edge` asserts `rmse_by_method` keys are exactly the 5 requested methods — consistent with the `methods=` restriction.
```
