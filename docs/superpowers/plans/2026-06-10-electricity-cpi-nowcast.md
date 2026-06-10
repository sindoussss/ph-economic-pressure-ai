# MoM Electricity-CPI Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nowcast month-on-month PH Electricity-CPI inflation before the PSA release using free energy-commodity prices + FX, with DM-vs-strongest-naive + a driver-only ablation + a preliminary-data robustness re-test — completing the predictability map.

**Architecture:** Reuse the generalized PSA PX-Web fetch (COICOP prefix `04.5.1` = Electricity) for the gold; a free Yahoo Brent+natgas+FX predictor panel; build a frame directly and reuse the existing `run_mom_nowcast` + `run_driver_only_ablation` (unchanged) + robustness re-test — identical structure to `food_nowcast`. Verdict reported either way.

**Tech Stack:** Python 3.10, requests, pandas, scikit-learn, statsmodels, pytest. Data: PSA OpenSTAT PX-Web, Yahoo Finance.

**Spec:** `docs/superpowers/specs/2026-06-10-electricity-cpi-nowcast-design.md`.

**Prereqs (on `master`):**
- `benchmark/psa_cpi.py` — generalized `_fetch_px_table(url, first_year, coicop_prefix)` (resolves commodity by `_resolve_commodity_id` matching label prefix `f'{coicop_prefix} -'`), `_resolve_commodity_id`, `fetch_food_cpi`/`fetch_transport_cpi`, `load_food_*`/`load_transport_*`, `HERE`, `PSA_TRANSPORT_URL_BACKCAST`, `PSA_TRANSPORT_URL_CURRENT`, `cpi_to_mom` imported. Confirmed live: commodity value `04.5.1 - Electricity (ND)` exists; prefix `04.5.1` resolves it (the more granular `04.5.1.0` starts with `04.5.1.0 -`, so it is NOT matched).
- `benchmark/nowcast.py` — reused UNCHANGED: `run_mom_nowcast(min_train, frame=…)` uses all non-`target` cols as features; `run_driver_only_ablation(min_train, frame=…)` drops `prev_mom`. Reserved frame cols: `prev_mom`, `target`.
- `benchmark/food_nowcast.py` — the structural template (build frame directly + full/robust ablation + `driver_edge_robust`).
- `benchmark/refresh_data.py::_yahoo_monthly(ticker, rng='10y')`, `HERE`. `build_food_features` is the template.
- `benchmark/report.py::build_report(... food_nowcast=None)` (trailing optional kwargs + `REQUIRED_KEYS` + `ARTIFACTS` + `load_report`); `benchmark/run.py` `main()` (runs food_nowcast); `ui/accuracy_view.py::food_nowcast_summary` (template).

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` first. `global_fuel_prices.xlsx` is gitignored — never stage it.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/electricity-cpi-nowcast
```

---

## File Structure
**Create:** `benchmark/electricity_nowcast.py`, `benchmark/data/psa_electricity_cpi_monthly.csv` (Task 2), `benchmark/data/electricity_features_monthly.csv` (Task 3); tests `test_electricity_nowcast.py` (+ a `psa_cpi` test addition).
**Modify:** `benchmark/psa_cpi.py`, `benchmark/refresh_data.py`, `benchmark/report.py`, `benchmark/run.py`, `ui/accuracy_view.py`; tests `test_psa_cpi.py`, `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: PSA electricity fetch + loaders

**Files:**
- Modify: `ph_economic_ai/benchmark/psa_cpi.py`
- Modify: `ph_economic_ai/tests/test_psa_cpi.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_psa_cpi.py`:
```python
from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom


def test_load_electricity_mom(tmp_path):
    import pytest
    p = tmp_path / 'elec.csv'
    p.write_text('date,electricity_cpi\n2018-01,100.0\n2018-02,103.0\n2018-03,103.0\n',
                 encoding='utf-8')
    mom = load_electricity_mom(p)
    assert mom['2018-02'] == pytest.approx(3.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -k electricity -v`
Expected: FAIL — `ImportError: cannot import name 'load_electricity_mom'`.

- [ ] **Step 3: Implement in psa_cpi.py**

Add near the food equivalents:
```python
ELECTRICITY_CSV = HERE / 'data' / 'psa_electricity_cpi_monthly.csv'


def fetch_electricity_cpi(out_csv: Path = ELECTRICITY_CSV) -> None:
    """Fetch monthly Electricity (COICOP 04.5.1) CPI from PSA OpenSTAT -> CSV."""
    series = {}
    series.update(_fetch_px_table(PSA_TRANSPORT_URL_BACKCAST, 1994, '04.5.1'))
    series.update(_fetch_px_table(PSA_TRANSPORT_URL_CURRENT, 2018, '04.5.1'))  # current wins
    if len(series) < 50:
        raise ValueError(f'electricity CPI series too short ({len(series)} rows)')
    df = (pd.DataFrame(sorted(series.items()), columns=['date', 'electricity_cpi'])
          .sort_values('date'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Wrote psa_electricity_cpi_monthly.csv ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def load_electricity_cpi(csv_path: Path = ELECTRICITY_CSV) -> pd.Series:
    """Monthly Electricity CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['electricity_cpi'].astype(float).values,
                  index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_electricity_mom(csv_path: Path = ELECTRICITY_CSV) -> pd.Series:
    """Month-over-month Electricity inflation % from the committed gold."""
    return cpi_to_mom(load_electricity_cpi(csv_path))
```
(If `fetch_food_cpi`'s real merge idiom differs, mirror it exactly with prefix `'04.5.1'` and column `electricity_cpi`.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/tests/test_psa_cpi.py
git commit -m "feat(benchmark): PSA electricity-CPI fetch + loaders (COICOP 04.5.1)"
```

---

## Task 2: Fetch + commit Electricity gold (network, one-off)

**Files:**
- Create (by running): `ph_economic_ai/benchmark/data/psa_electricity_cpi_monthly.csv`

- [ ] **Step 1: Fetch the electricity gold (network)**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import fetch_electricity_cpi; fetch_electricity_cpi()"`
Expected: `Wrote psa_electricity_cpi_monthly.csv (N rows, YYYY-MM..YYYY-MM)`. N may be ~89 (2018+) if the backcast lacks `04.5.1`, or ~350–390 if it has it — either is acceptable. If the `04.5.1` label isn't found, read the error (lists labels) and report; do NOT fabricate.

- [ ] **Step 2: Sanity-check**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import load_electricity_cpi, load_electricity_mom; c=load_electricity_cpi(); m=load_electricity_mom(); print('rows',len(c),c.index[0],c.index[-1]); print('2018 mean', round(c[[i for i in c.index if i.startswith('2018')]].mean(),1)); print('mom n',len(m),'std',round(m.std(),3))"`
Expected: **2018 mean ≈ 100** (base year confirms the right series); sensible MoM std (electricity is volatile, so std may be larger than food). If 2018 mean is far from 100, the wrong group was selected — STOP and report.

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/benchmark/data/psa_electricity_cpi_monthly.csv
git status --short
git commit -m "feat(benchmark): freeze PSA Electricity-CPI gold (PX-Web, COICOP 04.5.1)"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 3: Energy predictor panel (network, one-off)

**Files:**
- Modify: `ph_economic_ai/benchmark/refresh_data.py`
- Create (by running): `ph_economic_ai/benchmark/data/electricity_features_monthly.csv`

- [ ] **Step 1: Append `build_electricity_features` to refresh_data.py**

```python
ELECTRICITY_FEATURES_OUT = HERE / 'data' / 'electricity_features_monthly.csv'


def build_electricity_features(rng: str = 'max') -> None:
    """Free energy predictor panel for the Electricity-CPI nowcast:
    Yahoo Brent + natural gas + USD/PHP -> data/electricity_features_monthly.csv."""
    cols = {'BZ=F': 'oil_price', 'NG=F': 'natgas', 'PHP=X': 'usd_php'}
    parts = [_yahoo_monthly(t, rng).rename(name) for t, name in cols.items()]
    base = pd.concat(parts, axis=1).dropna().reset_index().rename(columns={'index': 'date'})
    base = base.sort_values('date')
    ELECTRICITY_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(ELECTRICITY_FEATURES_OUT, index=False)
    print(f'Wrote electricity_features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')
```

- [ ] **Step 2: Build the panel (network)**

Run: `python -c "from ph_economic_ai.benchmark.refresh_data import build_electricity_features; build_electricity_features()"`
Expected: `Wrote electricity_features_monthly.csv (N rows, ~2007-xx..2026-xx)`, N ≈ 170–220 (Brent starts ~2007). If a ticker fails, retry once; if persistent, STOP and report.

- [ ] **Step 3: Verify overlap with the electricity gold**

Run: `python -c "import pandas as pd; from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom; f=pd.read_csv('ph_economic_ai/benchmark/data/electricity_features_monthly.csv',dtype={'date':str}).set_index('date'); m=load_electricity_mom(); ov=f.index.intersection(m.index); print('cols',list(f.columns)); print('feat',len(f),'| elec MoM',len(m),'| overlap',len(ov),'=> backtest n ~',len(ov)-24)"`
Expected: columns `['oil_price','natgas','usd_php']`; overlap ~120+ (depends on whether electricity gold starts 1994 or 2018).

- [ ] **Step 4: Commit**

```bash
git add ph_economic_ai/benchmark/refresh_data.py ph_economic_ai/benchmark/data/electricity_features_monthly.csv
git status --short
git commit -m "feat(benchmark): free energy predictor panel for electricity nowcast"
```
Confirm xlsx NOT staged.

---

## Task 4: Electricity nowcast runner (with robustness)

**Files:**
- Create: `ph_economic_ai/benchmark/electricity_nowcast.py`
- Test: `ph_economic_ai/tests/test_electricity_nowcast.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_electricity_nowcast.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.electricity_nowcast as en


def test_run_electricity_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2008-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    oil = 60 + np.cumsum(rng.normal(0, 1, n))
    feats = pd.DataFrame({
        'oil_price': oil,
        'natgas': 3 + np.cumsum(rng.normal(0, 0.1, n)),
        'usd_php': 50 + np.cumsum(rng.normal(0, 0.1, n)),
    }, index=idx)
    mom = pd.Series(0.3 * np.r_[0.0, np.diff(oil)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(en, 'load_electricity_mom', lambda: mom)
    res = en.run_electricity_nowcast(min_train=24, features=feats, prelim_months=6)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']
    assert 'panel' not in res['mom']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_electricity_nowcast.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.electricity_nowcast`.

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/electricity_nowcast.py`:
```python
"""MoM Electricity-CPI nowcast — energy pass-through, tested honestly.

Builds a frame from free energy predictors (Brent, natural gas, FX) and the PSA
Electricity-CPI target, then reuses the existing MoM nowcast runners (unchanged)
with the same preliminary-data robustness re-test as the Food nowcast.
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation, run_mom_nowcast
from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom

ELECTRICITY_FEATURES_CSV = Path(__file__).parent / 'data' / 'electricity_features_monthly.csv'


def load_electricity_features(csv_path: Path = ELECTRICITY_FEATURES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={'date': str}).set_index('date').sort_index()


def _build_electricity_frame(features: pd.DataFrame) -> pd.DataFrame:
    tgt = load_electricity_mom()
    base = pd.DataFrame({
        'oil': features['oil_price'], 'natgas': features['natgas'],
        'fx': features['usd_php'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base['prev_mom'] = base['target'].shift(1)
    cols = ['oil', 'natgas', 'fx', 'prev_mom', 'target']
    return base[cols].dropna()


def run_electricity_nowcast(min_train: int = 24, features=None, prelim_months: int = 6) -> dict:
    """Electricity-MoM nowcast + driver-only ablation + trailing-preliminary
    robustness re-test. `driver_edge_robust` is the canonical verdict."""
    feats = load_electricity_features() if features is None else features
    frame = _build_electricity_frame(feats)
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

Run: `python -m pytest ph_economic_ai/tests/test_electricity_nowcast.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/electricity_nowcast.py ph_economic_ai/tests/test_electricity_nowcast.py
git commit -m "feat(benchmark): electricity-CPI nowcast runner (energy drivers + robustness)"
```

---

## Task 5: Report key

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_electricity_nowcast():
    rep = build_report(
        date_range=('2007-08', '2025-05'), n_months=200,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 200},
        data_hash='abc123',
        electricity_nowcast={'n': 170, 'driver_edge': False, 'driver_edge_robust': False,
                             'mom': {'verdict': 'no_better_than_naive'},
                             'driver_ablation': {'verdict': 'no_better_than_naive',
                                                 'driver_edge': False},
                             'robust': {'prelim_months_dropped': 6, 'n': 164,
                                        'driver_edge': False,
                                        'driver_ablation': {'verdict': 'no_better_than_naive'}}},
    )
    assert rep['electricity_nowcast']['n'] == 170
    assert 'electricity_nowcast' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_electricity_nowcast -v`
Expected: FAIL — unexpected keyword argument `electricity_nowcast`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`: (a) append `'electricity_nowcast'` to `REQUIRED_KEYS`; (b) add `electricity_nowcast=None` as the LAST parameter of `build_report` (after `food_nowcast=None`); (c) just before `'limitations'`:
```python
        'electricity_nowcast': electricity_nowcast if electricity_nowcast is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the electricity-CPI nowcast result"
```

---

## Task 6: Run the electricity nowcast in run.py (real result)

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`

- [ ] **Step 1: Wire into run.py**

In `main()`, after the food-nowcast block and BEFORE `rep = report.build_report(`, add:
```python
    # -- MoM Electricity-CPI nowcast (energy pass-through) --
    try:
        from ph_economic_ai.benchmark import electricity_nowcast as elec_mod
        elec_res = elec_mod.run_electricity_nowcast(MIN_TRAIN)
        _em = elec_res['mom']
        print(f"Electricity nowcast (n={elec_res['n']}): mom={_em['verdict']} "
              f"best={_em.get('best_method')} skill={_em.get('best_skill_vs_naive')} "
              f"DM p={_em.get('dm_p')} | driver_edge_robust={elec_res['driver_edge_robust']}")
    except FileNotFoundError:
        elec_res = {'verdict': 'not_run', 'reason': 'electricity gold/features missing'}
        print('Electricity nowcast: not_run (gold or features CSV missing)')
```
In the `report.build_report(` call add:
```python
        electricity_nowcast=elec_res,
```
After the existing artifact writes add:
```python
    import json as _json9
    (report.ARTIFACTS / 'electricity_nowcast_table.json').write_text(
        _json9.dumps(elec_res, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Run end-to-end (a few minutes)**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints an `Electricity nowcast (n=...): ...` line (NOT not_run); writes `electricity_nowcast_table.json`; no errors. Fix only run.py wiring on error; if a real bug surfaces in electricity_nowcast/nowcast, STOP and report.

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; import json; print(json.dumps(load_report()['electricity_nowcast'], indent=2))"`
Record `n`, the `mom` dict, `driver_edge` (full), `robust`, and `driver_edge_robust`. **This is the answer** — note whether electricity shows a robust edge or is efficient. Report either way.

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (suite currently fully green). Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/electricity_nowcast_table.json
git status --short
git commit -m "feat(benchmark): run the electricity-CPI nowcast; report the result"
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
        'electricity_nowcast': {'n': 170, 'driver_edge': False, 'driver_edge_robust': False,
                                'mom': {'verdict': 'no_better_than_naive', 'best_method': 'arima'},
                                'driver_ablation': {'verdict': 'no_better_than_naive',
                                                    'driver_edge': False},
                                'robust': {'prelim_months_dropped': 6, 'n': 164,
                                           'driver_edge': False,
                                           'driver_ablation': {'verdict': 'no_better_than_naive'}}},
```
Add the test:
```python
def test_view_shows_electricity_nowcast(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.electricity_nowcast_summary()
    assert '170' in s
    assert 'driver_edge_robust=False' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_electricity_nowcast -v`
Expected: FAIL — no attribute `electricity_nowcast_summary`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add the method (next to `food_nowcast_summary`):
```python
    def electricity_nowcast_summary(self) -> str:
        if not self._report:
            return ''
        E = self._report.get('electricity_nowcast') or {}
        if not E or E.get('verdict') == 'not_run':
            return ''
        mom = E.get('mom') or {}
        robust = bool(E.get('driver_edge_robust'))
        driver_txt = ('significant energy driver edge' if robust
                      else 'no robust energy driver edge')
        caveat = ''
        if E.get('driver_edge') and not robust:
            caveat = ' (full-sample edge was a preliminary-data artifact)'
        return (f"Electricity-CPI nowcast (n={E.get('n')}): MoM {mom.get('verdict')} "
                f"(best {mom.get('best_method')}); "
                f"{driver_txt} (driver_edge_robust={robust}){caveat}.")
```
(b) In `_build`, inside `if self._report is not None:`, after the food-nowcast block, add:
```python
            _en = self.electricity_nowcast_summary()
            if _en:
                enl = QLabel('<b>Electricity-CPI nowcast (energy→inflation)</b><br>' + _en)
                enl.setWordWrap(True)
                col.addWidget(enl)
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
git commit -m "feat(ui): show the electricity-CPI nowcast in the Accuracy view"
```

---

## Task 8: Record the outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-electricity-cpi-nowcast-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 6 Step 3**

Replace the `[n]`/`[m]`/`[x]`/`[p]`/`[bool]`/`[k]`/`[nr]` markers with the measured values from `electricity_nowcast_table.json`, and select the matching interpretation bullet (robust True → genuine edge; efficient/not robust → honest negative). Copy numbers verbatim; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-10-electricity-cpi-nowcast-design.md
git commit -m "docs: record measured electricity-CPI nowcast result"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass.

- [ ] **Result legible from report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; E=load_report()['electricity_nowcast']; print('n', E.get('n'), '| mom', E['mom']['verdict'], '| energy driver_edge_robust', E.get('driver_edge_robust'))"`
Expected: prints the electricity nowcast verdict + whether the energy driver edge is robust — the honest answer either way.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 gold loader (PSA `04.5.1`) → Tasks 1–2; §3.2 energy predictor panel → Task 3; §4.1 psa fetch+loaders → Task 1; §4.2 build_electricity_features → Task 3; §4.3 electricity_nowcast runner + robustness → Task 4; §4.4 report/run/view → Tasks 5–7; §6 error handling (not_run guard, label-resolution error, reserved names) → Tasks 1, 4, 6; §7 testing → Tasks 1, 4, 5, 7; §8 deliverables → all; §9 write-up → Task 8.

**Placeholder scan:** §9 markers filled at Task 8 from real output (empirical). Tasks 2–3 have no unit tests by design (network refresh). No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `load_electricity_mom()` (Task 1) reused in Task 4 (`en.load_electricity_mom` monkeypatch + `_build_electricity_frame`). `run_electricity_nowcast(min_train=24, features=None, prelim_months=6) -> {n, mom, driver_ablation, driver_edge, robust, driver_edge_robust}` consistent across Tasks 4, 6, 7. Frame cols `['oil','natgas','fx','prev_mom','target']` — only `prev_mom`/`target` reserved by the runners. `build_report(... electricity_nowcast=)` matches Task 5 (def) and Task 6 (call); report key `electricity_nowcast` consistent across Tasks 5–7. The view guards both the `not_run` dict and the full result. `_fetch_px_table(url, first_year, coicop_prefix)` (3-arg, generalized) used in Task 1 with prefix `'04.5.1'`.
```
