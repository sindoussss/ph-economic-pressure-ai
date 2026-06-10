# MoM Transport-CPI Nowcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nowcast month-on-month PH **Transport** CPI inflation before the PSA release using free within-month fuel/FX data, and test honestly (DM vs the strongest naive baseline + a driver-only ablation) whether the fuel→inflation pass-through is a significant, exploitable nowcast.

**Architecture:** A free, automated PSA OpenSTAT PX-Web loader produces a committed monthly Transport-CPI gold series; the existing MoM nowcast runners (`build_nowcast_frame`, `run_mom_nowcast`, `run_driver_only_ablation`) are reused unchanged with this new target and the existing long feature panel. Report/run/view extended; verdict reported either way.

**Tech Stack:** Python 3.10, requests, pandas, numpy, scikit-learn, statsmodels, pytest. Data: PSA OpenSTAT PX-Web (JSON), Yahoo Finance (existing).

**Spec:** `docs/superpowers/specs/2026-06-10-transport-cpi-nowcast-design.md`.

**Prereqs (on branch — see first task):**
- `benchmark/nowcast.py::build_nowcast_frame(target_loader=None, prev_col='prev_inflation', features=None)`, `run_mom_nowcast(min_train, baseline_pool, frame, methods)`, `run_driver_only_ablation(min_train, frame=None)`. The frame columns are `['oil','fx','fuel',prev_col,'target']`; `build_nowcast_frame` maps `features['oil_price']→oil, ['usd_php']→fx, ['gas_price']→fuel` and joins `target_loader()` as `target`.
- `benchmark/targets.py::cpi_to_mom(cpi_index_series) -> pd.Series` (MoM inflation %, drops first row).
- `benchmark/longsample.py::load_long_features()` → long feature DataFrame indexed `YYYY-MM` with `oil_price,usd_php,gas_price,demand_index`.
- `benchmark/report.py::build_report(...)` with trailing optional kwargs + `REQUIRED_KEYS` tuple + `ARTIFACTS` path + `load_report()`.
- `benchmark/run.py` `main()` (has `MIN_TRAIN`, builds `rep`, writes artifacts). `ui/accuracy_view.py::AccuracyView` (`self._report`, `*_summary` methods).
- Confirmed live: PSA PX-Web table at `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/2018NEW/0012M4ACP28.px` ("CPI by Commodity Group, 2018=100, 1994–present") is reachable and returns JSON metadata on GET.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing. `global_fuel_prices.xlsx` is gitignored — never stage it.

**Task 0 (branch):** Before Task 1, create the working branch:
```bash
git checkout master && git pull && git checkout -b feature/transport-cpi-nowcast
```

---

## File Structure
**Create:** `benchmark/psa_cpi.py`, `benchmark/transport_nowcast.py`, `benchmark/data/psa_transport_cpi_monthly.csv` (by running Task 2); tests `test_psa_cpi.py`, `test_transport_nowcast.py`.
**Modify:** `benchmark/report.py`, `benchmark/run.py`, `ui/accuracy_view.py`; tests `test_report.py`, `test_accuracy_view.py`; spec §9.

---

## Task 1: PSA CPI parsing helpers (pure, TDD)

**Files:**
- Create: `ph_economic_ai/benchmark/psa_cpi.py`
- Test: `ph_economic_ai/tests/test_psa_cpi.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_psa_cpi.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.benchmark.psa_cpi import _label_to_ym, load_transport_cpi, load_transport_mom


def test_label_to_ym_handles_formats():
    assert _label_to_ym('1994M01') == '1994-01'
    assert _label_to_ym('1994 M01') == '1994-01'
    assert _label_to_ym('January 1994') == '1994-01'
    assert _label_to_ym('1994 January') == '1994-01'
    assert _label_to_ym('2018-03') == '2018-03'
    assert _label_to_ym('not a date') is None


def test_load_transport_cpi_and_mom(tmp_path):
    p = tmp_path / 't.csv'
    p.write_text('date,transport_cpi\n2018-01,100.0\n2018-02,101.0\n2018-03,101.0\n',
                 encoding='utf-8')
    idx = load_transport_cpi(p)
    assert list(idx.index) == ['2018-01', '2018-02', '2018-03']
    assert idx['2018-02'] == pytest.approx(101.0)
    mom = load_transport_mom(p)
    # MoM % drops the first row; Feb = (101/100 - 1)*100 = 1.0, Mar = 0.0
    assert mom['2018-02'] == pytest.approx(1.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.psa_cpi`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/psa_cpi.py`:

```python
"""PSA OpenSTAT Transport-CPI gold loader (free, official, citable).

Fetches monthly Transport CPI (by commodity group, 2018=100) from the PSA
OpenSTAT PX-Web API and freezes it as a committed CSV. The MoM transform of this
series is the nowcast target for the fuel->inflation pass-through.
"""
import re
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.targets import cpi_to_mom

HERE = Path(__file__).parent
TRANSPORT_CSV = HERE / 'data' / 'psa_transport_cpi_monthly.csv'

_MONTHS = {m.lower(): i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}
_ABBR = {m[:3].lower(): i for m, i in _MONTHS.items()}


def _label_to_ym(label: str):
    """Normalise a PX-Web time label to 'YYYY-MM', or None if unparseable.

    Handles: '1994M01', '1994 M01', 'January 1994', '1994 January', '2018-03'."""
    s = str(label).strip()
    m = re.fullmatch(r'(\d{4})-(\d{2})', s)
    if m:
        return f'{m.group(1)}-{m.group(2)}'
    m = re.fullmatch(r'(\d{4})\s*M(\d{1,2})', s, re.IGNORECASE)
    if m:
        return f'{m.group(1)}-{int(m.group(2)):02d}'
    m = re.fullmatch(r'([A-Za-z]+)\s+(\d{4})', s)        # Month YYYY
    if m:
        mo = _MONTHS.get(m.group(1).lower()) or _ABBR.get(m.group(1)[:3].lower())
        if mo:
            return f'{m.group(2)}-{mo:02d}'
    m = re.fullmatch(r'(\d{4})\s+([A-Za-z]+)', s)        # YYYY Month
    if m:
        mo = _MONTHS.get(m.group(2).lower()) or _ABBR.get(m.group(2)[:3].lower())
        if mo:
            return f'{m.group(1)}-{mo:02d}'
    return None


def load_transport_cpi(csv_path: Path = TRANSPORT_CSV) -> pd.Series:
    """Monthly Transport CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['transport_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_transport_mom(csv_path: Path = TRANSPORT_CSV) -> pd.Series:
    """Month-over-month Transport inflation % from the committed gold."""
    return cpi_to_mom(load_transport_cpi(csv_path))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/tests/test_psa_cpi.py
git commit -m "feat(benchmark): PSA transport-CPI parsing helpers (label->ym, loaders)"
```

---

## Task 2: Fetch + commit the Transport-CPI gold (network, one-off)

**Files:**
- Modify: `ph_economic_ai/benchmark/psa_cpi.py` (add `fetch_transport_cpi`)
- Create (by running): `ph_economic_ai/benchmark/data/psa_transport_cpi_monthly.csv`

(No unit test — network refresh, matching `build_long_features`/`build_world_bank_csv`.)

- [ ] **Step 1: Inspect the live PX-Web metadata (discovery)**

Run:
```bash
python -c "import requests,json; H={'User-Agent':'Mozilla/5.0'}; u='https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/2018NEW/0012M4ACP28.px'; m=requests.get(u,headers=H,timeout=20).json(); [print(v['code'],'|',v['text'],'| values:',list(zip(v['values'][:4],v['valueTexts'][:4])),'...') for v in m['variables']]"
```
Read the output: note (a) the **commodity-group** variable's `code` and the value id whose text is "Transport"; (b) the **geolocation/area** variable's code and the value id for "Philippines"; (c) the **time** variable's code and the format of its `valueTexts` (e.g. `1994 M01`). If the time label format is NOT covered by `_label_to_ym` (Task 1), extend `_label_to_ym` and re-run its test before continuing.

- [ ] **Step 2: Add `fetch_transport_cpi` to psa_cpi.py**

Append to `ph_economic_ai/benchmark/psa_cpi.py`:

```python
PSA_TRANSPORT_URL = ('https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/'
                     '2018NEW/0012M4ACP28.px')
_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Content-Type': 'application/json'}


def _find_value(var: dict, *needles: str):
    """Return the value id whose label contains any needle (case-insensitive)."""
    for vid, txt in zip(var['values'], var['valueTexts']):
        low = txt.strip().lower()
        if any(n in low for n in needles):
            return vid
    raise ValueError(f"no value matching {needles} in '{var['text']}'; "
                     f"available: {var['valueTexts'][:12]}")


def _parse_jsonstat_series(payload: dict) -> dict:
    """json-stat2 with a single non-time selection -> {YYYY-MM: value}."""
    ids, sizes, dim, val = (payload['id'], payload['size'],
                            payload['dimension'], payload['value'])
    ti = next(i for i, s in enumerate(sizes) if s > 1)   # the time dimension
    cat = dim[ids[ti]]['category']
    index, label = cat['index'], cat.get('label', {})
    pairs = sorted(index.items(), key=lambda kv: kv[1])  # (key, position)
    out = {}
    for key, pos in pairs:
        v = val[pos] if pos < len(val) else None
        if v is None:
            continue
        ym = _label_to_ym(label.get(key, key))
        if ym:
            out[ym] = float(v)
    return out


def fetch_transport_cpi(out_csv: Path = TRANSPORT_CSV) -> None:
    """Fetch monthly Transport CPI from PSA OpenSTAT and freeze to CSV."""
    import json
    import requests

    meta = requests.get(PSA_TRANSPORT_URL, headers={'User-Agent': 'Mozilla/5.0'},
                        timeout=30).json()
    by_text = {v['text'].strip().lower(): v for v in meta['variables']}

    def pick(*needles):
        for t, v in by_text.items():
            if any(n in t for n in needles):
                return v
        raise ValueError(f"no variable matching {needles}; have {list(by_text)}")

    comm = pick('commodity')
    geo = pick('geolocation', 'region', 'area', 'location')
    time_var = pick('month', 'period', 'time', 'date')
    other = [v for v in meta['variables']
             if v['code'] not in {comm['code'], geo['code'], time_var['code']}]

    query = [
        {'code': geo['code'], 'selection': {'filter': 'item',
         'values': [_find_value(geo, 'philippines')]}},
        {'code': comm['code'], 'selection': {'filter': 'item',
         'values': [_find_value(comm, 'transport')]}},
        {'code': time_var['code'], 'selection': {'filter': 'all', 'values': ['*']}},
    ]
    for v in other:  # any extra variable -> take its first value
        query.append({'code': v['code'],
                      'selection': {'filter': 'item', 'values': [v['values'][0]]}})

    body = {'query': query, 'response': {'format': 'json-stat2'}}
    resp = requests.post(PSA_TRANSPORT_URL, headers=_HEADERS, data=json.dumps(body),
                         timeout=60)
    resp.raise_for_status()
    series = _parse_jsonstat_series(resp.json())
    if len(series) < 100:
        raise ValueError(f'transport CPI series too short ({len(series)} rows) — '
                         'check PX-Web selection')
    df = (pd.DataFrame(sorted(series.items()), columns=['date', 'transport_cpi'])
          .sort_values('date'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Wrote psa_transport_cpi_monthly.csv ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})')
```

- [ ] **Step 3: Run the fetch (network)**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import fetch_transport_cpi; fetch_transport_cpi()"`
Expected: prints `Wrote psa_transport_cpi_monthly.csv (N rows, 1994-01..YYYY-MM)` with N ≈ 350–390. If it raises on value/variable matching, read the error (it lists available labels), adjust the `_find_value`/`pick` needles to the real labels, and re-run. Do NOT fabricate the CSV.

- [ ] **Step 4: Sanity-check the gold**

Run: `python -c "from ph_economic_ai.benchmark.psa_cpi import load_transport_cpi, load_transport_mom; c=load_transport_cpi(); m=load_transport_mom(); print('rows',len(c),'range',c.index[0],c.index[-1]); print('2018 mean (~100?)', round(c[[i for i in c.index if i.startswith(\"2018\")]].mean(),1)); print('mom n',len(m),'mom std',round(m.std(),3))"`
Expected: ~350–390 rows; the 2018 mean is near 100 (the base year); MoM has a sensible non-zero std. If the 2018 mean is far from 100, the wrong commodity group/series was selected — fix the selection and re-fetch.

- [ ] **Step 5: Commit (code + committed gold CSV)**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/benchmark/data/psa_transport_cpi_monthly.csv
git status --short
git commit -m "feat(benchmark): fetch + freeze PSA Transport-CPI gold (PX-Web, 1994-present)"
```
Confirm `global_fuel_prices.xlsx` NOT staged.

---

## Task 3: Transport nowcast runner

**Files:**
- Create: `ph_economic_ai/benchmark/transport_nowcast.py`
- Test: `ph_economic_ai/tests/test_transport_nowcast.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_transport_nowcast.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.transport_nowcast as tn


def test_run_transport_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2010-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(7)
    gas = 45 + np.cumsum(rng.normal(0, 0.5, n))
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    mom = pd.Series(0.5 * np.r_[0.0, np.diff(gas)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(tn, 'load_transport_mom', lambda: mom)
    res = tn.run_transport_nowcast(min_train=24, features=feats)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge'], bool)
    assert 'panel' not in res['mom']
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_transport_nowcast.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.benchmark.transport_nowcast`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/benchmark/transport_nowcast.py`:

```python
"""MoM Transport-CPI nowcast — the fuel->inflation pass-through, tested honestly.

Reuses the existing MoM nowcast runners with the PSA Transport-CPI target and the
long feature panel. The driver-only ablation is the key test: does observable
within-month fuel nowcast transport inflation beyond persistence?
"""
from ph_economic_ai.benchmark.nowcast import (
    build_nowcast_frame, run_driver_only_ablation, run_mom_nowcast,
)
from ph_economic_ai.benchmark.psa_cpi import load_transport_mom
from ph_economic_ai.benchmark.longsample import load_long_features


def run_transport_nowcast(min_train: int = 24, features=None) -> dict:
    """Run the Transport-MoM nowcast + driver-only ablation. Returns
    {n, mom, driver_ablation, driver_edge} with heavy internals dropped."""
    feats = load_long_features() if features is None else features
    frame = build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom',
                                features=feats)
    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    drop = ('panel', 'calibration')
    return {
        'n': int(mom.get('n', len(frame))),
        'mom': {k: v for k, v in mom.items() if k not in drop},
        'driver_ablation': {k: v for k, v in abl.items() if k not in drop},
        'driver_edge': bool(abl.get('driver_edge', False)),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_transport_nowcast.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/transport_nowcast.py ph_economic_ai/tests/test_transport_nowcast.py
git commit -m "feat(benchmark): transport-CPI nowcast runner (reuses MoM pipeline)"
```

---

## Task 4: Report key

**Files:**
- Modify: `ph_economic_ai/benchmark/report.py`
- Modify: `ph_economic_ai/tests/test_report.py` (append)

- [ ] **Step 1: Append the test**

```python
def test_report_includes_transport_nowcast():
    rep = build_report(
        date_range=('2007-08', '2025-05'), n_months=200,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 200},
        data_hash='abc123',
        transport_nowcast={'n': 180, 'driver_edge': True,
                           'mom': {'verdict': 'beats_best_naive', 'best_method': 'ridge',
                                   'best_skill_vs_naive': 0.2, 'dm_p': 0.004},
                           'driver_ablation': {'verdict': 'beats_best_naive',
                                               'driver_edge': True}},
    )
    assert rep['transport_nowcast']['driver_edge'] is True
    assert 'transport_nowcast' in REQUIRED_KEYS
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_report.py::test_report_includes_transport_nowcast -v`
Expected: FAIL — unexpected keyword argument `transport_nowcast`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/benchmark/report.py`:
(a) Append `'transport_nowcast'` to the `REQUIRED_KEYS` tuple.
(b) Add `transport_nowcast=None` as the LAST parameter of `build_report`.
(c) In the returned dict, add just before `'limitations'`:
```python
        'transport_nowcast': transport_nowcast if transport_nowcast is not None else {},
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_report.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/report.py ph_economic_ai/tests/test_report.py
git commit -m "feat(benchmark): report carries the transport-CPI nowcast result"
```

---

## Task 5: Run the transport nowcast in run.py (real result)

**Files:**
- Modify: `ph_economic_ai/benchmark/run.py`

- [ ] **Step 1: Wire into run.py**

In `ph_economic_ai/benchmark/run.py::main()`, after the existing MoM/longsample blocks and BEFORE the `rep = report.build_report(` call, add:
```python
    # -- MoM Transport-CPI nowcast (fuel -> inflation pass-through) --
    try:
        from ph_economic_ai.benchmark import transport_nowcast as transport_mod
        transport_res = transport_mod.run_transport_nowcast(MIN_TRAIN)
        _tm = transport_res['mom']
        print(f"Transport nowcast (n={transport_res['n']}): mom={_tm['verdict']} "
              f"best={_tm.get('best_method')} skill={_tm.get('best_skill_vs_naive')} "
              f"DM p={_tm.get('dm_p')} | driver_edge={transport_res['driver_edge']}")
    except FileNotFoundError:
        transport_res = {'verdict': 'not_run', 'reason': 'transport gold missing'}
        print('Transport nowcast: not_run (psa_transport_cpi_monthly.csv missing)')
```
In the `report.build_report(` call, add:
```python
        transport_nowcast=transport_res,
```
After the existing artifact writes, add:
```python
    import json as _json7
    (report.ARTIFACTS / 'transport_nowcast_table.json').write_text(
        _json7.dumps(transport_res, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Run end-to-end on real data**

Run: `python -m ph_economic_ai.benchmark.run`
Expected: prints a `Transport nowcast (n=...): ...` line (n large, NOT `not_run` since the gold exists from Task 2); writes `transport_nowcast_table.json`; no errors.

- [ ] **Step 3: Record the real result (verbatim)**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; import json; print(json.dumps(load_report()['transport_nowcast'], indent=2))"`
Record `n`, the `mom` verdict/best_method/skill/DM p, and `driver_edge`. **This is the experiment's answer** — note whether `driver_edge` is True (the bold positive) or the series is efficient (honest negative). Report it either way.

- [ ] **Step 4: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the known pre-existing `test_main_window.py::test_on_run_requested_accepts_4_args`. Report counts.

- [ ] **Step 5: Commit (code + regenerated artifacts)**

```bash
git add ph_economic_ai/benchmark/run.py ph_economic_ai/benchmark/artifacts/accuracy_report.json ph_economic_ai/benchmark/artifacts/transport_nowcast_table.json
git status --short
git commit -m "feat(benchmark): run the transport-CPI nowcast; report the result"
```

---

## Task 6: Surface in the Accuracy view

**Files:**
- Modify: `ph_economic_ai/ui/accuracy_view.py`
- Modify: `ph_economic_ai/tests/test_accuracy_view.py`

- [ ] **Step 1: Add the test**

In `ph_economic_ai/tests/test_accuracy_view.py`, add to the `_report` helper's `rep` dict:
```python
        'transport_nowcast': {'n': 180, 'driver_edge': True,
                              'mom': {'verdict': 'beats_best_naive', 'best_method': 'ridge',
                                      'best_skill_vs_naive': 0.2, 'dm_p': 0.004},
                              'driver_ablation': {'verdict': 'beats_best_naive',
                                                  'driver_edge': True}},
```
Add the test:
```python
def test_view_shows_transport_nowcast(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.transport_nowcast_summary()
    assert '180' in s
    assert 'ridge' in s
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_accuracy_view.py::test_view_shows_transport_nowcast -v`
Expected: FAIL — no attribute `transport_nowcast_summary`.

- [ ] **Step 3: Implement**

In `ph_economic_ai/ui/accuracy_view.py`:
(a) Add a method to `AccuracyView`:
```python
    def transport_nowcast_summary(self) -> str:
        if not self._report:
            return ''
        T = self._report.get('transport_nowcast') or {}
        if not T or T.get('verdict') == 'not_run':
            return ''
        mom = T.get('mom') or {}
        return (f"Transport-CPI nowcast (n={T.get('n')}): {mom.get('verdict')} "
                f"(best {mom.get('best_method')}, skill {mom.get('best_skill_vs_naive')}, "
                f"DM p={mom.get('dm_p')}); fuel driver_edge={T.get('driver_edge')}.")
```
(b) In `_build`, inside `if self._report is not None:`, after the longer-sample block, add:
```python
            _tn = self.transport_nowcast_summary()
            if _tn:
                tnl = QLabel('<b>Transport-CPI nowcast (fuel→inflation)</b><br>' + _tn)
                tnl.setWordWrap(True)
                col.addWidget(tnl)
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
git commit -m "feat(ui): show the transport-CPI nowcast in the Accuracy view"
```

---

## Task 7: Record the outcome in the spec

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-transport-cpi-nowcast-design.md`

- [ ] **Step 1: Fill §9 with the real result from Task 5 Step 3**

Replace the `[n]`/`[m]`/`[x]`/`[p]`/`[bool]`/`[r]`/`[rb]` placeholders with the measured values from `transport_nowcast_table.json`, and select the matching interpretation bullet (driver_edge True → the bold positive; efficient → honest negative). Copy numbers verbatim; do not invent.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-10-transport-cpi-nowcast-design.md
git commit -m "docs: record measured transport-CPI nowcast result"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing UI failure.

- [ ] **Result legible from the report**

Run: `python -c "from ph_economic_ai.benchmark.report import load_report; T=load_report()['transport_nowcast']; print('n', T.get('n'), '| mom', T['mom']['verdict'], T['mom'].get('best_method'), '| fuel driver_edge', T.get('driver_edge'))"`
Expected: prints the transport nowcast verdict + whether the fuel driver edge is significant — the honest answer either way.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 gold loader (PX-Web) → Tasks 1–2. §3.2 predictors (long features) → Task 3. §4.1 `psa_cpi` (`_label_to_ym`, `load_transport_cpi`, `load_transport_mom`, `fetch_transport_cpi`) → Tasks 1–2. §4.2 `transport_nowcast.run_transport_nowcast` → Task 3. §4.3 report/run/view → Tasks 4–6. §6 error handling (not_run guard, fetch validation, label matching) → Tasks 2, 5. §7 testing (parser, MoM transform, wiring) → Tasks 1, 3. §8 deliverables → all. §9 write-up → Task 7.

**Placeholder scan:** §9's bracketed markers are filled at Task 7 from real output (accepted empirical pattern). Task 2 has no unit test by design (network refresh). No other red-flag placeholders; all code steps contain complete code.

**Type consistency:** `load_transport_mom()` defined in Task 1, reused in Task 3 (`tn.load_transport_mom` monkeypatched) and `transport_nowcast.py`. `_label_to_ym` (Task 1) reused by `fetch_transport_cpi`/`_parse_jsonstat_series` (Task 2). `run_transport_nowcast(min_train=24, features=None) -> {n, mom, driver_ablation, driver_edge}` consistent across Tasks 3, 5, 6. `build_report(... transport_nowcast=)` matches Task 4 (def) and Task 5 (call). Report key `transport_nowcast` consistent across Tasks 4–6. The view guards both the `not_run` dict and the full result. `build_nowcast_frame(target_loader=, prev_col='prev_mom', features=)` and `run_mom_nowcast(frame=)` / `run_driver_only_ablation(frame=)` match the existing nowcast signatures.
```
