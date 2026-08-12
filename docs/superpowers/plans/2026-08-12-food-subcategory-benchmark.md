# Food Sub-Category Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six new PSA CPI sub-category targets (rice, meat, fish, dairy & eggs, vegetables, sugar) to the validated benchmark, backtested through the exact same statistical pipeline (mean-corrected baseline pool, Diebold-Mariano significance, selection-holdout protocol) every other target in this benchmark already goes through.

**Architecture:** Refactor `benchmark/psa_cpi.py`'s three duplicated fetch functions into one parameterized fetcher, add six thin wrappers over it. Add `benchmark/food_subcategory_nowcast.py`, a single parameterized module (not six file copies) that reuses `nowcast.run_mom_nowcast` exactly as `food_nowcast.py` does today. Pre-register the six backtests before running any of them, per this project's `DEC-010` discipline.

**Tech Stack:** Python, pandas, requests (PSA OpenSTAT PX-Web API), pytest. No new dependencies.

## Global Constraints

- The `fetch_X_cpi()` refactor (Task 1) must produce byte-identical CSVs for the three existing series (transport, food, electricity) — verified by SHA-256 hash before and after, not by row count or spot-check.
- Every new CSV writer uses `lineterminator='\n'` explicitly (this benchmark's own established rule — `pandas.to_csv()` defaults to `os.linesep`, which is CRLF on Windows and silently breaks the checksums `write_record()`/`verify()` depend on).
- No category's forecast may be described as "predictable" anywhere until it has passed `selection.run_selection_holdout` — this plan builds the code and the pre-registration only; it does **not** execute the actual live-data backtest run. That is a separate, owner-gated action per `DEC-010` ("pre-register, then wait for explicit go-ahead before running" — this project's own established discipline, most recently applied to the nine remaining headline verdicts in `docs/preregistration/2026-08-08-selection-holdout-remaining-headline-verdicts.md`).
- Every new fetcher writes a provenance sidecar via `benchmark/provenance.py::write_record`, matching every other committed CSV in this benchmark.

---

### Task 1: Refactor `psa_cpi.py`'s three fetchers into one shared function

**Files:**
- Modify: `ph_economic_ai/benchmark/psa_cpi.py`
- Test: `ph_economic_ai/tests/test_psa_cpi.py`

**Interfaces:**
- Produces: `fetch_cpi_subcategory(coicop_prefix: str, out_csv: Path, column_name: str, source_label: str, min_rows: int = 50) -> None` — used by every fetcher in this plan, including Task 2's six new ones.

- [ ] **Step 1: Write the failing test for the shared fetcher's row-count guard**

Add to `ph_economic_ai/tests/test_psa_cpi.py`:

```python
from ph_economic_ai.benchmark.psa_cpi import fetch_cpi_subcategory


def test_fetch_cpi_subcategory_raises_on_too_few_rows(tmp_path, monkeypatch):
    def fake_fetch_px_table(url, first_year, coicop_prefix):
        return {'2020-01': 100.0, '2020-02': 101.0}  # only 2 rows

    import ph_economic_ai.benchmark.psa_cpi as psa_cpi
    monkeypatch.setattr(psa_cpi, '_fetch_px_table', fake_fetch_px_table)

    out = tmp_path / 'tiny.csv'
    with pytest.raises(ValueError, match='too short'):
        fetch_cpi_subcategory('99.9', out, 'tiny_cpi', 'test source', min_rows=50)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py::test_fetch_cpi_subcategory_raises_on_too_few_rows -v`
Expected: FAIL with `ImportError: cannot import name 'fetch_cpi_subcategory'`

- [ ] **Step 3: Write the shared fetcher, keeping the three existing functions as thin wrappers**

In `ph_economic_ai/benchmark/psa_cpi.py`, replace the body of `fetch_transport_cpi`, `fetch_food_cpi`, and `fetch_electricity_cpi` (lines 163-286 in the current file) with:

```python
def fetch_cpi_subcategory(coicop_prefix: str, out_csv: Path, column_name: str,
                          source_label: str, min_rows: int = 50) -> None:
    """Fetch one PSA OpenSTAT COICOP series (backcast + current tables spliced
    on the overlap) and freeze it to CSV with a provenance sidecar. Shared by
    every fetch_X_cpi() wrapper in this module -- extracted so a new series is
    a four-line wrapper, not a ~25-line copy of the same fetch/splice/write
    logic three functions already duplicated."""
    series_back = _fetch_px_table(PSA_TRANSPORT_URL_BACKCAST, first_year=1994, coicop_prefix=coicop_prefix)
    series_curr = _fetch_px_table(PSA_TRANSPORT_URL_CURRENT, first_year=2018, coicop_prefix=coicop_prefix)

    # Merge; current table takes precedence for any overlap (2018 overlap)
    combined = {**series_back, **series_curr}

    if len(combined) < min_rows:
        raise ValueError(f'{column_name} series too short ({len(combined)} rows) — '
                         'check PX-Web selection')

    df = (pd.DataFrame(sorted(combined.items()), columns=['date', column_name])
          .sort_values('date'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, lineterminator='\n')
    write_record(out_csv, source=source_label,
                 params={'coicop_prefix': coicop_prefix, 'base_year': '2018=100',
                         'endpoints': 'backcast + current PX-Web tables'},
                 transformations=['fetch both PX-Web tables',
                                  'filter to the COICOP prefix',
                                  'splice backcast and current on the overlap',
                                  'label months YYYY-MM, sort'],
                 units='CPI index (2018=100)',
                 notes=f'PSA gold target for the {column_name} nowcast.')
    print(f'Wrote {out_csv.name} ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def fetch_transport_cpi(out_csv: Path = TRANSPORT_CSV) -> None:
    """Fetch monthly Transport CPI from PSA OpenSTAT and freeze to CSV.

    Combines two PX-Web tables:
    - 0012M4ACP28.px : Jan 1994 – Dec 2017 (backcasted 2018-base values)
    - 0012M4ACP22.px : Jan 2018 – present  (official 2018-base series)
    """
    fetch_cpi_subcategory('07', out_csv, 'transport_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 07 Transport',
                          min_rows=100)


def fetch_food_cpi(out_csv: Path = FOOD_CSV) -> None:
    """Fetch monthly Food (COICOP 01) CPI from PSA OpenSTAT and freeze to CSV."""
    fetch_cpi_subcategory('01', out_csv, 'food_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01 Food and non-alcoholic beverages',
                          min_rows=100)


def fetch_electricity_cpi(out_csv: Path = ELECTRICITY_CSV) -> None:
    """Fetch monthly Electricity (COICOP 04.5.1) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('04.5.1', out_csv, 'electricity_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 04.5.1 Electricity',
                          min_rows=50)
```

Leave `load_transport_cpi`, `load_transport_mom`, `load_food_cpi`, `load_food_mom`, `load_electricity_cpi`, `load_electricity_mom` exactly as they are — only the three `fetch_*` functions change.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: All tests PASS, including the new one and all pre-existing ones (`test_label_to_ym_handles_formats`, `test_load_transport_cpi_and_mom`, `test_resolve_commodity_id_by_coicop_prefix`, `test_resolve_commodity_id_missing_raises`, `test_load_food_mom`, `test_load_electricity_mom`)

- [ ] **Step 5: Verify the refactor produced byte-identical output for the three existing series**

Run:
```bash
python -c "
import hashlib
from pathlib import Path
for name in ['psa_transport_cpi_monthly.csv', 'psa_food_cpi_monthly.csv', 'psa_electricity_cpi_monthly.csv']:
    p = Path('ph_economic_ai/benchmark/data') / name
    print(name, hashlib.sha256(p.read_bytes()).hexdigest())
"
```
Record the three hashes. Then run:
```bash
python -c "
from ph_economic_ai.benchmark import psa_cpi
psa_cpi.fetch_transport_cpi()
psa_cpi.fetch_food_cpi()
psa_cpi.fetch_electricity_cpi()
"
```
Re-run the hash command. Expected: all three hashes identical before and after. If any differ, do not proceed — the refactor changed behavior and Step 3 has a bug.

Then run `git diff ph_economic_ai/benchmark/data/` — expected: no output (nothing changed on disk), confirming the live re-fetch reproduced the committed files exactly.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/tests/test_psa_cpi.py
git commit -m "refactor(benchmark): share PSA CPI fetch logic across the three series

fetch_transport_cpi/fetch_food_cpi/fetch_electricity_cpi were ~25-line
near-copies of the same fetch/splice/write logic. Extracted to
fetch_cpi_subcategory(coicop_prefix, ...), confirmed byte-identical output
for all three committed CSVs before and after."
```

---

### Task 2: Add six new PSA CPI sub-category fetch/load pairs

**Files:**
- Modify: `ph_economic_ai/benchmark/psa_cpi.py`
- Test: `ph_economic_ai/tests/test_psa_cpi.py`

**Interfaces:**
- Consumes: `fetch_cpi_subcategory` from Task 1.
- Produces: `fetch_rice_cpi`, `fetch_meat_cpi`, `fetch_fish_cpi`, `fetch_dairy_eggs_cpi`, `fetch_vegetables_cpi`, `fetch_sugar_cpi` (each `(out_csv: Path = <CATEGORY>_CSV) -> None`); `load_rice_cpi`, `load_rice_mom`, ... (same six-category pattern) `(csv_path: Path = <CATEGORY>_CSV) -> pd.Series`; module constants `RICE_CSV`, `MEAT_CSV`, `FISH_CSV`, `DAIRY_EGGS_CSV`, `VEGETABLES_CSV`, `SUGAR_CSV`.

- [ ] **Step 1: Write the failing tests for all six `load_X_mom` functions**

Add to `ph_economic_ai/tests/test_psa_cpi.py`:

```python
from ph_economic_ai.benchmark.psa_cpi import (
    load_rice_mom, load_meat_mom, load_fish_mom, load_dairy_eggs_mom,
    load_vegetables_mom, load_sugar_mom,
)

_SUBCATEGORY_LOADERS = {
    'rice_cpi': load_rice_mom, 'meat_cpi': load_meat_mom,
    'fish_cpi': load_fish_mom, 'dairy_eggs_cpi': load_dairy_eggs_mom,
    'vegetables_cpi': load_vegetables_mom, 'sugar_cpi': load_sugar_mom,
}


@pytest.mark.parametrize('column,loader', _SUBCATEGORY_LOADERS.items())
def test_load_subcategory_mom(tmp_path, column, loader):
    p = tmp_path / f'{column}.csv'
    p.write_text(f'date,{column}\n2018-01,100.0\n2018-02,104.0\n2018-03,104.0\n',
                 encoding='utf-8')
    mom = loader(p)
    assert mom['2018-02'] == pytest.approx(4.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py::test_load_subcategory_mom -v`
Expected: FAIL with `ImportError: cannot import name 'load_rice_mom'`

- [ ] **Step 3: Add the six fetch/load pairs**

In `ph_economic_ai/benchmark/psa_cpi.py`, after `fetch_electricity_cpi`/`load_electricity_cpi`/`load_electricity_mom` (end of the current file), add:

```python
# ---------------------------------------------------------------------------
# Food sub-categories (COICOP 01.1.x) — monthly index (2018=100)
#
# Confirmed live against openstat.psa.gov.ph's own "Commodity Description"
# dimension (not assumed from documentation): these six COICOP codes exist
# as directly selectable series in the same PX-Web table the top-level food
# series already comes from. PSA's CPI does not split poultry from
# beef/pork — '01.1.2 Meat' is one category at every COICOP depth this API
# exposes, a permanent limitation, not a gap to work around here.
# ---------------------------------------------------------------------------

RICE_CSV = HERE / 'data' / 'psa_rice_cpi_monthly.csv'
MEAT_CSV = HERE / 'data' / 'psa_meat_cpi_monthly.csv'
FISH_CSV = HERE / 'data' / 'psa_fish_cpi_monthly.csv'
DAIRY_EGGS_CSV = HERE / 'data' / 'psa_dairy_eggs_cpi_monthly.csv'
VEGETABLES_CSV = HERE / 'data' / 'psa_vegetables_cpi_monthly.csv'
SUGAR_CSV = HERE / 'data' / 'psa_sugar_cpi_monthly.csv'


def fetch_rice_cpi(out_csv: Path = RICE_CSV) -> None:
    """Fetch monthly Rice (COICOP 01.1.1.12) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.1.12', out_csv, 'rice_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.1.12 Rice')


def load_rice_cpi(csv_path: Path = RICE_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['rice_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_rice_mom(csv_path: Path = RICE_CSV) -> pd.Series:
    return cpi_to_mom(load_rice_cpi(csv_path))


def fetch_meat_cpi(out_csv: Path = MEAT_CSV) -> None:
    """Fetch monthly Meat (COICOP 01.1.2) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.2', out_csv, 'meat_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.2 Meat')


def load_meat_cpi(csv_path: Path = MEAT_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['meat_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_meat_mom(csv_path: Path = MEAT_CSV) -> pd.Series:
    return cpi_to_mom(load_meat_cpi(csv_path))


def fetch_fish_cpi(out_csv: Path = FISH_CSV) -> None:
    """Fetch monthly Fish and other seafood (COICOP 01.1.3) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.3', out_csv, 'fish_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.3 Fish and other seafood')


def load_fish_cpi(csv_path: Path = FISH_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['fish_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_fish_mom(csv_path: Path = FISH_CSV) -> pd.Series:
    return cpi_to_mom(load_fish_cpi(csv_path))


def fetch_dairy_eggs_cpi(out_csv: Path = DAIRY_EGGS_CSV) -> None:
    """Fetch monthly Milk, dairy products & eggs (COICOP 01.1.4) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.4', out_csv, 'dairy_eggs_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.4 Milk, other dairy products and eggs')


def load_dairy_eggs_cpi(csv_path: Path = DAIRY_EGGS_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['dairy_eggs_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_dairy_eggs_mom(csv_path: Path = DAIRY_EGGS_CSV) -> pd.Series:
    return cpi_to_mom(load_dairy_eggs_cpi(csv_path))


def fetch_vegetables_cpi(out_csv: Path = VEGETABLES_CSV) -> None:
    """Fetch monthly Vegetables, tubers, plantains & pulses (COICOP 01.1.7) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.7', out_csv, 'vegetables_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.7 Vegetables, tubers, plantains, cooking bananas and pulses')


def load_vegetables_cpi(csv_path: Path = VEGETABLES_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['vegetables_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_vegetables_mom(csv_path: Path = VEGETABLES_CSV) -> pd.Series:
    return cpi_to_mom(load_vegetables_cpi(csv_path))


def fetch_sugar_cpi(out_csv: Path = SUGAR_CSV) -> None:
    """Fetch monthly Sugar, confectionery & desserts (COICOP 01.1.8) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.8', out_csv, 'sugar_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.8 Sugar, confectionery and desserts')


def load_sugar_cpi(csv_path: Path = SUGAR_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['sugar_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_sugar_mom(csv_path: Path = SUGAR_CSV) -> pd.Series:
    return cpi_to_mom(load_sugar_cpi(csv_path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_psa_cpi.py -v`
Expected: All tests PASS (7 parametrized `test_load_subcategory_mom` cases plus all pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/psa_cpi.py ph_economic_ai/tests/test_psa_cpi.py
git commit -m "feat(benchmark): add six PSA food sub-category CPI fetchers

Rice (01.1.1.12), Meat (01.1.2), Fish (01.1.3), Dairy & eggs (01.1.4),
Vegetables (01.1.7), Sugar (01.1.8) — same fetch_cpi_subcategory mechanism
the three existing series use. Fetch/load functions only; no CSVs written
yet, no live network call made by this commit."
```

---

### Task 3: `food_subcategory_nowcast.py` — one parameterized module for all six categories

**Files:**
- Create: `ph_economic_ai/benchmark/food_subcategory_nowcast.py`
- Test: `ph_economic_ai/tests/test_food_subcategory_nowcast.py`

**Interfaces:**
- Consumes: `nowcast.run_mom_nowcast`, `nowcast.run_driver_only_ablation` (unmodified, existing), `psa_cpi.load_rice_mom` etc. from Task 2, `calendar_index.calendar_lag`/`require_complete_calendar` (existing), `food_nowcast.load_food_features` (existing — same predictor set reused, no new predictors).
- Produces: `CATEGORIES: list[str]` (`['rice', 'meat', 'fish', 'dairy_eggs', 'vegetables', 'sugar']`); `run_subcategory_nowcast(category: str, min_train: int = 24, features=None) -> dict` — same return shape `run_food_nowcast` already returns (`n`, `mom`, `driver_ablation`, `driver_edge`, `robust`, `driver_edge_robust`).

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_food_subcategory_nowcast.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

import ph_economic_ai.benchmark.food_subcategory_nowcast as fsn


def test_categories_list_has_six_entries():
    assert fsn.CATEGORIES == ['rice', 'meat', 'fish', 'dairy_eggs', 'vegetables', 'sugar']


@pytest.mark.parametrize('category', fsn.CATEGORIES)
def test_run_subcategory_nowcast_wires_through(monkeypatch, category):
    n = 130
    idx = pd.date_range('2005-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(hash(category) % (2**31))
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
    loader_name = fsn._LOADER_BY_CATEGORY[category]
    monkeypatch.setattr(fsn.psa_cpi, loader_name, lambda: mom)

    res = fsn.run_subcategory_nowcast(category, min_train=24, features=feats)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']
    assert 'panel' not in res['mom']


def test_run_subcategory_nowcast_rejects_unknown_category():
    with pytest.raises(ValueError, match='unknown category'):
        fsn.run_subcategory_nowcast('durian', min_train=24, features=pd.DataFrame())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_food_subcategory_nowcast.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ph_economic_ai.benchmark.food_subcategory_nowcast'`

- [ ] **Step 3: Write the implementation**

Create `ph_economic_ai/benchmark/food_subcategory_nowcast.py`:

```python
"""MoM nowcast for PSA's food CPI sub-categories, tested honestly.

One parameterized module for all six categories rather than six near-copies
of food_nowcast.py — the only thing that varies per category is which PSA
series is the target; the predictor frame (global agri-futures + oil + FX)
and the whole statistical pipeline (mean-corrected baseline pool, DM test,
preliminary-data robustness re-test) are exactly food_nowcast.py's, reused
via nowcast.run_mom_nowcast / run_driver_only_ablation as-is.

Deliberately reuses the existing food predictor set rather than researching
bespoke per-category predictors — real future work, out of scope here (see
docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md §2).
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark import psa_cpi
from ph_economic_ai.benchmark.calendar_index import (
    calendar_lag, require_complete_calendar)
from ph_economic_ai.benchmark.food_nowcast import FOOD_FEATURES_CSV, load_food_features
from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation, run_mom_nowcast

CATEGORIES = ['rice', 'meat', 'fish', 'dairy_eggs', 'vegetables', 'sugar']

# Maps a category name to the psa_cpi loader function name that returns its
# MoM series. Read via getattr(psa_cpi, ...) rather than a direct import list
# so tests can monkeypatch psa_cpi.load_rice_mom etc. in place, the same
# pattern food_nowcast's own tests already use for load_food_mom.
_LOADER_BY_CATEGORY = {
    'rice': 'load_rice_mom', 'meat': 'load_meat_mom', 'fish': 'load_fish_mom',
    'dairy_eggs': 'load_dairy_eggs_mom', 'vegetables': 'load_vegetables_mom',
    'sugar': 'load_sugar_mom',
}


def _build_subcategory_frame(category: str, features: pd.DataFrame) -> pd.DataFrame:
    """Same shape as food_nowcast._build_food_frame: the category's own CPI
    MoM as target, the existing food predictor frame, calendar_lag for
    prev_mom (not a row lag — the feature panel skips months, per every
    other nowcast frame builder's own comment on this exact point)."""
    if category not in CATEGORIES:
        raise ValueError(f'unknown category {category!r}, expected one of {CATEGORIES}')
    loader = getattr(psa_cpi, _LOADER_BY_CATEGORY[category])
    tgt = loader()
    base = pd.DataFrame({
        'rice': features['rice'], 'wheat': features['wheat'], 'corn': features['corn'],
        'soybean': features['soybean'], 'oil': features['oil_price'],
        'fx': features['usd_php'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base['prev_mom'] = calendar_lag(base['target'], 1)
    cols = ['rice', 'wheat', 'corn', 'soybean', 'oil', 'fx', 'prev_mom', 'target']
    return require_complete_calendar(base[cols].dropna(), f'{category} nowcast frame')


def run_subcategory_nowcast(category: str, min_train: int = 24, features=None,
                            prelim_months: int = 6) -> dict:
    """One category's MoM nowcast + driver-only ablation + trailing-preliminary
    robustness re-test. Same return shape as food_nowcast.run_food_nowcast."""
    if category not in CATEGORIES:
        raise ValueError(f'unknown category {category!r}, expected one of {CATEGORIES}')
    feats = load_food_features() if features is None else features
    frame = _build_subcategory_frame(category, feats)
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

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_food_subcategory_nowcast.py -v`
Expected: All 8 tests PASS (`test_categories_list_has_six_entries`, 6 parametrized `test_run_subcategory_nowcast_wires_through`, `test_run_subcategory_nowcast_rejects_unknown_category`)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/benchmark/food_subcategory_nowcast.py ph_economic_ai/tests/test_food_subcategory_nowcast.py
git commit -m "feat(benchmark): food sub-category nowcast, one module for all six

Reuses nowcast.run_mom_nowcast and the existing food predictor frame exactly
as food_nowcast.py does — no new statistical machinery, no bespoke
per-category predictors. No live data touched by this commit."
```

---

### Task 4: Pre-registration document for the six backtests

**Files:**
- Create: `docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md`

**Interfaces:**
- Consumes: `selection.run_selection_holdout`, `selection.DEFAULT_HOLDOUT_FRAC`, `selection.MIN_HOLDOUT_PREDICTIONS` (all existing, unmodified), `nowcast.PANEL_METHODS`, `nowcast.BASELINE_POOL` (existing).
- Produces: nothing importable — this is a doc-only task establishing what Task 5 (a separate, owner-gated follow-up, not part of this plan) will run.

- [ ] **Step 1: Check feasibility for all six categories before writing the document**

Run:
```bash
python -c "
from ph_economic_ai.benchmark.food_subcategory_nowcast import CATEGORIES, _build_subcategory_frame
from ph_economic_ai.benchmark.food_nowcast import load_food_features
feats = load_food_features()
for cat in CATEGORIES:
    try:
        frame = _build_subcategory_frame(cat, feats)
        print(cat, len(frame))
    except FileNotFoundError as e:
        print(cat, 'NO DATA YET —', e)
"
```
Expected: `FileNotFoundError` for all six (the CSVs don't exist yet — Task 5 fetches them). This confirms the frame-building code path is reachable and fails for the *expected* reason (missing data), not a bug. Record this in the document's "Feasibility" section as "checked, not yet fetchable — Task 5 fetches the six series first."

- [ ] **Step 2: Write the pre-registration document**

Create `docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md`, following this project's established pre-registration format (`docs/preregistration/2026-08-08-selection-holdout-remaining-headline-verdicts.md`):

```markdown
# Pre-registration: selection-holdout test of six PSA food sub-categories

Written 2026-08-12, before any of the six series below have even been
fetched. Extends `RSK-004`'s protocol (`benchmark/selection.py`) to six new
targets that did not exist when the original eleven headline verdicts were
tested.

## Disclosure, first

**I have not seen any result for any of these six targets.** Unlike the
2026-08-08 pre-registration (which pre-registered the *method* for targets
whose full-sample verdict was already published), these six series have
never been fetched, backtested, or looked at in any form before this
document. This is a genuinely blind pre-registration.

## What is being tested

Six targets, each run through both a full nowcast and a driver-only
ablation (`benchmark/food_subcategory_nowcast.py::run_subcategory_nowcast`,
Task 3), then through `selection.run_selection_holdout` on the same frame:

| Target | Setup | Frame | Candidates | Baseline pool |
|---|---|---|---|---|
| Rice MoM | full nowcast | `food_subcategory_nowcast._build_subcategory_frame('rice', ...)` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Rice MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Meat MoM | full nowcast | same pattern, category='meat' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Meat MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Fish MoM | full nowcast | same pattern, category='fish' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Fish MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Dairy & eggs MoM | full nowcast | same pattern, category='dairy_eggs' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Dairy & eggs MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Vegetables MoM | full nowcast | same pattern, category='vegetables' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Vegetables MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Sugar MoM | full nowcast | same pattern, category='sugar' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Sugar MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |

Same candidate/baseline-pool split full-nowcast vs. driver-only rows already
use for food/electricity/transport (`mean` moves from candidate to baseline
either way).

## What is unchanged

`run_selection_holdout` itself: not modified. `min_train=24`,
`holdout_frac=0.30` (`DEFAULT_HOLDOUT_FRAC`), `MIN_HOLDOUT_PREDICTIONS=12`,
confirmation criterion `holdout_skill > 0 and holdout_p < 0.05 and
dm_stat < 0` — all identical to every other target this protocol has ever
been run against.

## Feasibility

Not yet checkable — none of the six series have been fetched. This document
is committed *before* Task 5 (fetching) runs, per this project's `DEC-010`
discipline: the frame sizes will be recorded here, in this same document,
once Task 5's fetch completes and before any backtest actually runs.

## Decision rule, stated before running

For each of the twelve rows (six categories × two setups) independently,
using `run_selection_holdout`'s own `confirmed` field:

| outcome | conclusion | action |
|---|---|---|
| `not_confirmed_on_holdout` | no selection-honest edge found for this category/setup | report as null in the benchmark artifacts; app-facing labels for this category stay "exploratory, not validated" |
| `confirmed_on_holdout` | a selection-honest edge survives | do **not** promote directly to a validated claim. Check against the audit family's Bonferroni threshold (twelve new tests added to the family), same treatment `fuel_audit`'s confirmation received. Flag to the owner before any manuscript or app wording changes. |

Twelve independent tests; no result on one changes the pre-registered
expectation for another.

## What would make this run uninformative

If any category's frame, once built in Task 5, has a holdout row count
under roughly 2x `MIN_HOLDOUT_PREDICTIONS` (i.e. under ~24), the honest
reading for that row is reduced power, not a verdict — recorded as such
rather than folded silently into the results table.

## Reported whatever the outcome

For every one of the twelve rows: selection-stage skill, holdout-stage
skill, shrinkage, holdout DM p-value, n and cut, and the verdict string
verbatim — the same fields every other `selection_holdout.json` entry
already carries.

## Mechanical steps, to be done only after this document is committed

1. **[Separate, owner-gated action — not part of this implementation plan.]**
   Fetch the six series live (`psa_cpi.fetch_rice_cpi()` etc.), commit the
   CSVs and provenance sidecars.
2. Record actual frame sizes here, in a "Feasibility, confirmed" section,
   before running any backtest.
3. Extend `selection.run()` with the twelve rows above.
4. Run `python -m ph_economic_ai.benchmark.selection`, writing the extended
   `selection_holdout.json`.
5. Report the verdict table verbatim in this document's own "Result"
   section (added after running, never edited into the sections above).

## Result

*(Not run. This section is added after Task 5's live fetch and the actual
backtest run — both deliberately outside this implementation plan's scope,
per `DEC-010`.)*
```

- [ ] **Step 3: Commit**

```bash
git add docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md
git commit -m "docs: pre-register the six food sub-category selection-holdout tests

Written before any of the six series have been fetched -- a genuinely blind
pre-registration, per DEC-010. Fetching the live data and running the actual
backtest are explicitly out of scope for this document and this plan; both
require the owner's separate go-ahead."
```

---

## What this plan deliberately does not do

Per `DEC-010` and the Global Constraints above: this plan does not fetch the six series from PSA's live API, does not run `selection.run()` with the new rows, and does not write any "Result" section. Those are real, separate actions — fetching fixes six specific numbers into this benchmark's committed record permanently, and running a pre-registered test cannot be undone by re-running it if the result is unwelcome. Both need the owner's explicit go-ahead, the same way the 2026-08-08 pre-registration was committed alone first and only run after a separate instruction. When ready, those two steps are exactly "Mechanical steps 1-5" in Task 4's document above.
