# Real Data Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace synthetic data generation with live fetching from Yahoo Finance (Brent crude + USD/PHP) and DOE data.gov.ph (Philippine retail gasoline prices), with a 24-hour local JSON cache and graceful error handling.

**Architecture:** A new `fetcher.py` module owns all HTTP calls and cache I/O. `data.py` re-exports `fetch_dataset()` so the rest of the app stays unchanged. `main.py`, `main_window.py`, and `sidebar.py` receive a one-line change each to thread the `data_source` label through to the status pill.

**Tech Stack:** `requests` 2.32 (already installed), Yahoo Finance unofficial v8 JSON API, DOE data.gov.ph CKAN API, `pathlib.Path` for cache file I/O.

---

## File Map

| File | Change |
|------|--------|
| `ph_economic_ai/fetcher.py` | **NEW** — all HTTP fetching, cache read/write, demand index |
| `ph_economic_ai/cache/.gitkeep` | **NEW** — ensures cache dir is tracked by git |
| `ph_economic_ai/tests/test_fetcher.py` | **NEW** — 5 tests for pure functions |
| `ph_economic_ai/data.py` | Add 1 import line (re-export `fetch_dataset`) |
| `ph_economic_ai/main.py` | Use `fetch_dataset()`, handle error, pass `data_source` |
| `ph_economic_ai/ui/main_window.py` | Accept `data_source` kwarg, pass to `SidebarWidget` |
| `ph_economic_ai/ui/sidebar.py` | Accept `data_source` kwarg, update footer pill |

All other files are **unchanged**.

---

## Task 1: Cache directory + fetcher skeleton + failing tests

**Files:**
- Create: `ph_economic_ai/cache/.gitkeep`
- Create: `ph_economic_ai/fetcher.py`
- Create: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Create cache directory and gitkeep**

```powershell
New-Item -ItemType Directory -Force ph_economic_ai/cache
"" | Out-File -Encoding utf8 ph_economic_ai/cache/.gitkeep
```

- [ ] **Step 2: Create `ph_economic_ai/fetcher.py` with stubs**

```python
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CACHE_PATH = Path(__file__).parent / 'cache' / 'data.json'
CACHE_TTL_HOURS = 24
FETCH_TIMEOUT = 8

_YAHOO_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json',
}


def fetch_dataset() -> tuple[pd.DataFrame, str]:
    raise NotImplementedError


def _fetch_all() -> pd.DataFrame:
    raise NotImplementedError


def _load_cache(cache_path: Path = CACHE_PATH) -> tuple[Optional[pd.DataFrame], bool]:
    raise NotImplementedError


def _save_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> None:
    raise NotImplementedError


def _fetch_yahoo(ticker: str) -> pd.Series:
    raise NotImplementedError


def _fetch_doe_prices() -> pd.Series:
    raise NotImplementedError


def _compute_demand(dates: list[str]) -> list[float]:
    raise NotImplementedError
```

- [ ] **Step 3: Create `ph_economic_ai/tests/test_fetcher.py` with 5 failing tests**

```python
import json
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.fetcher import _compute_demand, _load_cache, _save_cache


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'date': ['2024-01', '2024-02'],
        'oil_price': [80.0, 82.0],
        'usd_php': [56.0, 56.5],
        'demand_index': [72.0, 68.0],
        'gas_price': [65.0, 66.0],
    })


def test_compute_demand_range():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    assert all(55.0 <= v <= 90.0 for v in values), f'Out of range: {values}'


def test_compute_demand_peaks():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    june = values[5]   # index 5 = June
    assert values[2] > june, f'March ({values[2]:.1f}) should be > June ({june:.1f})'
    assert values[11] > june, f'December ({values[11]:.1f}) should be > June ({june:.1f})'


def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / 'data.json'
    df = _sample_df()
    _save_cache(df, cache_path=cache_file)
    loaded_df, _ = _load_cache(cache_path=cache_file)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        loaded_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_fresh_cache_is_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    _save_cache(_sample_df(), cache_path=cache_file)
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert is_fresh


def test_stale_cache_is_not_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    payload = {
        'fetched_at': (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
        'data': _sample_df().to_dict(orient='records'),
    }
    cache_file.write_text(json.dumps(payload), encoding='utf-8')
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert not is_fresh
```

- [ ] **Step 4: Run tests — expect FAIL (NotImplementedError)**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_fetcher.py -v
```

Expected: 5 errors (`NotImplementedError`)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/cache/.gitkeep ph_economic_ai/fetcher.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: scaffold fetcher module and cache directory with failing tests"
```

---

## Task 2: Implement `_compute_demand()`

**Files:**
- Modify: `ph_economic_ai/fetcher.py` (replace `_compute_demand` stub)

- [ ] **Step 1: Replace the `_compute_demand` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _compute_demand(dates: list[str]) -> list[float]:
    raise NotImplementedError
```

With:
```python
def _compute_demand(dates: list[str]) -> list[float]:
    """
    Seasonal demand index calibrated to Philippine driving patterns.
    Peaks: March-April (~82, Holy Week/summer), December (~78, Christmas).
    Trough: June-July (~55, rainy season).
    """
    result = []
    for date_str in dates:
        month = int(date_str[5:7])
        value = (
            60.0
            + 18.0 * math.sin(2 * math.pi * (month - 3) / 12)
            + 8.0 * math.sin(4 * math.pi * (month - 1) / 12)
        )
        result.append(round(max(55.0, min(90.0, value)), 1))
    return result
```

- [ ] **Step 2: Run tests — expect 2 pass, 3 still fail**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_fetcher.py -v
```

Expected: `test_compute_demand_range PASSED`, `test_compute_demand_peaks PASSED`, 3 still error.

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/fetcher.py
git commit -m "feat: implement seasonal demand index computation"
```

---

## Task 3: Implement `_save_cache()` and `_load_cache()`

**Files:**
- Modify: `ph_economic_ai/fetcher.py` (replace two stubs)

- [ ] **Step 1: Replace the `_save_cache` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _save_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> None:
    raise NotImplementedError
```

With:
```python
def _save_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'data': df.to_dict(orient='records'),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
```

- [ ] **Step 2: Replace the `_load_cache` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _load_cache(cache_path: Path = CACHE_PATH) -> tuple[Optional[pd.DataFrame], bool]:
    raise NotImplementedError
```

With:
```python
def _load_cache(cache_path: Path = CACHE_PATH) -> tuple[Optional[pd.DataFrame], bool]:
    if not cache_path.exists():
        return None, False
    try:
        raw = json.loads(cache_path.read_text(encoding='utf-8'))
        fetched_at = datetime.fromisoformat(raw['fetched_at'])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        is_fresh = (datetime.now(timezone.utc) - fetched_at) < timedelta(hours=CACHE_TTL_HOURS)
        return pd.DataFrame(raw['data']), is_fresh
    except Exception:
        return None, False
```

- [ ] **Step 3: Run tests — expect all 5 pass**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_fetcher.py -v
```

Expected: `5 passed`

- [ ] **Step 4: Run full suite to confirm no regressions**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -v
```

Expected: `32 passed` (27 original + 5 new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/fetcher.py
git commit -m "feat: implement cache save and load with TTL check"
```

---

## Task 4: Network fetchers + `fetch_dataset()` orchestrator

**Files:**
- Modify: `ph_economic_ai/fetcher.py` (replace 4 remaining stubs)

No automated tests for network functions. Manual verification at end.

- [ ] **Step 1: Replace the `_fetch_yahoo` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _fetch_yahoo(ticker: str) -> pd.Series:
    raise NotImplementedError
```

With:
```python
def _fetch_yahoo(ticker: str) -> pd.Series:
    """
    Fetch 5 years of monthly close prices from Yahoo Finance.
    Returns Series indexed by 'YYYY-MM' strings, values rounded to 2dp.
    """
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    r = requests.get(
        url,
        params={'interval': '1mo', 'range': '5y'},
        headers=_YAHOO_HEADERS,
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    result = r.json()['chart']['result'][0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']
    dates = [
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m')
        for ts in timestamps
    ]
    series = pd.Series(closes, index=dates, dtype=float).dropna().round(2)
    series.index.name = None
    return series
```

- [ ] **Step 2: Replace the `_fetch_doe_prices` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _fetch_doe_prices() -> pd.Series:
    raise NotImplementedError
```

With:
```python
def _fetch_doe_prices() -> pd.Series:
    """
    Fetch DOE weekly pump prices from data.gov.ph CKAN API.
    Finds RON 95 gasoline column dynamically, aggregates to monthly averages.
    Returns Series indexed by 'YYYY-MM' strings.
    """
    _DOE_HEADERS = {'User-Agent': 'PH-EconAI/1.0'}

    # Discover resource ID
    search_r = requests.get(
        'https://data.gov.ph/api/3/action/package_search',
        params={'q': 'retail pump prices petroleum', 'rows': 5},
        headers=_DOE_HEADERS,
        timeout=FETCH_TIMEOUT,
    )
    search_r.raise_for_status()
    results = search_r.json()['result']['results']
    if not results or not results[0].get('resources'):
        raise ValueError('DOE pump price dataset not found on data.gov.ph')
    resource_id = results[0]['resources'][0]['id']

    # Fetch records
    data_r = requests.get(
        'https://data.gov.ph/api/3/action/datastore_search',
        params={'resource_id': resource_id, 'limit': 2000},
        headers=_DOE_HEADERS,
        timeout=FETCH_TIMEOUT,
    )
    data_r.raise_for_status()
    records = data_r.json()['result']['records']
    if not records:
        raise ValueError('DOE dataset returned no records')

    # Identify date and RON 95 price columns dynamically
    sample = records[0]
    date_col = next(
        (k for k in sample if 'date' in k.lower() or 'period' in k.lower()), None
    )
    price_col = next(
        (k for k in sample if 'ron 95' in k.lower() or 'ron95' in k.lower()), None
    )
    if not price_col:
        price_col = next(
            (k for k in sample if 'gasoline' in k.lower() and k != date_col), None
        )
    if not date_col or not price_col:
        raise ValueError(
            f'Cannot identify columns. Available: {list(sample.keys())}'
        )

    # Parse records, aggregate to monthly average
    rows = []
    for rec in records:
        try:
            price = float(str(rec[price_col]).replace(',', '').strip())
            raw_date = str(rec[date_col]).strip()
            for fmt in ('%m/%d/%Y', '%B %d, %Y', '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%y'):
                try:
                    dt = datetime.strptime(raw_date, fmt)
                    rows.append({'month': dt.strftime('%Y-%m'), 'price': price})
                    break
                except ValueError:
                    continue
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        raise ValueError('No valid records parsed from DOE dataset')

    monthly = pd.DataFrame(rows).groupby('month')['price'].mean().round(2)
    monthly.index.name = None
    return monthly
```

- [ ] **Step 3: Replace the `_fetch_all` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def _fetch_all() -> pd.DataFrame:
    raise NotImplementedError
```

With:
```python
def _fetch_all() -> pd.DataFrame:
    """Fetch from all sources and inner-join on YYYY-MM, compute demand."""
    oil = _fetch_yahoo('BZ=F')
    usd = _fetch_yahoo('PHP=X')
    gas = _fetch_doe_prices()

    df = pd.DataFrame({
        'oil_price': oil,
        'usd_php': usd,
        'gas_price': gas,
    }).dropna()

    df.index.name = 'date'
    df = df.reset_index()
    df['demand_index'] = _compute_demand(df['date'].tolist())
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'oil_price', 'usd_php', 'demand_index', 'gas_price']]
```

- [ ] **Step 4: Replace the `fetch_dataset` stub**

In `ph_economic_ai/fetcher.py`, replace:
```python
def fetch_dataset() -> tuple[pd.DataFrame, str]:
    raise NotImplementedError
```

With:
```python
def fetch_dataset() -> tuple[pd.DataFrame, str]:
    """
    Returns (df, data_source).
    data_source is one of: 'Live Data' | 'Cached' | 'Cached · Stale'
    Raises RuntimeError if fetch fails and no cache exists at all.
    """
    df, is_fresh = _load_cache()
    if df is not None and is_fresh:
        return df, 'Cached'

    try:
        fresh_df = _fetch_all()
        _save_cache(fresh_df)
        return fresh_df, 'Live Data'
    except Exception:
        if df is not None:
            return df, 'Cached · Stale'
        raise RuntimeError(
            'Could not load economic data.\n'
            'Please check your internet connection and try again.'
        )
```

- [ ] **Step 5: Manual smoke test — verify fetch works**

```
.venv\Scripts\python -c "from ph_economic_ai.fetcher import fetch_dataset; df, src = fetch_dataset(); print(src); print(df.tail())"
```

Expected: prints `Live Data` (first run) or `Cached` (subsequent), then last 5 rows of DataFrame with real oil/usd/gas values. Gas prices should be in the ₱60–₱80 range, oil in $70–$100 range, USD/PHP in ₱55–₱65 range.

If the DOE fetch fails (network/API change), debug by running:
```
.venv\Scripts\python -c "
import requests
r = requests.get('https://data.gov.ph/api/3/action/package_search', params={'q': 'retail pump prices petroleum', 'rows': 2}, headers={'User-Agent': 'PH-EconAI/1.0'}, timeout=10)
import json; print(json.dumps(r.json()['result']['results'][0]['resources'][0], indent=2))
"
```
This shows the resource metadata — check `id` and `name` fields.

- [ ] **Step 6: Run full test suite — confirm still 32 passing**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -v
```

Expected: `32 passed`

- [ ] **Step 7: Commit**

```bash
git add ph_economic_ai/fetcher.py
git commit -m "feat: implement Yahoo Finance and DOE fetchers with fetch_dataset orchestrator"
```

---

## Task 5: Wire into `data.py`, `main.py`, `main_window.py`, `sidebar.py`

**Files:**
- Modify: `ph_economic_ai/data.py` (line 1 — add import)
- Modify: `ph_economic_ai/main.py` (full replacement)
- Modify: `ph_economic_ai/ui/main_window.py` (constructor signature)
- Modify: `ph_economic_ai/ui/sidebar.py` (constructor + footer pill)

- [ ] **Step 1: Add `fetch_dataset` re-export to `data.py`**

Add this line at the top of `ph_economic_ai/data.py` (after existing imports):
```python
from ph_economic_ai.fetcher import fetch_dataset  # noqa: F401
```

The full top of `data.py` becomes:
```python
import numpy as np
import pandas as pd
from ph_economic_ai.fetcher import fetch_dataset  # noqa: F401


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    # ... rest unchanged ...
```

- [ ] **Step 2: Rewrite `ph_economic_ai/main.py`**

```python
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

from ph_economic_ai.data import fetch_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    X, y, _, _ = build_features(df)
    regressor = ml.train(X, y)

    window = MainWindow(df=df, regressor=regressor, data_source=data_source)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Update `MainWindow.__init__` in `ph_economic_ai/ui/main_window.py`**

Change the constructor signature from:
```python
class MainWindow(QMainWindow):
    def __init__(self, df, regressor, parent=None):
```

To:
```python
class MainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data', parent=None):
```

And change the sidebar construction (find `self._sidebar = SidebarWidget()`):
```python
        self._sidebar = SidebarWidget(data_source=data_source)
```

- [ ] **Step 4: Update `SidebarWidget.__init__` in `ph_economic_ai/ui/sidebar.py`**

Change the constructor signature from:
```python
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(190)
        self.setStyleSheet('background: #FFFFFF;')
        self._buttons: list[tuple[QPushButton, int]] = []
        self._active_idx = 0
        self._build()
```

To:
```python
    def __init__(self, data_source: str = 'Live Data', parent=None):
        super().__init__(parent)
        self.setFixedWidth(190)
        self.setStyleSheet('background: #FFFFFF;')
        self._buttons: list[tuple[QPushButton, int]] = []
        self._active_idx = 0
        self._data_source = data_source
        self._build()
```

- [ ] **Step 5: Update the footer pill in `_build()` in `sidebar.py`**

Find and replace the footer lines:
```python
        footer = QLabel('  ●  Trained · Offline')
        footer.setStyleSheet(
            'font-size:10px; color:#4A90E2; font-weight:600;'
            'background:#EBF4FF; border-radius:10px;'
            'padding:4px 8px; margin:12px 14px;'
        )
```

With:
```python
        _PILL_COLOR = {
            'Live Data': '#4A90E2',
            'Cached': '#888888',
            'Cached · Stale': '#E0A84A',
        }.get(self._data_source, '#888888')
        footer = QLabel(f'  ●  {self._data_source}')
        footer.setStyleSheet(
            f'font-size:10px; color:{_PILL_COLOR}; font-weight:600;'
            'background:#EBF4FF; border-radius:10px;'
            'padding:4px 8px; margin:12px 14px;'
        )
```

- [ ] **Step 6: Run full test suite**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -v
```

Expected: `32 passed`

- [ ] **Step 7: Commit**

```bash
git add ph_economic_ai/data.py ph_economic_ai/main.py ph_economic_ai/ui/main_window.py ph_economic_ai/ui/sidebar.py
git commit -m "feat: wire real data fetching into app — live data pill in sidebar"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by task |
|---|---|
| `fetcher.py` new module | Task 1 |
| `_compute_demand()` seasonal peaks/trough | Task 2 |
| `_save_cache()` + `_load_cache()` with TTL | Task 3 |
| `_fetch_yahoo()` — Brent crude `BZ=F` | Task 4 |
| `_fetch_yahoo()` — USD/PHP `PHP=X` | Task 4 |
| `_fetch_doe_prices()` — CKAN dynamic resource ID | Task 4 |
| `fetch_dataset()` — cache → fetch → stale → error chain | Task 4 |
| `data.py` re-exports `fetch_dataset` | Task 5 |
| `main.py` handles `RuntimeError` with QMessageBox | Task 5 |
| `MainWindow` accepts `data_source` | Task 5 |
| `SidebarWidget` shows Live Data / Cached / Cached · Stale | Task 5 |
| `cache_path` param on cache functions (for testability) | Tasks 1–3 |
| `generate_dataset()` kept for test suite | Unchanged |

**Placeholder scan:** No TBD, TODO, or vague steps. Every code block is complete and runnable.

**Type consistency:** `fetch_dataset()` returns `tuple[pd.DataFrame, str]` — consumed identically in Task 4 (smoke test) and Task 5 (`main.py`). `_load_cache(cache_path=...)` and `_save_cache(df, cache_path=...)` signatures used consistently in Tasks 3 and 4. `data_source: str` threaded from `main.py` → `MainWindow` → `SidebarWidget` with consistent parameter name throughout.
