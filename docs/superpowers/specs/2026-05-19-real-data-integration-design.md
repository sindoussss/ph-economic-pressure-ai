# Real Data Integration — Design Spec

**Date:** 2026-05-19
**Status:** Approved
**Feature:** Replace synthetic data generation with live fetching from Yahoo Finance and DOE Philippines

---

## 1. Overview

Replace `generate_dataset()` as the live data source with `fetch_dataset()`, which fetches real Philippine economic data from Yahoo Finance (oil price, USD/PHP rate) and the DOE data.gov.ph open data portal (retail gasoline prices). A local JSON cache ensures the app starts fast and survives temporary network outages. The synthetic `generate_dataset()` is kept in `data.py` for the test suite only.

---

## 2. Files Changed

| File | Change |
|------|--------|
| `ph_economic_ai/fetcher.py` | **NEW** — all HTTP fetching, cache read/write |
| `ph_economic_ai/data.py` | Add `fetch_dataset()`; keep `generate_dataset()` for tests |
| `ph_economic_ai/cache/data.json` | **NEW** — cached fetch result (created at runtime) |
| `ph_economic_ai/cache/.gitkeep` | **NEW** — ensures cache dir is tracked by git |
| `ph_economic_ai/main.py` | Call `fetch_dataset()` instead of `generate_dataset()`; handle no-data error |
| `ph_economic_ai/ui/sidebar.py` | Accept and display `data_source` label in status pill |

Everything else — `preprocessing.py`, `model.py`, `explanation.py`, all other UI files — is **unchanged**. The DataFrame schema is identical.

---

## 3. Data Sources

### 3.1 Oil Price — Yahoo Finance (Brent Crude)

- **Ticker:** `BZ=F`
- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/BZ=F?interval=1mo&range=5y`
- **Column:** `oil_price` (USD/bbl, monthly close)
- **No API key required**

### 3.2 USD/PHP Exchange Rate — Yahoo Finance

- **Ticker:** `PHP=X` (gives PHP per 1 USD directly)
- **Endpoint:** `https://query1.finance.yahoo.com/v8/finance/chart/PHP=X?interval=1mo&range=5y`
- **Column:** `usd_php` (PHP per USD, monthly close)
- **No API key required**

### 3.3 Philippine Gasoline Price — DOE via data.gov.ph

- **Product:** RON 95 gasoline, Metro Manila (reference price)
- **Step 1 — discover resource ID dynamically:**
  ```
  GET https://data.gov.ph/api/3/action/package_search?q=retail+pump+prices+petroleum&rows=5
  ```
  Extract the resource ID from the first result's first resource entry.
- **Step 2 — fetch records:**
  ```
  GET https://data.gov.ph/api/3/action/datastore_search?resource_id=<id>&limit=2000
  ```
- **Aggregation:** Weekly prices → monthly average (mean of all weeks in each month)
- **Column:** `gas_price` (PHP/liter)
- **No API key required**

### 3.4 Demand Index — Computed (No Fetch)

Seasonal factor derived from month number, calibrated to Philippine driving patterns:

```python
month = 1..12
demand = 60 + 18 * sin(2π * (month - 3) / 12) + 8 * sin(4π * (month - 1) / 12)
demand = clip(demand, 55, 90)
```

Peak values: March–April (~82, Holy Week/summer), December (~78, Christmas travel).
Trough: June–July (~55, rainy season).

- **Column:** `demand_index` (0–100 scale, same range as before)
- **Always available — no network dependency**

---

## 4. `fetcher.py` — Module Interface

```python
CACHE_PATH = Path(__file__).parent / 'cache' / 'data.json'
CACHE_TTL_HOURS = 24
FETCH_TIMEOUT = 8  # seconds per request

def fetch_dataset() -> tuple[pd.DataFrame, str]:
    """
    Returns (df, data_source) where data_source is one of:
      'Live Data' | 'Cached' | 'Cached · Stale'
    Raises RuntimeError if fetch fails and no cache exists.
    """

def _load_cache() -> tuple[pd.DataFrame | None, bool]:
    """Returns (df, is_fresh). df is None if cache missing."""

def _save_cache(df: pd.DataFrame) -> None:
    """Save df + current timestamp to CACHE_PATH."""

def _fetch_yahoo(ticker: str) -> pd.Series:
    """Fetch monthly close prices for ticker. Returns Series indexed by YYYY-MM."""

def _fetch_doe_prices() -> pd.Series:
    """Fetch DOE weekly pump prices, aggregate to monthly. Returns Series indexed by YYYY-MM."""

def _compute_demand(dates: list[str]) -> pd.Series:
    """Compute seasonal demand index for each YYYY-MM date string."""
```

---

## 5. Cache Format

`ph_economic_ai/cache/data.json`:

```json
{
  "fetched_at": "2026-05-19T14:30:00",
  "data": [
    {"date": "2021-06", "oil_price": 71.23, "usd_php": 49.87, "demand_index": 58.2, "gas_price": 54.10},
    ...
  ]
}
```

---

## 6. Startup Sequence (updated `main.py`)

```
1. Try fetch_dataset()
   a. Fresh cache (< 24h) → load cache, data_source = "Cached"
   b. Stale/missing cache → fetch from APIs
      - Success → save cache, data_source = "Live Data"
      - Failure + stale cache → load stale cache, data_source = "Cached · Stale"
      - Failure + no cache → raise RuntimeError

2. If RuntimeError caught in main():
   → show QMessageBox.critical("Could not load economic data.
      Please check your internet connection and try again.")
   → sys.exit(1)

3. Pass data_source string to MainWindow constructor
   → `MainWindow(df=df, regressor=regressor, data_source=data_source)`
   → `MainWindow.__init__` gains `data_source: str = 'Live Data'` parameter
4. MainWindow passes data_source to SidebarWidget:
   → `SidebarWidget(data_source=data_source)`
```

---

## 7. Sidebar Status Pill (updated `sidebar.py`)

`SidebarWidget.__init__` gains a `data_source: str = 'Live Data'` parameter.

| `data_source` | Pill text | Color |
|---|---|---|
| `'Live Data'` | `● Live Data` | Blue (`#4A90E2`) |
| `'Cached'` | `● Cached` | Gray (`#888888`) |
| `'Cached · Stale'` | `● Cached · Stale` | Orange (`#E0A84A`) |

---

## 8. DataFrame Contract (unchanged)

The returned DataFrame has identical schema to the synthetic version:

| Column | Type | Notes |
|--------|------|-------|
| `date` | str | `YYYY-MM` format |
| `oil_price` | float | USD/bbl |
| `usd_php` | float | PHP per USD |
| `demand_index` | float | 55–90 seasonal index |
| `gas_price` | float | PHP/liter |

Rows: however many complete months exist in the intersection of all three real sources (expected ~48–60 months for 5-year range).

---

## 9. Testing

Existing tests in `tests/test_data.py` continue to use `generate_dataset()` — no changes needed.

Two new tests added to `tests/test_fetcher.py`:
- `test_compute_demand_range` — seasonal values stay within 55–90
- `test_compute_demand_peaks` — March and December are higher than June

No network calls in tests. `_fetch_yahoo` and `_fetch_doe_prices` are not unit-tested (they are integration-tested manually by running the app).

---

## 10. Out of Scope

- Automatic background refresh while the app is running
- Showing the fetch timestamp in the UI
- Per-source error messages (fetch failure is all-or-nothing)
- Support for regions other than Metro Manila gasoline price
- Historical data beyond 5 years
