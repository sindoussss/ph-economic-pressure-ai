# PH Economic AI — Expansion Design Spec

**Date:** 2026-05-19
**Status:** Approved
**Feature:** Add 4 new economic indicators, upgrade ML model, add Dashboard tabs (Indicators + Model Insights)

---

## 1. Overview

Expand `ph_economic_ai` with four new real Philippine economic indicators (PSEi, CPI, BSP lending rate, OFW remittances), upgrade the ML model from `RandomForestRegressor` to `HistGradientBoostingRegressor`, and surface the new data and model insights via two new tabs inside the existing Dashboard page.

No new sidebar buttons. No new top-level pages. All new content is immediately visible when the user opens the app on the Dashboard.

---

## 2. New Data Sources

### 2.1 PSEi — Yahoo Finance

- **Ticker:** `^PSEi`
- **Endpoint:** same `_fetch_yahoo()` helper, no new code
- **Column:** `psei` (index points, monthly close)
- **Frequency:** monthly

### 2.2 CPI Inflation — World Bank API

- **Indicator:** `FP.CPI.TOTL.ZG` (annual % change)
- **Endpoint:** `https://api.worldbank.org/v2/country/PHL/indicator/FP.CPI.TOTL.ZG?format=json&per_page=100`
- **Column:** `cpi` (percent, annual → forward-filled to monthly)
- **Frequency:** annual, forward-filled

### 2.3 BSP Lending Rate — World Bank API

- **Indicator:** `FR.INR.LEND` (lending interest rate %)
- **Endpoint:** `https://api.worldbank.org/v2/country/PHL/indicator/FR.INR.LEND?format=json&per_page=100`
- **Column:** `bsp_rate` (percent, annual → forward-filled to monthly)
- **Frequency:** annual, forward-filled
- **Note:** Closest freely available proxy for BSP overnight policy rate; no API key required

### 2.4 OFW Remittances — World Bank API

- **Indicator:** `BX.TRF.PWKR.CD.DT` (personal remittances received, current USD)
- **Endpoint:** `https://api.worldbank.org/v2/country/PHL/indicator/BX.TRF.PWKR.CD.DT?format=json&per_page=100`
- **Column:** `remittances` (USD billions, annual → forward-filled to monthly)
- **Frequency:** annual, forward-filled

### 2.5 Forward-Fill Logic

World Bank returns one value per year (e.g., `2023 → 6.1%`). We broadcast that value across all 12 months of that year. At dataset edges (most recent year with partial data), the last available annual value is carried forward.

```python
# pseudo-code for _forward_fill_annual(annual_series, monthly_index)
# annual_series: pd.Series indexed by 'YYYY'
# monthly_index: list of 'YYYY-MM' strings
# returns: pd.Series indexed by 'YYYY-MM'
```

---

## 3. Expanded DataFrame Schema

| Column | Type | Source | Notes |
|--------|------|---------|-------|
| `date` | str | — | `YYYY-MM` |
| `oil_price` | float | Yahoo `BZ=F` | USD/bbl |
| `usd_php` | float | Yahoo `PHP=X` | PHP per USD |
| `demand_index` | float | computed | 55–90 seasonal |
| `gas_price` | float | Yahoo `RB=F` derived | PHP/liter (target) |
| `psei` | float | Yahoo `^PSEi` | index points |
| `cpi` | float | World Bank | annual % forward-filled |
| `bsp_rate` | float | World Bank | annual % forward-filled |
| `remittances` | float | World Bank | USD billions forward-filled |

Rows: intersection of all sources after join + dropna (expected ~48–60 months, limited by World Bank annual coverage).

---

## 4. Files Changed

| File | Change |
|------|--------|
| `ph_economic_ai/fetcher.py` | Add `_fetch_world_bank(indicator_id)`, `_fetch_psei()`, `_forward_fill_annual()`; update `_fetch_all()` to build 9-column DataFrame |
| `ph_economic_ai/utils/preprocessing.py` | `build_features()` uses all 7 feature columns: `oil_price`, `usd_php`, `demand_index`, `psei`, `cpi`, `bsp_rate`, `remittances` |
| `ph_economic_ai/model.py` | Swap `RandomForestRegressor` → `HistGradientBoostingRegressor`; add `get_feature_importances(model, feature_names) -> dict[str, float]`; add `cross_val_rmse(model, X, y) -> float` |
| `ph_economic_ai/main.py` | Capture `cv_rmse` from `ml.cross_val_rmse(X, y)`; pass to `MainWindow` |
| `ph_economic_ai/ui/main_window.py` | Accept `cv_rmse: float` parameter; pass to `DashboardPage` |
| `ph_economic_ai/ui/dashboard.py` | Add `QTabWidget` with three tabs: Overview (existing), Indicators, Model Insights |
| `ph_economic_ai/tests/test_fetcher.py` | Add `test_forward_fill_annual`, `test_world_bank_parse` |
| `ph_economic_ai/tests/test_model.py` | Add `test_model_is_hgb`, `test_feature_importances_shape`, `test_cross_val_rmse_positive` |

Everything else — `data.py`, `explanation.py`, `sidebar.py`, `pressure.py`, `charts.py`, `agent_graph.py` — is **unchanged**.

---

## 5. ML Model

### 5.1 Model Type

Replace `RandomForestRegressor(n_estimators=100, random_state=42)` with `HistGradientBoostingRegressor(random_state=42)`.

**Why:** Handles NaN natively (important for forward-filled annual data at edges), generally higher accuracy than RandomForest on structured tabular data, same scikit-learn API, no new dependencies.

### 5.2 New Functions in `model.py`

```python
def get_feature_importances(model, feature_names: list[str]) -> dict[str, float]:
    """Returns feature name → importance (0–1), sorted descending."""
    importances = model.feature_importances_
    total = importances.sum()
    normalized = importances / total if total > 0 else importances
    return dict(sorted(zip(feature_names, normalized), key=lambda x: x[1], reverse=True))

def cross_val_rmse(X, y, cv: int = 5) -> float:
    """5-fold CV on a fresh HistGradientBoostingRegressor, return mean RMSE."""
    # Uses cross_val_score with neg_root_mean_squared_error
    # Returns positive float (RMSE in PHP/liter)
```

### 5.3 Confidence Band

The 6-month forecast in the Model Insights tab shows a ±CV-RMSE shaded band around the predicted line. The CV-RMSE is computed once at startup and passed through to the UI. It represents the model's average error on held-out data — an honest, minimal confidence representation.

### 5.4 6-Month Forecast

Forecast is generated by extrapolating each feature column 6 months forward using `last known value` (flat projection). This is intentionally simple — the goal is to show directional trend, not precision forecasting.

---

## 6. Dashboard Tab Layout

`DashboardPage.__init__` adds a `QTabWidget` as the top-level layout container.

### Tab 1: Overview (unchanged)
Existing dashboard content: trend chart, 4 mini-cards, simulation panel, right panel. No changes.

### Tab 2: Indicators
A `QScrollArea` containing a `QGridLayout` (2 columns) of 7 matplotlib `FigureCanvasQTAgg` charts.

| Row | Col 0 | Col 1 |
|-----|-------|-------|
| 0 | Oil Price (USD/bbl) | USD/PHP Rate |
| 1 | Demand Index | PSEi |
| 2 | CPI Inflation (%) | BSP Lending Rate (%) |
| 3 | OFW Remittances (USD bn) | — |

Each chart: line plot, title, y-axis label, subtle grid, no legend (title is self-explanatory). Fixed height 200px per chart. Tab is labeled **"Indicators"**.

### Tab 3: Model Insights
Two sections stacked vertically:

**Top — Feature Importance (horizontal bar chart)**
- 7 bars, sorted by importance descending
- Existing indicators (`oil_price`, `usd_php`, `demand_index`) colored `#4A90E2` (blue)
- New indicators (`psei`, `cpi`, `bsp_rate`, `remittances`) colored `#9B59B6` (purple)
- X-axis: 0.0–1.0 (normalized importance)
- Title: "Feature Importance"

**Bottom — 6-Month Forecast**
- Historical gas price line (last 12 months, gray)
- Forecast line (next 6 months, blue)
- Shaded band: forecast ± CV-RMSE (light blue, alpha 0.25)
- Text annotation: `f"CV-RMSE: ₱{cv_rmse:.2f}/liter"`
- Title: "6-Month Gas Price Forecast (indicative)"

Tab is labeled **"Model Insights"**.

---

## 7. Error Handling

- If any World Bank fetch fails, `_fetch_all()` raises `ValueError` (caught by `fetch_dataset()` → falls back to cache, same as today)
- If PSEi fetch fails, same fallback
- No partial-success mode — fetch is all-or-nothing to keep the DataFrame schema consistent

---

## 8. Testing

### New tests in `test_fetcher.py`

```python
def test_forward_fill_annual_basic():
    # annual series {2022: 5.0, 2023: 6.0}
    # monthly index ['2022-06', '2022-12', '2023-03']
    # expect [5.0, 5.0, 6.0]

def test_world_bank_parse_returns_series():
    # given sample WB JSON response, _parse_world_bank_response() returns pd.Series
    # indexed by 'YYYY', values are floats, NaN rows dropped
```

### New tests in `test_model.py`

```python
def test_model_is_hgb():
    # train() returns HistGradientBoostingRegressor instance

def test_feature_importances_shape():
    # get_feature_importances(model, names) returns dict with len == len(names)
    # all values sum to ~1.0

def test_cross_val_rmse_positive():
    # cross_val_rmse() returns float > 0
```

No network calls in tests. `_fetch_world_bank` and `_fetch_psei` are not unit-tested (integration-tested by running the app).

---

## 9. Out of Scope

- Regional breakdown of new indicators
- Automatic background refresh
- Per-indicator error messages
- Showing fetch timestamps for each source
- Model retraining while app is running
- Any indicator other than the 4 specified
