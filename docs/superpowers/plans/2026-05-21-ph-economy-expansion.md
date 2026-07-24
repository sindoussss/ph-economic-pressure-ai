# PH Economy Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand ph_economic_ai from a single gas-price predictor into a three-sector Philippine economic cascade simulator covering Gas, Food, and Electricity with weather-aware ML and per-sector LLM debates.

**Architecture:** Three independent `HistGradientBoostingRegressor` models run in order (Gas → Food → Electricity); Gas's prediction is passed as a feature into Food and Electricity models at training and inference time. A new Economy Overview bento tab displays all three sectors. Three Open-Meteo weather zones (Central Luzon, Bicol, Davao) are fetched and production-weighted to drive the Food model.

**Tech Stack:** PyQt6, scikit-learn HistGradientBoostingRegressor, Ollama (qwen2.5:7b synthesizer), Open-Meteo REST API, FAO FAOSTAT API, matplotlib sparklines, requests.

**Data separation invariant:** `gas_pred`, `food_pred`, and `electricity_pred` are transient inference-time values only. They must never be written to `cache/data.json` or used as training labels.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `ph_economic_ai/fetcher.py` | Modify | Add `_fetch_open_meteo()`, `_seasonal_weather_fallback()`, `_fetch_fao_food()`, `_derive_food_from_gas()`, `_derive_electricity()`; update `_fetch_all()` |
| `ph_economic_ai/data.py` | Modify | Add synthetic `rainfall_mm`, `temp_c`, `food_price_idx`, `electricity_rate` columns to `generate_dataset()` |
| `ph_economic_ai/utils/preprocessing.py` | Modify | Add `build_gas_features()`, `build_food_features()`, `build_electricity_features()`, `build_all_features()`; keep `build_features()` as alias |
| `ph_economic_ai/model.py` | Modify | Add `train_sector()` alias |
| `ph_economic_ai/main.py` | Modify | Train 3 models; pass `regressors` dict to `SimMainWindow` |
| `ph_economic_ai/engine/debate.py` | Modify | Add `FOOD_AGENTS`, `ELECTRICITY_AGENTS`, `SynthesizerThread` |
| `ph_economic_ai/ui/economy_overview.py` | **Create** | Full bento Economy Overview widget |
| `ph_economic_ai/ui/main_window.py` | Modify | Accept `regressors` param; add Economy Overview tab; wire synthesizer thread |
| `ph_economic_ai/tests/test_fetcher.py` | Modify | Add weather + food + electricity fetch tests |
| `ph_economic_ai/tests/test_preprocessing.py` | Modify | Add food/electricity feature builder tests |

---

## Task 1: 3-Zone Open-Meteo Weather Fetcher

**Files:**
- Modify: `ph_economic_ai/fetcher.py`
- Modify: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_fetcher.py`:

```python
from unittest.mock import patch, MagicMock
from ph_economic_ai.fetcher import _fetch_open_meteo, _seasonal_weather_fallback

def _make_meteo_response(rain_val: float, temp_val: float):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        'monthly': {
            'time': ['2024-01-01', '2024-07-01'],
            'precipitation_sum': [rain_val, rain_val],
            'temperature_2m_mean': [temp_val, temp_val],
        }
    }
    return mock

def test_fetch_open_meteo_makes_three_requests():
    with patch('ph_economic_ai.fetcher.requests.get',
               return_value=_make_meteo_response(100.0, 27.0)) as mock_get:
        _fetch_open_meteo()
    assert mock_get.call_count == 3

def test_fetch_open_meteo_weighted_sum():
    """Central Luzon 100mm×0.45 + Bicol 80mm×0.25 + Davao 60mm×0.30 = 83mm."""
    responses = [
        _make_meteo_response(100.0, 26.0),  # Central Luzon weight 0.45
        _make_meteo_response(80.0,  28.0),  # Bicol weight 0.25
        _make_meteo_response(60.0,  30.0),  # Davao weight 0.30
    ]
    with patch('ph_economic_ai.fetcher.requests.get', side_effect=responses):
        rain, temp = _fetch_open_meteo()
    assert abs(rain['2024-01'] - 83.0) < 0.5   # 45 + 20 + 18
    assert abs(temp['2024-01'] - 27.5) < 0.2   # 11.7 + 7.0 + 9.0

def test_seasonal_weather_fallback_july():
    """July is peak wet season — rainfall norm should be highest."""
    monthly_index = [f'2024-{m:02d}' for m in range(1, 13)]
    rain, temp = _seasonal_weather_fallback(monthly_index)
    assert rain['2024-07'] > rain['2024-01']   # July wetter than January
    assert rain['2024-07'] == 200.0
    assert temp['2024-01'] == 26.5

def test_seasonal_fallback_covers_all_months():
    monthly_index = [f'2024-{m:02d}' for m in range(1, 13)]
    rain, temp = _seasonal_weather_fallback(monthly_index)
    assert len(rain) == 12
    assert len(temp) == 12
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_fetcher.py::test_fetch_open_meteo_makes_three_requests -v
```

Expected: `ImportError` or `AttributeError` — `_fetch_open_meteo` not defined yet.

- [ ] **Step 3: Implement in fetcher.py**

Add after the existing imports and constants at the top of `ph_economic_ai/fetcher.py`:

```python
from datetime import datetime, timedelta  # add timedelta to existing datetime import

OPEN_METEO_URL = 'https://archive-api.open-meteo.com/v1/archive'

# (lat, lon, production_weight) — weights sum to 1.0
_WEATHER_ZONES = [
    (15.58, 121.10, 0.45),  # Central Luzon / Nueva Ecija (rice belt)
    (13.42, 123.41, 0.25),  # Bicol Region
    ( 7.07, 125.61, 0.30),  # Davao / Mindanao
]

# Weighted seasonal norms Jan-Dec (mm rainfall, °C temp)
_RAINFALL_NORMS_MM = [30.0, 25.0, 40.0, 70.0, 120.0, 180.0, 200.0, 190.0, 160.0, 120.0, 80.0, 40.0]
_TEMP_NORMS_C      = [26.5, 27.0, 28.0, 29.5, 30.0, 29.5, 29.0, 28.5, 28.5, 28.0, 27.5, 26.5]
```

Then add these two functions (before `_fetch_all`):

```python
def _fetch_open_meteo() -> tuple[pd.Series, pd.Series]:
    """Fetch monthly rainfall (mm) and avg temp (°C) weighted across 3 PH agricultural zones."""
    end = datetime.now()
    start = end - timedelta(days=5 * 365)
    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')

    weighted_rain: dict[str, float] = {}
    weighted_temp: dict[str, float] = {}

    for lat, lon, weight in _WEATHER_ZONES:
        r = requests.get(
            OPEN_METEO_URL,
            params={
                'latitude':  lat,
                'longitude': lon,
                'start_date': start_str,
                'end_date':   end_str,
                'monthly': 'precipitation_sum,temperature_2m_mean',
                'timezone': 'Asia/Manila',
            },
            timeout=FETCH_TIMEOUT,
        )
        r.raise_for_status()
        monthly = r.json()['monthly']
        for date_str, rain, temp in zip(
            monthly['time'],
            monthly['precipitation_sum'],
            monthly['temperature_2m_mean'],
        ):
            ym = date_str[:7]  # YYYY-MM-DD → YYYY-MM
            if rain is not None:
                weighted_rain[ym] = weighted_rain.get(ym, 0.0) + rain * weight
            if temp is not None:
                weighted_temp[ym] = weighted_temp.get(ym, 0.0) + temp * weight

    rain_s = pd.Series(weighted_rain, dtype=float).round(1)
    temp_s = pd.Series(weighted_temp, dtype=float).round(2)
    rain_s.index.name = None
    temp_s.index.name = None
    return rain_s, temp_s


def _seasonal_weather_fallback(monthly_index: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return hardcoded seasonal norms when Open-Meteo is unreachable."""
    rain = {ym: _RAINFALL_NORMS_MM[int(ym[5:7]) - 1] for ym in monthly_index}
    temp = {ym: _TEMP_NORMS_C[int(ym[5:7]) - 1] for ym in monthly_index}
    return pd.Series(rain, dtype=float), pd.Series(temp, dtype=float)
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest ph_economic_ai/tests/test_fetcher.py -k "meteo or fallback" -v
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/fetcher.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: add 3-zone Open-Meteo weather fetcher with seasonal fallback"
```

---

## Task 2: FAO Food Price Fetcher + Derivation Fallback

**Files:**
- Modify: `ph_economic_ai/fetcher.py`
- Modify: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_fetcher.py`:

```python
from ph_economic_ai.fetcher import _fetch_fao_food, _derive_food_from_gas

def test_fetch_fao_food_parses_response():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        'data': [
            {'Year': '2022', 'Value': 115.3},
            {'Year': '2023', 'Value': 118.7},
        ]
    }
    with patch('ph_economic_ai.fetcher.requests.get', return_value=mock):
        series = _fetch_fao_food()
    assert series['2022'] == pytest.approx(115.3)
    assert series['2023'] == pytest.approx(118.7)

def test_fetch_fao_food_raises_on_empty():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {'data': []}
    with patch('ph_economic_ai.fetcher.requests.get', return_value=mock):
        with pytest.raises(ValueError, match='no food price data'):
            _fetch_fao_food()

def test_derive_food_from_gas_increases_with_gas_spike():
    monthly_index = ['2024-01', '2024-02', '2024-03']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 70.0, '2024-03': 75.0})
    rain = pd.Series({'2024-01': 30.0, '2024-02': 25.0, '2024-03': 40.0})
    idx = _derive_food_from_gas(gas, rain, monthly_index, last_known_idx=100.0)
    # Gas rose 5 PHP/L from Jan→Feb, so food index should increase
    assert idx['2024-02'] > idx['2024-01']

def test_derive_food_clips_at_floor():
    monthly_index = ['2024-01']
    gas = pd.Series({'2024-01': 65.0})
    rain = pd.Series({'2024-01': 30.0})
    idx = _derive_food_from_gas(gas, rain, monthly_index, last_known_idx=80.0)
    assert idx['2024-01'] >= 80.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_fetcher.py::test_fetch_fao_food_parses_response -v
```

Expected: `ImportError` — `_fetch_fao_food` not defined yet.

- [ ] **Step 3: Add import to test file**

Add `import pytest` to the imports in `test_fetcher.py` if not already present.

- [ ] **Step 4: Implement in fetcher.py**

Add constant after `OPEN_METEO_URL`:

```python
FAO_URL = 'https://fenixservices.fao.org/faostat/api/v1/en/data/CP'
```

Add these two functions after `_seasonal_weather_fallback`:

```python
def _fetch_fao_food() -> pd.Series:
    """Fetch Philippines annual food CPI from FAO FAOSTAT. Returns Series indexed by 'YYYY'."""
    current_year = datetime.now().year
    r = requests.get(
        FAO_URL,
        params={
            'area':        '101',   # Philippines
            'element':     '5530',  # CPI
            'item':        '23013', # Food
            'year':        ','.join(str(y) for y in range(2018, current_year + 1)),
            'output_type': 'objects',
        },
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json().get('data', [])
    if not data:
        raise ValueError('FAO returned no food price data for Philippines')
    annual = {}
    for row in data:
        year  = str(row.get('Year', ''))
        value = row.get('Value')
        if year and value is not None:
            annual[year] = float(value)
    series = pd.Series(annual, dtype=float).dropna()
    series.index.name = None
    return series


def _derive_food_from_gas(
    gas_series: pd.Series,
    rain_series: pd.Series,
    monthly_index: list[str],
    last_known_idx: float = 100.0,
) -> pd.Series:
    """Derive food price index from gas prices and rainfall when FAO data is unavailable."""
    norm_rain = pd.Series(
        {ym: _RAINFALL_NORMS_MM[int(ym[5:7]) - 1] for ym in monthly_index},
        dtype=float,
    )
    actual_rain = rain_series.reindex(monthly_index).fillna(norm_rain)
    rain_deficit = ((norm_rain - actual_rain) / norm_rain).clip(0.0, 1.0)
    gas_delta = gas_series.reindex(monthly_index).diff().fillna(0.0)
    idx = last_known_idx + (gas_delta * 0.22) + (rain_deficit * 0.15)
    return idx.clip(lower=80.0).round(2)
```

- [ ] **Step 5: Run tests to confirm they pass**

```
pytest ph_economic_ai/tests/test_fetcher.py -k "fao or derive_food" -v
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```
git add ph_economic_ai/fetcher.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: add FAO food price fetcher and gas+weather derivation fallback"
```

---

## Task 3: Electricity Rate Derivation

**Files:**
- Modify: `ph_economic_ai/fetcher.py`
- Modify: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_fetcher.py`:

```python
from ph_economic_ai.fetcher import _derive_electricity

def test_derive_electricity_base_rate():
    """With no gas change, electricity stays at base rate."""
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 65.0})
    rate = _derive_electricity(gas, monthly_index)
    assert rate['2024-02'] == pytest.approx(11.20, abs=0.05)

def test_derive_electricity_rises_with_gas():
    """A +5 PHP/L gas increase → +0.90 PHP/kWh electricity increase."""
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 70.0})
    rate = _derive_electricity(gas, monthly_index)
    assert rate['2024-02'] > rate['2024-01']
    assert abs(rate['2024-02'] - rate['2024-01'] - 0.90) < 0.05  # 5 × 0.18

def test_derive_electricity_never_below_floor():
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 30.0, '2024-02': 10.0})  # extreme drop
    rate = _derive_electricity(gas, monthly_index)
    assert all(r >= 8.0 for r in rate.values)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_fetcher.py::test_derive_electricity_base_rate -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement in fetcher.py**

Add constant after `FAO_URL`:

```python
_ELECTRICITY_BASE_RATE = 11.20  # PHP/kWh — calibrated to Meralco 2024 average
```

Add function after `_derive_food_from_gas`:

```python
def _derive_electricity(gas_series: pd.Series, monthly_index: list[str]) -> pd.Series:
    """Derive monthly electricity rate from gas price movements.

    Each +1 PHP/L in gas → +0.18 PHP/kWh in electricity rate,
    reflecting Meralco's ~18% oil-linked generation cost pass-through.
    """
    gas = gas_series.reindex(monthly_index).ffill()
    gas_delta = gas.diff().fillna(0.0)
    rate = (_ELECTRICITY_BASE_RATE + gas_delta * 0.18).clip(lower=8.0).round(2)
    return rate
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest ph_economic_ai/tests/test_fetcher.py -k "electricity" -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/fetcher.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: add electricity rate derivation from gas pass-through"
```

---

## Task 4: Wire New Columns into _fetch_all() and generate_dataset()

**Files:**
- Modify: `ph_economic_ai/fetcher.py`
- Modify: `ph_economic_ai/data.py`
- Modify: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Write failing test**

Add to `ph_economic_ai/tests/test_fetcher.py`:

```python
from ph_economic_ai.data import generate_dataset

def test_generate_dataset_has_new_columns():
    df = generate_dataset()
    for col in ['rainfall_mm', 'temp_c', 'food_price_idx', 'electricity_rate']:
        assert col in df.columns, f'Missing column: {col}'
    assert df['rainfall_mm'].between(0, 500).all()
    assert df['temp_c'].between(20, 40).all()
    assert df['food_price_idx'].between(80, 200).all()
    assert df['electricity_rate'].between(8, 20).all()
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest ph_economic_ai/tests/test_fetcher.py::test_generate_dataset_has_new_columns -v
```

Expected: `AssertionError: Missing column: rainfall_mm`.

- [ ] **Step 3: Update generate_dataset() in data.py**

In `ph_economic_ai/data.py`, update `generate_dataset()` to add synthetic new columns. Replace the existing function body:

```python
import math  # already imported if not, add it

def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120

    dates = pd.date_range('2024-01', periods=n, freq='MS').strftime('%Y-%m').tolist()

    oil = np.empty(n)
    oil[0] = 85.0
    for i in range(1, n):
        oil[i] = oil[i - 1] + rng.normal(0.2, 1.8)
    oil = np.clip(oil, 75.0, 105.0)

    usd = np.empty(n)
    usd[0] = 56.5
    for i in range(1, n):
        drift = (oil[i] - oil[i - 1]) * 0.04
        usd[i] = usd[i - 1] + drift + rng.normal(0.0, 0.25)
    usd = np.clip(usd, 54.0, 62.0)

    t = np.arange(n)
    demand = 72.0 + 9.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0.0, 2.5, n)
    demand = np.clip(demand, 55.0, 90.0)

    gas = (
        62.0
        + (oil - 75.0) * 0.38
        + (usd - 54.0) * 0.75
        + (demand - 55.0) * 0.10
        + rng.normal(0.0, 0.4, n)
    )
    gas = np.clip(gas, 62.0, 82.0)

    # Synthetic weather: seasonal pattern for PH agricultural zones
    months = [int(d[5:7]) for d in dates]
    rainfall_norms = [30.0, 25.0, 40.0, 70.0, 120.0, 180.0, 200.0, 190.0, 160.0, 120.0, 80.0, 40.0]
    rainfall = np.array([rainfall_norms[m - 1] for m in months]) + rng.normal(0.0, 10.0, n)
    rainfall = np.clip(rainfall, 0.0, 300.0).round(1)

    temp_norms = [26.5, 27.0, 28.0, 29.5, 30.0, 29.5, 29.0, 28.5, 28.5, 28.0, 27.5, 26.5]
    temp = np.array([temp_norms[m - 1] for m in months]) + rng.normal(0.0, 0.5, n)
    temp = np.clip(temp, 22.0, 36.0).round(2)

    # Synthetic food price index: base + gas pass-through + rainfall impact
    gas_delta = np.diff(gas, prepend=gas[0])
    rain_deficit = np.clip((np.array([rainfall_norms[m - 1] for m in months]) - rainfall)
                           / np.array([rainfall_norms[m - 1] for m in months]), 0.0, 1.0)
    food_idx = np.empty(n)
    food_idx[0] = 100.0
    for i in range(1, n):
        food_idx[i] = food_idx[i - 1] + gas_delta[i] * 0.22 + rain_deficit[i] * 0.15
    food_idx = np.clip(food_idx, 80.0, 180.0).round(2)

    # Synthetic electricity rate: base + gas pass-through
    elec = np.empty(n)
    elec[0] = 11.20
    for i in range(1, n):
        elec[i] = elec[i - 1] + gas_delta[i] * 0.18
    elec = np.clip(elec, 8.0, 18.0).round(2)

    return pd.DataFrame({
        'date':             dates,
        'oil_price':        oil.round(2),
        'usd_php':          usd.round(2),
        'demand_index':     demand.round(1),
        'gas_price':        gas.round(2),
        'rainfall_mm':      rainfall,
        'temp_c':           temp,
        'food_price_idx':   food_idx,
        'electricity_rate': elec,
    })
```

- [ ] **Step 4: Update _fetch_all() in fetcher.py**

Replace the existing `_fetch_all()` function:

```python
def _fetch_all() -> pd.DataFrame:
    oil  = _fetch_yahoo('BZ=F')
    usd  = _fetch_yahoo('PHP=X')
    gas  = _fetch_doe_prices(usd_php=usd)
    psei = _fetch_psei()

    cpi_annual = _fetch_world_bank('FP.CPI.TOTL.ZG')
    bsp_annual = _fetch_world_bank('FR.INR.LEND')
    rem_annual = _fetch_world_bank('BX.TRF.PWKR.CD.DT')

    base = pd.DataFrame({'oil_price': oil, 'usd_php': usd, 'gas_price': gas}).dropna()
    monthly_index = base.index.tolist()

    cpi        = _forward_fill_annual(cpi_annual, monthly_index)
    bsp_rate   = _forward_fill_annual(bsp_annual, monthly_index)
    remittances = (_forward_fill_annual(rem_annual, monthly_index) / 1e9).round(2)

    # ── Weather (3-zone weighted; seasonal fallback on failure) ───────────────
    try:
        rainfall, temp = _fetch_open_meteo()
    except Exception:
        rainfall, temp = _seasonal_weather_fallback(monthly_index)

    norm_rain = pd.Series(
        {ym: _RAINFALL_NORMS_MM[int(ym[5:7]) - 1] for ym in monthly_index}, dtype=float
    )
    norm_temp = pd.Series(
        {ym: _TEMP_NORMS_C[int(ym[5:7]) - 1] for ym in monthly_index}, dtype=float
    )
    rainfall = rainfall.reindex(monthly_index).fillna(norm_rain)
    temp     = temp.reindex(monthly_index).fillna(norm_temp)

    # ── Food price index (FAO annual forward-filled; derivation fallback) ─────
    try:
        fao_annual = _fetch_fao_food()
        food_price_idx = _forward_fill_annual(fao_annual, monthly_index)
        if food_price_idx.isna().any():
            last = float(food_price_idx.dropna().iloc[-1]) if not food_price_idx.dropna().empty else 100.0
            derived = _derive_food_from_gas(gas, rainfall, monthly_index, last_known_idx=last)
            food_price_idx = food_price_idx.fillna(derived)
    except Exception:
        food_price_idx = _derive_food_from_gas(gas, rainfall, monthly_index)

    # ── Electricity rate (gas pass-through derivation) ────────────────────────
    electricity_rate = _derive_electricity(gas, monthly_index)

    df = pd.DataFrame({
        'oil_price':        oil,
        'usd_php':          usd,
        'gas_price':        gas,
        'psei':             psei,
        'cpi':              cpi,
        'bsp_rate':         bsp_rate,
        'remittances':      remittances,
        'rainfall_mm':      rainfall,
        'temp_c':           temp,
        'food_price_idx':   food_price_idx,
        'electricity_rate': electricity_rate,
    }).dropna(subset=['oil_price', 'usd_php', 'gas_price'])

    df.index.name = 'date'
    df = df.reset_index()
    df['demand_index'] = _compute_demand(df['date'].tolist())
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'oil_price', 'usd_php', 'demand_index', 'gas_price',
               'psei', 'cpi', 'bsp_rate', 'remittances',
               'rainfall_mm', 'temp_c', 'food_price_idx', 'electricity_rate']]
```

- [ ] **Step 5: Run tests**

```
pytest ph_economic_ai/tests/test_fetcher.py -v
```

Expected: all existing + new tests PASS.

- [ ] **Step 6: Commit**

```
git add ph_economic_ai/fetcher.py ph_economic_ai/data.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: wire weather, food, and electricity columns into fetch_all and generate_dataset"
```

---

## Task 5: Food and Electricity Feature Builders

**Files:**
- Modify: `ph_economic_ai/utils/preprocessing.py`
- Modify: `ph_economic_ai/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_preprocessing.py`:

```python
import numpy as np
from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import (
    build_gas_features, build_food_features,
    build_electricity_features, build_all_features,
)

def _full_df():
    return generate_dataset()

def test_build_gas_features_backward_compat():
    """build_gas_features must return same shape as old build_features on synthetic data."""
    df = _full_df()
    X, y, cols, _ = build_gas_features(df)
    assert 'prev_gas_price' in cols
    assert 'gas_price' not in cols   # target is not a feature
    assert X.shape[0] == len(df) - 1  # one row dropped for lag
    assert not np.isnan(X).any()

def test_build_food_features_shape():
    df = _full_df()
    gas_pred = df['gas_price'].values
    X, y, cols, _ = build_food_features(df, gas_pred)
    assert 'food_price_idx_lag1' in cols
    assert 'gas_pred' in cols
    assert 'food_price_idx' not in cols   # target only
    assert X.shape[1] == len(cols)
    assert not np.isnan(X).any()

def test_build_food_features_scalar_gas_pred():
    """Should also accept a scalar gas_pred for single-point inference."""
    df = _full_df()
    X, y, cols, _ = build_food_features(df, gas_pred=70.0)
    assert X.shape[1] == len(cols)
    assert not np.isnan(X).any()

def test_build_electricity_features_shape():
    df = _full_df()
    gas_pred = df['gas_price'].values
    X, y, cols, _ = build_electricity_features(df, gas_pred)
    assert 'electricity_rate_lag1' in cols
    assert 'gas_pred' in cols
    assert 'electricity_rate' not in cols
    assert not np.isnan(X).any()

def test_build_all_features_returns_three_sectors():
    df = _full_df()
    gas_pred = df['gas_price'].values
    result = build_all_features(df, gas_pred)
    assert set(result.keys()) == {'gas', 'food', 'electricity'}
    for sector, (X, y, cols, _) in result.items():
        assert X.shape[0] > 0, f'{sector} X is empty'
        assert not np.isnan(X).any(), f'{sector} X has NaN'

def test_build_features_alias_unchanged():
    """Existing build_features() must still work (backward compat)."""
    from ph_economic_ai.utils.preprocessing import build_features
    df = _full_df()
    X, y, cols, _ = build_features(df)
    assert 'prev_gas_price' in cols
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_preprocessing.py::test_build_food_features_shape -v
```

Expected: `ImportError` — `build_food_features` not defined.

- [ ] **Step 3: Implement in preprocessing.py**

Replace the entire contents of `ph_economic_ai/utils/preprocessing.py`:

```python
import numpy as np
import pandas as pd


def build_gas_features(df: pd.DataFrame):
    """Feature builder for the gas price model."""
    df = df.copy()
    df['prev_gas_price'] = df['gas_price'].shift(1)
    df = df.dropna(subset=['prev_gas_price']).reset_index(drop=True)
    base_cols  = ['oil_price', 'usd_php', 'demand_index']
    extra_cols = ['psei', 'cpi', 'bsp_rate', 'remittances']
    available  = [c for c in extra_cols if c in df.columns]
    feature_cols = base_cols + available + ['prev_gas_price']
    X = df[feature_cols].values.astype(float)
    y = df['gas_price'].values.astype(float)
    return X, y, feature_cols, df


def build_features(df: pd.DataFrame):
    """Backward-compatible alias for build_gas_features."""
    return build_gas_features(df)


def build_food_features(df: pd.DataFrame, gas_pred):
    """Feature builder for the food price index model.

    gas_pred: array-like (same length as df) for training,
              or scalar float for single-point inference.
              Must NOT be persisted into df or cache (data separation invariant).
    """
    df = df.copy()
    df['food_price_idx_lag1'] = df['food_price_idx'].shift(1)
    df = df.dropna(subset=['food_price_idx', 'food_price_idx_lag1']).reset_index(drop=True)

    if np.isscalar(gas_pred):
        df['gas_pred'] = float(gas_pred)
    else:
        arr = np.asarray(gas_pred, dtype=float)
        df['gas_pred'] = arr[-len(df):]  # align tail to post-dropna length

    feature_cols = [
        'oil_price', 'usd_php', 'cpi', 'rainfall_mm', 'temp_c',
        'food_price_idx_lag1', 'gas_pred',
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(float)
    y = df['food_price_idx'].values.astype(float)
    return X, y, available, df


def build_electricity_features(df: pd.DataFrame, gas_pred):
    """Feature builder for the electricity rate model.

    gas_pred: array-like (same length as df) for training,
              or scalar float for single-point inference.
              Must NOT be persisted into df or cache (data separation invariant).
    """
    df = df.copy()
    df['electricity_rate_lag1'] = df['electricity_rate'].shift(1)
    df = df.dropna(subset=['electricity_rate', 'electricity_rate_lag1']).reset_index(drop=True)

    if np.isscalar(gas_pred):
        df['gas_pred'] = float(gas_pred)
    else:
        arr = np.asarray(gas_pred, dtype=float)
        df['gas_pred'] = arr[-len(df):]

    feature_cols = [
        'oil_price', 'usd_php', 'bsp_rate', 'electricity_rate_lag1', 'gas_pred',
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(float)
    y = df['electricity_rate'].values.astype(float)
    return X, y, available, df


def build_all_features(df: pd.DataFrame, gas_pred) -> dict:
    """Orchestrator — returns {sector: (X, y, cols, df)} for all three sectors."""
    return {
        'gas':         build_gas_features(df),
        'food':        build_food_features(df, gas_pred),
        'electricity': build_electricity_features(df, gas_pred),
    }


def compute_index(current_oil: float, current_usd: float, current_demand: float,
                  df: pd.DataFrame) -> tuple:
    """Return (pressure_index 0-100, oil_delta, usd_delta, demand_norm)."""
    oil_mean, oil_std = df['oil_price'].mean(), df['oil_price'].std()
    usd_mean, usd_std = df['usd_php'].mean(), df['usd_php'].std()

    oil_deltas   = (df['oil_price'] - oil_mean) / oil_std
    usd_deltas   = (df['usd_php'] - usd_mean) / usd_std
    demand_norms = df['demand_index'] / 100.0
    raw_series   = oil_deltas * 0.50 + usd_deltas * 0.30 + demand_norms * 0.20
    raw_min, raw_max = float(raw_series.min()), float(raw_series.max())

    oil_delta    = float((current_oil - oil_mean) / oil_std)
    usd_delta    = float((current_usd - usd_mean) / usd_std)
    demand_norm  = float(current_demand / 100.0)
    raw = oil_delta * 0.50 + usd_delta * 0.30 + demand_norm * 0.20

    if raw_max > raw_min:
        normalized = (raw - raw_min) / (raw_max - raw_min) * 100.0
    else:
        normalized = 50.0

    return float(np.clip(normalized, 0.0, 100.0)), oil_delta, usd_delta, demand_norm


def pressure_band(index: float) -> str:
    if index <= 30:
        return 'Stable'
    elif index <= 60:
        return 'Rising'
    elif index <= 80:
        return 'High'
    return 'Critical'
```

- [ ] **Step 4: Run all preprocessing tests**

```
pytest ph_economic_ai/tests/test_preprocessing.py -v
```

Expected: all tests PASS (existing + new).

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/utils/preprocessing.py ph_economic_ai/tests/test_preprocessing.py
git commit -m "feat: add food and electricity feature builders; keep build_features alias"
```

---

## Task 6: Train 3 Sector Models

**Files:**
- Modify: `ph_economic_ai/model.py`
- Modify: `ph_economic_ai/main.py`

- [ ] **Step 1: Add train_sector alias to model.py**

In `ph_economic_ai/model.py`, add after the existing `train()` function:

```python
def train_sector(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Train a sector-specific model. Identical to train(); exists for naming clarity."""
    return train(X, y)
```

- [ ] **Step 2: Update main.py**

Replace the entire `main()` function in `ph_economic_ai/main.py`:

```python
def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    # ── Gas model (existing) ──────────────────────────────────────────────────
    X_gas, y_gas, _, _ = build_features(df)
    gas_regressor = ml.train(X_gas, y_gas)
    cv_rmse = ml.cross_val_rmse(X_gas, y_gas)

    # ── Food and Electricity models ───────────────────────────────────────────
    # Use observed gas_price as training proxy for gas_pred (data separation invariant:
    # these are historical observations, NOT stored predictions).
    gas_pred_train = df['gas_price'].values

    regressors: dict = {'gas': gas_regressor}

    X_food, y_food, _, _ = build_food_features(df, gas_pred_train)
    if len(X_food) > 0:
        regressors['food'] = ml.train_sector(X_food, y_food)

    X_elec, y_elec, _, _ = build_electricity_features(df, gas_pred_train)
    if len(X_elec) > 0:
        regressors['electricity'] = ml.train_sector(X_elec, y_elec)

    window = SimMainWindow(
        df=df,
        regressor=gas_regressor,
        regressors=regressors,
        data_source=data_source,
        cv_rmse=cv_rmse,
    )
    window.showMaximized()
    sys.exit(app.exec())
```

Also update the imports at the top of `main.py` to include the new feature builders:

```python
from ph_economic_ai.utils.preprocessing import build_features, build_food_features, build_electricity_features
```

- [ ] **Step 3: Verify imports still resolve**

```
python -c "from ph_economic_ai.main import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```
git add ph_economic_ai/model.py ph_economic_ai/main.py
git commit -m "feat: train food and electricity sector models alongside gas in main.py"
```

---

## Task 7: Food and Electricity Debate Agents

**Files:**
- Modify: `ph_economic_ai/engine/debate.py`
- Modify: `ph_economic_ai/tests/test_debate.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_debate.py`:

```python
from ph_economic_ai.engine.debate import FOOD_AGENTS, ELECTRICITY_AGENTS

def test_food_agents_count():
    assert len(FOOD_AGENTS) == 4

def test_food_agents_have_estimate_format():
    for agent in FOOD_AGENTS:
        assert 'ESTIMATE:' in agent.system_prompt

def test_electricity_agents_count():
    assert len(ELECTRICITY_AGENTS) == 4

def test_electricity_agents_have_estimate_format():
    for agent in ELECTRICITY_AGENTS:
        assert 'ESTIMATE:' in agent.system_prompt

def test_food_agents_use_main_model():
    from ph_economic_ai.engine.debate import _MAIN_MODEL
    for agent in FOOD_AGENTS:
        assert agent.model == _MAIN_MODEL

def test_electricity_agents_use_main_model():
    from ph_economic_ai.engine.debate import _MAIN_MODEL
    for agent in ELECTRICITY_AGENTS:
        assert agent.model == _MAIN_MODEL
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_debate.py::test_food_agents_count -v
```

Expected: `ImportError` — `FOOD_AGENTS` not defined.

- [ ] **Step 3: Add FOOD_AGENTS and ELECTRICITY_AGENTS to debate.py**

Add after `DEFAULT_AGENTS` in `ph_economic_ai/engine/debate.py`:

```python
FOOD_AGENTS: list[Agent] = [
    Agent(
        name='Agri Analyst',
        role='Crop supply, harvest cycles, import dependency',
        system_prompt=(
            'You are an agricultural economist specializing in Philippine food markets. '
            'Using the gas price context and weather data provided, estimate the monthly '
            'food price index CHANGE. '
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
        ),
        rag_sources=['neda_2024_2026'],
    ),
    Agent(
        name='Supply Chain Expert',
        role='Transport cost pass-through from fuel prices',
        system_prompt=(
            'You are a logistics expert analyzing how fuel price changes cascade into '
            'Philippine food distribution costs. Using the gas price context, estimate '
            'transport cost contribution to food price change. '
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
        ),
        rag_sources=['YahooFinanceCrude'],
    ),
    Agent(
        name='Weather Interpreter',
        role='Rainfall and temperature effects on crop yields',
        system_prompt=(
            'You are a climate-agriculture analyst. Using the rainfall and temperature '
            'data provided (weighted average across Central Luzon, Bicol, and Davao), '
            'assess crop stress and estimate weather-driven food price pressure. '
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
        ),
        rag_sources=[],
    ),
    Agent(
        name='Trade Policy Critic',
        role='Tariff, NFA buffer stock, import quota impact',
        system_prompt=(
            'You are a trade policy analyst focused on Philippine food security. '
            'Challenge or support previous estimates based on NFA buffer stocks, '
            'import quotas, and tariff policy. '
            'End your response with exactly one line: ESTIMATE: +X.X% or ESTIMATE: -X.X%'
        ),
        rag_sources=['neda_2024_2026'],
    ),
]

ELECTRICITY_AGENTS: list[Agent] = [
    Agent(
        name='Energy Economist',
        role='Generation mix, fuel cost pass-through',
        system_prompt=(
            'You are an energy economist specializing in Philippine power markets. '
            'Using the gas price context, estimate the monthly electricity rate change (PHP/kWh). '
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
        ),
        rag_sources=['YahooFinanceCrude'],
    ),
    Agent(
        name='Grid Analyst',
        role='Meralco capacity, demand-supply balance',
        system_prompt=(
            'You are a grid operations analyst for the Philippine electricity market. '
            'Assess demand-supply balance and its effect on Meralco distribution charges. '
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
        ),
        rag_sources=[],
    ),
    Agent(
        name='Regulatory Expert',
        role='ERC rate review cycles, stranded cost recovery',
        system_prompt=(
            'You are a regulatory affairs expert specializing in ERC proceedings. '
            'Analyze pending rate reviews and stranded cost recovery affecting the next billing period. '
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
        ),
        rag_sources=['neda_2024_2026'],
    ),
    Agent(
        name='Demand Forecaster',
        role='Industrial and residential load outlook',
        system_prompt=(
            'You are a demand forecasting analyst for Meralco service area. '
            'Estimate load growth and its effect on WESM spot prices. '
            'End your response with exactly one line: ESTIMATE: +₱X.XX/kWh or ESTIMATE: -₱X.XX/kWh'
        ),
        rag_sources=[],
    ),
]
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_debate.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/engine/debate.py ph_economic_ai/tests/test_debate.py
git commit -m "feat: add FOOD_AGENTS and ELECTRICITY_AGENTS to debate engine"
```

---

## Task 8: Economy Synthesizer Thread

**Files:**
- Modify: `ph_economic_ai/engine/debate.py`
- Modify: `ph_economic_ai/tests/test_debate.py`

- [ ] **Step 1: Write failing tests**

Add to `ph_economic_ai/tests/test_debate.py`:

```python
from unittest.mock import patch, MagicMock
from ph_economic_ai.engine.debate import SynthesizerThread

def _make_chunk(text: str):
    return {'message': {'content': text}}

def test_synthesizer_emits_tokens():
    thread = SynthesizerThread(
        gas_verdict='Gas up ₱2.50/L.',
        food_verdict='Food index rising 3%.',
        elec_verdict='Electricity up ₱0.45/kWh.',
    )
    tokens = []
    thread.token_ready.connect(tokens.append)

    with patch('ph_economic_ai.engine.debate.ollama.chat',
               return_value=[_make_chunk('Summary'), _make_chunk(' text.')]):
        thread.run()

    assert ''.join(tokens) == 'Summary text.'

def test_synthesizer_finished_signal():
    thread = SynthesizerThread(
        gas_verdict='Gas verdict.',
        food_verdict='Food verdict.',
        elec_verdict='Electricity verdict.',
    )
    results = []
    thread.finished.connect(results.append)

    with patch('ph_economic_ai.engine.debate.ollama.chat',
               return_value=[_make_chunk('Done.')]):
        thread.run()

    assert results == ['Done.']
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest ph_economic_ai/tests/test_debate.py::test_synthesizer_emits_tokens -v
```

Expected: `ImportError` — `SynthesizerThread` not defined.

- [ ] **Step 3: Add SynthesizerThread to debate.py**

Add after `ELECTRICITY_AGENTS` in `ph_economic_ai/engine/debate.py`:

```python
_SYNTHESIZER_MODEL = 'qwen2.5:7b'


class SynthesizerThread(QThread):
    token_ready = pyqtSignal(str)
    finished    = pyqtSignal(str)

    def __init__(self, gas_verdict: str, food_verdict: str, elec_verdict: str, parent=None):
        super().__init__(parent)
        self._gas   = gas_verdict
        self._food  = food_verdict
        self._elec  = elec_verdict

    def run(self):
        messages = [
            {
                'role': 'system',
                'content': (
                    'You are a Philippine macroeconomic analyst synthesizing expert sector analysis. '
                    'Write 3-5 sentences summarizing the cascade effect across gas, food, and electricity. '
                    'Focus on the household impact. Be specific about direction and magnitude.'
                ),
            },
            {
                'role': 'user',
                'content': (
                    f'GAS SECTOR ANALYSIS:\n{self._gas}\n\n'
                    f'FOOD SECTOR ANALYSIS:\n{self._food}\n\n'
                    f'ELECTRICITY SECTOR ANALYSIS:\n{self._elec}\n\n'
                    'Provide a 3-5 sentence macro summary of the cascade effect on Philippine households.'
                ),
            },
        ]
        full_text = ''
        for chunk in ollama.chat(model=_SYNTHESIZER_MODEL, messages=messages, stream=True):
            token = chunk['message']['content']
            full_text += token
            self.token_ready.emit(token)
        self.finished.emit(full_text)
```

- [ ] **Step 4: Run tests**

```
pytest ph_economic_ai/tests/test_debate.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/engine/debate.py ph_economic_ai/tests/test_debate.py
git commit -m "feat: add SynthesizerThread for macro summary across all three sectors"
```

---

## Task 9: Economy Overview Bento Widget

**Files:**
- Create: `ph_economic_ai/ui/economy_overview.py`

No unit tests for this task — PyQt6 widgets require a running QApplication; manual visual verification is the gate (see Step 3).

- [ ] **Step 1: Create ph_economic_ai/ui/economy_overview.py**

```python
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QSizePolicy, QVBoxLayout, QWidget,
)


# ── Palette ───────────────────────────────────────────────────────────────────
_STABLE   = '#27AE60'
_RISING   = '#E0A84A'
_HIGH     = '#E74C3C'
_CARD_BG  = '#FFFFFF'
_PAGE_BG  = '#F7F8FA'
_BORDER   = '#EAEAEA'
_TEXT_DIM = '#999999'
_TEXT_HI  = '#1A1A2E'

_PRESSURE_COLOR = {'Stable': _STABLE, 'Rising': _RISING, 'High': _HIGH, 'Critical': _HIGH}


def _pressure_color(pressure: str) -> str:
    return _PRESSURE_COLOR.get(pressure, _RISING)


def _card(radius: int = 12) -> QFrame:
    f = QFrame()
    f.setStyleSheet(
        f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:{radius}px;'
    )
    return f


def _label(text: str, size: int = 11, bold: bool = False, color: str = _TEXT_HI) -> QLabel:
    lbl = QLabel(text)
    weight = '700' if bold else '400'
    lbl.setStyleSheet(f'font-size:{size}px; font-weight:{weight}; color:{color}; border:none;')
    return lbl


# ── Sparkline canvas ──────────────────────────────────────────────────────────

class _Sparkline(FigureCanvasQTAgg):
    def __init__(self, color: str, parent=None):
        fig = Figure(figsize=(2.5, 0.9), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0, 0.1, 1, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._ax.axis('off')
        self._color = color
        super().__init__(fig)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def plot(self, values: list[float]):
        self._ax.clear()
        self._ax.axis('off')
        if len(values) >= 2:
            xs = list(range(len(values)))
            self._ax.plot(xs, values, color=self._color, linewidth=2.0)
            self._ax.fill_between(xs, values, min(values), alpha=0.15, color=self._color)
        self.draw()


# ── Sector card ───────────────────────────────────────────────────────────────

class SectorCard(QFrame):
    def __init__(self, title: str, unit: str, spark_color: str, parent=None):
        super().__init__(parent)
        self._unit = unit
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(4)

        self._title_lbl = _label(title.upper(), size=10, color=_TEXT_DIM)
        lay.addWidget(self._title_lbl)

        self._value_lbl = _label('—', size=28, bold=True)
        lay.addWidget(self._value_lbl)

        self._delta_lbl = _label('—', size=13, color=_TEXT_DIM)
        lay.addWidget(self._delta_lbl)

        self._spark = _Sparkline(spark_color)
        lay.addWidget(self._spark)

        self._signal_lbl = _label('—', size=11, color=_TEXT_DIM)
        lay.addWidget(self._signal_lbl)

        self._pending()

    def _pending(self):
        self._value_lbl.setText('Analyzing…')
        self._value_lbl.setStyleSheet(
            f'font-size:18px; font-weight:400; color:{_TEXT_DIM}; border:none;'
        )

    def update_data(
        self,
        value: float,
        delta: float,
        history: list[float],
        signal_text: str,
        pressure: str,
    ):
        color = _pressure_color(pressure)
        self._value_lbl.setText(f'{value:.2f} {self._unit}')
        self._value_lbl.setStyleSheet(
            f'font-size:26px; font-weight:700; color:{_TEXT_HI}; border:none;'
        )
        arrow = '↑' if delta >= 0 else '↓'
        sign  = '+' if delta >= 0 else ''
        self._delta_lbl.setText(f'{sign}{delta:.2f}  {arrow}')
        self._delta_lbl.setStyleSheet(
            f'font-size:13px; font-weight:600; color:{color}; border:none;'
        )
        self._spark.plot(history)
        self._signal_lbl.setText(signal_text)


# ── Weather panel ─────────────────────────────────────────────────────────────

class _WeatherPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)
        lay.addWidget(_label('WEATHER SIGNAL', size=10, color=_TEXT_DIM))

        self._rain_lbl = _label('Rainfall: —', size=12)
        self._temp_lbl = _label('Avg Temp: —', size=12)
        lay.addWidget(self._rain_lbl)
        lay.addWidget(self._temp_lbl)

        fig = Figure(figsize=(3, 0.7), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0.02, 0.1, 0.96, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._ax.axis('off')
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedHeight(65)
        lay.addWidget(self._canvas)

    def update_data(self, rainfall_history: list[float], temp_history: list[float]):
        if rainfall_history:
            self._rain_lbl.setText(f'Rainfall: {rainfall_history[-1]:.0f} mm')
        if temp_history:
            self._temp_lbl.setText(f'Avg Temp: {temp_history[-1]:.1f} °C')
        self._ax.clear()
        self._ax.axis('off')
        if len(rainfall_history) >= 2:
            xs = list(range(len(rainfall_history)))
            self._ax.bar(xs, rainfall_history, color='#4A90E2', alpha=0.7, width=0.8)
        self._canvas.draw()


# ── Gas→Food influence panel ──────────────────────────────────────────────────

class _InfluencePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)
        lay.addWidget(_label('GAS → FOOD INFLUENCE', size=10, color=_TEXT_DIM))

        self._transport_lbl = _label('Transport cost: —', size=12)
        self._rainfall_lbl  = _label('Rainfall deficit: —', size=12)
        lay.addWidget(self._transport_lbl)
        lay.addWidget(self._rainfall_lbl)

        fig = Figure(figsize=(3, 0.7), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0.05, 0.1, 0.92, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedHeight(65)
        lay.addWidget(self._canvas)

    def update_data(self, gas_delta: float, rainfall_deficit_pct: float):
        transport = gas_delta * 0.22
        rainfall  = rainfall_deficit_pct * 0.15
        self._transport_lbl.setText(f'Transport cost:    {transport:+.2f} idx pts')
        self._rainfall_lbl.setText( f'Rainfall deficit:  {rainfall:+.2f} idx pts')
        self._ax.clear()
        labels = ['Transport', 'Rainfall']
        values = [transport, rainfall]
        colors = [_RISING, '#4A90E2']
        bars = self._ax.barh(labels, values, color=colors, height=0.5)
        self._ax.axvline(0, color=_TEXT_DIM, linewidth=0.8)
        self._ax.axis('off')
        self._canvas.draw()


# ── Macro summary banner ──────────────────────────────────────────────────────

class _SummaryBanner(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:#FFFBF0; border:1px solid {_BORDER};'
            'border-left:4px solid #E0A84A; border-radius:10px;'
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        self._lbl = QLabel('Awaiting sector analysis…')
        self._lbl.setStyleSheet(
            f'font-size:12px; font-style:italic; color:{_TEXT_DIM}; border:none;'
        )
        self._lbl.setWordWrap(True)
        lay.addWidget(self._lbl)

    def set_text(self, text: str):
        self._lbl.setText(text)
        self._lbl.setStyleSheet(
            f'font-size:12px; font-style:italic; color:{_TEXT_HI}; border:none;'
        )


# ── Main Economy Overview widget ──────────────────────────────────────────────

class EconomyOverviewWidget(QWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df
        self.setStyleSheet(f'background:{_PAGE_BG};')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)

        # ── Macro summary banner ──────────────────────────────────────────────
        self._summary = _SummaryBanner()
        outer.addWidget(self._summary)

        # ── Three sector cards ────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        self._gas_card  = SectorCard('Gas',         '₱/L',   '#4A90E2')
        self._food_card = SectorCard('Food Index',  'pts',   '#27AE60')
        self._elec_card = SectorCard('Electricity', '₱/kWh', '#E0A84A')

        for card in (self._gas_card, self._food_card, self._elec_card):
            cards_row.addWidget(card)
        outer.addLayout(cards_row, stretch=3)

        # ── Bottom row: influence + weather ───────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(14)
        self._influence = _InfluencePanel()
        self._weather   = _WeatherPanel()
        bottom_row.addWidget(self._influence, stretch=1)
        bottom_row.addWidget(self._weather,   stretch=1)
        outer.addLayout(bottom_row, stretch=2)

        # Pre-populate from historical data
        self._populate_from_df()

    def _history(self, col: str, n: int = 6) -> list[float]:
        if col not in self._df.columns:
            return []
        return self._df[col].dropna().tail(n).tolist()

    def _populate_from_df(self):
        df = self._df

        if 'gas_price' in df.columns and len(df) >= 2:
            hist = self._history('gas_price')
            delta = float(df['gas_price'].iloc[-1] - df['gas_price'].iloc[-2])
            self._gas_card.update_data(
                value=float(df['gas_price'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text='Pressure: —',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'food_price_idx' in df.columns and len(df) >= 2:
            hist = self._history('food_price_idx')
            delta = float(df['food_price_idx'].iloc[-1] - df['food_price_idx'].iloc[-2])
            self._food_card.update_data(
                value=float(df['food_price_idx'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text=f'Rainfall: {df["rainfall_mm"].iloc[-1]:.0f} mm' if 'rainfall_mm' in df.columns else '—',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'electricity_rate' in df.columns and len(df) >= 2:
            hist = self._history('electricity_rate')
            delta = float(df['electricity_rate'].iloc[-1] - df['electricity_rate'].iloc[-2])
            self._elec_card.update_data(
                value=float(df['electricity_rate'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text='Fuel share: 18%',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'rainfall_mm' in df.columns:
            self._weather.update_data(
                rainfall_history=self._history('rainfall_mm'),
                temp_history=self._history('temp_c') if 'temp_c' in df.columns else [],
            )

        if 'gas_price' in df.columns and len(df) >= 2:
            gas_delta = float(df['gas_price'].iloc[-1] - df['gas_price'].iloc[-2])
            rain_norm = 100.0
            rain_actual = float(df['rainfall_mm'].iloc[-1]) if 'rainfall_mm' in df.columns else rain_norm
            deficit_pct = max(0.0, (rain_norm - rain_actual) / rain_norm)
            self._influence.update_data(gas_delta, deficit_pct)

    # ── Public update slots ───────────────────────────────────────────────────

    def update_gas(self, result: dict):
        """Called when gas debate completes. result keys: value, delta, history, pressure, verdict."""
        self._gas_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text=f'Pressure: {result.get("pressure", "—")}',
            pressure=result.get('pressure', 'Stable'),
        )

    def update_food(self, result: dict):
        self._food_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text=result.get('signal_text', '—'),
            pressure=result.get('pressure', 'Stable'),
        )

    def update_electricity(self, result: dict):
        self._elec_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text='Fuel share: 18%',
            pressure=result.get('pressure', 'Stable'),
        )

    def update_summary(self, text: str):
        self._summary.set_text(text)
```

- [ ] **Step 2: Verify the widget imports cleanly**

```
python -c "from ph_economic_ai.ui.economy_overview import EconomyOverviewWidget; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```
git add ph_economic_ai/ui/economy_overview.py
git commit -m "feat: add EconomyOverviewWidget bento dashboard (gas, food, electricity, weather)"
```

---

## Task 10: Wire Economy Overview into main_window.py

**Files:**
- Modify: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Update sidebar.py to add Economy Overview as index 0**

The sidebar's `_STAGES` list drives both navigation labels and 0-based indices emitted by `stage_changed`. Adding Economy Overview at index 0 shifts all existing stages by 1 — which matches the new stack layout (overview=0, stage1=1, …, stage5=5).

In `ph_economic_ai/ui/sidebar.py`, replace `_STAGES` and update `_locked`:

```python
_STAGES = [
    (0, 'Economy Overview', 'Live sector cascade'),
    (1, 'Graph Building',   'Build knowledge base'),
    (2, 'Environment',      'Configure scenario & agents'),
    (3, 'Simulation',       'Run agent debate'),
    (4, 'Report',           'View results'),
    (5, 'Interact',         'Adjust & explore'),
]
```

And in `SidebarWidget.__init__`, update the locked set:

```python
self._locked: set[int] = {3, 4, 5}  # stages 4-6 (0-based 3,4,5) locked until first run
```

No other changes to sidebar.py are needed — `stage_changed` still emits the 0-based index.

- [ ] **Step 2: Update SimMainWindow to accept regressors and add the new tab**

Replace the entire `ph_economic_ai/ui/main_window.py`:

```python
from pathlib import Path

from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget

from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import DEFAULT_AGENTS, FOOD_AGENTS, ELECTRICITY_AGENTS, SynthesizerThread
from ph_economic_ai.engine.swarm import SwarmThread
from ph_economic_ai.ui.sidebar import SidebarWidget
from ph_economic_ai.ui.economy_overview import EconomyOverviewWidget
from ph_economic_ai.ui.stage1_rag import Stage1RagPanel
from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel
from ph_economic_ai.ui.stage3_canvas import Stage3CanvasPanel
from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
from ph_economic_ai.ui.stage5_interact import Stage5InteractPanel


class SimMainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, regressors: dict | None = None, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._regressors = regressors or {'gas': regressor}
        self._cv_rmse = cv_rmse
        self._rag = RagEngine()
        self._agents = list(DEFAULT_AGENTS)
        self._last_scenario: dict = {}
        self._last_swarm_mode: bool = False
        self._last_parallel_n: int = 4
        self._swarm_thread: SwarmThread | None = None
        self._synth_thread: SynthesizerThread | None = None
        self._gas_verdict: str = ''
        self._food_verdict: str = ''
        self._elec_verdict: str = ''

        self.setWindowTitle('PH Economic Pressure Simulation Engine')
        self.setMinimumSize(1200, 720)
        self.setStyleSheet('background:#F7F8FA;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget()
        self._sidebar.stage_changed.connect(self._on_stage_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # ── Economy Overview (index 0 in stack) ───────────────────────────────
        self._economy_overview = EconomyOverviewWidget(self._df)

        self._stage1 = Stage1RagPanel(self._rag)
        self._stage2 = Stage2SetupPanel(self._agents)
        self._stage3 = Stage3CanvasPanel(self._rag, self._agents, self._regressor,
                                         self._df, self._cv_rmse)
        self._stage4 = Stage4ReportPanel()
        self._stage5 = Stage5InteractPanel(self._rag, self._agents, self._regressor,
                                           self._df, self._cv_rmse)

        self._stage3_container = QStackedWidget()
        self._stage3_container.addWidget(self._stage3)
        self._stage3_swarm = Stage3SwarmPanel()
        self._stage3_container.addWidget(self._stage3_swarm)

        # Stack order: 0=overview, 1=stage1, 2=stage2, 3=stage3, 4=stage4, 5=stage5
        for widget in (self._economy_overview, self._stage1, self._stage2,
                       self._stage3_container, self._stage4, self._stage5):
            self._stack.addWidget(widget)

        # Wire signals
        self._stage3.simulation_complete.connect(self._on_simulation_complete)
        self._stage3_swarm.swarm_complete.connect(self._on_swarm_complete)
        self._stage5.rerun_requested.connect(self._on_rerun_requested)
        self._stage2.run_requested.connect(self._on_run_requested)

        corpus_path = Path(__file__).parent.parent / 'assets' / 'corpus' / 'neda_2024_2026.txt'
        if corpus_path.exists():
            self._rag.add_text('neda_2024_2026', corpus_path.read_text(encoding='utf-8'))

    def _on_stage_changed(self, idx: int):
        # Sidebar emits 0-based index; overview is at stack index 0,
        # so sidebar stages 1-5 map to stack 1-5 as before.
        self._stack.setCurrentIndex(idx)

    def _on_rerun_requested(self, scenario_dict: dict):
        from ph_economic_ai.ui.stage2_setup import Scenario
        sc = Scenario(
            oil_pct=scenario_dict.get('oil_pct', 5.0),
            usd_pct=scenario_dict.get('usd_pct', 2.0),
            bsp_rate=scenario_dict.get('bsp_rate', 6.5),
            demand_index=scenario_dict.get('demand_index', 72.0),
        )
        self._on_run_requested(sc, self._agents, self._last_swarm_mode, self._last_parallel_n)

    def _on_run_requested(self, scenario, agents, swarm_mode: bool = False,
                          parallel_n: int = 4):
        self._last_scenario = scenario.to_dict()
        self._last_swarm_mode = swarm_mode
        self._last_parallel_n = parallel_n
        self._sidebar.set_active(3)   # stage3 is now sidebar index 3 (overview=0 added)
        self._stack.setCurrentIndex(3)
        if swarm_mode:
            self._stage3_container.setCurrentIndex(1)
            self._stage3_swarm.reset()
            thread = SwarmThread(self._rag, self._last_scenario, parallel_n=parallel_n)
            self._stage3_swarm.connect_thread(thread)
            thread.error_occurred.connect(lambda msg: print(f'Swarm error: {msg}'))
            self._swarm_thread = thread
            thread.start()
        else:
            self._stage3_container.setCurrentIndex(0)
            self._stage3.start_simulation(scenario, agents)

    def _on_simulation_complete(self, responses):
        consensus = self._stage3.engine.consensus()
        self._gas_verdict = consensus
        self._stage4.populate(responses, consensus, self._regressor,
                              self._df, self._cv_rmse,
                              self._stage3.scenario())
        self._stage5.update_context(responses, self._stage3.scenario())
        self._stage5.set_debate_engine(self._stage3.engine)
        self._sidebar.unlock_stages([3, 4, 5])
        self._sidebar.set_active(4)
        self._stack.setCurrentIndex(4)
        self._run_synthesizer_if_ready()

    def _on_swarm_complete(self, master_verdict):
        self._gas_verdict = master_verdict
        self._stage4.populate_swarm(
            master_verdict, self._regressor, self._df, self._cv_rmse,
            self._last_scenario,
        )
        self._stage5.set_swarm_context(master_verdict, self._last_scenario)
        self._sidebar.unlock_stages([3, 4, 5])
        self._sidebar.set_active(4)
        self._stack.setCurrentIndex(4)
        self._run_synthesizer_if_ready()

    def _run_synthesizer_if_ready(self):
        """Launch the Economy Synthesizer once the gas debate verdict is available."""
        if not self._gas_verdict:
            return
        # For Phase 1, food/electricity verdicts are empty strings — the synthesizer
        # will summarize gas only if the other debates haven't run yet.
        self._synth_thread = SynthesizerThread(
            gas_verdict=self._gas_verdict,
            food_verdict=self._food_verdict or '(Food sector debate not yet run.)',
            elec_verdict=self._elec_verdict or '(Electricity sector debate not yet run.)',
        )
        self._synth_thread.finished.connect(self._economy_overview.update_summary)
        self._synth_thread.start()
```

- [ ] **Step 3: Verify imports resolve**

```
python -c "from ph_economic_ai.ui.main_window import SimMainWindow; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run the full test suite**

```
pytest ph_economic_ai/tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```
git add ph_economic_ai/ui/sidebar.py ph_economic_ai/ui/main_window.py
git commit -m "feat: wire EconomyOverviewWidget and SynthesizerThread into main_window; add sidebar Economy Overview entry"
```

---

## Task 11: Smoke Test — Launch the App

This task has no unit tests. Its gate is visual: the app must open, the Economy Overview tab must show populated sector cards, and the existing stages must still work.

- [ ] **Step 1: Launch the app**

```
python -m ph_economic_ai.main
```

- [ ] **Step 2: Verify Economy Overview tab**

- Three sector cards visible (Gas, Food, Electricity) with values populated from historical data
- Weather Signal panel shows a rainfall bar chart
- Gas → Food Influence panel shows a horizontal bar chart
- Macro summary banner shows "Awaiting sector analysis…"

- [ ] **Step 3: Run a simulation and verify synthesizer fires**

- Navigate to Stage 2 → set a scenario → click Run
- After the gas debate completes, Stage 4 appears as before
- Switch to Economy Overview tab — macro summary banner should update with the synthesizer's text

- [ ] **Step 4: Fix any sidebar index offset issues found in Step 2 of Task 10**

If stage navigation is broken (clicking sidebar stages shows wrong panels), adjust `_on_stage_changed` and the `set_active` / `unlock_stages` calls in `_on_run_requested` and `_on_simulation_complete` to use the correct stack indices.

- [ ] **Step 5: Final commit**

```
git add -u
git commit -m "fix: adjust sidebar stage indices after Economy Overview tab insertion"
```

Only run this step if fixes were needed in Step 4. If the app worked first time, skip it.
