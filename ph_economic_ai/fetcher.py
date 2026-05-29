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

OPEN_METEO_URL = 'https://archive-api.open-meteo.com/v1/archive'
FAO_URL = 'https://fenixservices.fao.org/faostat/api/v1/en/data/CP'
_ELECTRICITY_BASE_RATE = 11.20  # PHP/kWh — calibrated to Meralco 2024 average

# (lat, lon, production_weight) — weights sum to 1.0
_WEATHER_ZONES = [
    (15.58, 121.10, 0.45),  # Central Luzon / Nueva Ecija (rice belt)
    (13.42, 123.41, 0.25),  # Bicol Region
    ( 7.07, 125.61, 0.30),  # Davao / Mindanao
]

# Weighted seasonal norms Jan-Dec (mm rainfall, °C temp)
_RAINFALL_NORMS_MM = [30.0, 25.0, 40.0, 70.0, 120.0, 180.0, 200.0, 190.0, 160.0, 120.0, 80.0, 40.0]
_TEMP_NORMS_C      = [26.5, 27.0, 28.0, 29.5, 30.0, 29.5, 29.0, 28.5, 28.5, 28.0, 27.5, 26.5]


def _parse_world_bank_response(payload: list) -> pd.Series:
    """Parse World Bank JSON array → Series indexed by 'YYYY', NaN dropped."""
    records = payload[1]
    data = {}
    for r in records:
        year = r.get('date')
        value = r.get('value')
        if year and value is not None:
            data[year] = float(value)
    series = pd.Series(data, dtype=float).dropna()
    series.index.name = None
    return series


def _forward_fill_annual(annual: pd.Series, monthly_index: list) -> pd.Series:
    """Broadcast annual values (keyed 'YYYY') to monthly 'YYYY-MM' strings."""
    sorted_years = sorted(annual.index)
    result = {}
    for ym in monthly_index:
        year = ym[:4]
        earlier = [y for y in sorted_years if y <= year]
        if earlier:
            result[ym] = annual[max(earlier)]
    return pd.Series(result, dtype=float)


def _fetch_world_bank(indicator_id: str) -> pd.Series:
    """Fetch annual World Bank indicator for Philippines. Returns Series indexed by 'YYYY'."""
    url = f'https://api.worldbank.org/v2/country/PHL/indicator/{indicator_id}'
    r = requests.get(
        url,
        params={'format': 'json', 'per_page': '100'},
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError(f'World Bank returned no data for {indicator_id!r}')
    return _parse_world_bank_response(payload)


def _fetch_psei() -> pd.Series:
    """Fetch PSEi monthly close prices from Yahoo Finance."""
    return _fetch_yahoo('^PSEi')


def fetch_dataset() -> tuple[pd.DataFrame, str]:
    df, is_fresh = _load_cache()
    if df is not None and is_fresh:
        return df, 'Cached'

    try:
        fresh_df = _fetch_all()
    except (requests.RequestException, OSError, ValueError):
        if df is not None:
            return df, 'Cached · Stale'
        raise RuntimeError(
            'Could not load economic data.\n'
            'Please check your internet connection and try again.'
        ) from None

    try:
        _save_cache(fresh_df)
    except OSError:
        pass  # cache write failed; still serve fresh data

    return fresh_df, 'Live Data'


def _fetch_open_meteo() -> tuple[pd.Series, pd.Series]:
    """Fetch monthly rainfall (mm) and avg temp (°C) weighted across 3 PH agricultural zones."""
    end = datetime.now()
    start = end - timedelta(days=5 * 365)
    start_str = start.strftime('%Y-%m-%d')
    end_str   = end.strftime('%Y-%m-%d')

    weighted_rain: dict[str, float] = {}
    weighted_temp: dict[str, float] = {}
    rain_weight:   dict[str, float] = {}
    temp_weight:   dict[str, float] = {}

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
            ym = date_str[:7]
            if rain is not None:
                weighted_rain[ym] = weighted_rain.get(ym, 0.0) + rain * weight
                rain_weight[ym]   = rain_weight.get(ym, 0.0) + weight
            if temp is not None:
                weighted_temp[ym] = weighted_temp.get(ym, 0.0) + temp * weight
                temp_weight[ym]   = temp_weight.get(ym, 0.0) + weight

    rain_s = pd.Series(
        {ym: v / rain_weight[ym] for ym, v in weighted_rain.items()},
        dtype=float,
    ).round(1)
    temp_s = pd.Series(
        {ym: v / temp_weight[ym] for ym, v in weighted_temp.items()},
        dtype=float,
    ).round(2)
    rain_s.index.name = None
    temp_s.index.name = None
    return rain_s, temp_s


def _seasonal_weather_fallback(monthly_index: list[str]) -> tuple[pd.Series, pd.Series]:
    """Return hardcoded seasonal norms when Open-Meteo is unreachable."""
    rain = {ym: _RAINFALL_NORMS_MM[int(ym[5:7]) - 1] for ym in monthly_index}
    temp = {ym: _TEMP_NORMS_C[int(ym[5:7]) - 1] for ym in monthly_index}
    return pd.Series(rain, dtype=float), pd.Series(temp, dtype=float)


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


def _derive_electricity(gas_series: pd.Series, monthly_index: list[str]) -> pd.Series:
    """Derive monthly electricity rate from gas price movements.

    Each +1 PHP/L in gas → +0.18 PHP/kWh in electricity rate,
    reflecting Meralco's ~18% oil-linked generation cost pass-through.
    """
    gas = gas_series.reindex(monthly_index).ffill()
    gas_delta = gas.diff().fillna(0.0)
    rate = (_ELECTRICITY_BASE_RATE + gas_delta * 0.18).clip(lower=8.0).round(2)
    return rate


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


def _save_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'fetched_at': datetime.now(timezone.utc).isoformat(),
        'data': df.to_dict(orient='records'),
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _fetch_yahoo(ticker: str) -> pd.Series:
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    r = requests.get(
        url,
        params={'interval': '1mo', 'range': '5y'},
        headers=_YAHOO_HEADERS,
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    results = (payload.get('chart') or {}).get('result') or []
    if not results:
        error = (payload.get('chart') or {}).get('error') or {}
        raise ValueError(f'Yahoo Finance returned no data for {ticker!r}: {error}')
    result = results[0]
    timestamps = result['timestamp']
    closes = result['indicators']['quote'][0]['close']
    dates = [
        datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m')
        for ts in timestamps
    ]
    series = pd.Series(closes, index=dates, dtype=float).dropna().round(2)
    series = series[~series.index.duplicated(keep='last')]
    series.index.name = None
    return series


def _fetch_doe_prices(usd_php: Optional[pd.Series] = None) -> pd.Series:
    # data.gov.ph CKAN API was decommissioned (site migrated to Angular SPA).
    # Use RBOB gasoline futures (Yahoo Finance RB=F) converted to PHP/liter as
    # the closest freely accessible proxy for Philippine retail pump prices.
    rbob = _fetch_yahoo('RB=F')                                    # USD per gallon, front-month futures
    if usd_php is None:
        usd_php = _fetch_yahoo('PHP=X')                            # PHP per USD

    combined = pd.concat(
        [rbob.rename('rbob'), usd_php.rename('usd_php')], axis=1
    ).dropna()
    # Calibrated to approximate DOE RON 95 Metro Manila retail (PHP/liter).
    # Formula: (RBOB USD/gal ÷ 3.785 L/gal × PHP/USD) × 1.35 + 12
    # The fixed term covers excise tax, VAT portion, and distribution margin.
    gas_php = (combined['rbob'] / 3.785 * combined['usd_php']) * 1.35 + 12
    gas_php = gas_php.round(2)
    gas_php.index.name = None
    return gas_php


def _compute_demand(dates: list[str]) -> list[float]:
    result = []
    for date_str in dates:
        month = int(date_str[5:7])
        value = (
            65.0
            + 17.0 * math.cos(2 * math.pi * (month - 3) / 12)
            + 6.0 * math.cos(2 * math.pi * (month - 12) / 12)
        )
        result.append(round(max(55.0, min(90.0, value)), 1))
    return result