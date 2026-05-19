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


def _fetch_all() -> pd.DataFrame:
    oil = _fetch_yahoo('BZ=F')
    usd = _fetch_yahoo('PHP=X')
    gas = _fetch_doe_prices(usd_php=usd)

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
