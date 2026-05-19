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
        _save_cache(fresh_df)
        return fresh_df, 'Live Data'
    except Exception:
        if df is not None:
            return df, 'Cached · Stale'
        raise RuntimeError(
            'Could not load economic data.\n'
            'Please check your internet connection and try again.'
        )


def _fetch_all() -> pd.DataFrame:
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


def _fetch_doe_prices() -> pd.Series:
    _DOE_HEADERS = {'User-Agent': 'PH-EconAI/1.0'}

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
