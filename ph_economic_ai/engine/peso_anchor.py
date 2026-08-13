"""Peso-anchored prices for four food sub-categories, confirmed live at
PSA OpenSTAT's 2M/2018NEW/ table family (2026-08-13 research pass,
docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md
Section 7). Never 2M/RP or 2M/NRP -- both confirmed frozen at 2021 during
that same research pass; using either would silently anchor the app to
five-year-old prices with no error.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent
CACHE_PATH = HERE.parent / 'cache' / 'bantay_cache.json'
STALE_AFTER_DAYS = 60

#: One specific commodity item per category -- not an average across the
#: many items PSA tracks per category (Fish alone has 57 species) -- chosen
#: for being the item most commonly referenced in PH price-watch contexts.
#: `table` is the PX-Web table id under 2M/2018NEW/; `commodity` is the
#: exact label text the table's own `valueTexts` carries for that item
#: (confirmed live during the 2026-08-13 research pass).
CATEGORY_ITEMS = {
    'rice': {
        'table': '0042M4ARN01.px',
        'commodity': 'RICE, REGULAR-MILLED, 1 KG',
    },
    'meat': {
        'table': '0042M4ARN09.px',
        'commodity': 'FRESH PORK, KASIM, 1 KG',
    },
    'fish': {
        'table': '0042M4ARN11.px',
        'commodity': 'FRESH FISH, ROUND SCAD, GALUNGGONG, MEDIUM, 1 KG',
    },
    'vegetables': {
        'table': '0042M4ARN05.px',
        'commodity': 'TOMATO, 1 KG',
    },
}


def project(anchor_price: float, pct_change: float) -> float:
    """anchor_price scaled by pct_change percent. Pure arithmetic -- a
    negative pct_change must lower the price, not just format with a minus
    sign; this is real multiplication, not a display concern."""
    return anchor_price * (1 + pct_change / 100)


def _load_cache(cache_path: Path = CACHE_PATH) -> dict:
    """The cache file as a dict, or {} if it doesn't exist yet or is
    corrupt -- a missing/bad cache means 'fetch live' (Task 2), never a
    crash."""
    try:
        return json.loads(cache_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(cache: dict, cache_path: Path = CACHE_PATH) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding='utf-8')


def _is_fresh(entry: dict, today: date) -> bool:
    """True if `entry` was fetched today. A same-day cache hit avoids a
    redundant network call (Task 2); anything older gets re-fetched."""
    fetched_on = entry.get('fetched_on')
    if not fetched_on:
        return False
    return fetched_on == today.isoformat()


def _is_usable_if_stale(entry: dict, today: date) -> bool:
    """True if `entry` is older than today but still within
    STALE_AFTER_DAYS -- used as a fallback when a live re-fetch fails
    (Task 2), never as a substitute for trying to fetch fresh data first.
    PSA's own normal reporting lag is ~30 days; this only refuses an entry
    once fetches have plausibly been failing for a while, per RSK-041's
    lesson about silent staleness."""
    fetched_on = entry.get('fetched_on')
    if not fetched_on:
        return False
    age_days = (today - date.fromisoformat(fetched_on)).days
    return 0 <= age_days <= STALE_AFTER_DAYS


import requests

from ph_economic_ai.benchmark.psa_cpi import _MONTHS

_PSA_HEADERS = {'User-Agent': 'Mozilla/5.0'}
_PSA_TABLE_URL = 'https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/2018NEW/{table}'


def _fetch_live(category: str) -> Optional[dict]:
    """One PSA OpenSTAT PX-Web call for `category`'s specific commodity
    item, under 2M/2018NEW/ -- the confirmed-live folder (never 2M/RP or
    2M/NRP, both confirmed frozen at 2021). Geolocation is always '0'
    (Philippines, national), matching every other PSA fetch already in
    this codebase. The commodity id is resolved at request time by
    matching CATEGORY_ITEMS[category]['commodity'] against the table's own
    valueTexts (label text, not a numeric index) -- the same resolution
    shape psa_cpi.py::_resolve_commodity_id already uses for CPI, since
    PX-Web value ids aren't stable/guessable across tables. Returns
    {'price': float, 'as_of': 'YYYY-MM'} for the most recent real month
    (Annual rows and blank cells skipped), or None on any failure -- never
    raises, since a network hiccup must fall back to cache (get_anchor),
    not crash the caller.

    Runs synchronously on whatever thread calls it. Called from the UI
    thread on a cache miss (Task 3) -- a deliberate, documented trade-off:
    this project's own history (RSK-040, RSK-056) shows background QThread
    lifecycle bugs are a real, recurring cost, and get_anchor's daily cache
    means this network call happens at most once per category per day, not
    on every card render."""
    if category not in CATEGORY_ITEMS:
        return None
    item = CATEGORY_ITEMS[category]
    url = _PSA_TABLE_URL.format(table=item['table'])
    try:
        meta = requests.get(url, headers=_PSA_HEADERS, timeout=20).json()
        by_code = {v['code']: v for v in meta['variables']}
        commodity_var = by_code['Commodity']
        commodity_id = None
        for vid, txt in zip(commodity_var['values'], commodity_var['valueTexts']):
            if txt.strip() == item['commodity']:
                commodity_id = vid
                break
        if commodity_id is None:
            return None

        year_var = by_code['Year']
        period_var = by_code['Period']
        month_ids = [pid for pid, txt in zip(period_var['values'], period_var['valueTexts'])
                    if txt != 'Annual']

        body = {
            'query': [
                {'code': 'Geolocation',
                 'selection': {'filter': 'item', 'values': ['0']}},
                {'code': 'Commodity',
                 'selection': {'filter': 'item', 'values': [commodity_id]}},
                {'code': 'Year',
                 'selection': {'filter': 'item', 'values': year_var['values']}},
                {'code': 'Period',
                 'selection': {'filter': 'item', 'values': month_ids}},
            ],
            'response': {'format': 'json'},
        }
        resp = requests.post(url, json=body, headers=_PSA_HEADERS, timeout=30)
        resp.raise_for_status()
        data = json.loads(resp.content.decode('utf-8-sig'))

        year_labels = year_var['valueTexts']
        period_labels = dict(zip(period_var['values'], period_var['valueTexts']))

        latest = None
        for row in data['data']:
            raw = row['values'][0]
            if raw in ('..', '', None):
                continue
            year_id, period_id = row['key'][2], row['key'][3]
            month_name = period_labels.get(period_id)
            if month_name is None or month_name == 'Annual':
                continue
            month_num = _MONTHS.get(month_name.lower())
            if month_num is None:
                continue
            year_idx = year_var['values'].index(year_id)
            yyyy = year_labels[year_idx]
            as_of = f'{yyyy}-{month_num:02d}'
            if latest is None or as_of > latest[1]:
                latest = (float(raw), as_of)

        if latest is None:
            return None
        price, as_of = latest
        return {'price': price, 'as_of': as_of}
    except (requests.RequestException, KeyError, ValueError, TypeError,
            IndexError, AttributeError):
        # IndexError is reachable from several spots above (e.g.
        # row['values'][0], row['key'][2:4], year_labels[year_idx])
        # whenever PSA returns a variable with an empty values/valueTexts
        # list or a data row with a short key. AttributeError is reachable
        # from txt.strip() in the Commodity-label matching loop whenever
        # PSA returns None for an untranslated/missing valueTexts entry --
        # a real PX-Web response shape, not hypothetical. Both are
        # malformed-but-not-impossible responses that must fall back to
        # cache (get_anchor), not crash the caller, same as every other
        # failure mode here.
        return None


def get_anchor(category: str, cache_path: Path = CACHE_PATH,
               today: Optional[date] = None) -> Optional[dict]:
    """Today's price for `category`: {'price': float, 'as_of': 'YYYY-MM',
    'fetched_on': 'YYYY-MM-DD'}, or None if `category` isn't one of the
    four in scope, the live fetch fails and no usable cache exists, or the
    only cached entry is older than STALE_AFTER_DAYS. `today` is
    injectable for tests; defaults to the real current date."""
    if category not in CATEGORY_ITEMS:
        return None
    today = today or date.today()
    cache = _load_cache(cache_path)
    entry = cache.get(category)
    if entry and _is_fresh(entry, today):
        return entry

    fetched = _fetch_live(category)
    if fetched is not None:
        entry = {**fetched, 'fetched_on': today.isoformat()}
        cache[category] = entry
        _save_cache(cache, cache_path)
        return entry

    if entry and _is_usable_if_stale(entry, today):
        return entry
    return None
