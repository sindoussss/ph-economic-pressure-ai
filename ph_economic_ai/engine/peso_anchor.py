"""Peso-anchored prices for four food sub-categories, confirmed live at
PSA OpenSTAT's 2M/2018NEW/ table family (2026-08-13 research pass,
docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md
Section 7). Never 2M/RP or 2M/NRP -- both confirmed frozen at 2021 during
that same research pass; using either would silently anchor the app to
five-year-old prices with no error.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from ph_economic_ai.benchmark.psa_cpi import _MONTHS

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


#: Display labels for the four PSA-anchored categories. Deliberately not
#: shared with the UI layer's own six-category label map (which also covers
#: dairy_eggs/sugar, outside this module's scope) -- this module stays free
#: of any UI-layer import, engine code never depends on ui/.
_DISPLAY_LABELS = {'rice': 'Rice', 'meat': 'Meat', 'fish': 'Fish', 'vegetables': 'Vegetables'}


def anchor_strip(subcategories: dict) -> Optional[dict]:
    """{'text': str, 'caption': str} for the four PSA-anchored categories
    that have both a live/cached price and this cycle's debate percentage in
    `subcategories`, or None if none do. A category missing either piece
    shows '—' in its own slot rather than inventing a stand-in value or
    suppressing the rest of the strip.

    Pure formatting -- no PyQt -- so Monitor and Report can both build their
    own widget from the returned strings without duplicating this logic.
    `get_anchor()` is only called for a category that actually has a
    percentage this cycle, same cost discipline the original inline version
    had."""
    parts = []
    as_of_months = []
    for category, label in _DISPLAY_LABELS.items():
        pct = subcategories.get(category)
        anchor = get_anchor(category) if pct is not None else None
        # .get(), not direct indexing: a malformed cache entry (e.g. missing
        # 'price' or 'as_of') must read as "unavailable for this category"
        # -- an em-dash -- not raise.
        price = anchor.get('price') if anchor is not None else None
        as_of = anchor.get('as_of') if anchor is not None else None
        if pct is not None and price is not None and as_of is not None:
            projected = project(price, pct)
            parts.append(f'{label} ₱{price:.2f} → ₱{projected:.2f}')
            as_of_months.append(as_of)
        else:
            parts.append(f'{label} —')

    if not as_of_months:
        return None
    # Each category's get_anchor() call is independent and can fall back to
    # a different cache age, so two categories in this same strip can
    # legitimately carry different as_of months. Report the OLDEST one
    # actually used -- a floor, not a claim every price shown is from that
    # exact month. 'YYYY-MM' strings sort correctly lexicographically.
    oldest_month = min(as_of_months)
    return {
        'text': '  ·  '.join(parts),
        'caption': (f'PSA retail price (as of {oldest_month} or later) × this '
                    "debate's forecast — exploratory projection, not a "
                    'validated prediction.'),
    }


def _load_cache(cache_path: Path = CACHE_PATH) -> dict:
    """The cache file as a dict, or {} if it doesn't exist yet, is corrupt,
    or (rare, but a real JSON file can hold a list/string/number just as
    easily as an object) doesn't even hold a dict -- a missing/bad/wrong-
    shaped cache means 'fetch live' (Task 2), never a crash."""
    try:
        data = json.loads(cache_path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


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


_PSA_HEADERS = {'User-Agent': 'Mozilla/5.0'}
_PSA_TABLE_URL = 'https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/2018NEW/{table}'


def _fetch_live(category: str) -> Optional[dict]:
    """One PSA OpenSTAT PX-Web call for `category`'s specific commodity
    item, under 2M/2018NEW/ -- the confirmed-live folder (never 2M/RP or
    2M/NRP, both confirmed frozen at 2021). Geolocation is always
    '000000000' (Philippines, national) -- this table family keys
    Geolocation by 9-digit PSGC code, not the small sequential index
    ('0', '1', ...) psa_cpi.py's own table family uses for the same
    concept; confirmed against the live API for all four category tables
    after a hardcoded '0' silently returned zero rows (PSA had no
    Geolocation value matching '0', so the query matched nothing and
    failed with no error). The commodity id is resolved at request time by
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
            logging.warning('peso_anchor: commodity %r not found in table %s for %s',
                            item['commodity'], item['table'], category)
            return None

        year_var = by_code['Year']
        period_var = by_code['Period']
        month_ids = [pid for pid, txt in zip(period_var['values'], period_var['valueTexts'])
                    if txt != 'Annual']

        body = {
            'query': [
                {'code': 'Geolocation',
                 'selection': {'filter': 'item', 'values': ['000000000']}},
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
            # Each row's own parsing (value extraction, price, and
            # year/period resolution) is isolated here, same shape as
            # psa_cpi.py::_fetch_px_table's own per-row try/except -- one
            # malformed row (an empty or missing `values` list, a comma-
            # formatted price, a dash, a data row with a short key, or a
            # `None` entry in the Year variable's labels -- the same
            # "untranslated/missing label" PX-Web shape already handled for
            # Commodity below) must skip only that row, not discard a
            # `latest` value already found from an earlier, perfectly good
            # row.
            try:
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
                # A `None` (or otherwise non-4-digit) year label must never
                # silently become a "None-02"-shaped as_of string that then
                # gets cached and rendered to a real user.
                if not yyyy or not str(yyyy).isdigit() or len(str(yyyy)) != 4:
                    continue
                price = float(raw)
            except (ValueError, TypeError, IndexError, KeyError, AttributeError):
                logging.debug('peso_anchor: skipped an unparseable row for %s', category)
                continue
            as_of = f'{yyyy}-{month_num:02d}'
            if latest is None or as_of > latest[1]:
                latest = (price, as_of)

        if latest is None:
            logging.warning('peso_anchor: no valid price rows found for %s', category)
            return None
        price, as_of = latest
        return {'price': price, 'as_of': as_of}
    except (requests.RequestException, KeyError, ValueError, TypeError,
            IndexError, AttributeError) as exc:
        # IndexError/KeyError are reachable from metadata parsing above
        # (e.g. by_code['Commodity'], commodity_var['valueTexts']) whenever
        # PSA returns a variable with an empty/missing values list --
        # per-row causes (row['values'][0], year_labels[year_idx]) are now
        # isolated inside the per-row try/except above and skip just that
        # row instead of reaching here. AttributeError is reachable from
        # txt.strip() in the Commodity-label matching loop
        # whenever PSA returns None for an untranslated/missing valueTexts
        # entry -- a real PX-Web response shape, not hypothetical. Both are
        # malformed-but-not-impossible responses that must fall back to
        # cache (get_anchor), not crash the caller, same as every other
        # failure mode here.
        logging.warning('peso_anchor: fetch failed for %s: %s', category, exc)
        return None


#: How long a failed fetch attempt suppresses the NEXT attempt for the same
#: category. Not the documented synchronous-fetch design (already accepted,
#: not in scope to relitigate) -- just a floor so a persistent PSA outage
#: doesn't repeat the same ~50s-per-category network call on every single
#: Monitor render within a short window. Keyed under a reserved
#: '_failed_attempts' top-level cache entry, never inside a category's own
#: entry, so it never rides along with (and is never mistaken for) a real
#: price reading.
_FAILURE_COOLDOWN = timedelta(hours=1)
_FAILURES_KEY = '_failed_attempts'


def _recently_failed(cache: dict, category: str, now: datetime) -> bool:
    """True if the last fetch attempt for `category` failed within
    `_FAILURE_COOLDOWN`. A `failed_at` timestamped in the future (clock
    skew, or a cache file copied between machines with different clocks)
    must not keep the cooldown active indefinitely -- same hazard
    `_is_usable_if_stale` already guards on the other side of the cache
    with its own `0 <= age_days` check."""
    failures = cache.get(_FAILURES_KEY)
    if not isinstance(failures, dict):
        return False
    failed_at_raw = failures.get(category)
    if not failed_at_raw:
        return False
    try:
        failed_at = datetime.fromisoformat(failed_at_raw)
    except (ValueError, TypeError):
        return False
    elapsed = now - failed_at
    return timedelta() <= elapsed < _FAILURE_COOLDOWN


def _mark_failed(cache: dict, category: str, now: datetime) -> None:
    cache.setdefault(_FAILURES_KEY, {})[category] = now.isoformat()


def _clear_failed(cache: dict, category: str) -> None:
    failures = cache.get(_FAILURES_KEY)
    if isinstance(failures, dict):
        failures.pop(category, None)


def get_anchor(category: str, cache_path: Path = CACHE_PATH,
               today: Optional[date] = None,
               now: Optional[datetime] = None) -> Optional[dict]:
    """Today's price for `category`: {'price': float, 'as_of': 'YYYY-MM',
    'fetched_on': 'YYYY-MM-DD'}, or None if `category` isn't one of the
    four in scope, the live fetch fails and no usable cache exists, or the
    only cached entry is older than STALE_AFTER_DAYS. `today` is
    injectable for tests; defaults to the real current date. `now` is
    injectable the same way, for the negative-cache cooldown check;
    defaults to the real current datetime.

    A fetch that fails negative-caches for `_FAILURE_COOLDOWN` regardless
    of whether a usable stale cache entry is found to fall back to: the
    next call within that window skips `_fetch_live` entirely rather than
    repeating the same doomed, slow network call on every render during a
    PSA outage -- including (especially) the dominant real-world case
    where a stale cache keeps every call succeeding anyway."""
    if category not in CATEGORY_ITEMS:
        return None
    today = today or date.today()
    now = now or datetime.now()
    cache = _load_cache(cache_path)
    entry = cache.get(category)
    if entry and _is_fresh(entry, today):
        return entry

    if _recently_failed(cache, category, now):
        if entry and _is_usable_if_stale(entry, today):
            return entry
        return None

    fetched = _fetch_live(category)
    if fetched is not None:
        entry = {**fetched, 'fetched_on': today.isoformat()}
        cache[category] = entry
        _clear_failed(cache, category)
        _save_cache(cache, cache_path)
        return entry

    _mark_failed(cache, category, now)
    _save_cache(cache, cache_path)

    if entry and _is_usable_if_stale(entry, today):
        return entry

    return None
