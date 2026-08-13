# Peso-Anchored Food Sub-Category Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For rice, meat, fish, and vegetables, Monitor's Food card shows a real current peso price alongside the existing percentage forecast, plus a projected price computed from the two. Dairy & eggs and sugar are provably untouched.

**Architecture:** A new, independent module (`ph_economic_ai/engine/peso_anchor.py`) fetches and caches one specific PSA commodity item per category from the confirmed-live `openstat.psa.gov.ph` `2M/2018NEW/` table family, with a 60-day staleness bound. `pressure_monitor.py`'s `_sector_card` reads this module directly and combines its output with `SectorReading.subcategories` (already shipped) at render time — no changes to `SectorReading`, `debate.py`, or `forum.py` at all.

**Tech Stack:** Python, PyQt6, `requests`, pytest. No new dependencies.

## Global Constraints

- The four fetchers must resolve against `2M/2018NEW/` specifically. `2M/RP` and `2M/NRP` are both confirmed frozen at 2021 (2026-08-13 research pass) and must never be used — fetching from either would silently anchor the app to five-year-old prices with no error.
- A category missing either half of the pair (peso price or debate percentage) never gets a fabricated stand-in value — renders `—`, the same convention the existing percentage-only line already uses.
- The exploratory-projection caption is present whenever the peso strip renders any real content, and absent when it renders none.
- No real network call in any test — every `requests.get`/`requests.post` call is mocked. This project's own history (most recently `RSK-056`'s investigation) is what happens when tests touch live services.
- The 60-day staleness cutoff must not fire on PSA's own normal ~30-day reporting lag — tested at the exact boundary (59/60/61 days), not just "old" vs "new."
- Geolocation is always the national figure (value id `'0'`, "Philippines"), matching every other PSA fetch already in this codebase.

---

### Task 1: `peso_anchor.py` — constants, projection math, cache read/write, staleness check

**Files:**
- Create: `ph_economic_ai/engine/peso_anchor.py`
- Test: `ph_economic_ai/tests/test_peso_anchor.py`

**Interfaces:**
- Produces: `CATEGORY_ITEMS: dict[str, dict]` (the four in-scope categories, each with `table` and `commodity` keys); `CACHE_PATH: Path`; `STALE_AFTER_DAYS: int = 60`; `project(anchor_price: float, pct_change: float) -> float`; `_load_cache(cache_path: Path = CACHE_PATH) -> dict`; `_save_cache(cache: dict, cache_path: Path = CACHE_PATH) -> None`; `_is_fresh(entry: dict, today: date) -> bool`; `_is_usable_if_stale(entry: dict, today: date) -> bool` — all consumed by Task 2's `get_anchor`.

- [ ] **Step 1: Write the failing tests**

Create `ph_economic_ai/tests/test_peso_anchor.py`:

```python
from datetime import date, timedelta

import pytest

from ph_economic_ai.engine import peso_anchor


def test_project_raises_the_price_for_a_positive_percent():
    assert peso_anchor.project(100.0, 5.0) == pytest.approx(105.0)


def test_project_lowers_the_price_for_a_negative_percent():
    """The sign has to actually flip the direction of the arithmetic, not
    just format with a minus -- this is real multiplication."""
    assert peso_anchor.project(100.0, -5.0) == pytest.approx(95.0)


def test_project_is_a_noop_at_zero_percent():
    assert peso_anchor.project(52.36, 0.0) == pytest.approx(52.36)


def test_load_cache_returns_empty_dict_when_file_is_missing(tmp_path):
    assert peso_anchor._load_cache(tmp_path / 'nope.json') == {}


def test_load_cache_returns_empty_dict_when_file_is_corrupt(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_text('{not valid json', encoding='utf-8')
    assert peso_anchor._load_cache(p) == {}


def test_save_then_load_cache_round_trips(tmp_path):
    p = tmp_path / 'cache.json'
    peso_anchor._save_cache({'rice': {'price': 52.36}}, p)
    assert peso_anchor._load_cache(p) == {'rice': {'price': 52.36}}


def test_save_cache_creates_the_parent_directory_if_missing(tmp_path):
    p = tmp_path / 'nested' / 'cache.json'
    peso_anchor._save_cache({'rice': {'price': 52.36}}, p)
    assert peso_anchor._load_cache(p) == {'rice': {'price': 52.36}}


def test_is_fresh_true_for_an_entry_fetched_today():
    today = date(2026, 8, 13)
    entry = {'fetched_on': '2026-08-13'}
    assert peso_anchor._is_fresh(entry, today) is True


def test_is_fresh_false_for_an_entry_fetched_yesterday():
    today = date(2026, 8, 13)
    entry = {'fetched_on': '2026-08-12'}
    assert peso_anchor._is_fresh(entry, today) is False


def test_is_fresh_false_for_a_missing_fetched_on():
    assert peso_anchor._is_fresh({}, date(2026, 8, 13)) is False


def test_is_usable_if_stale_true_at_the_59_day_boundary():
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=59)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is True


def test_is_usable_if_stale_true_at_exactly_60_days():
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=60)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is True


def test_is_usable_if_stale_false_at_61_days():
    """The boundary itself, not just 'a long time ago' -- 61 days is where
    RSK-041's lesson actually bites: PSA's own ~30-day lag is well inside
    this, so the cutoff must not fire on normal reporting cadence."""
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=61)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is False


def test_is_usable_if_stale_false_for_a_missing_fetched_on():
    assert peso_anchor._is_usable_if_stale({}, date(2026, 8, 13)) is False


def test_category_items_covers_exactly_the_four_confirmed_categories():
    assert set(peso_anchor.CATEGORY_ITEMS) == {'rice', 'meat', 'fish', 'vegetables'}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ph_economic_ai/tests/test_peso_anchor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ph_economic_ai.engine.peso_anchor'`

- [ ] **Step 3: Write the implementation**

Create `ph_economic_ai/engine/peso_anchor.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ph_economic_ai/tests/test_peso_anchor.py -v`
Expected: All 16 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/peso_anchor.py ph_economic_ai/tests/test_peso_anchor.py
git commit -m "feat(engine): peso_anchor scaffolding -- constants, projection math, cache

Pure functions and cache read/write only, no network yet. project()
verified to actually flip the price direction on a negative percent, not
just format one. The 60-day staleness bound tested at its exact boundary
(59/60/61 days), not just 'old' vs 'new' -- RSK-041 already taught this
project what a silent staleness bug costs."
```

---

### Task 2: `peso_anchor.py` — live fetch and the `get_anchor` orchestrator

**Files:**
- Modify: `ph_economic_ai/engine/peso_anchor.py`
- Test: `ph_economic_ai/tests/test_peso_anchor.py`

**Interfaces:**
- Consumes: `CATEGORY_ITEMS`, `_load_cache`, `_save_cache`, `_is_fresh`, `_is_usable_if_stale` (Task 1); `_MONTHS` from `ph_economic_ai.benchmark.psa_cpi` (existing, reused rather than duplicated).
- Produces: `_fetch_live(category: str) -> Optional[dict]` (returns `{'price': float, 'as_of': 'YYYY-MM'}` or `None`, never raises); `get_anchor(category: str, cache_path: Path = CACHE_PATH, today: Optional[date] = None) -> Optional[dict]` (returns `{'price': float, 'as_of': str, 'fetched_on': str}` or `None`) — consumed by Task 3's UI integration.

- [ ] **Step 1: Write the failing tests**

Add to `ph_economic_ai/tests/test_peso_anchor.py`:

```python
import json as jsonlib
from unittest.mock import MagicMock


def test_fetch_live_returns_none_for_a_category_not_in_scope():
    assert peso_anchor._fetch_live('sugar') is None


def _fake_meta_response(commodity_label):
    r = MagicMock()
    r.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['0'], 'valueTexts': ['Philippines']},
            {'code': 'Commodity', 'values': ['5'], 'valueTexts': [commodity_label]},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': ['2018', '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    return r


def _fake_post_response(rows):
    r = MagicMock()
    r.content = jsonlib.dumps({'data': rows}).encode('utf-8')
    r.raise_for_status = lambda: None
    return r


def test_fetch_live_parses_a_successful_response(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    # key = [Geolocation, Commodity, Year, Period] value-ids;
    # year id '1' -> 2019 (index 1), period id '1' -> February.
    post = _fake_post_response([{'key': ['0', '5', '1', '1'], 'values': ['55.41']}])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


def test_fetch_live_picks_the_most_recent_row_when_several_exist(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '0', '0'], 'values': ['50.00']},   # 2018-01
        {'key': ['0', '5', '1', '1'], 'values': ['55.41']},   # 2019-02, latest
        {'key': ['0', '5', '0', '11'], 'values': ['51.00']},  # not present here, illustrative
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


def test_fetch_live_skips_annual_and_blank_rows(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '1', '12'], 'values': ['999.00']},  # Annual, must be skipped
        {'key': ['0', '5', '0', '0'], 'values': ['..']},       # blank, must be skipped
        {'key': ['0', '5', '0', '1'], 'values': ['48.20']},    # 2018-02, the only real row
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 48.20, 'as_of': '2018-02'}


def test_fetch_live_returns_none_when_the_network_call_raises(monkeypatch):
    def raise_error(*a, **kw):
        raise peso_anchor.requests.RequestException('boom')
    monkeypatch.setattr(peso_anchor.requests, 'get', raise_error)
    assert peso_anchor._fetch_live('rice') is None


def test_fetch_live_returns_none_when_the_commodity_label_is_not_found(monkeypatch):
    meta = _fake_meta_response('SOME OTHER ITEM, 1 KG')
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    assert peso_anchor._fetch_live('rice') is None


def test_get_anchor_returns_none_for_a_category_not_in_scope():
    assert peso_anchor.get_anchor('sugar') is None


def test_get_anchor_uses_a_fresh_same_day_cache_without_fetching(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    peso_anchor._save_cache(
        {'rice': {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'}},
        cache_path)

    def fail_if_called(category):
        raise AssertionError('should not fetch when cache is fresh')
    monkeypatch.setattr(peso_anchor, '_fetch_live', fail_if_called)

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'}


def test_get_anchor_fetches_and_caches_when_todays_entry_is_missing(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    monkeypatch.setattr(peso_anchor, '_fetch_live',
                        lambda category: {'price': 52.90, 'as_of': '2026-08'})

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 52.90, 'as_of': '2026-08', 'fetched_on': '2026-08-13'}
    assert peso_anchor._load_cache(cache_path)['rice']['price'] == 52.90


def test_get_anchor_falls_back_to_a_usable_stale_cache_when_the_fetch_fails(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    peso_anchor._save_cache(
        {'rice': {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}},
        cache_path)
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}


def test_get_anchor_returns_none_when_fetch_fails_and_cache_is_too_stale(tmp_path, monkeypatch):
    from datetime import timedelta
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    too_old = (today - timedelta(days=61)).isoformat()
    peso_anchor._save_cache(
        {'rice': {'price': 45.00, 'as_of': '2026-06', 'fetched_on': too_old}}, cache_path)
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)

    assert peso_anchor.get_anchor('rice', cache_path=cache_path, today=today) is None


def test_get_anchor_returns_none_when_fetch_fails_and_no_cache_exists(tmp_path, monkeypatch):
    cache_path = tmp_path / 'nope.json'
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)
    assert peso_anchor.get_anchor('rice', cache_path=cache_path, today=date(2026, 8, 13)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ph_economic_ai/tests/test_peso_anchor.py -v`
Expected: FAIL with `AttributeError: module 'ph_economic_ai.engine.peso_anchor' has no attribute '_fetch_live'` (and similar for `get_anchor`).

- [ ] **Step 3: Write the implementation**

Add to `ph_economic_ai/engine/peso_anchor.py` (after `_is_usable_if_stale`):

```python
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

        geo_var = by_code['Geolocation']
        year_var = by_code['Year']
        period_var = by_code['Period']
        month_ids = [pid for pid, txt in zip(period_var['values'], period_var['valueTexts'])
                    if txt != 'Annual']

        body = {
            'query': [
                {'code': 'Geolocation',
                 'selection': {'filter': 'item', 'values': [geo_var['values'][0]]}},
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
    except (requests.RequestException, KeyError, ValueError, TypeError):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ph_economic_ai/tests/test_peso_anchor.py -v`
Expected: All 27 tests PASS (16 from Task 1 + 11 new).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/peso_anchor.py ph_economic_ai/tests/test_peso_anchor.py
git commit -m "feat(engine): peso_anchor live fetch and get_anchor orchestration

_fetch_live reuses psa_cpi.py's own commodity-label-resolution shape,
targets 2M/2018NEW/ specifically (never the two confirmed-stale sibling
folders), and never raises -- get_anchor falls back to a usable cache
entry on any fetch failure, and to None only when neither a fresh fetch
nor a usable cache exists. No real network call in any test."
```

---

### Task 3: Monitor's Food card shows the peso strip

**Files:**
- Modify: `ph_economic_ai/ui/pressure_monitor.py`
- Test: `ph_economic_ai/tests/test_monitor.py`

**Interfaces:**
- Consumes: `peso_anchor.get_anchor`, `peso_anchor.project` (Task 2); `SectorReading.subcategories`, `_CATEGORY_DISPLAY_LABELS` (both existing, unmodified).

- [ ] **Step 1: Write the failing tests**

Read `ph_economic_ai/tests/test_monitor.py`'s existing `test_food_card_shows_subcategory_breakdown` (around line 76) first to confirm the exact fixture/import pattern this file uses, then add immediately after it:

```python
def test_food_card_shows_peso_anchor_strip(app, monkeypatch):
    """Rice/meat/fish/vegetables get a peso anchor + projected price; a
    category missing either the anchor or the debate percentage shows '—'
    in that slot, never inventing a stand-in value."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.engine import peso_anchor
    from PyQt6.QtWidgets import QLabel

    def fake_get_anchor(category, *a, **kw):
        prices = {
            'rice': {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
            'meat': {'price': 185.40, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
            'fish': None,  # simulate a fetch failure for this one category
            'vegetables': {'price': 62.10, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
        }
        return prices.get(category)
    monkeypatch.setattr(peso_anchor, 'get_anchor', fake_get_anchor)

    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('food', 'rising', 0.4, '%', 64,
                      estimates=[0.3, 0.4, 0.5],
                      subcategories={'rice': 0.3, 'meat': -0.3, 'fish': 0.8,
                                     'vegetables': 0.5})
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))

    assert 'Rice ₱52.36 → ₱52.52' in texts
    assert 'Meat ₱185.40 → ₱184.85' in texts
    assert 'Fish —' in texts   # anchor fetch failed even though a percentage exists
    assert 'Vegetables ₱62.10 → ₱62.41' in texts
    assert 'exploratory projection, not a validated prediction' in texts


def test_food_card_omits_the_peso_strip_when_no_category_has_both_pieces(app, monkeypatch):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.engine import peso_anchor
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(peso_anchor, 'get_anchor', lambda category, *a, **kw: None)

    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('food', 'rising', 0.4, '%', 64,
                      estimates=[0.3], subcategories={'sugar': 0.1})  # not a peso-anchor category
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'exploratory projection' not in texts


def test_gas_card_is_unaffected_by_the_peso_strip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from PyQt6.QtWidgets import QLabel

    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.20, '₱/L', 70, estimates=[1.2])
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'exploratory projection' not in texts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ph_economic_ai/tests/test_monitor.py::test_food_card_shows_peso_anchor_strip ph_economic_ai/tests/test_monitor.py::test_food_card_omits_the_peso_strip_when_no_category_has_both_pieces ph_economic_ai/tests/test_monitor.py::test_gas_card_is_unaffected_by_the_peso_strip -v`
Expected: The first test FAILs with `AssertionError` (no peso text present yet). The other two currently PASS vacuously (nothing to assert against yet) — confirm this, then proceed; they'll stay meaningful once Step 3 lands.

- [ ] **Step 3: Add the peso strip**

In `ph_economic_ai/ui/pressure_monitor.py`, add the import near the top of the file (alongside the existing `from ph_economic_ai.ui import honesty as _honesty`-style imports):

```python
from ph_economic_ai.engine import peso_anchor
```

In `_sector_card`, insert immediately before `return card` (after the existing `note` widget that reads "Separate per-category reads..."), still inside the `if r.sector == 'food' and getattr(r, 'subcategories', None):` block:

```python
            anchor_parts = []
            as_of_month = None
            for category in ('rice', 'meat', 'fish', 'vegetables'):
                label = _CATEGORY_DISPLAY_LABELS[category]
                pct = r.subcategories.get(category)
                anchor = peso_anchor.get_anchor(category) if pct is not None else None
                if anchor is not None and pct is not None:
                    projected = peso_anchor.project(anchor['price'], pct)
                    anchor_parts.append(
                        f"{label} ₱{anchor['price']:.2f} → ₱{projected:.2f}")
                    as_of_month = anchor['as_of']
                else:
                    anchor_parts.append(f'{label} —')

            if as_of_month is not None:
                strip = QLabel('  ·  '.join(anchor_parts))
                strip.setStyleSheet(f'color:{_T2};font-size:11px;margin-top:8px;')
                strip.setWordWrap(True)
                lay.addWidget(strip)

                caption = QLabel(
                    f'PSA retail price (as of {as_of_month}) × this debate\'s '
                    'forecast — exploratory projection, not a validated prediction.')
                caption.setStyleSheet(
                    f'color:{_T3};font-size:10px;font-style:italic;margin-top:2px;')
                caption.setWordWrap(True)
                lay.addWidget(caption)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ph_economic_ai/tests/test_monitor.py -v`
Expected: All tests PASS, including the three new ones and every pre-existing test in the file (gas/electricity cards, and the existing percentage-only breakdown test, must render exactly as before).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/pressure_monitor.py ph_economic_ai/tests/test_monitor.py
git commit -m "feat(ui): Monitor's Food card shows the peso anchor + projection

Rice/meat/fish/vegetables each get \"Label ₱anchor → ₱projected\" when
both the PSA price and this cycle's debate percentage are available;
either missing shows '—' for that category, never a fabricated value.
One shared caption names the anchor's as-of month and states the
projection is exploratory, not validated -- present only when the strip
has real content. Dairy & eggs, sugar, gas, and electricity are
untouched; the new code path lives entirely inside the existing
food-only gate."
```

---

## Final verification

- [ ] Run the complete affected-area suite:

```bash
python -m pytest ph_economic_ai/tests/test_peso_anchor.py ph_economic_ai/tests/test_monitor.py ph_economic_ai/tests/test_psa_cpi.py ph_economic_ai/tests/test_main_window.py -v
```
Expected: All PASS.

- [ ] Run the full repo suite: `python -m pytest ph_economic_ai/tests -q --no-header`
Expected: All PASS, count higher than the pre-plan baseline by exactly the number of new tests added across all three tasks (27 in `test_peso_anchor.py` + 3 in `test_monitor.py` = 30).

- [ ] Visually verify by hand (per this session's own established method — do not force `QT_QPA_PLATFORM=offscreen`, confirmed to render text as tofu boxes on this machine): build a `SimMainWindow` with a populated store, populate the Monitor page with a `PressureBrief` whose food `SectorReading` has all four peso-anchor categories present, and confirm the strip reads correctly, including the projected-price arithmetic actually matching what a reader would compute by hand.

- [ ] Confirm `ph_economic_ai/cache/bantay_cache.json` is genuinely gitignored (already an existing `.gitignore` entry, predating this plan) by running the app once, letting a real fetch populate the file, then `git status` — the file must not appear as untracked-and-stageable content the app would accidentally commit.
