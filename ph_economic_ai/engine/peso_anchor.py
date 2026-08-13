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
