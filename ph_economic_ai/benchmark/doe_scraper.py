"""Scrape DOE weekly retail gasoline prices and aggregate to monthly.

`parse_doe_prices()` and `to_monthly()` are pure and unit-tested. `fetch_doe()`
performs the network call (thin, not unit-tested).

KNOWN LIMITATION (verified 2026-06): `doe.gov.ph/oil-monitor` currently returns
HTTP 500; only `legacy.doe.gov.ph/oil-monitor` responds (200). That page,
however, does NOT embed the prevailing-price table inline — the weekly figures
live in linked bulletin files (PDF/Excel), so `parse_doe_prices()` typically
finds zero rows against the live HTML. The parser remains correct for any DOE
markup that DOES contain inline dated RON95 rows (older layouts, mirrored
tables). Populating the live track record from DOE therefore needs a bulletin
(PDF) parser as a follow-up; until then the World Bank frozen backtest carries
the headline accuracy claim, exactly as the design intends.
"""
import re
from collections import defaultdict

import requests

DOE_URL = 'https://legacy.doe.gov.ph/oil-monitor'
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Matches a date and a nearby gasoline price. Tolerant of intervening HTML tags.
_ROW_RE = re.compile(
    r'(\d{4}-\d{2}-\d{2}).*?(?:RON\s*95|Gasoline)\D*([0-9]{2,3}\.[0-9]{2})',
    re.IGNORECASE | re.DOTALL,
)


def parse_doe_prices(html: str) -> dict:
    """Return {YYYY-MM-DD: php_per_liter} for RON95/Gasoline rows found."""
    out = {}
    for m in _ROW_RE.finditer(html):
        out[m.group(1)] = float(m.group(2))
    return out


def to_monthly(daily: dict) -> dict:
    """Average daily/weekly prices into {YYYY-MM: mean_php_per_liter}."""
    buckets = defaultdict(list)
    for date_str, val in daily.items():
        buckets[date_str[:7]].append(val)
    return {ym: round(sum(v) / len(v), 2) for ym, v in buckets.items()}


def fetch_doe(timeout: int = 8) -> dict:
    """Network fetch -> monthly prices. Returns {} on failure or empty parse."""
    try:
        r = requests.get(DOE_URL, headers=_HEADERS, timeout=timeout)
        r.raise_for_status()
        return to_monthly(parse_doe_prices(r.text))
    except Exception:
        return {}
