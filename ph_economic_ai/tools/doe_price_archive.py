"""Index DOE's regional retail-price archive, so `Q-ENG-009` Phase 2 has actuals.

The app forecasts a per-region price CHANGE and has never been graded on one.
`ground_truth` scores the national figure only, and the seventeen regional numbers
on the map are thirteen arithmetic derivations of four debated estimates. Phase 0
labelled that. Phase 1 is the missing dependent variable.

## What the archive is

DOE's Oil Industry Management Bureau publishes weekly retail pump prices per
region as PDFs, listed under `category=Price+Monitoring` across the `lfo`, `vfo`,
`mfo` and `oimb` field-office pages. Enumerated 2026-07-31: **2713 distinct
documents spanning 2017 to 2026**, per province, city and brand, with RON 95 named
as its own row.

Coverage is not uniform and the gap is permanent:

    NCR              petro_ncr, petro_mm, ncr-price-monitoring     2017 to 2026
    South Luzon      petro_sluz, per PROVINCE                      2020 to 2025
    Visayas          petro_vis                                     2018 to 2025
    Mindanao         petro_min                                     2018 to 2025
    Central Luzon    NOTHING
    North Luzon      NOTHING

DOE publishes nothing north of NCR, verified with controls: zero files for Region
III, I, II or CAR against 1243 South Luzon, 470 NCR, 348 Visayas, 333 Mindanao. Of
the four region groups the swarm debates, three are covered and Central Luzon
cannot be validated from this source at all.

## Why the filenames need a real parser

The naming drifted over seven years and at least eight conventions are live:

    petro_ncr_2020_february_18          year, full month, day
    petro_ncr_2023-jun-01               year, short month, day
    petro_mm_2017_mar31                 month and day RUN TOGETHER
    petro_mm_2017_june6                 same, full month
    petro_sluz_2021-apr-08_laguna       province suffix
    petro_ncr_2024-apr-16-22            a DATE RANGE, not a day
    ncr-price-monitoring-05192026-pdf   MMDDYYYY, no separators
    47-lfro-price-monitoring-november-25-2025-pdf   sequence prefix

A file whose date cannot be read is useless for a backtest, so the parse rate is
reported rather than assumed, and unparsed names are listed rather than dropped
silently.

    python -m ph_economic_ai.tools.doe_price_archive --index
    python -m ph_economic_ai.tools.doe_price_archive --index --json out.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
from pathlib import Path
from typing import Optional

import requests

LISTINGS = ('lfo', 'vfo', 'mfo', 'oimb')
LISTING_URL = ('https://doe.gov.ph/site/{site}/articles/group/liquid-fuels'
               '?category=Price+Monitoring&display_type=Card')
HEADERS = {'User-Agent': 'Mozilla/5.0 (research; PH fuel price study)'}

ARTIFACT = (Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts'
            / 'doe_price_archive.json')

#: Filename fragment to the area it covers. Ordered: `petro_min` must be tested
#: before any looser pattern, and `petro-vis` is a one-off typo of `petro_vis`.
SERIES = (
    ('petro_ncr', 'NCR'),
    ('petro_mm', 'NCR'),
    ('ncr-price-monitoring', 'NCR'),
    ('petro_sluz', 'South Luzon'),
    ('petro_vis', 'Visayas'),
    ('petro-vis', 'Visayas'),
    ('petro_min', 'Mindanao'),
    ('lfro-price-monitoring', 'Luzon field office'),
    ('vfo-lf-price-monitoring', 'Visayas'),
    ('vfo-price-monitoring', 'Visayas'),
    ('mfo-price-monitoring', 'Mindanao'),
    ('region-iv-a', 'South Luzon'),
    ('region-iv-b', 'South Luzon'),
    ('region-v', 'South Luzon'),
)

_MONTHS = {m: i for i, m in enumerate(
    ('january february march april may june july august september october '
     'november december').split(), 1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})
#: DOE writes September four ways. `sept` alone accounted for 26 undated files.
_MONTHS['sept'] = 9
_MONTH_RE = '|'.join(sorted(_MONTHS, key=len, reverse=True))


def area_of(name: str) -> Optional[str]:
    for frag, area in SERIES:
        if frag in name:
            return area
    return None


def date_of(name: str) -> Optional[dt.date]:
    """First date in the filename, or None.

    A RANGE like `2024-apr-16-22` yields its START, because that is the week the
    prices took effect and it is what `vintage.fuel_cycle_start` aligns to.
    """
    n = name.lower()

    # YYYY-MM-DD all numeric, e.g. petro_ncr_2022-06-23
    m = re.search(r'(20\d\d)[-_](\d{1,2})[-_](\d{1,2})(?!\d)', n)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # MM-DD-YYYY separated, e.g. ncr-price-monitoring-07-01-2025
    m = re.search(r'(?<!\d)(\d{1,2})[-_](\d{1,2})[-_](20\d\d)(?!\d)', n)
    if m:
        try:
            return dt.date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    # MMDDYY, six digits, only where the series name makes it unambiguous:
    # vfo-lf-price-monitoring-102825 is 28 October 2025.
    m = re.search(r'price-monitoring-(?:for-)?(\d{2})(\d{2})(\d{2})(?!\d)', n)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            try:
                return dt.date(2000 + yy, mm, dd)
            except ValueError:
                pass

    # MMDDYYYY with no separators, e.g. ncr-price-monitoring-05192026
    m = re.search(r'(?<!\d)(\d{2})(\d{2})(20\d\d)(?!\d)', n)
    if m:
        mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            try:
                return dt.date(yy, mm, dd)
            except ValueError:
                pass

    # year first: 2020_february_18, 2023-jun-01, 2017_mar31, 2017_june6
    m = re.search(rf'(20\d\d)[_\-\s]*({_MONTH_RE})[_\-\s]*(\d{{1,2}})(?!\d)', n)
    if m:
        try:
            return dt.date(int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3)))
        except ValueError:
            return None

    # month first: november-25-2025, august-9-to-15 has no year and fails here
    m = re.search(rf'({_MONTH_RE})[_\-\s]*(\d{{1,2}})[_\-\s]+(20\d\d)', n)
    if m:
        try:
            return dt.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            return None

    # Month first with a RANGE, e.g. april-15-21-2025. Tried LAST and deliberately
    # so: placed before the simple shape it read `november-25-2025` as November 2,
    # splitting the 25 across the range and the year. An ambiguous pattern must
    # never pre-empt an unambiguous one.
    # august-9-to-15-2025. The start of the range is the effective week.
    m = re.search(rf'({_MONTH_RE})[_\-\s]*(\d{{1,2}})[_\-\s]*(?:to[_\-\s]*)?'
                  rf'\d{{1,2}}[_\-\s]*(20\d\d)', n)
    if m:
        try:
            return dt.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
        except ValueError:
            pass
    return None


def province_of(name: str) -> Optional[str]:
    """The province suffix on a South Luzon file, if present."""
    m = re.match(r'petro_sluz_[\d a-z\-_]*?[\-_](\d{1,2})[\-_](.+?)(?:-pdf)?$',
                 name.lower())
    if not m:
        return None
    tail = m.group(2)
    tail = re.sub(r'^(to[\-_]\d+[\-_]?)', '', tail)
    tail = tail.strip('-_')
    return tail.replace('_', ' ').replace('-', ' ') or None


# ── The PDF table ─────────────────────────────────────────────────────────────

#: Product rows DOE prints. RON 95 is the one the app forecasts; the rest are
#: kept because a grade-selection defect on the national baseline came from
#: pooling incomparable products, and having them separated makes that impossible
#: here.
PRODUCTS = ('RON 100', 'RON 97', 'RON 95', 'RON 91',
            'DIESEL PLUS', 'DIESEL', 'KEROSENE')

#: Words that mark the header row, which carries the brand column positions.
#: The brand set VARIES between documents, UNIOIL in one week and PHOENIX in
#: another, so the columns are read from each file rather than hardcoded.
_HEADER_MARKERS = ('PROVINCE', 'PRODUCT')

#: Vertical tolerance when grouping words into a row. `RON 95` and its prices sit
#: three points apart in the sample, so a strict y match splits a row in half and
#: silently drops every price on it.
_ROW_TOL = 4.0


def _rows_from_words(words) -> list[list]:
    """Group (x0, y0, ...) word tuples into visual rows, y-tolerant.

    `RON 97` and its prices sit four points apart in the sample, so a strict y
    match splits a row in half and drops every price on it.
    """
    out: list[list] = []
    for w in sorted(words, key=lambda w: (w[1], w[0])):
        if out and abs(w[1] - out[-1][0][1]) <= _ROW_TOL:
            out[-1].append(w)
        else:
            out.append([w])
    return [sorted(r, key=lambda w: w[0]) for r in out]


#: Column boundaries in PDF points, read off the header row: PROVINCE at x=22,
#: CITY/MUNICIPALITY at x=84, PRODUCT at x=144, the first brand at x=214.
_X_CITY, _X_PRODUCT, _X_PRICES = 82.0, 140.0, 212.0

#: Where the table stops. DOE closes each sheet with monitoring metadata printed
#: in the province and city columns.
_FOOTER_RE = re.compile(r'DATE\s+MONITOR|PRICES?\s+AS\s+OF|PREPARED\s+BY|NOTE\s*:')


def _num(token: str):
    try:
        v = float(token.replace(',', ''))
    except ValueError:
        return None
    # 0.00 is DOE's "not sold in this city", not a price of zero.
    return v if 0.0 < v < 500.0 else None


def _row_prices(row) -> tuple:
    """(low, high, common) for one product row, located by layout not by order.

    The published range is written `58.45 - 66.60`, so the widest-right hyphen
    brackets it, and anything numeric further right is the COMMON PRICE column.
    DOE writes `None` there when it declines to name one.

    Located this way rather than by taking the last few numbers because the
    PRODUCT label is itself numeric: an earlier version read the 95 of `RON 95`
    as a price and reported a common price of 95.00.
    """
    dashes = [w for w in row if w[4] == '-' and w[0] > _X_PRICES]
    if not dashes:
        return None, None, None
    dash_x = max(d[0] for d in dashes)
    nums = [(w[0], _num(w[4])) for w in row if w[0] > _X_PRICES]
    nums = [(x, v) for x, v in nums if v is not None]
    left = [v for x, v in nums if x < dash_x]
    right = [(x, v) for x, v in nums if x > dash_x]
    low = left[-1] if left else None
    high = right[0][1] if right else None
    common = right[1][1] if len(right) > 1 else None
    return low, high, common


def _province_bands(page) -> list[float]:
    """Y positions of the horizontal rules bounding the PROVINCE column cells.

    The table HAS ruling lines; only the text layer lacks them. Reading the
    drawings gives the exact cell a province owns, which nearest-centre
    assignment could only approximate: a province label is centred against every
    city block it contains, so at a boundary the nearest label can belong to the
    neighbouring province. One city in the sample landed in the wrong one.

    Returns an empty list when a document has no usable rules, and the caller
    falls back to proximity rather than dropping the page.
    """
    ys: list[float] = []
    for item in page.get_drawings():
        for op in item['items']:
            if op[0] == 'l':
                a, b = op[1], op[2]
                if abs(a.y - b.y) < 0.6 and abs(b.x - a.x) > 10:
                    lo, hi = min(a.x, b.x), max(a.x, b.x)
                    if lo < 40 and hi > 60:
                        ys.append(a.y)
            elif op[0] == 're':
                r = op[1]
                if r.x0 < 40 and r.x1 > 60:
                    ys.extend((r.y0, r.y1))
    # Rules are drawn as near-duplicate pairs a fraction of a point apart.
    out: list[float] = []
    for y in sorted(ys):
        if not out or y - out[-1] > 1.5:
            out.append(y)
    return out


def _band_of(y: float, bands: list[float]) -> Optional[int]:
    """Index of the province cell containing `y`, or None."""
    for i in range(len(bands) - 1):
        if bands[i] <= y < bands[i + 1]:
            return i
    return None


def parse_price_pdf(content: bytes) -> list[dict]:
    """Rows of {province, city, product, low, high, common} from one DOE PDF.

    `common` is DOE's own COMMON PRICE column, preferred over any average this
    code could compute: an average over brand columns weights a station that
    happens to report equally with one that does not, and DOE already publishes
    the figure it stands behind. It is often absent, written `None`, in which case
    the range is what the document offers.

    Parsed in BLOCKS rather than row by row. Each city gets seven product rows,
    RON 100 through KEROSENE, and its label is printed once, vertically centred
    against the block rather than on its first row. Carrying a label forward
    row-by-row therefore attaches it to the wrong city, which is what an earlier
    version did.

    Province cells come from the DRAWING layer. The text layer has no rules, but
    the table is drawn with them, and reading them is what settled an assignment
    that proximity could only approximate.
    """
    import fitz

    doc = fitz.open(stream=content, filetype='pdf')
    out: list[dict] = []
    for page in doc:
        rows = _rows_from_words(page.get_text('words'))

        # Nothing above the column header is data. The sheet's letterhead sits in
        # the same x range as the province and city columns, so parsing from the
        # top made "Republic DEPARTMENT" a province.
        first = next((i for i, r in enumerate(rows)
                      if 'PROVINCE' in ' '.join(w[4] for w in r).upper()
                      and 'PRODUCT' in ' '.join(w[4] for w in r).upper()), None)
        if first is None:
            continue

        provinces: list[tuple[float, str]] = []   # (y centre, label)
        blocks: list[dict] = []
        block = None
        pending_prov: list[tuple[float, str]] = []

        for row in rows[first + 1:]:
            text = ' '.join(w[4] for w in row).upper()
            if 'MUNICIPALITY' in text or text.startswith('CITY/'):
                continue
            if _FOOTER_RE.search(text):
                break
            product = next((p for p in PRODUCTS if p in text), None)

            if product == PRODUCTS[0]:
                block = {'y0': row[0][1], 'y1': row[0][1], 'city': [], 'rows': []}
                blocks.append(block)

            for w in row:
                if _num(w[4]) is not None or ':' in w[4]:
                    continue
                if w[0] < _X_CITY:
                    pending_prov.append((w[1], w[4]))
                elif w[0] < _X_PRODUCT and block is not None:
                    block['city'].append(w[4])

            if product and block is not None:
                block['rows'].append((product, *_row_prices(row)))
                block['y1'] = row[0][1]

        # A province name may WRAP onto two lines, `Camarines` then `Norte` with a
        # city printed between them, so words within a few points of each other
        # are one label.
        for y, word in sorted(pending_prov):
            if provinces and y - provinces[-1][0] <= 14.0:
                provinces[-1] = (provinces[-1][0], provinces[-1][1] + ' ' + word)
            else:
                provinces.append((y, word))

        # A province is printed ONCE, vertically centred against however many city
        # blocks it contains, so it can appear below the first city it covers.
        # Nearest-centre assignment handles that; carrying forward does not.
        bands = _province_bands(page)
        # Label per province CELL, from the ruling lines. Falls back to nearest
        # centre only when a document has no usable rules.
        by_band: dict = {}
        for y, label in provinces:
            idx = _band_of(y, bands)
            if idx is not None:
                by_band[idx] = label

        for b in blocks:
            centre = (b['y0'] + b['y1']) / 2
            prov = by_band.get(_band_of(centre, bands))
            if prov is None:
                prov = (min(provinces, key=lambda pv: abs(pv[0] - centre))[1]
                        if provinces else None)
            city = ' '.join(b['city']).strip() or None
            for product, low, high, common in b['rows']:
                out.append({'province': prov, 'city': city, 'product': product,
                            'low': low, 'high': high, 'common': common})
    return out


def list_documents() -> dict[str, str]:
    """{filename: url} across every field-office listing.

    Documents live under several owner paths, `/documents/d/guest/`,
    `/documents/d/oimb/`, `/documents/d/lfo/`. An earlier scrape hard-coded
    `guest` and silently dropped four fifths of the corpus while looking like a
    listing cap, so the URL is kept whole rather than rebuilt from the name.
    """
    out: dict[str, str] = {}
    for site in LISTINGS:
        r = requests.get(LISTING_URL.format(site=site), headers=HEADERS, timeout=60)
        r.raise_for_status()
        for href in re.findall(r'href=["\']([^"\']+)["\']', r.text):
            if 'prod-cms' in href and '/documents/' in href:
                out.setdefault(href.rsplit('/', 1)[-1], href)
    return out


def build_index(docs: dict[str, str]) -> dict:
    rows, unparsed = [], []
    for name, url in sorted(docs.items()):
        area, when = area_of(name), date_of(name)
        if area is None or when is None:
            unparsed.append(name)
            continue
        rows.append({'name': name, 'url': url, 'area': area,
                     'date': when.isoformat(), 'province': province_of(name)})
    return {'rows': rows, 'unparsed': unparsed, 'total': len(docs)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--index', action='store_true')
    ap.add_argument('--json', type=Path, default=ARTIFACT)
    args = ap.parse_args()
    if not args.index:
        ap.print_help()
        return 0

    print('Enumerating DOE price-monitoring listings...')
    docs = list_documents()
    idx = build_index(docs)
    rows, unparsed = idx['rows'], idx['unparsed']
    print(f'  {idx["total"]} documents, {len(rows)} indexed, '
          f'{len(unparsed)} unparsed ({100 * len(rows) / max(idx["total"], 1):.1f}% parsed)\n')

    by_area = collections.defaultdict(list)
    for r in rows:
        by_area[r['area']].append(r['date'][:4])
    print(f'{"area":<22}{"files":>7}  years')
    for area in sorted(by_area, key=lambda a: -len(by_area[a])):
        yrs = sorted(set(by_area[area]))
        print(f'{area:<22}{len(by_area[area]):>7}  {yrs[0]} to {yrs[-1]}')

    if unparsed:
        print(f'\n{len(unparsed)} names could not be dated or placed. '
              f'A backtest cannot use these; they are listed, not dropped:')
        for n in unparsed[:12]:
            print(f'   {n[:80]}')
        if len(unparsed) > 12:
            print(f'   ... and {len(unparsed) - 12} more')

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(idx, indent=2), encoding='utf-8')
    print(f'\nWrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
