"""Fetch the DOE archive into a cache, and build the regional RON 95 panel.

The second half of `Q-ENG-009` Phase 1. `doe_price_archive` established that the
source exists and can be read; this turns 2713 remote PDFs into a committed panel
that Phase 2 can regress, with a provenance record per `DEC-020`.

## Why a cache rather than a fetch-and-parse pass

Three reasons, all learned rather than assumed:

* **The parser is not finished changing.** Five defects came out of the table
  layout and each needed the ORIGINAL bytes to reproduce. Re-downloading 2713
  files per fix is both slow and rude to DOE's server.
* **A run must be resumable.** At one request per second the corpus is most of an
  hour, and a run that loses everything on a timeout will be run less carefully.
* **The bytes are the evidence.** `ADR-008` fixed a national series whose inputs
  could not be rebuilt from what was committed. Caching the source documents and
  hashing them means a regional row can always be traced to the page it came from.

The cache is deliberately NOT committed: 650 MB of PDFs for 1271 documents. The
manifest and the panel are, and the manifest carries a SHA-256 per file so a
re-fetch can prove the cache still holds what it claims.

## The publication date is not the pricing week

Only 1767 of 2520 dated files fall on a Tuesday. 538 are Thursday, and there is a
tail through every other weekday. A monitoring date is when DOE walked the
stations, not when the price took effect, so every date is snapped BACK to the
Tuesday 06:00 boundary that opened the cycle it falls in. A Monday file belongs to
the cycle that opened six days earlier, not to the one starting tomorrow.

Getting this backwards would shift a third of the corpus by one week, which is
`DEC-045`'s failure mode one level up: not a wrong file, a wrongly-placed one.

## What the panel contains

RON 95 only, selected by name per `DEC-043`, at the grain DOE publishes: one row
per document, province, city and pricing cycle. **Aggregating cities to a regional
level is a Phase 2 decision and is deliberately not made here.** `DEC-010` and
`DEC-021` both exist because a modelling choice made in passing is a choice nobody
declared, and choosing between a median city, a mean, and a population weight
changes the dependent variable.

## Coverage is not what the date range says

The panel spans 2019 to 2026, and no region has that. NCR runs the whole of it at
about 50 weeks a year. Western Visayas has 1, 1, 2 and 7 weeks for 2019 to 2022
and then 36 to 46 a year from 2023, because the earlier weeks were published as
scans with no text layer. Davao sits between the two.

So `--build` reports weeks PER YEAR per debated region, and Central Luzon appears
in that table with a zero rather than being left out. A regression that read the
headline span as its window would fit three regions on three different periods.

Roughly 400 cached documents yield no rows: most are those scans, the rest use
pre-2020 layouts. They are counted and named, never dropped quietly.

    python -m ph_economic_ai.tools.doe_price_series --fetch
    python -m ph_economic_ai.tools.doe_price_series --fetch --areas NCR --limit 20
    python -m ph_economic_ai.tools.doe_price_series --build
    python -m ph_economic_ai.tools.doe_price_series --verify
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import requests

from ph_economic_ai.benchmark.paths import DATA_DIR, artifact, data
from ph_economic_ai.benchmark.provenance import sha256_file, write_record
from ph_economic_ai.engine import price_calendar, vintage
from ph_economic_ai.tools.doe_price_archive import (
    ARTIFACT as INDEX_ARTIFACT, HEADERS, area_of, date_of, parse_price_pdf,
)

#: Source PDFs. Gitignored: about 250 MB, and rebuildable from the manifest.
CACHE_DIR = DATA_DIR / 'doe_cache'

#: What was fetched, when, and what it hashed to. Committed; small.
MANIFEST = artifact('doe_price_cache.json')

#: The dependent variable Phase 2 regresses. Committed.
PANEL = data('doe_regional_ron95.csv')

PRODUCT = 'RON 95'

#: The provinces DOE's Visayas and Mindanao sheets can legitimately name.
#: Reference data, not a correction table: a province either IS one of these or
#: the cell was not read.
#:
#: Some documents carry an OCR'd text layer rather than a typeset one, and it
#: produces province names that are wrong in a way no schema catches --
#: `Llnao dal Norto`, `Zamboanga dsl Nodo`, `ilitamlt Occldontrl`. Each appears
#: once or twice against thousands of clean rows, so they never move an
#: aggregate enough to be noticed, and they silently split one province's series
#: into several.
_VISAYAS = (
    'Aklan', 'Antique', 'Capiz', 'Guimaras', 'Iloilo', 'Negros Occidental',
    'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor',
    'Biliran', 'Eastern Samar', 'Leyte', 'Northern Samar', 'Samar',
    'Southern Leyte',
)
_MINDANAO = (
    'Zamboanga del Norte', 'Zamboanga del Sur', 'Zamboanga Sibugay',
    'Bukidnon', 'Camiguin', 'Lanao del Norte', 'Misamis Occidental',
    'Misamis Oriental',
    'Davao de Oro', 'Davao del Norte', 'Davao del Sur', 'Davao Occidental',
    'Davao Oriental',
    'Cotabato', 'Sarangani', 'South Cotabato', 'Sultan Kudarat',
    'Agusan del Norte', 'Agusan del Sur', 'Dinagat Islands',
    'Surigao del Norte', 'Surigao del Sur',
    'Basilan', 'Lanao del Sur', 'Maguindanao', 'Sulu', 'Tawi-Tawi',
)
PROVINCES = _VISAYAS + _MINDANAO

#: Names DOE uses for a province listed above under a different one. Both are
#: the province's own names, not near-misses: `Compostela Valley` was renamed
#: Davao de Oro in 2019 and `North Cotabato` is the common form of Cotabato.
_ALIASES = {
    'compostela valley': 'Davao de Oro',
    'north cotabato': 'Cotabato',
    'maguindanao del norte': 'Maguindanao',
    'maguindanao del sur': 'Maguindanao',
    'western samar': 'Samar',
}

_CANONICAL = {p.lower(): p for p in PROVINCES}
_CANONICAL.update(_ALIASES)


def canonical_province(raw: Optional[str]) -> Optional[str]:
    """`raw` as a province DOE actually covers, or None.

    Case and spacing are normalised, because the same sheet writes `ILOILO` and
    `Iloilo`. Nothing else is: a name that is not on the list is REFUSED rather
    than matched to its nearest neighbour. `lloilo`, with a lowercase L for the
    I, is one edit from Iloilo and one edit from nothing in particular, and
    guessing which is the same move `DEC-045` refuses for dates.
    """
    if not raw:
        return None
    return _CANONICAL.get(' '.join(raw.split()).lower())


#: The swarm's four debated region groups, as province sets. Administrative fact,
#: not a modelling choice: which provinces make up Western Visayas is published,
#: whereas how to combine their prices into one regional number is the Phase 2
#: decision this layer refuses to make.
#:
#: Central Luzon is listed with an EMPTY set on purpose. Leaving it out entirely
#: would let a coverage report read as complete while a quarter of the swarm's
#: debating capacity has no source at all (`DEC-044`).
DEBATED_REGIONS = {
    'NCR': frozenset(),                       # the area IS the region; no provinces
    'Central Luzon': frozenset(),             # DOE publishes nothing north of NCR
    'Western Visayas': frozenset({
        'Aklan', 'Antique', 'Capiz', 'Guimaras', 'Iloilo', 'Negros Occidental'}),
    'Davao Region': frozenset({
        'Davao de Oro', 'Davao del Norte', 'Davao del Sur', 'Davao Occidental',
        'Davao Oriental'}),
}


#: Every PH region as its province set, for the regions DOE's panel can reach.
#: Administrative fact, published by PSA, not a modelling choice. `DEBATED_REGIONS`
#: is the swarm's four; this is the full table `swarm.ALL_REGIONS` assigns a
#: freight multiplier to, so each multiplier can be checked against prices.
#:
#: The four with an EMPTY set are empty because DOE publishes nothing north of
#: NCR. They are listed rather than omitted so a coverage table cannot read as
#: complete (`DEC-044`).
REGION_PROVINCES: dict[str, frozenset] = {
    'NCR': frozenset(),                        # the area IS the region
    'Ilocos Region': frozenset(),              # no DOE series
    'Cagayan Valley': frozenset(),             # no DOE series
    'Central Luzon': frozenset(),              # no DOE series
    'CAR': frozenset(),                        # no DOE series
    'CALABARZON': frozenset({
        'Cavite', 'Laguna', 'Batangas', 'Rizal', 'Quezon'}),
    'MIMAROPA': frozenset({
        'Occidental Mindoro', 'Oriental Mindoro', 'Marinduque', 'Romblon',
        'Palawan'}),
    'Bicol Region': frozenset({
        'Albay', 'Camarines Norte', 'Camarines Sur', 'Catanduanes', 'Masbate',
        'Sorsogon'}),
    'Western Visayas': frozenset({
        'Aklan', 'Antique', 'Capiz', 'Guimaras', 'Iloilo', 'Negros Occidental'}),
    'Central Visayas': frozenset({
        'Bohol', 'Cebu', 'Negros Oriental', 'Siquijor'}),
    'Eastern Visayas': frozenset({
        'Biliran', 'Eastern Samar', 'Leyte', 'Northern Samar', 'Samar',
        'Southern Leyte'}),
    'Zamboanga': frozenset({
        'Zamboanga del Norte', 'Zamboanga del Sur', 'Zamboanga Sibugay'}),
    'Northern Mindanao': frozenset({
        'Bukidnon', 'Camiguin', 'Lanao del Norte', 'Misamis Occidental',
        'Misamis Oriental'}),
    'Davao Region': frozenset({
        'Davao de Oro', 'Davao del Norte', 'Davao del Sur', 'Davao Occidental',
        'Davao Oriental'}),
    'SOCCSKSARGEN': frozenset({
        'Cotabato', 'Sarangani', 'South Cotabato', 'Sultan Kudarat'}),
    'Caraga': frozenset({
        'Agusan del Norte', 'Agusan del Sur', 'Dinagat Islands',
        'Surigao del Norte', 'Surigao del Sur'}),
    'BARMM': frozenset({
        'Basilan', 'Lanao del Sur', 'Maguindanao', 'Sulu', 'Tawi-Tawi'}),
}


def coverage(rows: list[dict]) -> dict:
    """Priced weeks per debated region per year.

    The number that decides Phase 2's window, and it cannot be read off the
    panel's overall date range: NCR runs 2019 to 2026 at roughly 50 weeks a year
    while Western Visayas has single-digit years before 2023, because those
    weeks were published as scans with no text layer. A regression that took the
    headline span for granted would silently fit three regions on three
    different periods.
    """
    out: dict[str, dict] = {}
    for region, provinces in DEBATED_REGIONS.items():
        if region == 'NCR':
            sub = [r for r in rows if r['area'] == 'NCR']
        else:
            sub = [r for r in rows if r['province'] in provinces] if provinces else []
        priced = [r for r in sub if r['common'] or r['high']]
        weeks = sorted({r['cycle'] for r in priced})
        out[region] = {
            'priced_rows': len(priced),
            'weeks': len(weeks),
            'first': weeks[0] if weeks else None,
            'last': weeks[-1] if weeks else None,
            'weeks_per_year': dict(sorted(collections.Counter(
                w[:4] for w in weeks).items())),
        }
    return out


#: The areas Phase 2 can actually use. South Luzon is 1249 files covering regions
#: the swarm derives rather than debates, so it is fetched only on request.
DEFAULT_AREAS = ('NCR', 'Visayas', 'Mindanao')

#: Politeness. 2713 documents on a government CMS is not a burst.
_DELAY = 0.7
_TIMEOUT = 90


# ── Placing a document in a pricing week ─────────────────────────────────────

def cycle_of(day: dt.date) -> dt.date:
    """The Tuesday 06:00 PHT boundary that opened the cycle containing `day`.

    Snaps BACKWARD. A Thursday file describes prices set on Tuesday; a Monday file
    describes prices set on the Tuesday six days before, not the one tomorrow.

    Delegates to `vintage.fuel_cycle_start` rather than recomputing the weekday
    arithmetic, so a regional actual and a run's own vintage can never disagree
    about which week they are in.
    """
    noon = dt.datetime(day.year, day.month, day.day, 12, 0, tzinfo=price_calendar.PH_TZ)
    return vintage.fuel_cycle_start(noon).date()


# ── Fetching ─────────────────────────────────────────────────────────────────

def cache_path(name: str) -> Path:
    return CACHE_DIR / name


def load_index() -> list[dict]:
    """The indexed rows from `doe_price_archive --index`."""
    if not INDEX_ARTIFACT.exists():
        raise SystemExit(
            f'No index at {INDEX_ARTIFACT}.\n'
            'Run: python -m ph_economic_ai.tools.doe_price_archive --index')
    return json.loads(INDEX_ARTIFACT.read_text(encoding='utf-8'))['rows']


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding='utf-8'))
    return {'schema_version': 1, 'records': {}}


def fetch_all(rows: Iterable[dict], manifest: dict, delay: float = _DELAY,
              limit: Optional[int] = None, session=None) -> dict:
    """Download anything not already cached. Returns counts.

    A file already in the cache costs NO request: that is what makes a run of this
    size resumable after an interruption, and it is why the default never
    re-fetches. Use `--verify` to check the cache against its hashes instead.
    """
    session = session or requests.Session()
    records = manifest['records']
    counts = collections.Counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows if not cache_path(r['name']).exists()]
    if limit is not None:
        todo = todo[:limit]
    counts['already_cached'] = sum(1 for r in rows if cache_path(r['name']).exists())

    for i, row in enumerate(todo, 1):
        name = row['name']
        try:
            resp = session.get(row['url'], headers=HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            body = resp.content
        except Exception as exc:
            records[name] = {'url': row['url'], 'error': f'{type(exc).__name__}: {exc}'}
            counts['failed'] += 1
            continue

        # A CMS that answers a missing document with an HTML error page returns
        # 200. Cached as a PDF it would parse to zero rows and read as a document
        # with no prices, which is the quiet failure this whole file guards.
        if not body.startswith(b'%PDF'):
            records[name] = {'url': row['url'], 'error': 'not a PDF',
                             'bytes': len(body)}
            counts['not_pdf'] += 1
            continue

        path = cache_path(name)
        path.write_bytes(body)
        # Only what the FETCH established. Area, date and cycle are derived from
        # the filename at build time instead of frozen here, because they are
        # interpretations and interpretations get corrected: `lfro` was read as
        # Luzon and is Mindanao. A record that froze the wrong area would have
        # needed 2713 re-downloads to fix a one-line mapping.
        records[name] = {
            'url': row['url'],
            'bytes': len(body),
            'sha256': sha256_file(path),
            'fetched_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
        }
        counts['fetched'] += 1
        if i % 25 == 0:
            print(f'  {i}/{len(todo)} fetched')
        time.sleep(delay)

    return counts


def verify_cache(manifest: dict) -> dict:
    """Re-hash every cached file against the manifest.

    `changed` is the one with teeth, for the same reason it is in
    `benchmark.provenance`: a file that no longer hashes to what the record claims
    means the record describes a different document, and a stale record reads as
    authoritative.
    """
    ok, changed, missing = [], [], []
    for name, rec in manifest['records'].items():
        if 'sha256' not in rec:
            continue
        path = cache_path(name)
        if not path.exists():
            missing.append(name)
        elif sha256_file(path) != rec['sha256']:
            changed.append(name)
        else:
            ok.append(name)
    return {'ok': ok, 'changed': changed, 'missing': missing}


# ── Building the panel ───────────────────────────────────────────────────────

def panel_rows(manifest: dict) -> tuple[list[dict], dict]:
    """RON 95 rows from every cached document, plus what went wrong.

    A document that parses to zero rows is COUNTED, not skipped. Silent zeroes are
    how a corpus looks fully covered while the panel behind it is empty.
    """
    out: list[dict] = []
    problems = {'unparsed': [], 'no_ron95': [], 'unplaceable': [],
                'no_province': 0, 'no_city': 0,
                'unrecognised_provinces': collections.Counter()}

    for name, rec in sorted(manifest['records'].items()):
        if 'sha256' not in rec:
            continue
        path = cache_path(name)
        if not path.exists():
            continue
        area, day = area_of(name), date_of(name)
        if area is None or day is None:
            problems['unplaceable'].append(name)
            continue
        try:
            parsed = parse_price_pdf(path.read_bytes())
        except Exception as exc:
            problems['unparsed'].append(f'{name}: {type(exc).__name__}: {exc}')
            continue
        if not parsed:
            problems['unparsed'].append(f'{name}: no rows')
            continue

        grade = [r for r in parsed if r['product'] == PRODUCT]
        if not grade:
            problems['no_ron95'].append(name)
            continue

        for r in grade:
            province = canonical_province(r['province'])
            if province is None:
                problems['no_province'] += 1
                if r['province']:
                    problems['unrecognised_provinces'][r['province']] += 1
            if r['city'] is None:
                problems['no_city'] += 1
            out.append({
                'cycle': cycle_of(day).isoformat(),
                'file_date': day.isoformat(),
                'area': area,
                'province': province or '',
                # The cell as printed, kept whatever happens to it above, so a
                # refused name can be inspected rather than only counted.
                'province_raw': r['province'] or '',
                'city': r['city'] or '',
                'low': r['low'], 'high': r['high'], 'common': r['common'],
                'source_file': name,
            })
    return out, problems


_FIELDS = ('cycle', 'file_date', 'area', 'province', 'province_raw', 'city',
           'low', 'high', 'common', 'source_file')


def write_panel(rows: list[dict], path: Path = PANEL) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['cycle'], r['area'],
                                                r['province'], r['city'])))
    return path


def collisions(rows: list[dict]) -> dict[tuple, int]:
    """(area, cycle, city) seen in more than one document.

    DOE republishes and corrects. Two documents covering the same city-week are two
    observations of one week, and averaging them silently would hide a correction.
    Reported so Phase 2 declares a rule rather than inheriting one.
    """
    seen = collections.Counter(
        (r['area'], r['cycle'], r['city']) for r in rows)
    files = collections.defaultdict(set)
    for r in rows:
        files[(r['area'], r['cycle'], r['city'])].add(r['source_file'])
    return {k: len(files[k]) for k, v in seen.items() if len(files[k]) > 1}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _fetch(args) -> int:
    rows = load_index()
    areas = tuple(args.areas) if args.areas else DEFAULT_AREAS
    rows = [r for r in rows if r['area'] in areas]
    print(f'{len(rows)} indexed documents in {", ".join(areas)}')

    manifest = load_manifest()
    counts = fetch_all(rows, manifest, delay=args.delay, limit=args.limit)
    manifest['updated_at'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n',
                        encoding='utf-8')

    print(f'  cached already {counts["already_cached"]}')
    print(f'  fetched        {counts["fetched"]}')
    if counts['failed']:
        print(f'  FAILED         {counts["failed"]}')
    if counts['not_pdf']:
        print(f'  not a PDF      {counts["not_pdf"]}  (server answered, wrong body)')
    print(f'\nmanifest {MANIFEST}')
    return 0


def _build(args) -> int:
    manifest = load_manifest()
    rows, problems = panel_rows(manifest)
    if not rows:
        print('No RON 95 rows. Fetch first.')
        return 1

    write_panel(rows)
    by_area = collections.Counter(r['area'] for r in rows)
    cycles = sorted({r['cycle'] for r in rows})

    print(f'{len(rows)} RON 95 rows, {len(cycles)} pricing weeks, '
          f'{cycles[0]} to {cycles[-1]}')
    print(f'{"area":<22}{"rows":>7}{"weeks":>8}  provinces')
    for area in sorted(by_area, key=lambda a: -by_area[a]):
        sub = [r for r in rows if r['area'] == area]
        provs = sorted({r['province'] for r in sub if r['province']})
        print(f'{area:<22}{len(sub):>7}{len(set(r["cycle"] for r in sub)):>8}  '
              f'{len(provs)}')

    print(f'\n{"debated region":<18}{"weeks":>7}  span                    per year')
    for region, cov in coverage(rows).items():
        if not cov['weeks']:
            print(f'{region:<18}{0:>7}  NO SOURCE AT ALL        '
                  f'(DEC-044: nothing published)')
            continue
        years = ' '.join(f'{y}:{n}' for y, n in cov['weeks_per_year'].items())
        print(f'{region:<18}{cov["weeks"]:>7}  {cov["first"]} to {cov["last"]}  '
              f'{years}')

    dupes = collisions(rows)
    if dupes:
        print(f'\n{len(dupes)} city-weeks appear in more than one document. '
              f'Phase 2 declares a rule; this layer does not pick one.')
    for label, items in (('could not parse', problems['unparsed']),
                         ('no RON 95 row', problems['no_ron95']),
                         ('cached but unplaceable by name', problems['unplaceable'])):
        if items:
            print(f'\n{len(items)} documents {label}:')
            for n in items[:8]:
                print(f'   {n[:100]}')
    if problems['no_province']:
        share = 100 * problems['no_province'] / len(rows)
        print(f'\n{problems["no_province"]} rows ({share:.0f}%) carry no usable '
              f'province. NCR has none by construction: the city IS the region.')
    bad = problems['unrecognised_provinces']
    if bad:
        print(f'\n{sum(bad.values())} rows name a province that is not one '
              f'({len(bad)} distinct). Kept in `province_raw`, refused in '
              f'`province`, never guessed at:')
        for name, count in bad.most_common(6):
            print(f'   {count:4}  {name[:60]!r}')

    write_record(
        PANEL,
        source='DOE Oil Industry Management Bureau, Prevailing Retail Pump Prices '
               '(weekly PDFs, category=Price+Monitoring across the lfo/vfo/mfo/oimb '
               'field-office listings)',
        params={'products': [PRODUCT],
                'areas': sorted({r['area'] for r in rows}),
                'documents_cached': len(manifest['records']),
                'documents_contributing': len({r['source_file'] for r in rows}),
                'index_artifact': INDEX_ARTIFACT.name,
                'coverage_by_debated_region': coverage(rows)},
        transformations=[
            'enumerate listings, index by filename (doe_price_archive)',
            'fetch each PDF to a local cache, SHA-256 per file',
            'parse the positional table; province cells from the PDF drawing layer',
            f'keep {PRODUCT} rows only, selected by name (DEC-043)',
            'snap each publication date back to the Tuesday 06:00 PHT fuel cycle '
            'it falls in (vintage.fuel_cycle_start)',
            'accept a province only if it is one DOE covers; the cell as printed '
            'is kept in province_raw and is never corrected by similarity',
        ],
        units='PHP per litre',
        notes='Per document, province, city and pricing week. Aggregation to a '
              'regional level is deliberately NOT applied: choosing between a '
              'median city, a mean and a population weight changes the dependent '
              'variable, and DEC-010 forbids making that choice after seeing '
              'results. DOE publishes nothing for Region III, I, II or CAR, so '
              'Central Luzon has no row here at any date. '
              f'{len(problems["unparsed"])} cached documents yielded no rows and '
              'are NOT in this file: most carry a scanned page with no text '
              'layer, the rest use pre-2020 layouts the parser does not read. '
              'Coverage is therefore uneven across years and has to be checked '
              'per region before any regression, not assumed from the date range.',
    )
    print(f'\npanel      {PANEL}')
    print(f'provenance {PANEL.name}.provenance.json')
    return 0


def _verify(args) -> int:
    result = verify_cache(load_manifest())
    print(f'ok      {len(result["ok"])}')
    for n in result['changed']:
        print(f'  CHANGED {n}  (cached bytes no longer match the manifest)')
    for n in result['missing']:
        print(f'  missing {n}')
    return 1 if result['changed'] else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fetch', action='store_true', help='download uncached documents')
    ap.add_argument('--build', action='store_true', help='build the panel from cache')
    ap.add_argument('--verify', action='store_true', help='re-hash the cache')
    ap.add_argument('--areas', nargs='*', help=f'default: {" ".join(DEFAULT_AREAS)}')
    ap.add_argument('--limit', type=int, help='stop after N new documents')
    ap.add_argument('--delay', type=float, default=_DELAY)
    args = ap.parse_args()

    if args.fetch:
        return _fetch(args)
    if args.build:
        return _build(args)
    if args.verify:
        return _verify(args)
    ap.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
