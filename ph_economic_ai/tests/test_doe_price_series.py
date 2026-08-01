"""Caching the DOE archive, and turning it into a panel.

`test_doe_price_archive` covers reading one document. This covers the layer that
decides WHICH week a document describes, what the cache is allowed to claim, and
what happens to a document that cannot be placed.

The failure mode is the same one as the parser's and it is worth stating again:
none of these produce an error. A file in the wrong week, a cache that says it
holds something it does not, an HTML error page stored as a PDF -- each yields a
panel that looks complete.

Network-free. The fetch itself is exercised by running the tool.
"""
import datetime as dt
import json

import pytest

from ph_economic_ai.tools import doe_price_series as series


# ── Which pricing week a document describes ──────────────────────────────────

@pytest.mark.parametrize('day,expected,label', [
    ('2025-11-25', '2025-11-25', 'a Tuesday opens its own cycle'),
    ('2025-11-26', '2025-11-25', 'Wednesday, one day in'),
    ('2025-11-27', '2025-11-25', 'Thursday, when 538 files are dated'),
    ('2025-11-30', '2025-11-25', 'Sunday, five days in'),
    ('2025-11-24', '2025-11-18', 'Monday belongs to the cycle already running'),
])
def test_a_publication_date_snaps_back_to_its_pricing_week(day, expected, label):
    """Only 1767 of 2520 dated files fall on a Tuesday. A monitoring date is when
    DOE walked the stations, not when the price took effect."""
    assert series.cycle_of(dt.date.fromisoformat(day)).isoformat() == expected, label


def test_a_monday_is_never_snapped_forward():
    """The regression this exists for. Snapping to the NEAREST Tuesday puts a
    Monday file in a cycle that had not started when the prices were observed,
    which would shift a fifth of the corpus one week into the future and make the
    backtest look prescient."""
    monday = dt.date(2025, 11, 24)
    assert series.cycle_of(monday) < monday


def test_every_cycle_start_is_a_tuesday():
    """Property, not a list of dates: whatever the input weekday, the answer is a
    Tuesday at or before it."""
    day = dt.date(2024, 1, 1)
    for _ in range(400):
        start = series.cycle_of(day)
        assert start.weekday() == 1, f'{day} landed on {start:%A}'
        assert 0 <= (day - start).days <= 6
        day += dt.timedelta(days=1)


# ── What the cache is allowed to claim ───────────────────────────────────────

class _Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, body):
        self._body = body
        self.calls = 0

    def get(self, url, **kw):
        self.calls += 1
        return _Response(self._body)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(series, 'CACHE_DIR', tmp_path / 'doe_cache')
    return tmp_path


_ROW = {'name': 'petro_ncr_2025_november_25', 'url': 'https://x/y',
        'area': 'NCR', 'date': '2025-11-25'}


def test_a_cached_record_holds_only_what_the_fetch_established(cache):
    """Area, date and cycle are INTERPRETATIONS and interpretations get corrected:
    `lfro` was read as Luzon and is Mindanao. Freezing them into the record would
    have made a one-line mapping fix cost 2713 re-downloads, so the record carries
    the bytes, the hash, the URL and the time, and nothing derived."""
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    rec = manifest['records']['petro_ncr_2025_november_25']
    assert set(rec) == {'url', 'bytes', 'sha256', 'fetched_at'}


def test_a_body_that_is_not_a_pdf_is_refused_rather_than_cached(cache):
    """A CMS answers a missing document with an HTML error page and status 200.
    Stored as a PDF it parses to zero rows and reads as a week with no prices."""
    manifest = {'records': {}}
    counts = series.fetch_all([_ROW], manifest, delay=0,
                              session=_Session(b'<!DOCTYPE html><h1>Not found'))
    assert counts['not_pdf'] == 1
    assert not series.cache_path(_ROW['name']).exists()
    assert 'error' in manifest['records'][_ROW['name']]


def test_an_already_cached_document_costs_no_request(cache):
    """What makes 2713 documents resumable after an interruption."""
    manifest = {'records': {}}
    first = _Session(b'%PDF-1.4 x')
    series.fetch_all([_ROW], manifest, delay=0, session=first)
    second = _Session(b'%PDF-1.4 x')
    series.fetch_all([_ROW], manifest, delay=0, session=second)
    assert first.calls == 1 and second.calls == 0


def test_a_cached_file_that_changed_underneath_is_reported(cache):
    """A stale record reads as authoritative, which is worse than no record --
    the same reason `benchmark.provenance.verify` exists."""
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    assert series.verify_cache(manifest)['changed'] == []

    series.cache_path(_ROW['name']).write_bytes(b'%PDF-1.4 DIFFERENT')
    assert series.verify_cache(manifest)['changed'] == [_ROW['name']]


def test_a_deleted_cache_file_is_reported_separately_from_a_changed_one(cache):
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    series.cache_path(_ROW['name']).unlink()
    result = series.verify_cache(manifest)
    assert result['missing'] == [_ROW['name']] and result['changed'] == []


# ── Building the panel ───────────────────────────────────────────────────────

def test_a_document_that_cannot_be_placed_never_reaches_the_panel(cache, monkeypatch):
    """`DEC-045` at the panel layer. A cached file whose name yields no date or no
    area is counted as unplaceable, not given a neighbour's week."""
    manifest = {'records': {}}
    series.fetch_all([{'name': 'region-iv-a-calabarzon', 'url': 'https://x/y',
                       'area': 'South Luzon', 'date': '2025-11-25'}],
                     manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    monkeypatch.setattr(series, 'parse_price_pdf', lambda b: [
        {'province': 'Cavite', 'city': 'Bacoor', 'product': 'RON 95',
         'low': 55.0, 'high': 60.0, 'common': 57.0}])
    rows, problems = series.panel_rows(manifest)
    assert rows == []
    assert problems['unplaceable'] == ['region-iv-a-calabarzon']


def test_a_document_that_parses_to_nothing_is_counted_not_skipped(cache, monkeypatch):
    """Silent zeroes are how a corpus looks fully covered while the panel behind
    it is empty."""
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    monkeypatch.setattr(series, 'parse_price_pdf', lambda b: [])
    rows, problems = series.panel_rows(manifest)
    assert rows == []
    assert len(problems['unparsed']) == 1


def test_a_document_with_no_ron95_row_is_named(cache, monkeypatch):
    """The app forecasts RON 95 and grades against it (`DEC-043`). A document
    carrying only diesel contributes nothing and has to say so."""
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    monkeypatch.setattr(series, 'parse_price_pdf', lambda b: [
        {'province': 'x', 'city': 'y', 'product': 'DIESEL',
         'low': 50.0, 'high': 55.0, 'common': 52.0}])
    rows, problems = series.panel_rows(manifest)
    assert rows == [] and problems['no_ron95'] == [_ROW['name']]


def test_a_panel_row_carries_the_document_it_came_from(cache, monkeypatch):
    """`ADR-008` fixed a national series whose inputs could not be rebuilt from
    what was committed. A regional row has to trace back to a page."""
    manifest = {'records': {}}
    series.fetch_all([_ROW], manifest, delay=0, session=_Session(b'%PDF-1.4 x'))
    monkeypatch.setattr(series, 'parse_price_pdf', lambda b: [
        {'province': 'Cavite', 'city': 'Bacoor', 'product': 'RON 95',
         'low': 55.0, 'high': 60.0, 'common': 57.0}])
    rows, _ = series.panel_rows(manifest)
    assert rows[0]['source_file'] == _ROW['name']
    assert rows[0]['cycle'] == '2025-11-25'
    assert rows[0]['area'] == 'NCR'


def test_two_documents_covering_one_city_week_are_reported_not_merged():
    """DOE republishes and corrects. Averaging two observations of one week would
    hide the correction, and picking one is a Phase 2 rule to be declared rather
    than inherited."""
    rows = [
        {'area': 'NCR', 'cycle': '2025-11-25', 'city': 'Manila',
         'source_file': 'a', 'common': 57.0},
        {'area': 'NCR', 'cycle': '2025-11-25', 'city': 'Manila',
         'source_file': 'b', 'common': 57.5},
        {'area': 'NCR', 'cycle': '2025-11-25', 'city': 'Pasig',
         'source_file': 'a', 'common': 56.0},
    ]
    dupes = series.collisions(rows)
    assert list(dupes) == [('NCR', '2025-11-25', 'Manila')]
    assert dupes[('NCR', '2025-11-25', 'Manila')] == 2


# ── A province is recognised or refused, never guessed ───────────────────────

def test_a_province_name_is_normalised_only_for_case_and_spacing():
    """The same sheet writes `ILOILO` and `Iloilo`."""
    assert series.canonical_province('ILOILO') == 'Iloilo'
    assert series.canonical_province('  negros   occidental ') == 'Negros Occidental'


def test_a_garbled_province_is_refused_rather_than_matched_to_its_nearest():
    """Some documents carry an OCR'd text layer, which yields `Llnao dal Norto`
    and `Zamboanga dsl Nodo`. Each appears once or twice against thousands of
    clean rows, so it never moves an aggregate enough to be noticed while
    silently splitting one province's series into several. `lloilo`, with a
    lowercase L for the I, is one edit from Iloilo and one edit from nothing in
    particular; guessing which is the move `DEC-045` refuses for dates."""
    for garbled in ('Llnao dal Norto', 'Zamboanga dsl Nodo', 'lloilo',
                    'ilitamlt Occldontrl', 'Misamis Occident al'):
        assert series.canonical_province(garbled) is None


def test_the_canonical_province_list_cannot_diverge_from_the_region_map():
    """It was written out a second time, covering Visayas and Mindanao only
    because those were the areas fetched at the time. When South Luzon arrived,
    its sixteen provinces were silently refused and 2847 rows naming Cavite,
    Palawan, Batangas and Albay were treated as unreadable cells.

    Two lists that must agree do not stay in agreement, so there is now one and
    this pins it."""
    derived = set().union(*series.REGION_PROVINCES.values())
    assert set(series.PROVINCES) == derived
    for province in ('Cavite', 'Palawan', 'Albay', 'Batangas', 'Camarines Sur'):
        assert series.canonical_province(province) == province


def test_a_wrap_fragment_is_not_a_province():
    """`Sur` and `Norte` alone are half a name, not a place."""
    for fragment in ('Sur', 'Norte', 'Misamis', 'Province', ''):
        assert series.canonical_province(fragment) is None


def test_a_renamed_province_maps_to_the_name_the_panel_uses():
    """Both are the province's OWN names across time, not near-misses:
    Compostela Valley was renamed Davao de Oro in 2019, and a series that split
    at the rename would show a seven-year province as two four-year ones."""
    assert series.canonical_province('Compostela Valley') == 'Davao de Oro'
    assert series.canonical_province('North Cotabato') == 'Cotabato'


def test_the_panel_keeps_the_cell_as_printed_even_when_it_refuses_it():
    """So a refused name can be inspected, not only counted."""
    assert 'province_raw' in series._FIELDS
    assert series._FIELDS.index('province') < series._FIELDS.index('province_raw')


# ── Coverage is per region and per year, never inferred from the span ────────

def test_coverage_is_reported_per_year_not_as_one_span():
    """NCR runs 2019 to 2026 at about 50 weeks a year while Western Visayas has
    single-digit years before 2023, because those weeks were published as scans
    with no text layer. One headline date range covers both and would let three
    regions be fitted on three different periods without anyone seeing it."""
    rows = [{'area': 'NCR', 'province': '', 'cycle': '2019-06-11',
             'common': 50.0, 'high': 55.0},
            {'area': 'NCR', 'province': '', 'cycle': '2026-07-21',
             'common': 60.0, 'high': 65.0},
            {'area': 'Visayas', 'province': 'Iloilo', 'cycle': '2026-07-21',
             'common': 61.0, 'high': 66.0}]
    cov = series.coverage(rows)
    assert cov['NCR']['weeks_per_year'] == {'2019': 1, '2026': 1}
    assert cov['Western Visayas']['weeks_per_year'] == {'2026': 1}


def test_a_region_with_no_source_is_reported_as_zero_not_omitted():
    """Omitting Central Luzon would let a coverage report read as complete while
    a quarter of the swarm's debating capacity has no source at all."""
    cov = series.coverage([{'area': 'NCR', 'province': '', 'cycle': '2026-07-21',
                            'common': 60.0, 'high': 65.0}])
    assert 'Central Luzon' in cov
    assert cov['Central Luzon']['weeks'] == 0
    assert cov['Central Luzon']['first'] is None


def test_an_unpriced_row_is_not_counted_as_coverage():
    """DOE writes a row for a city that reported nothing. Counting it makes a
    week look observed when no price was published."""
    cov = series.coverage([{'area': 'NCR', 'province': '', 'cycle': '2026-07-21',
                            'common': None, 'high': None}])
    assert cov['NCR']['weeks'] == 0


def test_the_panel_never_carries_an_aggregate_of_its_own(tmp_path):
    """Aggregating cities to a regional level changes the dependent variable, and
    `DEC-010` forbids choosing that after seeing results. The panel is published
    at the grain DOE publishes: one row per document, province, city and week."""
    path = series.write_panel([
        {'cycle': '2025-11-25', 'file_date': '2025-11-27', 'area': 'NCR',
         'province': '', 'city': 'Manila', 'low': 55.0, 'high': 60.0,
         'common': 57.0, 'source_file': 'a'}], tmp_path / 'panel.csv')
    header = path.read_text(encoding='utf-8').splitlines()[0]
    assert header.split(',') == list(series._FIELDS)
    for banned in ('mean', 'median', 'regional_average', 'aggregate'):
        assert banned not in header
