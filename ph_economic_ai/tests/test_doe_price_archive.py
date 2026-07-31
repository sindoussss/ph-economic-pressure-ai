"""Indexing DOE's regional price archive, and reading its PDFs.

The app forecasts a per-region price CHANGE and has never been graded on one:
`ground_truth` scores the national figure only. This is the dependent variable
Phase 2 needs, and both halves of getting it are error-prone in ways that fail
QUIETLY, so they are pinned here.

Filenames drifted over seven years across ten date conventions. A file whose date
cannot be read is useless for a backtest; a file whose date is read WRONG is
worse, because it silently lands in another week.

The PDFs are a positional table with no ruling lines in the text layer, so every
column boundary is inferred from x coordinates. Three separate defects came out
of that and each produced plausible-looking output.

Network-free. The listing enumeration and the PDF fetch are exercised by running
the tool; these cover the parsing that decides whether the data is right.
"""
import datetime as dt

import pytest

from ph_economic_ai.tools.doe_price_archive import (
    PRODUCTS, _num, _row_prices, _rows_from_words, area_of, date_of,
)


# ── Dating a filename ────────────────────────────────────────────────────────

@pytest.mark.parametrize('name,expected', [
    ('petro_ncr_2020_february_18', dt.date(2020, 2, 18)),      # full month
    ('petro_ncr_2023-jun-01', dt.date(2023, 6, 1)),            # short month
    ('petro_mm_2017_mar31', dt.date(2017, 3, 31)),             # run together
    ('petro_mm_2017_june6', dt.date(2017, 6, 6)),              # run together, full
    ('petro_min_2020_sept_01-pdf', dt.date(2020, 9, 1)),       # DOE's fourth spelling
    ('petro_ncr_2022-06-23', dt.date(2022, 6, 23)),            # all numeric
    ('ncr-price-monitoring-07-01-2025-pdf', dt.date(2025, 7, 1)),   # MM-DD-YYYY
    ('ncr-price-monitoring-05192026-pdf', dt.date(2026, 5, 19)),    # MMDDYYYY
    ('vfo-lf-price-monitoring-102825', dt.date(2025, 10, 28)),      # MMDDYY
    ('47-lfro-price-monitoring-november-25-2025-pdf', dt.date(2025, 11, 25)),
])
def test_every_naming_convention_dates(name, expected):
    assert date_of(name) == expected


def test_a_date_range_yields_its_start():
    """`petro_ncr_2024-apr-16-22` covers a week. The start is when the prices took
    effect and is what `vintage.fuel_cycle_start` aligns to."""
    assert date_of('petro_ncr_2024-apr-16-22') == dt.date(2024, 4, 16)
    assert date_of('region-iv-a-batangas-rizal-quezon-as-of-april-15-21-2025-pdf') \
        == dt.date(2025, 4, 15)


def test_an_unambiguous_date_is_never_pre_empted_by_the_range_pattern():
    """The regression this ordering exists for. Placed before the simple shape,
    the range pattern read `november-25-2025` as November 2, splitting the 25
    across the day and the year. It looked like a working parser."""
    assert date_of('47-lfro-price-monitoring-november-25-2025-pdf').day == 25


def test_a_file_with_no_year_is_refused_rather_than_guessed():
    """166 `region-*` files carry no year. A backtest that guesses one puts prices
    in the wrong week, which is worse than dropping the file."""
    assert date_of('region-iv-b-mimaropa-as-of-august-9-to-15-pdf') is None
    assert date_of('region-iv-a-calabarzon') is None
    assert date_of('region-v-bicol-23-pdf') is None


def test_an_impossible_date_is_refused():
    assert date_of('petro_ncr_2020_february_31') is None


# ── Placing a filename ───────────────────────────────────────────────────────

@pytest.mark.parametrize('name,area', [
    ('petro_ncr_2020_february_18', 'NCR'),
    ('petro_mm_2017_january_17', 'NCR'),
    ('ncr-price-monitoring-05192026-pdf', 'NCR'),
    ('petro_sluz_2021-apr-08_laguna', 'South Luzon'),
    ('petro_vis_2019_august_20-pdf', 'Visayas'),
    ('petro-vis_2022-may-10', 'Visayas'),
    ('vfo-lf-price-monitoring-102825', 'Visayas'),
    ('petro_min_2018_november_13-pdf', 'Mindanao'),
    ('mfo-price-monitoring-april-15-2025-pdf', 'Mindanao'),
])
def test_a_filename_places_itself(name, area):
    assert area_of(name) == area


def test_an_unplaceable_name_is_refused():
    """`petro_2019_february_26` names no region. Assigning it to one would put
    another area's prices into a region's series."""
    assert area_of('petro_2019_february_26') is None
    assert area_of('oil-monitor-as-of-28-july-2026-pdf') is None


def test_no_series_maps_to_central_luzon():
    """DOE publishes nothing north of NCR: zero files for Region III, I, II or CAR
    against controls of 1243, 470, 348 and 333. A mapping that invented Central
    Luzon coverage would make an unfalsifiable region look validated."""
    from ph_economic_ai.tools.doe_price_archive import SERIES
    areas = {a for _, a in SERIES}
    assert 'Central Luzon' not in areas
    assert 'North Luzon' not in areas


# ── Reading one product row ──────────────────────────────────────────────────

def _w(x, y, text):
    return (x, y, x + 20, y + 8, text, 0, 0, 0)


def test_a_product_number_is_never_read_as_a_price():
    """`RON 95` has a numeric label. The first version took the last few numbers
    on the row and reported a common price of exactly 95.00."""
    row = [_w(144, 188, 'RON'), _w(157, 188, '95'),
           _w(613, 188, '58.10'), _w(785, 188, '58.10'),
           _w(810, 188, '-'), _w(822, 188, '58.10'), _w(873, 188, 'None')]
    low, high, common = _row_prices(row)
    assert (low, high) == (58.10, 58.10)
    assert common is None
    assert 95.0 not in (low, high, common)


def test_the_common_column_is_read_when_present():
    row = [_w(144, 200, 'DIESEL'), _w(785, 200, '53.30'), _w(810, 200, '-'),
           _w(822, 200, '63.15'), _w(873, 200, '58.00')]
    assert _row_prices(row) == (53.30, 63.15, 58.00)


def test_zero_is_not_sold_here_rather_than_free():
    """DOE writes 0.00 for a product a city does not sell."""
    assert _num('0.00') is None
    assert _num('58.10') == 58.10


def test_a_row_with_no_range_yields_nothing_rather_than_a_guess():
    assert _row_prices([_w(144, 168, 'KEROSENE')]) == (None, None, None)


# ── Grouping words into rows ─────────────────────────────────────────────────

def test_a_product_and_its_prices_are_one_row_despite_the_y_offset():
    """`RON 97` sits at y=176 and its prices at y=180. A strict y match splits the
    row and drops every price on it."""
    words = [_w(144, 176.0, 'RON'), _w(157, 176.0, '97'),
             _w(785, 180.0, '0.00'), _w(810, 180.0, '-')]
    rows = _rows_from_words(words)
    assert len(rows) == 1


def test_genuinely_separate_rows_stay_separate():
    words = [_w(144, 176.0, 'RON'), _w(144, 200.0, 'DIESEL')]
    assert len(_rows_from_words(words)) == 2


def test_products_are_ordered_so_diesel_plus_wins_over_diesel():
    """`DIESEL PLUS` contains `DIESEL`, so a substring test in the wrong order
    labels every DIESEL PLUS row as DIESEL and overwrites a real product."""
    assert PRODUCTS.index('DIESEL PLUS') < PRODUCTS.index('DIESEL')
