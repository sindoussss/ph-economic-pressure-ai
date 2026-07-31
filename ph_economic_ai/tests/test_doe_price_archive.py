"""Indexing DOE's regional price archive, and reading its PDFs.

The app forecasts a per-region price CHANGE and has never been graded on one:
`ground_truth` scores the national figure only. This is the dependent variable
Phase 2 needs, and both halves of getting it are error-prone in ways that fail
QUIETLY, so they are pinned here.

Filenames drifted over seven years across ten date conventions. A file whose date
cannot be read is useless for a backtest; a file whose date is read WRONG is
worse, because it silently lands in another week.

The PDFs are a positional table. The TEXT layer has no ruling lines, so row and
column boundaries have to be inferred, and NINE separate defects came out of that,
every one producing plausible-looking output rather than an error. The DRAWING
layer does have rules, which is what settled both axes after proximity and
measured constants could not.

The recurring shape is worth naming, because it is why these tests are written
against the exact document that produced each defect: a positional parser does not
fail loudly. It returns a real place name in the wrong column, a real price under
the wrong city, or nothing at all from a document whose layout shifted.

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


def test_a_month_name_beats_a_numeric_shape():
    """`june` is a month and can be nothing else; `2-8-2026` is a month and a day
    only by convention. Tried the other way round,
    `ncr-price-monitoring-for-june-2-8-2026` read as 8 February 2026 -- a June
    sheet in a February pricing week, carrying June prices. It then won the
    panel's later-filename tie-break, overwrote the correct February data, and
    moved that week's regional level by 18 PHP/L on its own.

    Second instance of one rule, after the range pattern read `november-25-2025`
    as November 2: an ambiguous pattern must never pre-empt an unambiguous one.
    Both produced a real date, in the right year, for the wrong week."""
    assert date_of('ncr-price-monitoring-for-june-2-8-2026-pdf-1') \
        == dt.date(2026, 6, 2)
    assert date_of('petro_min_2019_april_02_04102019') == dt.date(2019, 4, 2)


def test_a_numeric_date_still_parses_when_no_month_is_named():
    """The reordering must not cost the numeric conventions, which are the only
    thing most of the corpus has."""
    assert date_of('ncr-price-monitoring-07-01-2025-pdf') == dt.date(2025, 7, 1)
    assert date_of('petro_ncr_2022-06-23') == dt.date(2022, 6, 23)
    assert date_of('ncr-price-monitoring-05192026-pdf') == dt.date(2026, 5, 19)
    assert date_of('vfo-lf-price-monitoring-102825') == dt.date(2025, 10, 28)


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


def test_a_city_block_does_not_depend_on_the_first_product_existing():
    """Blocks were opened on `PRODUCTS[0]`, RON 100. A sheet that lists no RON
    100 opened no block and returned NOTHING:
    `petro_min_2020_february_11` has zero RON 100 rows and 18 RON 95 ones, and
    lost every one of them without raising. A block starts where the product
    ORDER restarts, which does not privilege any single product."""
    from ph_economic_ai.tools.doe_price_archive import PRODUCTS as P
    starts = []
    last = len(P)
    for product in ('RON 97', 'RON 95', 'DIESEL', 'RON 97', 'RON 95', 'DIESEL'):
        rank = P.index(product)
        if rank <= last:
            starts.append(product)
        last = rank
    assert starts == ['RON 97', 'RON 97'], 'two cities, neither listing RON 100'


# ── Province cells come from the ruling lines ────────────────────────────────

def test_a_block_belongs_to_the_province_cell_it_sits_in():
    """The table HAS ruling lines; only the text layer lacks them. Nearest-centre
    assignment could only approximate, because a province label is centred against
    every city block it contains, so at a boundary the nearest label belongs to
    the NEIGHBOURING province. That put Ligao, which is in Albay, under
    Catanduanes."""
    from ph_economic_ai.tools.doe_price_archive import _band_of

    bands = [168.0, 295.0, 358.0, 420.0, 485.0]
    assert _band_of(226.0, bands) == 0      # Camarines Sur label
    assert _band_of(321.0, bands) == 1      # Masbate
    assert _band_of(384.0, bands) == 2      # Sorsogon
    assert _band_of(441.0, bands) == 3      # Camarines, wrapped
    assert _band_of(453.0, bands) == 3      # ...Norte, same cell


def test_a_position_outside_every_cell_is_refused():
    """So the caller falls back to proximity rather than inventing a province."""
    from ph_economic_ai.tools.doe_price_archive import _band_of
    bands = [168.0, 295.0]
    assert _band_of(100.0, bands) is None
    assert _band_of(400.0, bands) is None
    assert _band_of(200.0, []) is None


def test_a_cell_boundary_belongs_to_the_cell_below_it():
    """Half-open intervals, so a block starting exactly on a rule cannot land in
    two provinces or in neither."""
    from ph_economic_ai.tools.doe_price_archive import _band_of
    bands = [100.0, 200.0, 300.0]
    assert _band_of(200.0, bands) == 1
    assert _band_of(199.9, bands) == 0


# ── Columns come from the page, not from a constant ──────────────────────────

class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Page:
    """Just enough of a fitz page for the column reader: vertical ruling lines."""

    def __init__(self, xs, hlines=()):
        self._xs, self._h = xs, hlines

    def get_drawings(self):
        out = [{'items': [('l', _P(x, 100.0), _P(x, 400.0))]} for x in self._xs]
        out += [{'items': [('l', _P(x0, y), _P(x1, y))]} for x0, x1, y in self._h]
        return out


def _hdr(x0, x1, y, text):
    return (x0, y, x1, y + 8, text, 0, 0, 0)


def test_column_boundaries_are_read_from_the_page_being_parsed():
    """They were three constants measured off one document: city at x=82. DOE
    scales the sheet per issue, and on a January 2026 file the province column
    ends at x=59 with cities starting at 63, so every city name fell left of 82
    and was read as a PROVINCE. The output was a list of real Philippine place
    names -- Jimenez, Titay, Lala -- which is why it read as correct."""
    from ph_economic_ai.tools.doe_price_archive import _columns

    small = [_hdr(28.8, 53.9, 121, 'PROVINCE'), _hdr(72.8, 83.8, 118, 'CITY'),
             _hdr(62.5, 96.7, 125, 'MUNICIPALITY'), _hdr(107.0, 130.8, 122, 'PRODUCT')]
    cols = _columns(small, _Page([21.6, 59.5, 98.8, 138.0]))
    assert cols.city == 59.5 and cols.product == 98.8
    assert cols.city < 63.0, 'a city at x=63 must not fall in the province column'

    large = [_hdr(38.3, 85.3, 228, 'PROVINCE'), _hdr(120.8, 141.4, 222, 'CITY'),
             _hdr(101.5, 165.6, 235, 'MUNICIPALITY'), _hdr(184.4, 229.0, 228, 'PRODUCT')]
    wide = _columns(large, _Page([25.3, 96.1, 169.3, 242.6]))
    assert wide.city == 96.1 and wide.product == 169.3
    assert wide.city != cols.city, 'two scales cannot share one boundary'


def test_columns_fall_back_to_the_header_midpoint_without_rules():
    """A document whose rules are unusable still parses, rather than being
    dropped for want of a drawing layer."""
    from ph_economic_ai.tools.doe_price_archive import _columns
    hdr = [_hdr(28.8, 53.9, 121, 'PROVINCE'), _hdr(72.8, 83.8, 118, 'CITY'),
           _hdr(62.5, 96.7, 125, 'MUNICIPALITY'), _hdr(107.0, 130.8, 122, 'PRODUCT')]
    cols = _columns(hdr, _Page([]))
    assert 53.9 < cols.city < 62.5
    assert 96.7 < cols.product < 107.0


def test_an_ncr_sheet_has_no_province_column_and_still_parses():
    """NCR is the region, so its sheets head the label column `AREA` and carry
    two columns where regional sheets carry three. Requiring PROVINCE returned no
    columns for every NCR document, and NCR is the one region Phase 2 cannot do
    without. The province column collapses to the left edge, so a city can never
    be read into it."""
    from ph_economic_ai.tools.doe_price_archive import _columns
    hdr = [_hdr(127.0, 160.0, 228, 'AREA'), _hdr(225.0, 270.0, 228, 'PRODUCT')]
    cols = _columns(hdr, _Page([84.8, 207.3, 308.5]))
    assert cols is not None
    assert cols.city == cols.left, 'no province column means an EMPTY one'
    assert cols.product == 207.3


def test_a_header_written_as_one_token_is_still_found():
    """`CITY / MUNICIPALITY` is three tokens on some sheets and the single token
    `CITY/MUNICIPALITY` on others. An exact-match lookup found the first and not
    the second, so 116 documents from 2024 to 2026 reported no label column and
    were dropped whole -- the most recent Visayas weeks, which is the series
    Phase 2 most wants."""
    from ph_economic_ai.tools.doe_price_archive import _columns
    joined = [_hdr(106.0, 140.0, 121, 'PROVINCE'),
              _hdr(149.0, 205.0, 121, 'CITY/MUNICIPALITY'),
              _hdr(214.0, 250.0, 121, 'PRODUCT')]
    cols = _columns(joined, _Page([100.0, 145.0, 210.0, 255.0]))
    assert cols is not None and cols.city == 145.0 and cols.product == 210.0


def test_a_data_row_naming_a_city_is_not_mistaken_for_a_header():
    """The window had to widen, because some sheets print `Province  Cities`
    three rows BELOW the PRODUCT row -- read as absent, the province column
    collapsed onto the left edge and the city column swallowed every province
    name on the page. Widening on the header WORDS alone was unsafe: `Davao City
    RON 100 54.00` offers the token `City`, and treating it as a header would
    skip the first city. A header names a column and carries neither a product
    nor a price."""
    from ph_economic_ai.tools.doe_price_archive import _is_header_row

    assert _is_header_row([_w(42, 200, 'Province'), _w(115, 200, 'Cities')])
    assert not _is_header_row(
        [_w(54, 300, 'Davao'), _w(74, 300, 'City'), _w(98, 300, 'RON'),
         _w(112, 300, '100'), _w(154, 300, '54.00')])
    assert not _is_header_row([_w(54, 300, 'City'), _w(154, 300, '54.00')])


def test_a_header_the_page_does_not_have_is_refused():
    from ph_economic_ai.tools.doe_price_archive import _columns
    assert _columns([_hdr(127.0, 160.0, 228, 'AREA')], _Page([84.8])) is None


def test_the_province_band_window_follows_the_page_scale():
    """The rules bounding a province cell only count if they span the province
    column, and that test was the literal `x < 40 and x > 60` of one document. On
    a page whose province column runs 21.6 to 59.5 no rule qualified, so the bands
    came back EMPTY and the page fell through to proximity -- which meant the
    wrong-province fix quietly stopped applying to every document at that scale.
    A silent fallback is the worst shape for a bug: the output stays plausible."""
    from ph_economic_ai.tools.doe_price_archive import Columns, _province_bands

    rules = [(21.6, 59.5, 168.0), (21.6, 59.5, 295.0), (21.6, 59.5, 358.0)]
    page = _Page([21.6, 59.5, 98.8], hlines=rules)
    small = Columns(left=21.6, city=59.5, product=98.8, prices=138.0)

    assert _province_bands(page, small) == [168.0, 295.0, 358.0]
    assert _province_bands(page) == [], 'the old fixed window sees nothing here'


# ── Wrapped labels reassemble in reading order ───────────────────────────────

def test_a_wrapped_province_reassembles_in_reading_order():
    """Sorted by (y, x). Sorting the raw tuples put the WORD third in the key, so
    two fragments on ONE line came back alphabetically, and uppercase sorts
    before lowercase: `Agusan del Sur` became `Agusan Sur del`. Every result was
    a real province with its words rearranged."""
    from ph_economic_ai.tools.doe_price_archive import _assemble_labels
    one_line = [(300.0, 40.0, 'Agusan'), (300.0, 70.0, 'del'), (300.0, 82.0, 'Sur')]
    assert _assemble_labels(one_line) == [(300.0, 'Agusan del Sur')]
    assert _assemble_labels([(300.0, 40.0, 'Santa'), (300.0, 66.0, 'Maria')]) \
        == [(300.0, 'Santa Maria')]


def test_a_province_wrapped_across_lines_is_still_one_label():
    """`Camarines` and `Norte` print on separate lines with a city between."""
    from ph_economic_ai.tools.doe_price_archive import _assemble_labels
    assert _assemble_labels([(300.0, 40.0, 'Camarines'), (312.0, 40.0, 'Norte')]) \
        == [(300.0, 'Camarines Norte')]


def test_two_provinces_far_apart_stay_two_labels():
    from ph_economic_ai.tools.doe_price_archive import _assemble_labels
    out = _assemble_labels([(300.0, 40.0, 'Albay'), (400.0, 40.0, 'Sorsogon')])
    assert [label for _y, label in out] == ['Albay', 'Sorsogon']


# ── Where the table stops ────────────────────────────────────────────────────

def test_the_monitoring_footer_ends_the_table():
    """`DATE OF MONITORING` needs the optional OF. Without it the footer was read
    as data: its words sat in the label column and appended themselves to the
    last city, which came out as `Date Pasay of City`."""
    from ph_economic_ai.tools.doe_price_archive import _FOOTER_RE
    assert _FOOTER_RE.search('DATE OF MONITORING: JANUARY 6-9, 2026')
    assert _FOOTER_RE.search('DATE MONITORED')


def test_the_ncr_summary_table_is_not_read_as_city_data():
    """NCR sheets print a SECOND table below the first, a region-wide summary.
    Its `Diesel` and `Kerosene` rows match as products and attached themselves to
    the last city on the page."""
    from ph_economic_ai.tools.doe_price_archive import _FOOTER_RE
    assert _FOOTER_RE.search('PREVAILING RETAIL PRICES OF PETROLEUM PRODUCTS NCR')


# ── Placing the live series ──────────────────────────────────────────────────

def test_the_lfro_series_is_mindanao_despite_its_name():
    """`lfro` reads as Luzon and the earlier mapping inferred exactly that, from
    the letter L. The 62 files are listed under the MINDANAO field office and
    their province column is 22 Mindanao provinces with no Luzon province on any
    page. These are the live 2025-2026 series, so mislabelling them would put
    Mindanao prices into a Luzon backtest."""
    assert area_of('47-lfro-price-monitoring-november-25-2025-pdf') == 'Mindanao'
    assert area_of('01-lfro-price-monitoring-january-06-2026-pdf') == 'Mindanao'


def test_no_area_label_names_a_region_it_cannot_evidence():
    """`Luzon field office` asserted an island group that the documents contradict.
    An area label is a claim about where the prices are from, so every value in
    SERIES has to be one the content supports."""
    from ph_economic_ai.tools.doe_price_archive import SERIES
    assert {a for _f, a in SERIES} == {'NCR', 'South Luzon', 'Visayas', 'Mindanao'}
