"""DOE's weekly Prior Notice on Price Adjustments -- the announced number.

This is the top of the certainty gradient. DOE publishes, every Monday, what each
oil company will charge from 6:00 AM Tuesday, per litre, per product. For six days
of every seven the "next week's price" question is a published fact, not a
forecast, and the app currently answers it by feeding three news HEADLINES about
the bulletin to a swarm of small language models.

Verified against the notice for the week of 2026-08-11: gasoline -4.70, diesel
-4.30, announced 2026-08-10 between 9:24 AM and 9:00 PM, effective 6:00 AM
2026-08-11.

Two properties of the real notice drive every design choice here:

  * `Total` is BOTH an oil company and the summary row. A parser that searches
    for the word takes the company's number and calls it the industry's. In the
    2026-08-11 notice they happen to agree, which is worse than if they differed
    -- the bug would ship green.
  * Companies may stagger an adjustment across days. Filpride/Mobility posted
    -1.70, -2.00 and -1.00 on three consecutive dates, which sums to the -4.70
    everyone else took in one step. Reading only the first row understates the
    week by 3.00 PHP/L.

The logic below is tested on structured rows, not on scraped text, because the
notice is client-rendered and no extracted-text sample exists yet. Keeping the
reasoning pure means the part that can be verified today is verified today, and
only the tokenizer waits on the fetch being solved.
"""
import datetime as dt
import pathlib

import pytest

from ph_economic_ai.benchmark import doe_adjustment as da


def _row(company, gasoline, diesel, kerosene=None, effectivity='2026-08-11',
         received='2026-08-10'):
    return {'company': company, 'gasoline': gasoline, 'diesel': diesel,
            'kerosene': kerosene, 'effectivity': effectivity, 'received': received}


#: The real notice, transcribed. 13 of 15 companies posted an identical move.
AUG_11 = [
    _row('Shell', -4.70, -4.30, -4.90),
    _row('City Oil', -4.70, -4.30, -4.88),
    _row('Total', -4.70, -4.30),                    # the COMPANY, not the summary
    _row('Petro Gazz', -4.70, -4.30),
    _row('Ecooil', -4.70, -4.30),
    _row('Phoenix', -4.70, -4.30),
    _row('Seaoil', -4.70, -4.30, -4.88),
    _row('Jetti', -4.70, -4.30),
    _row('Unioil', -4.70, -4.30),
    _row('Petron', -4.70, -4.30, -4.90),
    _row('PTT', -4.70, -4.30),
    _row('Clean Fuel', -5.00, -5.00),               # outlier
    _row('Caltex', -4.70, -4.30, -4.88),
    _row('Flying V', -4.70, -4.30, -4.88),
    _row('Filpride/Mobility', -1.70, -1.30, effectivity='2026-08-11'),
    _row('Filpride/Mobility', -2.00, -2.00, effectivity='2026-08-12'),
    _row('Filpride/Mobility', -1.00, -1.00, effectivity='2026-08-13'),
]


# ── The week the notice covers ───────────────────────────────────────────────

def test_reads_the_week_from_the_heading():
    start, end = da.parse_week_header('For the week August 11-17, 2026')
    assert start == dt.date(2026, 8, 11)
    assert end == dt.date(2026, 8, 17)


def test_a_week_spanning_two_months_is_read_correctly():
    start, end = da.parse_week_header('For the week July 28-August 3, 2026')
    assert start == dt.date(2026, 7, 28)
    assert end == dt.date(2026, 8, 3)


def test_a_heading_without_a_week_is_refused():
    with pytest.raises(ValueError):
        da.parse_week_header('Prior Notice on Price Adjustments')


# ── The Total/Total collision ────────────────────────────────────────────────

def test_the_company_named_total_is_not_mistaken_for_the_summary():
    """The trap. `Total` is an oil company AND the label of the summary row.

    Here the company posts a number that is NOT the industry move, so a parser
    that matches on the word alone returns -9.99 and looks perfectly healthy.
    """
    rows = [_row('Shell', -4.70, -4.30),
            _row('Total', -9.99, -9.99),          # the company, deliberately odd
            _row('Petron', -4.70, -4.30),
            _row('Caltex', -4.70, -4.30)]
    result = da.industry_adjustment(rows)
    assert result['gasoline'] == pytest.approx(-4.70)
    assert result['basis'] == 'modal'


def test_an_explicit_summary_row_is_preferred_over_the_mode():
    rows = [_row('Shell', -4.70, -4.30), _row('Petron', -4.70, -4.30)]
    result = da.industry_adjustment(rows, summary={'gasoline': -4.65, 'diesel': -4.30})
    assert result['gasoline'] == pytest.approx(-4.65)
    assert result['basis'] == 'summary'


# ── Staggered adjustments ────────────────────────────────────────────────────

def test_a_staggered_company_is_summed_not_truncated():
    """Filpride/Mobility spread -4.70 across three days. Taking the first row
    reports -1.70 and understates the week by 3.00 PHP/L."""
    staggered = [r for r in AUG_11 if r['company'] == 'Filpride/Mobility']
    assert da.company_total(staggered, 'gasoline') == pytest.approx(-4.70)
    assert da.company_total(staggered, 'diesel') == pytest.approx(-4.30)


def test_the_staggered_company_agrees_with_the_industry_once_summed():
    result = da.industry_adjustment(AUG_11)
    per_company = result['by_company']
    assert per_company['Filpride/Mobility']['gasoline'] == pytest.approx(-4.70)
    assert per_company['Filpride/Mobility']['gasoline'] == pytest.approx(result['gasoline'])


# ── The real notice end to end ───────────────────────────────────────────────

def test_the_2026_08_11_notice_reads_minus_4_70():
    result = da.industry_adjustment(AUG_11)
    assert result['gasoline'] == pytest.approx(-4.70)
    assert result['diesel'] == pytest.approx(-4.30)
    assert result['n_companies'] == 15
    assert result['consensus'] >= 0.85          # 13 of 15 posted the modal move


def test_kerosene_is_optional_and_never_invented():
    """Most companies left kerosene blank. A missing product is None, not 0.0 --
    a zero would be read as "no change announced", which is a different claim."""
    result = da.industry_adjustment(AUG_11)
    assert result['kerosene'] is not None            # five companies did report it
    sparse = da.industry_adjustment([_row('Shell', -4.70, -4.30)])
    assert sparse['kerosene'] is None


def test_an_outlier_does_not_move_the_industry_number():
    """Clean Fuel posted -5.00 against everyone else's -4.70. The mode is used
    precisely so one company cannot drag the published figure."""
    result = da.industry_adjustment(AUG_11)
    assert result['gasoline'] == pytest.approx(-4.70)
    assert 'Clean Fuel' in result['outliers']


def test_no_consensus_is_reported_rather_than_averaged():
    """If the companies genuinely disagree there is no single announced number,
    and inventing a mean would manufacture one."""
    rows = [_row('A', -1.0, -1.0), _row('B', -3.0, -3.0), _row('C', -5.0, -5.0)]
    result = da.industry_adjustment(rows)
    assert result['consensus'] < 0.5
    assert result['basis'] == 'no_consensus'
    assert result['gasoline'] is None


def test_an_empty_notice_yields_nothing_rather_than_zero():
    result = da.industry_adjustment([])
    assert result['gasoline'] is None
    assert result['basis'] == 'empty'


# ── The real document, not a transcription of it ─────────────────────────────

FIXTURE = (pathlib.Path(__file__).parent / 'fixtures'
           / 'doe_notice_2026-08-11.txt')


def _notice_text():
    return FIXTURE.read_text(encoding='utf-8')


def test_the_real_notice_text_yields_the_published_figures():
    """End to end on `pypdf` output from the actual DOE notice, not on rows
    typed from the screenshot. Everything above tests the reasoning; this tests
    that the reasoning ever meets the document."""
    parsed = da.parse_notice_text(_notice_text())
    result = da.industry_adjustment(parsed['rows'], summary=parsed['summary'])
    assert result['gasoline'] == pytest.approx(-4.70)
    assert result['diesel'] == pytest.approx(-4.30)
    assert result['basis'] == 'summary'


def test_the_week_is_read_although_the_heading_extracts_last():
    """`For the week August 11-17, 2026` appears AFTER every data row in
    extraction order. Anything scanning forward from the heading finds nothing.
    """
    parsed = da.parse_notice_text(_notice_text())
    assert parsed['week'] == (dt.date(2026, 8, 11), dt.date(2026, 8, 17))


def test_the_summary_row_is_recognised_by_having_no_date():
    """Every data row carries its effectivity date on the money line; the
    table's bottom line carries only money. That is the one unambiguous signal
    in the flattened text, and it is what lets the headline avoid depending on
    company attribution at all.
    """
    parsed = da.parse_notice_text(_notice_text())
    assert parsed['summary'] is not None
    assert parsed['summary']['gasoline'] == pytest.approx(-4.70)
    assert parsed['summary']['diesel'] == pytest.approx(-4.30)


def test_every_data_row_is_captured():
    """Fourteen companies moved once and one spread its move over three days,
    so the table holds seventeen data rows. Losing the staggered ones is the
    failure that understates a week."""
    parsed = da.parse_notice_text(_notice_text())
    assert len(parsed['rows']) == 17


def test_the_staggered_rows_survive_extraction_reordering():
    """Filpride/Mobility's -1.70, -2.00 and -1.00 extract BEFORE its name, so
    they cannot be attributed by position. They must still be present."""
    parsed = da.parse_notice_text(_notice_text())
    gasoline = sorted(r['gasoline'] for r in parsed['rows'])
    for expected in (-2.00, -1.70, -1.00):
        assert any(g == pytest.approx(expected) for g in gasoline), (
            f'staggered row {expected} lost in extraction')
