"""DOE's weekly Prior Notice on Price Adjustments -- the announced number.

This is the top of the certainty gradient. DOE publishes, every Monday, what each
oil company will charge from 6:00 AM Tuesday, per litre, per product. For six days
of every seven the "next week's price" question is a published fact, not a
forecast. The app read three news HEADLINES about the bulletin instead until
PR #38 put the number on the gas card and PR #39 replaced the headline feed in
the swarm prompt with the figure itself.

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

The reasoning below is tested on structured rows AND, since PR #37, on the real
`pypdf` output of the 2026-08-11 notice committed at
`fixtures/doe_notice_2026-08-11.txt`. Splitting the two is deliberate: the row
logic is verifiable without a document, and the fixture proves the reasoning ever
meets one.
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


# ── The summary row was exempt from every check ──────────────────────────────
#
# Found 2026-08-21, investigating a gap the Accuracy Roadmap had recorded as
# unexplained: summary-basis weeks are measurably worse against observed retail
# than modal ones, 1.71 MAD against 0.94.
#
# The cause is `industry_adjustment`'s ordering. `if summary:` returned BEFORE
# the consensus floor, so the summary row was exempt from the one check that
# decides whether an announced figure exists at all. The modal path carried two
# guards, the floor and the clustered mode; the summary path carried none.
#
# `test_the_2026_08_18_notice_has_no_summary_row` below already states half of
# this -- "with a summary row the mode is never consulted". Skipping the mode is
# deliberate and correct, because the summary row is immune to the company
# attribution errors PDF extraction causes. Skipping the FLOOR was not intended.
#
# In the 24 committed weeks the split is total: every modal week sits at 0.714
# consensus or better, and every week below that is summary-basis. Seven weeks
# published a figure the filings do not support, including one read from a
# single company's filing and three where the largest bloc was under a third:
#
#     2026-03-24   +11.00 gasoline   10 companies   consensus 0.30
#     2026-03-31    +2.60 gasoline   11 companies   consensus 0.27
#     2026-04-07    +6.00 gasoline   12 companies   consensus 0.25
#     2026-07-21    +3.65 gasoline    1 company     consensus 1.00
#
# The last is the sharper one. A share computed over a single filing is 1.0 by
# construction, so the number that is supposed to express agreement reports
# perfect agreement precisely when there is nobody to agree with. Publishing
# that as the industry move is the borrowed-authority failure `RSK-023` cost
# three withdrawn grades for, and worse here because the figure carries DOE's
# certainty rather than a model's.

def test_a_summary_row_does_not_exempt_a_week_from_the_consensus_floor():
    """The defect. Companies disagree three ways; a summary row is present."""
    rows = [_row('A', -1.0, -1.0), _row('B', -3.0, -3.0), _row('C', -5.0, -5.0)]
    result = da.industry_adjustment(rows, summary={'gasoline': 11.0, 'diesel': 18.0})
    assert result['basis'] == 'no_consensus', (
        'a summary row must not publish a figure the filings contradict')
    assert result['gasoline'] is None


def test_one_filing_is_not_a_consensus():
    """2026-07-21. One company, so the share is 1.0 by construction."""
    result = da.industry_adjustment([_row('A', -2.0, -2.0)],
                                    summary={'gasoline': 3.65, 'diesel': 10.68})
    assert result['n_companies'] == 1
    assert result['basis'] == 'no_consensus'
    assert result['gasoline'] is None


def test_a_tie_is_not_a_consensus():
    """2026-03-10 and 03-17: two companies, one each, a coin flip at 0.5."""
    rows = [_row('A', -1.0, -1.0), _row('B', -9.0, -9.0)]
    result = da.industry_adjustment(rows, summary={'gasoline': 9.0, 'diesel': 21.0})
    assert result['consensus'] == pytest.approx(0.5)
    assert result['basis'] == 'no_consensus'


def test_the_guard_does_not_disable_the_summary_row():
    """The risk in tightening a floor is silencing the path it guards.

    Where the filings agree, the summary is still preferred over the mode, which
    is the whole reason it is read: it is immune to the attribution errors PDF
    extraction introduces.
    """
    rows = [_row('A', -4.70, -4.30), _row('B', -4.70, -4.30), _row('C', -4.70, -4.30)]
    result = da.industry_adjustment(rows, summary={'gasoline': -4.72, 'diesel': -4.31})
    assert result['basis'] == 'summary'
    assert result['gasoline'] == pytest.approx(-4.72)


def test_no_committed_week_publishes_a_figure_its_filings_do_not_support():
    """The data regression, against the committed series.

    Applying the corrected guard to what is on file, not to a fixture.
    """
    import csv
    with open(da.ANNOUNCEMENTS_CSV, encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    assert rows, 'the announced series is empty'

    offenders = [
        r['week_start'] for r in rows
        if r.get('gasoline') not in (None, '')
        and (float(r['consensus']) <= da.MIN_CONSENSUS or int(r['n_companies']) < da.MIN_FILINGS)
    ]
    assert not offenders, (
        'weeks publishing an announced figure their own filings do not support: '
        + ', '.join(offenders))


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


# ── Reaching the screen ──────────────────────────────────────────────────────

def test_an_announcement_is_found_for_any_day_in_its_week():
    rows = [{'week_start': '2026-08-11', 'week_end': '2026-08-17',
             'gasoline': '-4.70', 'diesel': '-4.30', 'kerosene': '',
             'basis': 'summary', 'n_companies': '14', 'consensus': '0.8571',
             'source_pdf': 'https://example/x-pdf'}]
    for day in (dt.date(2026, 8, 11), dt.date(2026, 8, 14), dt.date(2026, 8, 17)):
        got = da.announcement_for(day, announcements=rows)
        assert got is not None and got['gasoline'] == pytest.approx(-4.70)


def test_no_announcement_outside_the_week_it_covers():
    """Silence is the correct answer. Showing last week's announced move for
    this week would be the `RSK-023` defect -- a number attached to the wrong
    period -- with the added harm that it reads as a fact rather than a guess.
    """
    rows = [{'week_start': '2026-08-11', 'week_end': '2026-08-17',
             'gasoline': '-4.70', 'diesel': '-4.30', 'kerosene': '',
             'basis': 'summary', 'n_companies': '14', 'consensus': '0.8571',
             'source_pdf': ''}]
    assert da.announcement_for(dt.date(2026, 8, 10), announcements=rows) is None
    assert da.announcement_for(dt.date(2026, 8, 18), announcements=rows) is None


def test_a_week_with_no_consensus_announces_nothing():
    """`industry_adjustment` returns None when companies disagree. That must
    stay None on screen rather than becoming a confident zero."""
    rows = [{'week_start': '2026-08-11', 'week_end': '2026-08-17',
             'gasoline': '', 'diesel': '', 'kerosene': '',
             'basis': 'no_consensus', 'n_companies': '3', 'consensus': '0.33',
             'source_pdf': ''}]
    got = da.announcement_for(dt.date(2026, 8, 12), announcements=rows)
    assert got is not None
    assert got['gasoline'] is None


def test_the_committed_series_loads():
    rows = da.load_announcements()
    assert rows, 'no announcements committed'
    assert all('week_start' in r for r in rows)


def test_the_announced_move_is_never_an_app_forecast():
    """The load-bearing safety property.

    The announced figure is what the oil companies published, not what this app
    predicted. If it ever reached the graded track record the app would show a
    near-perfect accuracy it did not earn -- the precise overclaim this project
    keeps retracting. `announcement_for` is therefore a READ, and nothing in it
    produces an estimate, an error, or a graded row.
    """
    import inspect

    from ph_economic_ai.engine import store as _store

    source = inspect.getsource(da)
    for forbidden in ('upsert_sector_grade', 'update_run_quality',
                      'find_and_grade', 'add_graded', 'record_prediction'):
        assert forbidden not in source, (
            f'{forbidden} appears in doe_adjustment: an announced number must '
            f'never enter the grading path')
    assert not hasattr(da, 'grade')


# ── How it is said on screen ─────────────────────────────────────────────────

def test_the_announced_line_says_it_is_not_a_forecast():
    """Borrowed authority is the risk. A published figure shown in the same
    voice as the app's own estimate lets a reader credit the app with DOE's
    certainty."""
    from ph_economic_ai.ui import honesty

    line = honesty.announced_adjustment_line(
        {'gasoline': -4.70, 'week_start': '2026-08-11',
         'week_end': '2026-08-17', 'n_companies': '14'})
    assert '-4.70' in line
    assert 'not this app' in line.lower()
    assert '2026-08-11' in line


def test_nothing_announced_says_nothing():
    from ph_economic_ai.ui import honesty
    assert honesty.announced_adjustment_line(None) == ''
    assert honesty.announced_adjustment_line({'gasoline': None}) == ''


def test_a_disagreeing_week_is_reported_as_such():
    from ph_economic_ai.ui import honesty
    line = honesty.announced_adjustment_line(
        {'gasoline': None, 'basis': 'no_consensus'})
    assert 'did not file a single common change' in line


# ── The swarm prompt reads the number, not headlines about it ────────────────

def test_the_prompt_block_carries_the_announced_figure():
    """The swarm used to receive two Google News titles about the DOE bulletin.
    It now receives the bulletin's number. A headline is a lossy account of a
    figure that is published exactly, and small models are the least able to
    recover the figure from the prose.
    """
    from ph_economic_ai.engine.live_data import LiveDataBrief

    brief = LiveDataBrief()
    block = brief.as_prompt_block({'oil_pct': 1.0, 'usd_pct': 0.0})
    assert 'DOE FUEL PRICE SIGNALS (latest headlines)' not in block


def test_the_headline_feed_is_gone():
    """Removed rather than left dormant: a dead fetch still costs a network
    call, a failure mode inside the thread pool, and a reader's attention."""
    from ph_economic_ai.engine import live_data

    assert not hasattr(live_data, 'fetch_doe_headlines')
    assert 'doe_news' not in vars(live_data.LiveDataBrief())


def test_the_announced_block_is_absent_when_nothing_is_announced(monkeypatch):
    """Outside a covered week the prompt says nothing about an announcement,
    rather than carrying a stale one or an empty heading."""
    from ph_economic_ai.engine import live_data

    monkeypatch.setattr(live_data, '_announced_today', lambda: None)
    block = live_data.LiveDataBrief().as_prompt_block({})
    assert 'ANNOUNCED' not in block.upper()


# ── A feed that stopped updating must not look like a quiet week ─────────────

_FEED = [{'week_start': '2026-08-11', 'week_end': '2026-08-17',
          'gasoline': -4.70, 'diesel': -4.30, 'kerosene': None,
          'basis': 'summary', 'n_companies': '14', 'consensus': 0.8571,
          'source_pdf': ''}]


def test_a_current_feed_is_not_stale():
    """Two days past the covered week is normal: the next notice publishes on
    Monday, so a grace period keeps a Tuesday check from crying wolf."""
    assert da.feed_is_stale(dt.date(2026, 8, 19), announcements=_FEED) is False


def test_a_feed_that_missed_a_week_is_stale():
    """DOE publishes weekly. More than seven days past the newest covered week
    means at least one notice was never fetched."""
    assert da.feed_is_stale(dt.date(2026, 8, 31), announcements=_FEED) is True


def test_an_empty_feed_is_stale_not_quiet():
    """The load-bearing case. Nothing on file and nothing on screen is
    indistinguishable from "no change announced this week", and the two call for
    opposite responses: one is normal, the other means the refresh stopped
    running. `series_is_stale` treats an empty series the same way, and for the
    same reason.
    """
    assert da.feed_is_stale(dt.date(2026, 8, 19), announcements=[]) is True


def test_the_screen_distinguishes_stale_from_quiet():
    """A reader who sees silence must be able to tell which silence it is."""
    from ph_economic_ai.ui import honesty

    quiet = honesty.announced_adjustment_line(None, stale=False)
    stale = honesty.announced_adjustment_line(None, stale=True)
    assert quiet == ''
    assert stale != ''
    assert 'out of date' in stale.lower() or 'not been updated' in stale.lower()


def test_a_stale_feed_never_shows_a_figure_anyway():
    """Staleness must not be papered over by displaying the last known number.
    That is the `RSK-023` defect: a figure attached to a period it does not
    describe, here wearing the authority of a published fact."""
    from ph_economic_ai.ui import honesty

    line = honesty.announced_adjustment_line(None, stale=True)
    assert '4.70' not in line
    assert '-' not in line.replace('out-of-date', '').replace('up-to-date', '')


# ── Durable discovery: the URL is derivable from the week ────────────────────

def test_a_week_inside_one_month_makes_the_short_slug():
    slugs = da.week_slugs(dt.date(2026, 8, 11), dt.date(2026, 8, 17))
    assert 'website-posting-itmsfuel-aug-11-17-pdf' in slugs


def test_a_week_across_two_months_names_both():
    slugs = da.week_slugs(dt.date(2026, 7, 28), dt.date(2026, 8, 3))
    assert 'website-posting-itmsfuel-jul-28-aug-3-pdf' in slugs


def test_both_month_spellings_are_offered():
    """DOE writes `jul-7-13` in July and `june-9-15` in June. Trying only the
    abbreviation reported eleven weeks missing that were in fact published --
    a gap that looked like DOE not posting and was our slug being wrong.
    """
    slugs = da.week_slugs(dt.date(2026, 6, 23), dt.date(2026, 6, 29))
    assert 'website-posting-itmsfuel-jun-23-29-pdf' in slugs
    assert 'website-posting-itmsfuel-june-23-29-pdf' in slugs


def test_days_are_not_zero_padded():
    """`jul-7-13`, never `jul-07-13`. The padded form 404s."""
    slugs = da.week_slugs(dt.date(2026, 7, 7), dt.date(2026, 7, 13))
    assert any('jul-7-13' in s for s in slugs)
    assert not any('jul-07-13' in s for s in slugs)


def test_weeks_back_from_walks_tuesdays_in_reverse():
    weeks = list(da.weeks_back_from(dt.date(2026, 8, 19), count=3))
    assert weeks[0][0] == dt.date(2026, 8, 18)      # the Tuesday on or before
    assert weeks[1][0] == dt.date(2026, 8, 11)
    assert weeks[2][0] == dt.date(2026, 8, 4)
    for start, end in weeks:
        assert (end - start).days == 6
        assert start.weekday() == 1                  # Tuesday


def test_a_week_is_seven_days_ending_monday():
    (start, end), = da.weeks_back_from(dt.date(2026, 8, 13), count=1)
    assert start == dt.date(2026, 8, 11)
    assert end == dt.date(2026, 8, 17)
    assert end.weekday() == 0                        # Monday


# ── Centavo-level disagreement is not disagreement ───────────────────────────
#
# Found 2026-08-19 while backfilling. The notice for the week of 2026-08-18 was
# filed by all fifteen companies, every one of them RAISING gasoline, and the
# whole industry sat inside ten centavos:
#
#     2.40 x2    2.49 x7    2.50 x6
#
# Exact-equality mode counts those as three different answers, so the largest
# bloc is 7 of 15, the 0.5 consensus floor rejects it, and the app reports the
# week as having no announced figure. It had one. That is the top of the
# certainty gradient reporting a published fact as unknown.

WEEK_2026_08_18 = (pathlib.Path(__file__).parent / 'fixtures'
                   / 'doe_notice_2026-08-18.txt')


def _aug_18():
    return da.parse_notice_text(WEEK_2026_08_18.read_text(encoding='utf-8'))


def test_the_2026_08_18_notice_has_no_summary_row():
    """The precondition. With a summary row the mode is never consulted, so this
    week only exercises the consensus rule because the document lacks one."""
    parsed = _aug_18()
    assert parsed['summary'] is None
    assert len(parsed['rows']) == 15


def test_centavo_differences_do_not_fragment_the_industry_figure():
    parsed = _aug_18()
    result = da.industry_adjustment(parsed['rows'], summary=parsed['summary'])
    assert result['basis'] == 'modal', (
        'fifteen companies within ten centavos is a consensus, not a dispute')
    assert result['gasoline'] == pytest.approx(2.49)
    assert result['diesel'] == pytest.approx(3.84)
    assert result['consensus'] == pytest.approx(1.0)
    assert result['outliers'] == []


def test_the_figure_reported_is_one_a_company_actually_posted():
    """Why this is a clustered MODE and not a cluster mean.

    The mean of the 2026-08-18 gasoline filings is 2.4753, a number no station
    charged. Grouping near-identical filings must not become a licence to
    average them.
    """
    parsed = _aug_18()
    result = da.industry_adjustment(parsed['rows'], summary=parsed['summary'])
    posted = {r['gasoline'] for r in parsed['rows'] if r['gasoline'] is not None}
    assert result['gasoline'] in posted


def test_the_tolerance_does_not_absorb_a_genuine_outlier():
    """The property the tolerance must not cost.

    Clean Fuel posted -5.00 against everyone else's -4.70 on 2026-08-11, a gap
    of 30 centavos. A tolerance wide enough to swallow that would defeat the
    reason the mode is used at all.
    """
    result = da.industry_adjustment(AUG_11)
    assert result['gasoline'] == pytest.approx(-4.70)
    assert 'Clean Fuel' in result['outliers']


def test_filings_further_apart_than_the_tolerance_stay_separate():
    """The boundary, stated as a test so it cannot drift silently."""
    tol = da.CONSENSUS_TOLERANCE_PHP
    near = [_row('A', 1.00, 1.00), _row('B', 1.00 + tol, 1.00 + tol),
            _row('C', 1.00 + tol, 1.00 + tol)]
    assert da.industry_adjustment(near)['basis'] == 'modal'

    far = [_row('A', 1.00, 1.00), _row('B', 1.00 + tol * 3, 1.00 + tol * 3),
           _row('C', 1.00 + tol * 6, 1.00 + tol * 6)]
    assert da.industry_adjustment(far)['basis'] == 'no_consensus'


def test_genuine_disagreement_is_still_refused():
    """Regression guard on the case the tolerance must not weaken."""
    rows = [_row('A', -1.0, -1.0), _row('B', -3.0, -3.0), _row('C', -5.0, -5.0)]
    result = da.industry_adjustment(rows)
    assert result['basis'] == 'no_consensus'
    assert result['gasoline'] is None
