"""DOE's weekly Prior Notice on Price Adjustments: the announced pump move.

**Why this sits at the top of the roadmap.** Every measured gain in this project
came from reading a determined quantity rather than predicting it: the generation
charge was 41 percent low until it was read from data, and weekly gas beats
"assume no change" by 14.7 percent while the monthly pipeline loses to it by 7.35.
This is the same move taken to its conclusion. DOE publishes, each Monday, what
every oil company will charge from 6:00 AM Tuesday, per litre, per product. For
six days of every seven, "what will fuel cost next week" is a published fact.

The app currently answers that question by fetching three Google News HEADLINES
about the bulletin (`engine/live_data.py:206`) and giving them to a swarm of small
local models as prompt text. The number itself is never read.

**What the notice looks like.** Verified against the week of 2026-08-11: fifteen
companies, each with the time their text message arrived, the time the change
takes effect, and a figure for gasoline, diesel and sometimes kerosene. The
industry moved -4.70 on gasoline and -4.30 on diesel, announced 2026-08-10.

Two properties of the real document shape everything here:

  * `Total` is BOTH an oil company and the label of the summary row. Matching on
    the word returns the company's figure and calls it the industry's. In the
    2026-08-11 notice the two agree, which is the dangerous case: the bug ships
    green and only surfaces on a week where the company differs.
  * Companies may stagger a move across days. Filpride/Mobility posted -1.70,
    -2.00 and -1.00 on three consecutive dates, summing to the -4.70 everyone
    else took at once. Reading the first row alone understates the week by 3.00.

**Scope.** This module reasons about rows. It does not fetch them: the notice is
client-rendered behind a JavaScript front end, its body is absent from the HTML,
the Nuxt payload and the document attachments, and the browser surface available
here refuses the host. Extraction is deliberately a separate, swappable layer so
the reasoning that CAN be verified today is verified today. `parse_rows` is a
best-effort tokenizer and is marked as awaiting a real extracted-text sample.
"""
from __future__ import annotations

import collections
import datetime as dt
import re
from typing import Iterable, Mapping, Optional, Sequence

#: Products the notice carries. Kerosene is frequently blank; a blank is absence
#: of an announcement, which is not the same claim as a zero adjustment.
PRODUCTS = ('gasoline', 'diesel', 'kerosene')

#: Fraction of companies that must post the same figure before it is reported as
#: THE announced move. Below this the notice does not contain one number, and
#: averaging would manufacture one. The 2026-08-11 notice sits at 14 of 15.
MIN_CONSENSUS = 0.5

#: Peso figures are centavo-precise; rounding sums here keeps float noise out of
#: the equality test the mode depends on.
_DP = 4

_MONTHS = {m: i for i, m in enumerate(
    ('january', 'february', 'march', 'april', 'may', 'june', 'july',
     'august', 'september', 'october', 'november', 'december'), start=1)}

_WEEK_RE = re.compile(
    r'week\s+([A-Za-z]+)\s+(\d{1,2})\s*[-–]\s*(?:([A-Za-z]+)\s+)?(\d{1,2})\s*,\s*(\d{4})',
    re.IGNORECASE)


def parse_week_header(text: str) -> tuple[dt.date, dt.date]:
    """(start, end) of the week a notice covers, from its own heading.

    Handles both `August 11-17, 2026` and the month-spanning
    `July 28-August 3, 2026`. Refuses rather than guessing: a notice whose week
    cannot be read cannot be filed against an outcome, and a wrong week is the
    defect `RSK-023` withdrew three grades for.
    """
    m = _WEEK_RE.search(text or '')
    if not m:
        raise ValueError(f'no "week <Month> <D>-<D>, <YYYY>" heading in {text!r}')
    m1, d1, m2, d2, year = m.group(1).lower(), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
    if m1 not in _MONTHS:
        raise ValueError(f'unknown month {m.group(1)!r}')
    start_month = _MONTHS[m1]
    end_month = _MONTHS[m2.lower()] if m2 and m2.lower() in _MONTHS else start_month
    # The trailing year belongs to the END date. A week running December into
    # January starts in the previous year.
    end_year = year
    start_year = year - 1 if end_month < start_month else year
    return dt.date(start_year, start_month, d1), dt.date(end_year, end_month, d2)


def company_total(rows: Iterable[Mapping], product: str) -> Optional[float]:
    """One company's total move for `product` across however many dates it used.

    Staggering is a real pattern, not an edge case: Filpride/Mobility split
    -4.70 into -1.70, -2.00 and -1.00 in the 2026-08-11 notice. Summing is what
    makes their week comparable to a company that moved once.
    """
    values = [r.get(product) for r in rows if r.get(product) is not None]
    return round(sum(values), _DP) if values else None


def _modal(values: Sequence[float]) -> tuple[Optional[float], float]:
    """(most common value, share of entries holding it)."""
    if not values:
        return None, 0.0
    counts = collections.Counter(values)
    value, n = counts.most_common(1)[0]
    return value, n / len(values)


def industry_adjustment(rows: Iterable[Mapping],
                        summary: Optional[Mapping] = None) -> dict:
    """The announced per-litre move for the week, with how it was determined.

    `basis` records which rule produced the number, because the rules are not
    equivalent and a reader cannot check the figure without it:

      * `summary`      -- the notice's own total row was supplied;
      * `modal`        -- the figure most companies posted;
      * `no_consensus` -- companies disagreed; no single announced number exists;
      * `empty`        -- nothing to read.

    The mode is used rather than the mean so one company cannot drag the
    published figure: Clean Fuel posted -5.00 against everyone else's -4.70 on
    2026-08-11, and an average would have reported -4.72, a number no station
    charged.
    """
    rows = list(rows or [])
    by_company: dict[str, dict] = {}
    for row in rows:
        by_company.setdefault(row['company'], []).append(row)

    totals = {
        company: {p: company_total(company_rows, p) for p in PRODUCTS}
        for company, company_rows in by_company.items()
    }

    result: dict = {
        'n_companies': len(by_company),
        'by_company': totals,
        'outliers': [],
        'consensus': 0.0,
    }
    for product in PRODUCTS:
        result[product] = None

    if not by_company:
        result['basis'] = 'empty'
        return result

    # Consensus is judged on gasoline: it is the product every company reports,
    # so it is the only one where a disagreement is unambiguous rather than an
    # artefact of who bothered to file.
    posted = [t['gasoline'] for t in totals.values() if t['gasoline'] is not None]
    modal_value, share = _modal(posted)
    result['consensus'] = round(share, 4)
    result['outliers'] = sorted(
        company for company, t in totals.items()
        if t['gasoline'] is not None and t['gasoline'] != modal_value)

    if summary:
        result['basis'] = 'summary'
        for product in PRODUCTS:
            if summary.get(product) is not None:
                result[product] = round(float(summary[product]), _DP)
        return result

    if share < MIN_CONSENSUS:
        result['basis'] = 'no_consensus'
        return result

    result['basis'] = 'modal'
    for product in PRODUCTS:
        values = [t[product] for t in totals.values() if t[product] is not None]
        value, _ = _modal(values)
        result[product] = value
    return result


# ── Extraction, validated against the 2026-08-11 notice ──────────────────────
#
# The notice is a PDF linked from a client-rendered article. `pypdf` flattens it
# to text whose shape is NOT what a reader of the rendered table would guess, and
# each surprise below cost a wrong assumption before the real sample was read:
#
#   * a row spans FOUR lines. The company and the time it filed are on one line,
#     the dates on the next two, and the money lands on the line carrying the
#     effectivity date: `August 11 -4.70 -4.30 -4.88`.
#   * the table HEADING extracts LAST, after every row, so anything that scans
#     forward from "For the week" finds nothing.
#   * a staggered company's rows extract BEFORE its name. Filpride/Mobility's
#     -1.70, -2.00 and -1.00 appear while the last named company is still Flying
#     V, so per-company attribution would silently credit them to the wrong firm
#     and, worse, sum Flying V to -9.40 and poison the mode.
#
# The summary row is what makes this tractable: it extracts as a money-only line
# with no date prefix, which nothing else in the document looks like. Preferring
# it sidesteps company attribution entirely for the number that matters.

_MONEY_RE = re.compile(r'-?\d+\.\d{2}\b')
_TIME_RE = re.compile(r'\d{1,2}:\d{2}\s*(?:AM|PM)', re.IGNORECASE)
_COMPANY_RE = re.compile(r'^(.+?)\s+\d{1,2}:\d{2}\s*(?:AM|PM)', re.IGNORECASE)
_DATED_MONEY_RE = re.compile(r'^[A-Za-z]+\s+\d{1,2}\b')


def parse_notice_text(text: str) -> dict:
    """`{'week': (start, end), 'rows': [...], 'summary': {...} | None}`.

    Validated against the notice for the week of 2026-08-11.

    **Company attribution is best effort and the headline does not rest on it.**
    PDF extraction reorders the staggered block, so a company that spread its
    move across days may have those rows credited to the firm above it. The
    summary row is preferred precisely because it is immune to that; the rows are
    returned for inspection and as a fallback, not as the primary reading.
    """
    rows: list[dict] = []
    summary: Optional[dict] = None
    company: Optional[str] = None

    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        money = _MONEY_RE.findall(line)

        if not money:
            named = _COMPANY_RE.match(line)
            if named:
                company = named.group(1).strip(' /|\t')
            continue

        values = [float(v) for v in money[:3]]
        record = dict(zip(('gasoline', 'diesel', 'kerosene'), values))
        record.setdefault('kerosene', None)

        head = line[:line.find(money[0])].strip()
        if not head:
            # Money with nothing before it is the table's own bottom line.
            summary = record
            continue
        if _DATED_MONEY_RE.match(head):
            rows.append({'company': company or 'unknown', **record})
            continue
        named = _COMPANY_RE.match(line)
        if named:
            company = named.group(1).strip(' /|\t')
        rows.append({'company': company or head, **record})

    try:
        week = parse_week_header(text)
    except ValueError:
        week = None
    return {'week': week, 'rows': rows, 'summary': summary}


def parse_notice_pdf(content: bytes) -> dict:
    """Parse a downloaded notice PDF. `pypdf` is imported lazily, as in
    `benchmark/meralco.py`, so importing this module never requires it."""
    import io

    import pypdf

    text = '\n'.join((page.extract_text() or '')
                     for page in pypdf.PdfReader(io.BytesIO(content)).pages)
    return parse_notice_text(text)
