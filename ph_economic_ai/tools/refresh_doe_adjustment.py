"""Build `benchmark/data/doe_price_adjustments.csv` from DOE's weekly notices.

The announced pump move, which is the top of the accuracy roadmap: for six days
of every seven, "what will fuel cost next week" is a published fact and the app
was guessing at it.

**Discovery is derived, not scraped.** The notice PDF's URL follows from the week
it covers, so this asks for a week directly instead of hunting the article
listing for a link. The listing route was tried first and abandoned for cause:
inside a single day it served 200-with-articles, 200-with-an-empty-shell and
HTTP 500, and notices age off its recent-articles window within a day or two of
publication. A weekly job built on it fails most weeks for reasons unrelated to
whether a notice exists. It also needed a headless browser to recover a link that
JavaScript writes after hydration; deriving the URL removes that dependency
altogether, so this tool is now plain `requests` plus `pypdf`.

Measured 2026-08-19: 23 consecutive weeks resolve, 2026-03-10 to 2026-08-11, with
no gaps once both month spellings are tried. DOE writes `jul-7-13` in July and
`june-9-15` in June, and trying only the abbreviation reported eleven published
weeks as missing.

    python -m ph_economic_ai.tools.refresh_doe_adjustment            # recent weeks
    python -m ph_economic_ai.tools.refresh_doe_adjustment --all      # full backfill
    python -m ph_economic_ai.tools.refresh_doe_adjustment --check    # is it stale?

**Running it weekly.** `--check` exits 1 when the committed announcements have
fallen behind OR when the last recorded run failed, so a scheduler can act
without parsing output. Both halves are needed: the staleness window is seven
days, so a refresh that starts failing on a Tuesday would otherwise look healthy
until the following Tuesday. Every run appends its outcome to
`logs/refresh_doe_adjustment.jsonl` via `tools/run_log.py`, which is what makes
the second half answerable at all. On Windows:

    schtasks /Create /TN "DOE price notice" /SC WEEKLY /D TUE /ST 08:00 ^
      /TR "cmd /c cd /d <repo> && python -m ph_economic_ai.tools.refresh_doe_adjustment"

The app does not depend on the scheduler having worked: `feed_is_stale` drives a
notice on the gas card, so a refresh that silently stopped is visible rather than
absorbed. A scheduled task can fail quietly; a staleness banner cannot.
"""
from __future__ import annotations

import csv
import datetime as dt
from typing import Optional

import requests

from ph_economic_ai.benchmark.doe_adjustment import (
    CONSENSUS_TOLERANCE_PHP, NOTICE_URL_BASE, industry_adjustment,
    parse_notice_pdf, week_slugs, weeks_back_from,
)
from ph_economic_ai.benchmark.paths import data
from ph_economic_ai.tools import run_log
from ph_economic_ai.benchmark.provenance import write_record

OUT = data('doe_price_adjustments.csv')

#: Key this tool's run history is filed under. See `tools/run_log.py`.
TOOL = 'refresh_doe_adjustment'

HEADERS = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/120.0.0.0 Safari/537.36')}

#: Weeks fetched by a routine run. Comfortably more than a scheduler needs, so a
#: few missed Tuesdays heal themselves rather than leaving a permanent hole.
RECENT_WEEKS = 6

#: Weeks probed by `--all`. The series starts around 2026-03-10; this reaches
#: past it so the walk finds the true beginning rather than a configured guess.
BACKFILL_WEEKS = 60

#: Consecutive misses before the walk concludes it has run off the end of the
#: series. Notices are weekly and uninterrupted where they exist, so a run this
#: long means the archive stops rather than that one week is absent.
STOP_AFTER_MISSES = 8


def fetch_week(start: dt.date, end: dt.date,
               session: Optional[requests.Session] = None) -> Optional[dict]:
    """The announced adjustment for one week, or None if not published."""
    get = (session or requests).get
    for slug in week_slugs(start, end):
        try:
            resp = get(NOTICE_URL_BASE + slug, headers=HEADERS, timeout=60)
        except requests.RequestException:
            continue
        if resp.status_code != 200 or 'pdf' not in resp.headers.get('content-type', ''):
            continue
        parsed = parse_notice_pdf(resp.content)
        result = industry_adjustment(parsed['rows'], summary=parsed['summary'])
        # The PDF's own heading wins over the requested week. They agree in every
        # observed case, but a slug that happened to resolve to the wrong
        # document would otherwise file its figures under a week it does not
        # describe -- the `RSK-023` defect, reached through the URL instead.
        week = parsed['week'] or (start, end)
        return {'week_start': week[0].isoformat(), 'week_end': week[1].isoformat(),
                'gasoline': result['gasoline'], 'diesel': result['diesel'],
                'kerosene': result['kerosene'], 'basis': result['basis'],
                'n_companies': result['n_companies'],
                'consensus': result['consensus'],
                'source_pdf': NOTICE_URL_BASE + slug}
    return None


def collect(weeks: int, stop_after: int = STOP_AFTER_MISSES) -> list[dict]:
    rows, misses = [], 0
    with requests.Session() as session:
        for start, end in weeks_back_from(count=weeks):
            row = fetch_week(start, end, session)
            if row:
                misses = 0
                rows.append(row)
                print(f'  {row["week_start"]}  gasoline {row["gasoline"]}  '
                      f'diesel {row["diesel"]}  ({row["basis"]})')
            else:
                misses += 1
                print(f'  {start}  not published')
                if misses >= stop_after:
                    print(f'  {stop_after} consecutive misses; end of series')
                    break
    return rows


def build(weeks: int = RECENT_WEEKS) -> None:
    print(f'probing {weeks} week(s) back from today')
    rows = collect(weeks)
    if not rows:
        # Nothing fetched is not the same as nothing existing, and a scheduler
        # reading only the exit code cannot tell them apart unless it is said.
        raise SystemExit(
            'No notice resolved for any probed week. DOE may be unreachable or '
            'the slug format may have changed again. Nothing was written; the '
            'existing CSV is untouched.')

    existing: dict[str, dict] = {}
    if OUT.exists():
        with open(OUT, encoding='utf-8') as fh:
            existing = {r['week_start']: r for r in csv.DictReader(fh)}
    for row in rows:
        existing[row['week_start']] = {k: ('' if v is None else v)
                                       for k, v in row.items()}

    fields = ['week_start', 'week_end', 'gasoline', 'diesel', 'kerosene',
              'basis', 'n_companies', 'consensus', 'source_pdf']
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow({f: existing[key].get(f, '') for f in fields})

    write_record(
        OUT,
        source=('DOE Oil Industry Management Bureau, Prior Notice on Price '
                'Adjustments (weekly PDF, URL derived from the week it covers)'),
        params={'url_base': NOTICE_URL_BASE,
                'slug': 'website-posting-itmsfuel-<week>-pdf',
                'month_spellings': 'abbreviated and full, both tried',
                'weeks_probed': weeks},
        transformations=[
            'derive the document slug from each Tuesday-to-Monday pricing week',
            'extract text with pypdf; read the covering week from the heading '
            'and prefer it over the requested week',
            'industry figure from the summary row where present, else the modal '
            'company figure; mode not mean so one company cannot drag it',
            f'the mode is taken over blocs of filings within '
            f'{CONSENSUS_TOLERANCE_PHP:.2f} PHP/L of each other rather than over '
            f'exactly equal values, and the reported figure is the mode WITHIN '
            f'the winning bloc, so it is always a value some company posted',
        ],
        units='PHP per litre, change for the week',
        notes=('The announced move, not a forecast. Announced Monday, effective '
               '6:00 AM Tuesday, and holds for the week. '
               'The `consensus` column is the share of companies inside the '
               'winning bloc, NOT the share posting an identical figure; the '
               'two differ wherever companies filed the same move to different '
               'centavos, and figures written before 2026-08-19 carry the '
               'stricter meaning.'),
    )
    print(f'\nWrote {OUT.name} ({len(existing)} week(s)) + provenance')
    return len(existing)


def check() -> int:
    """Is this job healthy? Exit 1 if the feed is stale OR the last run failed.

    Separate from `build` so a scheduler can ask the cheap question without any
    network round trip, and so a monitoring job never mutates the repository.

    **Why the last run counts, not only the feed's age.** The two come apart for
    a full week. `feed_is_stale` allows seven days past the newest covered week,
    one publication cycle, so a refresh that starts crashing on a Tuesday leaves
    the committed feed looking current until the following Tuesday. A
    staleness-only check reports success across that whole week over a job that
    has already stopped working. That is the silence this exists to end, and it
    is what happened on 2026-08-19: the run failed, returned 1 to a scheduler
    that records exit codes nobody reads, and left nothing behind.

    Only the MOST RECENT run decides health. A failure that a later run fixed is
    history, not an outstanding alarm.
    """
    from ph_economic_ai.benchmark.doe_adjustment import (
        feed_is_stale, load_announcements)

    rows = load_announcements()
    stale = feed_is_stale()
    newest = max((r.get('week_end') or '' for r in rows), default='(none)')
    print(f'{len(rows)} week(s) on file, newest covers to {newest}')
    print('STALE: the weekly refresh has not run recently' if stale else 'up to date')
    print(run_log.describe(TOOL))

    last = run_log.last_record(TOOL)
    broken = last is not None and not last.get('ok')
    if broken:
        print(f'  traceback:\n{last.get("traceback", "(none recorded)")}')
    if stale or broken:
        print(f'  full history: {run_log.log_path(TOOL)}')
    return 1 if (stale or broken) else 0


if __name__ == '__main__':
    import sys

    args = sys.argv[1:]
    if '--check' in args:
        raise SystemExit(check())

    weeks = BACKFILL_WEEKS if '--all' in args else RECENT_WEEKS
    # The scheduler discards stdout, so an unwrapped traceback reaches nobody.
    # `logged_run` re-raises, which keeps the non-zero exit the task reports.
    with run_log.logged_run(TOOL, weeks=weeks, args=args) as record:
        record['weeks_on_file'] = build(weeks)
