"""Grade the stored run backlog against already-published PSA history.

`PSAMonthlyCheckerThread` grades months as they settle, but runs already in the
store predate it. This walks them once so the backlog counts too.

    python -m ph_economic_ai.tools.grade_monthly_backlog [--dry-run] [--db PATH]

Idempotent: a month already in `sector_grades` is never regraded, so a second
run reports 0.

Remember the ceiling this enforces: **one graded sample per calendar month, no
matter how many runs fall inside it.** A backlog of dozens of runs spanning two
months is two samples, and twelve independent months is twelve months of
wall-clock. See `engine/ground_truth_monthly` for why that is the point rather
than a limitation.
"""
from __future__ import annotations

import argparse

from ph_economic_ai.engine.ground_truth_monthly import (
    SECTOR_COLUMN,
    find_and_grade_months,
    grade_verdict_monthly,
    load_series,
    series_is_stale,
)
from ph_economic_ai.engine.interval import GRADED_SECTORS, MIN_GRADED_FOR_CALIBRATION
from ph_economic_ai.engine.store import AgentTrustStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--db', default=None, help='path to trust.db')
    ap.add_argument('--dry-run', action='store_true',
                    help='show what would be graded without writing')
    args = ap.parse_args(argv)

    store = AgentTrustStore(args.db)
    print(f'{store.count_runs()} stored runs\n')

    for sector in sorted(GRADED_SECTORS & set(SECTOR_COLUMN)):
        series = load_series(sector)
        span = f'{min(series.index)}..{max(series.index)}' if len(series) else 'empty'
        stale = '  [STALE - refresh the PSA CSV]' if series_is_stale(series) else ''
        print(f'-- {sector}: outcome series {span}{stale}')

        verdict = grade_verdict_monthly(store, sector, series)
        for s in verdict['gradable']:
            print(f'   {s["month"]}: est {s["estimate"]:+.3f} vs actual '
                  f'{s["actual"]:+.3f} -> abs error {s["abs_error"]:.3f} '
                  f'({s["n_runs"]} run(s) collapsed to 1 sample)')
        for b in verdict['blocked']:
            print(f'   {b["month"]}: not graded [{b["obstacle"]}] - {b["reason"]}')

        if args.dry_run:
            print(f'   dry run - {len(verdict["gradable"])} month(s) would be graded')
        else:
            print(f'   graded {find_and_grade_months(store, sector, series)} new month(s)')

        n = store.count_sector_graded_months(sector)
        short = max(0, MIN_GRADED_FOR_CALIBRATION - n)
        state = 'CALIBRATED' if not short else 'stated prior'
        tail = f'; ~{short} more month(s) needed' if short else ''
        print(f'   {n}/{MIN_GRADED_FOR_CALIBRATION} independent months - band is a '
              f'{state}{tail}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
