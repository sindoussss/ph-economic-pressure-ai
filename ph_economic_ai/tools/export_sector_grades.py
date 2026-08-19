"""Export this machine's graded months to the committed grade file.

`trust.db` is gitignored, so before this the twelve-month calibration threshold
was per installation: a fresh checkout restarted the clock at zero and the
evidence died with the laptop that produced it. Food stood at 2/12 with about ten
months to go, and every clone began again.

Run after grading, and commit the result:

    python -m ph_economic_ai.tools.grade_monthly_backlog
    python -m ph_economic_ai.tools.export_sector_grades

**What this is allowed to export.** Graded MONTHS: the sector, the month, the
median estimate behind it, the settled outcome, the absolute error, and how many
runs stood behind that one sample. Not runs, not prompts, not model output. The
row is evidence about the app's error, which is what the band needs and the only
thing the owner's 2026-08-19 ruling covers.

**It never invents a month.** Only months the local store actually graded are
written, and `shared_grades.merge_grades` keys on month so re-importing them
somewhere else cannot turn one calendar month into two. That dedup rule is the
constraint the whole monthly path was built for and it has to survive the round
trip, not just the original grading.
"""
from __future__ import annotations

from ph_economic_ai.benchmark.provenance import write_record
from ph_economic_ai.benchmark.shared_grades import (
    FIELDS, SHARED_CSV, load_shared, merge_grades, write_shared,
)
from ph_economic_ai.engine.interval import GRADED_SECTORS

#: Gas grades per run against a weekly DOE price rather than per calendar month,
#: so it carries no month key to merge on and is deliberately not exported.
MONTHLY_SECTORS = tuple(s for s in sorted(GRADED_SECTORS) if s != 'gas')


def export(store=None) -> int:
    if store is None:
        from ph_economic_ai.engine.store import AgentTrustStore
        store = AgentTrustStore()

    existing = load_shared()
    total = 0
    merged_all: list[dict] = []
    for sector in MONTHLY_SECTORS:
        local = store.get_sector_grades(sector)
        merged = merge_grades(existing, local, sector)
        origin = {r['month'] for r in local}
        print(f'  {sector:12s} local {len(local):3d}  shared {len(merged) - len(origin):3d}'
              f'  merged {len(merged):3d}')
        merged_all.extend(merged)
        total += len(merged)

    # Anything already committed for a sector this machine does not grade is
    # carried through untouched. An export from a partial installation must not
    # delete another's evidence.
    kept = [r for r in existing if r.get('sector') not in MONTHLY_SECTORS]
    if kept:
        print(f'  carried through {len(kept)} row(s) for other sectors')
    merged_all.extend(kept)

    if not merged_all:
        raise SystemExit('no graded months to export; nothing written')

    rows = write_shared(merged_all)
    write_record(
        SHARED_CSV,
        source=('This app\'s own graded months, exported from the local '
                'trust.db sector_grades table'),
        params={'sectors': list(MONTHLY_SECTORS), 'fields': list(FIELDS)},
        transformations=[
            'one row per sector and calendar month, never per run',
            'merged with any already-committed rows, keyed on month so a month '
            'graded in two places stays one sample',
            'a local row wins a conflict: it is this installation\'s own '
            'observation of that month',
        ],
        units='abs_error in the sector\'s own unit (food: percentage points)',
        notes=('Shareable because the errors are a property of the app rather '
               'than of the installation (owner ruling, 2026-08-19). Gas is '
               'excluded: it grades per run against a weekly price and has no '
               'month key to merge on.'),
    )
    print(f'\nWrote {SHARED_CSV.name} ({rows} row(s)) + provenance')
    return rows


if __name__ == '__main__':
    export()
