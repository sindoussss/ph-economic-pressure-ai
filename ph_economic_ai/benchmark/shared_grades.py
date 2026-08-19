"""Graded months shared across installations.

`trust.db` is gitignored, so `MIN_GRADED_FOR_CALIBRATION = 12` was per machine.
No fresh checkout and no CI run could ever display a calibrated band, and twelve
months of accumulated evidence died with the laptop that produced it. Food sat at
2/12 with roughly ten months to go, and every new clone restarted that clock at
zero.

**The decision this rests on.** Whether the graded months may be shared is not a
storage question, it is a claim about what the error distribution belongs to. A
band calibrated on one person's runs describes that installation's model errors,
and this app runs local models whose behaviour varies by machine. Shipping
someone else's history as "verified" would be the borrowed-authority failure
`RSK-023` cost three withdrawn grades for. The owner ruled on 2026-08-19 that the
errors are a property of the APP, so the months are shareable evidence, and this
module exists on that ruling rather than on a convenience argument.

**The month rule survives the merge.** One calendar month is one sample however
many runs produced it, which is the constraint the whole monthly grading path was
built for. A month present in both the shared file and the local store is still
ONE month. Counting it twice would inflate the apparent evidence by exactly the
factor the rule removes, silently, on the number the band advertises.

**A local measurement outranks a shared one.** Where both graded the same month,
the local row is this installation's own observation. Shared evidence seeds a gap
it has; it never overwrites something the machine measured itself.
"""
from __future__ import annotations

import csv
from typing import Iterable, Mapping, Optional

from ph_economic_ai.benchmark.paths import data

SHARED_CSV = data('sector_grades.csv')

#: Written in this order so a diff of the committed file is readable.
FIELDS = ('sector', 'month', 'estimate', 'actual', 'abs_error', 'n_runs',
          'graded_at')

_NUMERIC = ('estimate', 'actual', 'abs_error')


def _coerce(row: Mapping) -> dict:
    out = dict(row)
    for key in _NUMERIC:
        try:
            out[key] = float(out[key]) if out.get(key) not in (None, '') else None
        except (TypeError, ValueError):
            out[key] = None
    try:
        out['n_runs'] = int(out['n_runs']) if out.get('n_runs') not in (None, '') else None
    except (TypeError, ValueError):
        out['n_runs'] = None
    return out


def load_shared(csv_path=SHARED_CSV) -> list[dict]:
    """Committed graded months, or `[]` when the file is absent.

    Absence is normal on a checkout that has never exported, and must behave
    exactly as before rather than crashing or reporting zero months when the
    local store holds some.
    """
    try:
        with open(csv_path, encoding='utf-8') as fh:
            return [_coerce(r) for r in csv.DictReader(fh)]
    except (FileNotFoundError, OSError):
        return []


def merge_grades(shared: Iterable[Mapping], local: Iterable[Mapping],
                 sector: str) -> list[dict]:
    """Graded months for `sector`, oldest first, one row per calendar month.

    Keyed on month precisely so the dedup rule cannot be lost in the union, and
    filtered to one sector because food's errors are percentage points while
    gas's are PHP/L -- mixing them is the unit hazard
    `interval.FALLBACK_HALFWIDTH` warns about.
    """
    by_month: dict[str, dict] = {}
    for row in shared or ():
        if row.get('sector') == sector and row.get('month'):
            by_month[row['month']] = dict(row)
    for row in local or ():
        if row.get('sector') == sector and row.get('month'):
            by_month[row['month']] = dict(row)      # local wins a conflict
    return [by_month[m] for m in sorted(by_month)]


def merged_errors(shared: Iterable[Mapping], local: Iterable[Mapping],
                  sector: str) -> list[float]:
    """Absolute errors newest first, matching `store.get_sector_graded_errors`.

    Same order as the store's own accessor so the two are interchangeable at the
    call site and a reader comparing them is not tripped by orientation.
    """
    merged = merge_grades(shared, local, sector)
    return [m['abs_error'] for m in reversed(merged)
            if m.get('abs_error') is not None]


def grade_origin(shared: Iterable[Mapping], local: Iterable[Mapping],
                 sector: str) -> dict:
    """How many of the merged months this machine actually observed.

    Twelve inherited months and twelve earned ones are different claims about the
    same band, and a reader told "calibrated on 12 months" deserves to know which
    they are looking at. The screen can then say so instead of implying the
    machine did work it did not do.
    """
    merged = merge_grades(shared, local, sector)
    local_months = {r['month'] for r in (local or ())
                    if r.get('sector') == sector and r.get('month')}
    return {'total': len(merged),
            'local': len([m for m in merged if m['month'] in local_months]),
            'shared_only': len([m for m in merged
                                if m['month'] not in local_months])}


def write_shared(grades: Iterable[Mapping], csv_path=SHARED_CSV) -> int:
    """Write the committed grade file. Returns the row count.

    Sorted by sector then month so the diff of a re-export shows what changed
    rather than a reshuffle.
    """
    rows = [r for r in grades if r.get('sector') and r.get('month')]
    rows.sort(key=lambda r: (r['sector'], r['month']))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS), lineterminator='\n',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
