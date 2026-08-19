"""Graded months shared across installations.

`trust.db` is gitignored, so `MIN_GRADED_FOR_CALIBRATION = 12` was per machine:
no fresh checkout and no CI run could ever show a calibrated band, and twelve
months of evidence died with the laptop that produced it.

The owner's ruling on 2026-08-19 is that the errors are a property of the APP,
not of the installation, so the graded months are shareable evidence. This
exports them to a committed CSV and merges that with whatever the local store
holds.

**The merge must not break the month rule.** One calendar month is one sample no
matter how many runs produced it -- the constraint this whole monthly path was
built for. A month present both in the shared file and in the local store is
still ONE month. Counting it twice would inflate the apparent evidence by exactly
the factor the rule exists to remove, and would do it silently, on the number the
band advertises.
"""
import datetime as dt

import pytest

from ph_economic_ai.benchmark import shared_grades as sg


def _grade(month, abs_error, sector='food', n_runs=1):
    return {'sector': sector, 'month': month, 'estimate': 0.5, 'actual': 0.4,
            'abs_error': abs_error, 'n_runs': n_runs, 'graded_at': '2026-08-19'}


# ── The month rule survives the merge ────────────────────────────────────────

def test_a_month_in_both_sources_counts_once():
    """The load-bearing test. 2026-07 graded locally AND shared is one month."""
    shared = [_grade('2026-06', 0.30), _grade('2026-07', 0.38)]
    local = [_grade('2026-07', 0.38), _grade('2026-08', 0.21)]
    merged = sg.merge_grades(shared, local, sector='food')
    assert [m['month'] for m in merged] == ['2026-06', '2026-07', '2026-08']


def test_the_local_measurement_wins_a_conflict():
    """Where both sources graded a month, the local store is this installation's
    own observation of it. Shared evidence seeds a gap; it does not overwrite a
    measurement the machine made itself."""
    shared = [_grade('2026-07', 9.99)]
    local = [_grade('2026-07', 0.38)]
    merged = sg.merge_grades(shared, local, sector='food')
    assert len(merged) == 1
    assert merged[0]['abs_error'] == pytest.approx(0.38)


def test_only_the_requested_sector_is_merged():
    """Food errors are percentage points and gas errors are PHP/L. Mixing them
    is the unit hazard `interval.FALLBACK_HALFWIDTH` exists to warn about."""
    shared = [_grade('2026-07', 0.38, sector='food'),
              _grade('2026-07', 1.90, sector='gas')]
    merged = sg.merge_grades(shared, [], sector='food')
    assert len(merged) == 1
    assert merged[0]['sector'] == 'food'


def test_merged_months_are_ordered_and_unique():
    shared = [_grade('2026-08', 0.2), _grade('2026-06', 0.3)]
    local = [_grade('2026-07', 0.4), _grade('2026-06', 0.3)]
    months = [m['month'] for m in sg.merge_grades(shared, local, sector='food')]
    assert months == sorted(set(months))


# ── Errors reaching the band ─────────────────────────────────────────────────

def test_errors_come_out_newest_first_like_the_store():
    shared = [_grade('2026-06', 0.30), _grade('2026-07', 0.38)]
    errors = sg.merged_errors(shared, [], sector='food')
    assert errors == [0.38, 0.30]


def test_an_absent_shared_file_changes_nothing():
    """A checkout without the CSV must behave exactly as before, not crash and
    not silently report zero months when the local store has some."""
    local = [_grade('2026-07', 0.38)]
    assert sg.merged_errors([], local, sector='food') == [0.38]
    assert sg.load_shared(csv_path='/nonexistent/path.csv') == []


# ── Provenance travels with the count ────────────────────────────────────────

def test_the_split_between_shared_and_local_is_reported():
    """A reader told "calibrated on 12 months" deserves to know how many of those
    months this machine actually observed. Twelve inherited months and twelve
    earned ones are different claims about the same band."""
    shared = [_grade('2026-05', 0.3), _grade('2026-06', 0.2)]
    local = [_grade('2026-06', 0.2), _grade('2026-07', 0.4)]
    origin = sg.grade_origin(shared, local, sector='food')
    assert origin['total'] == 3
    assert origin['local'] == 2
    assert origin['shared_only'] == 1
