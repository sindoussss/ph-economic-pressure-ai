"""Does the regional derivation beat doing nothing?

Pre-registered at `docs/preregistration/2026-08-01-regional-accuracy.md`, then
re-registered on the basis-guarded panel at
`docs/preregistration/2026-08-07-regional-accuracy-basis-guarded-rerun.md`
(`RSK-034`).

`build_arms` had no test at all before the re-run, which is how it took until
the re-run to notice it computed its own week-over-week change directly from
`regional_levels()` and never consulted `regional_actual`'s basis guard --
so the "re-run" would silently have scored the exact same, unguarded
computation a second time.
"""
import datetime as dt

from ph_economic_ai.tools.regional_accuracy import REFERENCE, build_arms


def _levels_and_bases(prices: dict, bases: dict) -> tuple[dict, dict]:
    """`prices`: {region: {date: price}}. `bases`: {region: {date: 'common' or
    'midpoint'}}, sparse -- a date with no entry carries no basis, same as a
    caller that never recorded one."""
    return prices, bases


def test_a_change_across_a_basis_switch_is_refused():
    d0, d1 = dt.date(2024, 1, 2), dt.date(2024, 1, 9)
    levels, bases = _levels_and_bases(
        prices={REFERENCE: {d0: 60.0, d1: 61.0},
                'CALABARZON': {d0: 58.0, d1: 60.0},
                # Unaffected control so the week survives `if not actuals`
                # and the assertion isolates CALABARZON specifically.
                'MIMAROPA': {d0: 62.0, d1: 63.0}},
        bases={REFERENCE: {d0: 'common', d1: 'common'},
               'CALABARZON': {d0: 'common', d1: 'midpoint'},
               'MIMAROPA': {d0: 'common', d1: 'common'}},
    )
    out = build_arms(levels, bases)
    assert d1 in out
    assert 'CALABARZON' not in out[d1]['_actual'], (
        'a change measured across a common/midpoint switch was not refused')
    assert 'MIMAROPA' in out[d1]['_actual']


def test_a_change_on_a_consistent_basis_is_kept():
    d0, d1 = dt.date(2024, 1, 2), dt.date(2024, 1, 9)
    levels, bases = _levels_and_bases(
        prices={REFERENCE: {d0: 60.0, d1: 61.0},
                'CALABARZON': {d0: 58.0, d1: 60.0}},
        bases={REFERENCE: {d0: 'common', d1: 'common'},
               'CALABARZON': {d0: 'common', d1: 'common'}},
    )
    out = build_arms(levels, bases)
    assert 'CALABARZON' in out[d1]['_actual']
    assert out[d1]['_actual']['CALABARZON'] == 2.0


def test_a_switch_in_the_reference_itself_drops_the_whole_week():
    """The national change feeds every arm for that week -- DELTA-EQUAL and
    DERIVATION are both built from it -- so a switch in NCR's own series is not
    a per-region problem, it invalidates the week."""
    d0, d1 = dt.date(2024, 1, 2), dt.date(2024, 1, 9)
    levels, bases = _levels_and_bases(
        prices={REFERENCE: {d0: 60.0, d1: 61.0},
                'CALABARZON': {d0: 58.0, d1: 59.0}},
        bases={REFERENCE: {d0: 'common', d1: 'midpoint'},
               'CALABARZON': {d0: 'common', d1: 'common'}},
    )
    out = build_arms(levels, bases)
    assert d1 not in out


def test_a_region_with_no_recorded_basis_is_not_refused():
    """Absence is not contradiction. A caller supplying levels with no basis
    information (or a region the panel never assigned a basis to) is not
    judged on one, matching `regional_actual`'s own rule."""
    d0, d1 = dt.date(2024, 1, 2), dt.date(2024, 1, 9)
    levels, bases = _levels_and_bases(
        prices={REFERENCE: {d0: 60.0, d1: 61.0},
                'CALABARZON': {d0: 58.0, d1: 60.0}},
        bases={REFERENCE: {d0: 'common', d1: 'common'}},
    )
    out = build_arms(levels, bases)
    assert 'CALABARZON' in out[d1]['_actual']


def test_day_matching_stays_exact_not_nearest_within_a_week():
    """`regional_actual` snaps to the nearest available day within a week of
    the target, which would confound the basis-guard re-run with a second,
    unregistered change to which weeks count as a pair. `build_arms` must keep
    requiring the exact day."""
    d0 = dt.date(2024, 1, 2)
    d1_exact = dt.date(2024, 1, 9)
    d1_near = dt.date(2024, 1, 10)  # one day off, inside regional_actual's window
    levels, bases = _levels_and_bases(
        prices={REFERENCE: {d0: 60.0, d1_exact: 61.0},
                'CALABARZON': {d0: 58.0, d1_near: 60.0},
                'MIMAROPA': {d0: 62.0, d1_exact: 63.0}},
        bases={REFERENCE: {d0: 'common', d1_exact: 'common'},
               'CALABARZON': {d0: 'common', d1_near: 'common'},
               'MIMAROPA': {d0: 'common', d1_exact: 'common'}},
    )
    out = build_arms(levels, bases)
    assert 'CALABARZON' not in out[d1_exact]['_actual'], (
        'a region observed the day after the target was matched anyway')
    assert 'MIMAROPA' in out[d1_exact]['_actual']
