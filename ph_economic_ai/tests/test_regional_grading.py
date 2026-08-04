"""Grading a regional forecast against DOE's published regional prices.

`ground_truth` scores the national figure and nothing else, which is why the
seventeen regional numbers have never been compared to anything and why
`Unsafe Claims` forbids saying they are validated. Phase 1 built the actual, so
the comparison is possible for the first time.

What it cannot do is settle `Q-ENG-009`. A region can be graded well while the
model producing it is still the wrong shape.
"""
import datetime as dt

import pytest

from ph_economic_ai.engine import regional_grading as rg


def _levels(region, start, n, step=1.0):
    day = start
    out = {}
    for i in range(n):
        out[day] = 60.0 + i * step
        day += dt.timedelta(days=7)
    return {region: out}


# ── The actual is a change between CONSECUTIVE weeks ─────────────────────────

def test_the_actual_is_the_change_over_the_target_week():
    levels = _levels('Davao Region', dt.date(2026, 1, 6), 4, step=0.5)
    got = rg.regional_actual('Davao Region', dt.date(2026, 1, 20), levels)
    assert got == pytest.approx(0.5)


def test_a_gap_before_the_target_yields_no_actual():
    """`ADR-003`'s defect. Differencing across a gap labels a three-week move as a
    one-week move, so a missing prior week means no grade rather than a wrong
    one."""
    levels = {'Davao Region': {dt.date(2026, 1, 6): 60.0,
                               dt.date(2026, 1, 27): 63.0}}
    assert rg.regional_actual('Davao Region', dt.date(2026, 1, 27), levels) is None


def test_a_region_with_no_series_yields_no_actual():
    """Central Luzon, Ilocos, Cagayan Valley and CAR, at any date."""
    assert rg.regional_actual('Central Luzon', dt.date(2026, 1, 20), {}) is None


def test_a_target_far_from_any_published_week_is_refused():
    """Graded against a price observed near ITS OWN target date, the rule
    `ground_truth` already applies nationally."""
    levels = _levels('Davao Region', dt.date(2026, 1, 6), 3)
    assert rg.regional_actual('Davao Region', dt.date(2026, 6, 1), levels) is None


def test_an_implausible_outcome_is_refused_as_truth():
    """An outcome the app would refuse to accept as an ESTIMATE must not be
    accepted as the TRUTH it is graded against. `RSK-018` is what happens when a
    grader trusts its own inputs: every stored run was scored against a hardcoded
    fallback and seven agents were benched on an outcome that never happened."""
    levels = {'Davao Region': {dt.date(2026, 1, 13): 60.0,
                               dt.date(2026, 1, 20): 95.0}}
    assert rg.regional_actual('Davao Region', dt.date(2026, 1, 20), levels) is None


# ── Ungradable regions are named, never dropped ──────────────────────────────

def test_ungradable_regions_are_returned_by_name():
    """A mean over whatever happened to be gradable describes two thirds of the
    map as if it were the map."""
    levels = _levels('Davao Region', dt.date(2026, 1, 6), 3)
    out = rg.grade_regional(
        {'Davao Region': 0.9, 'Central Luzon': 1.0, 'CAR': 1.0}, dt.date(2026, 1, 20),
        levels)
    assert set(out['graded']) == {'Davao Region'}
    assert set(out['ungraded']) == {'Central Luzon', 'CAR'}
    assert out['n_graded'] == 1 and out['n_ungraded'] == 2
    for reason in out['ungraded'].values():
        assert 'no DOE series' in reason


def test_a_region_graded_on_an_assumed_premium_is_flagged():
    """The score is real; what it says about the model is weaker, because part of
    the number being graded came from a constant nobody measured."""
    levels = _levels('BARMM', dt.date(2026, 1, 6), 3)
    out = rg.grade_regional({'BARMM': 1.0}, dt.date(2026, 1, 20), levels)
    assert out['graded']['BARMM']['premium_assumed'] is True

    levels = _levels('Davao Region', dt.date(2026, 1, 6), 3)
    out = rg.grade_regional({'Davao Region': 1.0}, dt.date(2026, 1, 20), levels)
    assert out['graded']['Davao Region']['premium_assumed'] is False


def test_no_gradable_region_reports_no_mean_rather_than_zero():
    out = rg.grade_regional({'Central Luzon': 1.0}, dt.date(2026, 1, 20), {})
    assert out['mean_score'] is None and out['mean_abs_error'] is None


# ── The scale is the national one, not a parallel invention ──────────────────

def test_the_score_uses_the_national_accuracy_function():
    """A regional grade on a different scale could not be read beside the
    national one."""
    from ph_economic_ai.engine.ground_truth import compute_accuracy_score
    levels = _levels('Davao Region', dt.date(2026, 1, 6), 3, step=1.0)
    out = rg.grade_regional({'Davao Region': 0.0}, dt.date(2026, 1, 20), levels)
    got = out['graded']['Davao Region']
    assert got['score'] == pytest.approx(compute_accuracy_score(0.0, got['actual']))


def test_the_plausibility_bound_is_imported_not_restated():
    """Two graders with two bounds drift, and the one that drifts is the one
    nobody is reading."""
    from ph_economic_ai.engine.debate import _MAX_REALISTIC_FUEL_PHP_L
    assert rg._MAX_PLAUSIBLE_CHANGE == _MAX_REALISTIC_FUEL_PHP_L


# ── Only city-grain rows ─────────────────────────────────────────────────────

def test_the_panel_reader_ignores_metro_aggregates(tmp_path):
    """A metro-wide row is not a city, and averaging a whole market with its own
    components moves the level without any price moving (`DEC-062`)."""
    csv_path = tmp_path / 'panel.csv'
    csv_path.write_text(
        'cycle,file_date,area,grain,province,province_raw,city,low,high,common,source_file\n'
        '2026-01-06,2026-01-06,NCR,metro,,,Metro Manila,40,50,,a\n'
        '2026-01-06,2026-01-06,NCR,city,,,Manila,60,60,60,b\n'
        '2026-01-06,2026-01-06,NCR,city,,,Pasig,61,61,61,b\n'
        '2026-01-06,2026-01-06,NCR,city,,,Taguig,62,62,62,b\n',
        encoding='utf-8')
    levels = rg.regional_levels(csv_path)
    assert levels['NCR'][dt.date(2026, 1, 6)] == pytest.approx(61.0)


def test_a_week_below_the_city_minimum_is_dropped(tmp_path):
    csv_path = tmp_path / 'panel.csv'
    csv_path.write_text(
        'cycle,file_date,area,grain,province,province_raw,city,low,high,common,source_file\n'
        '2026-01-06,2026-01-06,NCR,city,,,Manila,60,60,60,b\n'
        '2026-01-06,2026-01-06,NCR,city,,,Pasig,61,61,61,b\n',
        encoding='utf-8')
    assert rg.regional_levels(csv_path).get('NCR', {}) == {}


# ── The pre-registered accuracy result, and what the note must say ───────────

def test_the_basis_note_states_the_accuracy_result():
    """The action the pre-registration fixed BEFORE the run: label, do not
    delete. Graded over 294 weeks the derivation's MAE is 1.311 against 1.217
    for assuming no regional change, and the corrected interval on the
    difference contains zero -- so the note must say no more accurate, not
    worse, and must not claim a win either."""
    from ph_economic_ai.ui.honesty import regional_basis
    note = regional_basis()
    assert 'no more accurate than assuming no regional change' in note
    assert 'measurably worse than not applying it' in note
    assert 'derived rather than forecast' in note
    for overclaim in ('validated', 'accurate forecast', 'outperforms'):
        assert overclaim not in note


def test_the_regional_figures_are_still_shown():
    """`DEC-021`'s spirit and the pre-registered rule: the number stays, the
    claim around it stops overreaching. Only the branch where the derivation is
    measurably WORSE than the best naive removes a figure, and the run did not
    land there."""
    from ph_economic_ai.engine.swarm import ALL_REGIONS, derive_regional_estimates
    derived = derive_regional_estimates(2.42, {0: 2.42})
    assert len(derived) == len(ALL_REGIONS)
    for group in ALL_REGIONS:
        assert group['name'] in derived


def test_the_level_construction_matches_the_premiums_module(tmp_path):
    """Two constructions of the same quantity that must agree, and did not.

    `regional_grading` neither deduplicated city-weeks nor recovered the South
    Luzon regions from their filenames, so it disagreed with
    `regional_level_premiums` on 22 of 1798 week-values and was missing about 183
    weeks each for CALABARZON, MIMAROPA and Bicol. That understated the accuracy
    test by 77 paired weeks.

    Pinned on a fixture rather than the committed panel, so the guard holds
    without a 4 MB read."""
    import csv as _csv
    from ph_economic_ai.tools.regional_level_premiums import (
        regional_levels_by_region)

    header = ('cycle,file_date,area,grain,province,province_raw,city,'
              'low,high,common,source_file\n')
    rows = header + ''.join(
        f'2026-01-06,2026-01-06,Visayas,city,Iloilo,Iloilo,C{i},60,60,{60 + i},b\n'
        for i in range(3))
    # The same city-week in two documents: dedup must keep the later filename.
    rows += '2026-01-06,2026-01-06,Visayas,city,Iloilo,Iloilo,C0,60,60,99,a\n'
    path = tmp_path / 'panel.csv'
    path.write_text(rows, encoding='utf-8')

    mine = rg.regional_levels(path)
    with open(path, encoding='utf-8') as fh:
        theirs = regional_levels_by_region(list(_csv.DictReader(fh)))
    assert mine == theirs, 'the two level constructions have diverged again'
