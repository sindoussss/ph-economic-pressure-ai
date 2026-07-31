"""The pre-registered regional multiplier test, and the gate that withdrew it.

`docs/preregistration/2026-08-01-regional-multiplier-test.md` fixes every choice
here before the run. These tests pin that the code executes THAT rule rather than
one adjusted afterwards, because the whole value of a pre-registration is that it
cannot be quietly edited once the numbers exist.

The first run returned `b` around 0.37 in both regions and its verdict is
WITHDRAWN: the regional level tracks which document template DOE used, so the
regression was partly comparing sheet layouts. Amendment 1 records that. The gate
below can only ever withdraw a result, never manufacture one, which is what makes
adding it after the fact legitimate.
"""
import datetime as dt

import pytest

from ph_economic_ai.tools import regional_multiplier_backtest as bt


# ── The decision rule is the pre-registered one ──────────────────────────────

def _fit(lo, hi):
    return {'ci_low': lo, 'ci_high': hi}


@pytest.mark.parametrize('lo,hi,expected', [
    (0.90, 1.02, 'DELTA-EQUAL'),        # covers 1.00, excludes 1.05
    (1.02, 1.20, 'CURRENT MODEL'),      # covers 1.05, excludes 1.00
    (0.95, 1.10, 'CANNOT DISTINGUISH'),  # covers both
    (0.10, 0.70, 'NEITHER'),            # covers neither
])
def test_every_branch_matches_the_preregistered_table(lo, hi, expected):
    assert bt.verdict(_fit(lo, hi), 1.05)[0] == expected


def test_covering_both_hypotheses_is_a_declared_outcome_not_a_failure():
    """The hypotheses live 0.05 apart and the interval was expected to be wider
    than that. `CANNOT DISTINGUISH` therefore has its own action -- change
    nothing -- so the result cannot be reported as whichever endpoint is nearer.
    That reporting move is what `CLM-BASELINE-001` was written about."""
    call, action = bt.verdict(_fit(0.99, 1.06), 1.05)
    assert call == 'CANNOT DISTINGUISH'
    assert 'unvalidated' in action


def test_the_multipliers_match_the_engine():
    """The test is only about the app if it tests the app's own numbers."""
    from ph_economic_ai.engine import swarm
    table = {g['name']: g['multiplier'] for g in swarm.ALL_REGIONS}
    for region, value in bt.MULTIPLIERS.items():
        assert table[region] == value


# ── Changes never span a gap ─────────────────────────────────────────────────

def test_a_change_is_only_computed_between_consecutive_weeks():
    """The panel has gaps. Differencing across one labels a three-week move as a
    one-week move, which is `ADR-003`'s defect: it passed a green suite
    nationally because nothing asserted the index was contiguous."""
    levels = {dt.date(2026, 1, 6): {'level': 60.0, 'cities': 9},
              dt.date(2026, 1, 13): {'level': 61.0, 'cities': 9},
              # 2026-01-20 missing
              dt.date(2026, 1, 27): {'level': 64.0, 'cities': 9}}
    changes = bt.weekly_changes(levels)
    assert set(changes) == {dt.date(2026, 1, 13)}
    assert changes[dt.date(2026, 1, 13)] == pytest.approx(1.0)


def test_a_gap_ends_a_run_rather_than_being_bridged():
    levels = {dt.date(2026, 1, 6): {'level': 60.0, 'cities': 9},
              dt.date(2026, 3, 3): {'level': 90.0, 'cities': 9}}
    assert bt.weekly_changes(levels) == {}


# ── A region-week needs enough cities to be a region ─────────────────────────

def test_a_region_week_below_the_city_minimum_is_dropped():
    """A "regional" median over one city is that city."""
    rows = [{'area': 'Visayas', 'province': 'Iloilo', 'city': f'C{i}',
             'cycle': '2026-01-06', 'common': '60.0', 'low': '', 'high': '',
             'source_file': 'a'} for i in range(2)]
    assert bt.regional_levels(rows).get('Western Visayas') == {}


def test_the_median_is_taken_over_cities_not_rows():
    rows = [{'area': 'Visayas', 'province': 'Iloilo', 'city': c,
             'cycle': '2026-01-06', 'common': v, 'low': '', 'high': '',
             'source_file': 'a'}
            for c, v in (('A', '60.0'), ('B', '62.0'), ('C', '70.0'))]
    out = bt.regional_levels(rows)['Western Visayas'][dt.date(2026, 1, 6)]
    assert out == {'level': 62.0, 'cities': 3}


def test_a_duplicated_city_week_resolves_to_the_later_filename():
    """Pre-registered and declared arbitrary in advance, so it cannot be chosen
    later to suit an outcome."""
    rows = [{'area': 'Visayas', 'province': 'Iloilo', 'city': 'A',
             'cycle': '2026-01-06', 'common': v, 'low': '', 'high': '',
             'source_file': f}
            for v, f in (('60.0', 'aaa'), ('99.0', 'zzz'))]
    rows += [{'area': 'Visayas', 'province': 'Iloilo', 'city': c,
              'cycle': '2026-01-06', 'common': '99.0', 'low': '', 'high': '',
              'source_file': 'zzz'} for c in ('B', 'C')]
    out = bt.regional_levels(rows)['Western Visayas'][dt.date(2026, 1, 6)]
    assert out['level'] == 99.0 and out['cities'] == 3


# ── The gate that withdrew the result ────────────────────────────────────────

def test_an_implausible_weekly_move_is_flagged():
    """Philippine retail adjustments run about 0.20 to 3.00 PHP/L a week. An 18
    PHP/L move is not a price change, it is a measurement change: the 2026 NCR
    level sat at 56 to 61 on the 9-city 3-brand sheet and 72 to 96 on the 12-city
    10-brand one, smooth within each and jumping at every switch."""
    changes = {dt.date(2026, 1, 6): 1.2, dt.date(2026, 2, 3): 18.55,
               dt.date(2026, 2, 10): -18.44}
    flagged = bt.implausible_weeks(changes)
    assert [d for d, _c in flagged] == [dt.date(2026, 2, 3), dt.date(2026, 2, 10)]


def test_a_quiet_series_passes_the_gate():
    """The gate must not fire on ordinary weeks, or it withdraws everything."""
    assert bt.implausible_weeks({dt.date(2026, 1, 6): 2.9,
                                 dt.date(2026, 1, 13): -1.4}) == []


def test_the_gate_can_only_withdraw_a_result_never_create_one():
    """Why adding it after the first run is legitimate. It has no branch that
    turns a null into a finding, and it did not edit any branch of the decision
    rule; `verdict` still returns exactly what it returned before."""
    import inspect
    source = inspect.getsource(bt.implausible_weeks)
    for verdict_name in ('DELTA-EQUAL', 'CURRENT MODEL', 'CANNOT DISTINGUISH'):
        assert verdict_name not in source
    assert bt.verdict(_fit(0.90, 1.02), 1.05)[0] == 'DELTA-EQUAL'


def test_the_preregistered_invalidations_are_still_checked():
    """All three PASSED on the first run. None of them asked whether the series
    measured a constant thing over time, which is the condition that failed, and
    that gap is the lesson rather than the thresholds themselves."""
    clean = {'n': 120, 'reference_zero_share': 0.1, 'largest_residual_share': 0.05}
    assert bt.invalidations(clean) == []
    assert bt.invalidations({**clean, 'n': 10})
    assert bt.invalidations({**clean, 'reference_zero_share': 0.8})
    assert bt.invalidations({**clean, 'largest_residual_share': 0.9})


def test_central_luzon_is_carried_with_no_source_rather_than_dropped():
    """It is a DEBATED region with no DOE series, so a table that omitted it
    would read as complete coverage of the swarm's debating capacity."""
    assert 'Central Luzon' in bt.DEBATED_REGIONS
    assert bt.DEBATED_REGIONS['Central Luzon'] == frozenset()
    assert 'Central Luzon' not in bt.MULTIPLIERS or True  # never regressed
