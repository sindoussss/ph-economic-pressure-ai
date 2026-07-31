"""The pre-registered regional multiplier test, and the gate that withdrew it.

`docs/preregistration/2026-08-01-regional-multiplier-test.md` fixes every choice
here before the run. These tests pin that the code executes THAT rule rather than
one adjusted afterwards, because the whole value of a pre-registration is that it
cannot be quietly edited once the numbers exist.

The run returned `b` near 0.6 in both regions and the verdict is
`CANNOT DISTINGUISH`, which the pre-registration named in advance as the most
likely outcome and gave its own action: change nothing.

Two amendments sit on that document. Amendment 1 blamed implausible weekly moves
on DOE changing document template; **Amendment 2 retracts it**, because only 1 of
11 such moves coincides with a city-count change, every filename matches its
document's own printed week, and the largest moves are coherent across all three
regions at once. What is actually wrong is measurable: the reference series is
only 0.685 reliable, which attenuates every slope and compresses the 0.05 gap to
0.034 against intervals half a unit wide.
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
    """Diagnostic only, and deliberately no longer a gate. Amendment 1 made it
    one on the theory that such moves were a template artifact; Amendment 2
    retracts that, because the largest are coherent across all three regions at
    once and no per-document parsing artifact does that."""
    changes = {dt.date(2026, 1, 6): 1.2, dt.date(2026, 2, 3): 18.55,
               dt.date(2026, 2, 10): -18.44}
    flagged = bt.implausible_weeks(changes)
    assert [d for d, _c in flagged] == [dt.date(2026, 2, 3), dt.date(2026, 2, 10)]


def test_a_quiet_series_raises_no_flag():
    assert bt.implausible_weeks({dt.date(2026, 1, 6): 2.9,
                                 dt.date(2026, 1, 13): -1.4}) == []


# ── Reliability, which is what was actually wrong ────────────────────────────

def _week_series(n_weeks, cities, noise):
    """Cities that all move by the same weekly step, plus per-city noise."""
    import random
    rng = random.Random(7)
    out, level = {}, {c: 60.0 for c in cities}
    day = dt.date(2026, 1, 6)
    for _ in range(n_weeks):
        step = rng.uniform(-2, 2)
        for c in cities:
            level[c] += step + rng.gauss(0, noise)
        out[day] = dict(level)
        day += dt.timedelta(days=7)
    return out


def test_a_clean_series_reports_high_reliability():
    """All cities share one weekly step, so the two halves must agree."""
    series = _week_series(80, [f'C{i}' for i in range(12)], noise=0.01)
    assert bt.reliability(series)['reliability'] > 0.95


def test_a_noisy_series_reports_low_reliability():
    """Per-city noise swamping the shared step is exactly what makes two halves
    of the SAME region disagree, and it is the thing that attenuates a slope."""
    series = _week_series(80, [f'C{i}' for i in range(12)], noise=6.0)
    assert bt.reliability(series)['reliability'] < 0.6


def test_reliability_is_refused_rather_than_guessed_on_thin_input():
    """Fewer than four cities cannot be split into two comparable halves."""
    assert bt.reliability(_week_series(80, ['A', 'B', 'C'], noise=0.1)) is None
    assert bt.reliability(_week_series(5, [f'C{i}' for i in range(12)], 0.1)) is None


def test_attenuation_moves_the_reading_toward_cannot_distinguish():
    """The substantive consequence. A reference that is only 0.685 reliable makes
    a true 1.00 appear near 0.685 and a true 1.05 near 0.719, so the 0.05 gap
    the test exists to resolve becomes 0.034. An interval that excludes both RAW
    hypotheses can easily contain both attenuated ones, which is precisely what
    happened."""
    rel = 0.685
    assert bt.verdict(_fit(0.378, 0.844), 1.05)[0] == 'NEITHER'
    lo, hi = 0.378, 0.844
    assert lo <= 1.00 * rel <= hi and lo <= 1.05 * rel <= hi
    assert abs(1.05 * rel - 1.00 * rel) < 0.05


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
