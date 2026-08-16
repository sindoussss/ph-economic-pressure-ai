"""The weekly pump-price backtest — the one horizon where this app has an edge.

The monthly pipeline scores -7.35% against "assume no change"
(`accuracy_report.json`): worse than doing nothing. That number has been read as
"these series are unforecastable", but it is a statement about ONE HORIZON. PH
pump prices reset weekly against a Singapore refined-product average computed
over the PRIOR week, so by the weekend most of the input determining Tuesday's
move is already published. Monthly aggregation destroys exactly that structure.

Measured here: +14.7% skill against no-change, HAC-DM t = -2.09, positive in
7 of 7 years (sign test p = 0.0078). That is the only positive, leakage-tested
forecasting result in this repository.

Every test below guards a way the result could be an illusion rather than an
edge, because a positive finding is where this project has historically gone
wrong (`corrected_predictability_map.json`: every nowcast verdict flipped to
null once the historical mean entered the baseline pool).
"""
import datetime as dt

import numpy as np
import pytest

from ph_economic_ai.benchmark import weekly_gas


# ── The target: a price change, not a panel change ───────────────────────────

def test_matched_panel_measures_prices_not_which_cities_reported():
    """A median over a CHANGING panel moves when the panel changes.

    Two weeks where every reporting city holds its price, but the second week
    adds an expensive city. A level-median jumps; the matched panel must read
    zero, because no price moved. This is the difference between measuring the
    market and measuring who filed a report.
    """
    rows = [
        {'cycle': '2026-01-06', 'city': 'A', 'common': '50.00'},
        {'cycle': '2026-01-06', 'city': 'B', 'common': '52.00'},
        {'cycle': '2026-01-06', 'city': 'C', 'common': '54.00'},
        {'cycle': '2026-01-13', 'city': 'A', 'common': '50.00'},
        {'cycle': '2026-01-13', 'city': 'B', 'common': '52.00'},
        {'cycle': '2026-01-13', 'city': 'C', 'common': '54.00'},
        {'cycle': '2026-01-13', 'city': 'D', 'common': '90.00'},   # newcomer
    ]
    changes = weekly_gas.matched_panel_changes(rows, min_cities=3)
    assert changes[dt.date(2026, 1, 13)] == pytest.approx(0.0)


def test_a_real_move_is_still_measured():
    rows = []
    for city, base in (('A', 50.0), ('B', 52.0), ('C', 54.0)):
        rows.append({'cycle': '2026-01-06', 'city': city, 'common': f'{base:.2f}'})
        rows.append({'cycle': '2026-01-13', 'city': city, 'common': f'{base + 1.5:.2f}'})
    changes = weekly_gas.matched_panel_changes(rows, min_cities=3)
    assert changes[dt.date(2026, 1, 13)] == pytest.approx(1.5)


def test_a_gap_week_is_not_differenced():
    """Differencing across a three-week gap labels a three-week move as one
    week's. `regional_actual` already refuses this nationally; the same rule
    applies here or the target is not what its name says.
    """
    rows = []
    for city, base in (('A', 50.0), ('B', 52.0), ('C', 54.0)):
        rows.append({'cycle': '2026-01-06', 'city': city, 'common': f'{base:.2f}'})
        rows.append({'cycle': '2026-01-27', 'city': city, 'common': f'{base + 3:.2f}'})
    changes = weekly_gas.matched_panel_changes(rows, min_cities=3)
    assert dt.date(2026, 1, 27) not in changes


def test_a_thin_week_is_dropped():
    rows = [
        {'cycle': '2026-01-06', 'city': 'A', 'common': '50.00'},
        {'cycle': '2026-01-13', 'city': 'A', 'common': '58.00'},
    ]
    assert weekly_gas.matched_panel_changes(rows, min_cities=3) == {}


# ── Leakage: the property that would silently invalidate everything ──────────

def test_features_never_see_the_week_they_predict():
    """The whole claim is that the input is published BEFORE the price moves.

    If the feature window reached the cycle date, the model would be reading the
    answer and the +14.7% would be an artifact. The window must close strictly
    before the adjustment, so a spike ON the cycle date cannot reach it.
    """
    cycle = dt.date(2026, 3, 17)
    quiet = {cycle - dt.timedelta(days=d): 100.0 for d in range(0, 21)}
    spiked = dict(quiet)
    for d in range(0, 2):                       # cycle day and the day before
        spiked[cycle - dt.timedelta(days=d)] = 999.0

    assert (weekly_gas.prior_window_change(quiet, cycle)
            == pytest.approx(weekly_gas.prior_window_change(spiked, cycle))), (
        'a price move on the cycle date reached the feature window — the '
        'backtest would be reading the answer it claims to predict')


def test_the_feature_window_still_sees_the_prior_week():
    """The mirror of the leakage test: cutting too much would measure nothing."""
    cycle = dt.date(2026, 3, 17)
    series = {cycle - dt.timedelta(days=d): 100.0 for d in range(0, 21)}
    for d in range(2, 9):                       # the week that DOES determine it
        series[cycle - dt.timedelta(days=d)] = 110.0
    assert weekly_gas.prior_window_change(series, cycle) != pytest.approx(0.0)


def test_walk_forward_never_fits_on_the_future():
    """Each prediction is made from strictly earlier rows. Fitting once on the
    whole sample and scoring it is the classic way to manufacture skill.
    """
    seen = []

    def spy(train_x, train_y, row_x):
        seen.append(len(train_y))
        return 0.0

    n = 80
    frame = {'target': list(np.arange(n, dtype=float)),
             'a': list(np.arange(n, dtype=float))}
    truth, _ = weekly_gas.walk_forward(frame, ['a'], min_train=52, fit_predict=spy)
    assert seen == list(range(52, n))
    assert len(truth) == n - 52


# ── The result itself, and the honesty that must travel with it ─────────────

def test_shuffled_commodities_have_no_skill():
    """The leakage detector, as a test rather than a one-off check.

    Permuting the commodity columns destroys their alignment with the target but
    keeps every other property of the pipeline. Skill must collapse. If a
    shuffled feature scores like a real one, the harness is leaking and no
    downstream number means anything.
    """
    report = weekly_gas.run_weekly_gas()
    null = report['shuffle_null']
    assert null['max_skill'] < 0.05, (
        f'shuffled features reached {null["max_skill"]:.1%} skill — the harness '
        f'is manufacturing an edge')
    assert report['skill'] > null['max_skill'], (
        'the real effect does not exceed the shuffled ceiling')


def test_weekly_beats_no_change_in_every_year():
    """The sign test is the robust core of the claim: it needs no HAC choice and
    no distributional assumption. 7 of 7 is p = 0.0078.
    """
    report = weekly_gas.run_weekly_gas()
    years = report['by_year']
    assert len(years) >= 6
    negative = {y: v['skill'] for y, v in years.items() if v['skill'] <= 0}
    assert not negative, f'years where the edge reverses: {negative}'


def test_the_headline_carries_its_own_holdout_verdict():
    """+14.7% is significant on the full sample and NOT confirmed on the strict
    two-stage holdout (t = -1.54). Reporting the first without the second is the
    overclaim this repo keeps retracting -- so the report cannot omit it.
    """
    report = weekly_gas.run_weekly_gas()
    assert 'holdout' in report
    assert report['holdout']['confirmed'] is False
    assert 'verdict' in report and 'not confirmed' in report['verdict'].lower()


def test_the_monthly_contrast_is_recorded():
    """The finding is comparative: the same app is -7.35% monthly and positive
    weekly. Stating the weekly number alone invites reading it as a general
    forecasting claim, which it is not.
    """
    report = weekly_gas.run_weekly_gas()
    assert report['monthly_skill_for_contrast'] < 0
    assert report['skill'] > 0


# ── The band the app actually shows ──────────────────────────────────────────

def test_the_stated_gas_band_is_consistent_with_real_weekly_outcomes():
    """`FALLBACK_HALFWIDTH['gas']` is displayed to users and had never been
    checked against an outcome.

    Its own comment sources it from "the ranges the prompts already describe as
    typical" -- a guess, and until this panel existed there was nothing to test
    it against. Measured here it turns out to be close to right:

        level  stated  covers
        0.50    0.60    46%
        0.80    1.40    82%
        0.90    2.00    90%
        0.95    2.60    93%

    So this test does not change the numbers -- it stops them drifting unnoticed
    the way the generation charge and `anchor_validation.json` both did. A prior
    nobody can check is a prior nobody can trust, however good it happens to be.

    **What this does NOT establish.** The errors here come from the walk-forward
    commodity model, not from the app's own swarm-plus-anchor path, which has
    zero graded gas forecasts (`RSK-023` withdrew all three). This is a sanity
    floor on the WIDTH, not a validation of the app's live band. Only grading
    real runs can do that, which is what `MIN_GRADED_FOR_CALIBRATION` is for.
    """
    import numpy as np

    from ph_economic_ai.engine.interval import FALLBACK_HALFWIDTH

    frame = weekly_gas.load_features()
    truth, pred = weekly_gas.walk_forward(frame, list(weekly_gas.COMMODITY_COLS))
    errors = np.abs(truth - pred)

    stated = FALLBACK_HALFWIDTH['gas']
    drifted = {}
    for level, half in stated.items():
        covered = float((errors <= half).mean())
        if abs(covered - level) > 0.10:
            drifted[level] = (half, round(covered, 3))

    assert not drifted, (
        'the stated weekly gas band no longer matches real outcomes '
        f'(level -> (stated_halfwidth, measured_coverage)): {drifted}')


def test_the_band_prior_is_not_silently_wider_than_the_outcome_series():
    """A band can also fail by being too generous: a 50% range that holds 90% of
    the time is not honest either, it just fails in the flattering direction.
    """
    import numpy as np

    from ph_economic_ai.engine.interval import FALLBACK_HALFWIDTH

    frame = weekly_gas.load_features()
    truth, pred = weekly_gas.walk_forward(frame, list(weekly_gas.COMMODITY_COLS))
    errors = np.abs(truth - pred)
    half_50 = FALLBACK_HALFWIDTH['gas'][0.5]
    assert float((errors <= half_50).mean()) < 0.75, (
        'the 50% band is behaving like a 75%+ band — too wide to mean what it says')
