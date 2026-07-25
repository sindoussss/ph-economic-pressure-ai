"""Minimum-detectable-effect for the efficiency nulls (M2 reviewer defense)."""
import numpy as np
import pytest

from ph_economic_ai.benchmark import power


def test_random_walk_model_has_near_zero_observed_skill():
    """A model that just repeats the last value IS the random walk — skill ~0."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0, 1, 60)) + 100
    y_pred = np.roll(y, 1)          # predict previous actual = random walk
    r = power.min_detectable_skill(y, y_pred)
    assert abs(r['observed_skill']) < 0.05


def test_mde_is_positive_and_reported_as_pct():
    rng = np.random.default_rng(1)
    y = np.cumsum(rng.normal(0, 1, 60)) + 100
    y_pred = np.roll(y, 1) + rng.normal(0, 0.5, 60)   # a real (noisy) forecast
    r = power.min_detectable_skill(y, y_pred)
    assert r['min_detectable_skill'] > 0
    assert r['min_detectable_skill_pct'] == pytest.approx(r['min_detectable_skill'] * 100, abs=0.1)


def test_more_data_lowers_the_detectable_effect():
    """Power rises with n, so the minimum detectable skill falls."""
    rng = np.random.default_rng(2)
    def mde(n):
        y = np.cumsum(rng.normal(0, 1, n)) + 100
        yp = np.roll(y, 1) + rng.normal(0, 0.3, n)   # slightly noisy RW-ish
        return power.min_detectable_skill(y, yp)['min_detectable_skill']
    assert mde(200) < mde(40)


def test_interpretation_states_the_honest_bound():
    r = power.run()['fuel_one_month_forecast']
    assert 'no detectable edge at this power' in r['interpretation']
    assert r['min_detectable_skill_pct'] > 0
    # observed skill on the flagship fuel null is ~0 (efficient)
    assert abs(r['observed_skill']) < 0.05


# ── Per-target power for the nowcast nulls ────────────────────────────────────

def test_mde_from_errors_is_baseline_agnostic():
    """The same model errors give a LARGER detectable effect against a stronger
    baseline — which is the whole reason the nowcast nulls must be bounded against
    the mean rather than the random walk."""
    rng = np.random.default_rng(7)
    e_model = rng.normal(0, 1.0, 200)
    e_weak = rng.normal(0, 2.0, 200)       # a poor baseline
    e_strong = rng.normal(0, 1.05, 200)    # a baseline close to the model
    weak = power.mde_from_errors(e_model, e_weak, 'weak')
    strong = power.mde_from_errors(e_model, e_strong, 'strong')
    assert weak['observed_skill'] > strong['observed_skill']
    assert weak['baseline'] == 'weak' and strong['baseline'] == 'strong'


def test_identical_errors_give_zero_observed_skill():
    e = np.random.default_rng(8).normal(0, 1, 100)
    r = power.mde_from_errors(e, e.copy(), 'self')
    assert r['observed_skill'] == pytest.approx(0.0, abs=1e-12)
    assert r['min_detectable_skill'] >= 0


def test_every_nowcast_null_is_bounded():
    """No null may be reported without a minimum-detectable-effect: an unbounded
    null is an unfalsifiable claim."""
    rows = power.run()['nowcast_nulls_vs_mean']
    expected = {'headline_mom', 'headline_mom_long', 'food_mom', 'electricity_mom',
                'electricity_driver_only', 'transport_driver_only'}
    assert expected <= set(rows)
    for label, r in rows.items():
        assert r['baseline'] == 'mean', f'{label} must be bounded vs the binding baseline'
        assert r['min_detectable_skill'] > 0, f'{label} has no usable power bound'
        assert r['n'] > 24
        assert r['best_candidate'] not in ('mean', 'random_walk', 'drift', 'seasonal_naive')


def test_observed_skill_stays_below_the_detectable_bound():
    """Consistency: every target is a null, so the observed skill must sit inside
    the band the test could not have resolved. If one ever exceeds its MDE, that
    target is a real positive and the map is wrong."""
    for label, r in power.run()['nowcast_nulls_vs_mean'].items():
        assert r['observed_skill'] < r['min_detectable_skill'], (
            f'{label}: observed {r["observed_skill"]:.3f} exceeds its MDE '
            f'{r["min_detectable_skill"]:.3f} — it is not a null')
