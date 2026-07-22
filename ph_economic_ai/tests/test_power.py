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
