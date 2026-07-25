import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark import baseline_theory as bt


def test_crossover_is_exactly_half():
    """A mean-predictor ties the random walk at rho = 0.5, wins below, loses above."""
    assert bt.spurious_skill(0.5) == pytest.approx(0.0, abs=1e-12)
    assert bt.spurious_skill(0.3) > 0
    assert bt.spurious_skill(0.7) < 0


def test_white_noise_gives_the_classic_30_percent():
    """rho = 0 -> 1 - 1/sqrt(2) ~ 29.3%. This is the magnitude that appeared as the
    audit's flagship 'electricity driver edge' (+28.3%)."""
    assert bt.spurious_skill(0.0) == pytest.approx(1 - 1 / np.sqrt(2), abs=1e-12)
    assert bt.spurious_skill(0.0) == pytest.approx(0.293, abs=0.001)


def test_skill_is_monotone_decreasing_in_rho():
    rhos = [-0.5, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
    skills = [bt.spurious_skill(r) for r in rhos]
    assert skills == sorted(skills, reverse=True)


def test_implied_rho_inverts_spurious_skill():
    for rho in (-0.3, 0.0, 0.25, 0.45):
        assert bt.implied_rho(bt.spurious_skill(rho)) == pytest.approx(rho, abs=1e-9)


def test_implied_rho_explains_the_flagship_edge():
    """The reported +28.3% electricity 'driver edge' is what a mean-predictor scores
    on a near-white-noise target — i.e. fully accounted for without any driver.
    Electricity MoM measures rho ~ +0.002, so the implied and measured values agree
    to a few hundredths and the 'edge' needs no driver to explain it."""
    implied = bt.implied_rho(0.283)
    assert abs(implied) < 0.05                  # essentially zero autocorrelation
    assert implied == pytest.approx(0.002, abs=0.05)   # matches the measured rho


def test_closed_form_matches_the_projects_own_backtest():
    """Validate against the real walk_forward estimator (expanding-window mean, not
    the true mu), which is what the audit actually runs."""
    for rho in (0.0, 0.3, 0.6):
        assert bt.simulate(rho, n=300, reps=8) == pytest.approx(
            bt.spurious_skill(rho), abs=0.03)


def test_non_stationary_rho_is_rejected():
    with pytest.raises(ValueError):
        bt.spurious_skill(1.0)


def test_lag1_autocorr_recovers_a_known_ar1():
    rng = np.random.default_rng(0)
    y = np.zeros(4000)
    for t in range(1, len(y)):
        y[t] = 0.4 * y[t - 1] + rng.normal(0, 1)
    assert bt.lag1_autocorr(y) == pytest.approx(0.4, abs=0.05)
    assert bt.lag1_autocorr([1.0]) == 0.0          # degenerate input is safe
