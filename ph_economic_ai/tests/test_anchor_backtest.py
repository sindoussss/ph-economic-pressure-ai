"""Tests for the anchor validation harness — the pure parts, no network."""
import numpy as np
import pytest

from ph_economic_ai.tools import anchor_backtest as ab


def _synthetic_panel(n=40, passthrough=0.8, seed=1):
    """Months where pump moves are `passthrough` × the mechanical anchor."""
    from ph_economic_ai.engine import anchoring
    rng = np.random.default_rng(seed)
    brent, fx, fuel = 90.0, 57.0, 60.0
    rows = [{'month': '2020-01', 'fuel': fuel, 'brent': brent, 'fx': fx}]
    for i in range(1, n):
        oil_pct = rng.normal(0, 6)
        brent = max(30.0, brent * (1 + oil_pct / 100))
        mech = anchoring.fuel_passthrough_anchor(
            oil_pct, 0.0, brent_usd=rows[-1]['brent'],
            fx_php_per_usd=fx, calibrated=False)
        fuel = fuel + passthrough * mech + rng.normal(0, 0.3)
        rows.append({'month': f'2020-{i:02d}', 'fuel': fuel, 'brent': brent, 'fx': fx})
    return rows


def test_robustness_sweep_passes():
    result = ab.robustness()
    assert result['passed'], result['failures'][:5]
    assert result['checks'] > 1000


def test_ols_recovers_a_known_slope():
    x = np.linspace(-5, 5, 50)
    y = 2.0 * x + 1.0
    slope, intercept, r2 = ab._ols(x, y)
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(1.0)
    assert r2 == pytest.approx(1.0)


def test_backtest_recovers_the_injected_passthrough():
    """If pump moves are 0.8× the mechanical anchor, the fitted slope must be
    ~0.8 — this is exactly how the real calibration is derived."""
    panel = _synthetic_panel(passthrough=0.8)
    bt = ab.backtest(panel)
    assert bt['ols_slope'] == pytest.approx(0.8, abs=0.15)
    assert bt['correlation'] > 0.8


def test_backtest_reports_directional_accuracy_in_range():
    bt = ab.backtest(_synthetic_panel())
    assert 0.0 <= bt['directional_accuracy'] <= 1.0


def test_weak_model_benefit_is_positive_on_synthetic_data():
    """Reconciliation must reduce the error of a hallucinating model."""
    wb = ab.weak_model_benefit(_synthetic_panel(n=60))
    assert wb['mae_anchored_php_l'] < wb['mae_raw_model_php_l']
    assert wb['improvement_pct'] > 0
