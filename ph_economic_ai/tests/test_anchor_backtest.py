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


def test_corr_significance_flags_a_strong_relationship():
    x = np.linspace(0, 10, 60)
    y = x + np.random.default_rng(0).normal(0, 1, 60)   # strong positive
    s = ab.corr_significance(x, y)
    assert s['r'] > 0.9
    assert s['p_value'] < 0.001
    assert s['significant'] is True
    assert s['ci95'][0] < s['r'] < s['ci95'][1]


def test_corr_significance_flags_noise_as_not_significant():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=40), rng.normal(size=40)     # independent
    s = ab.corr_significance(x, y)
    assert s['p_value'] > 0.05
    assert s['significant'] is False


def test_slope_significance_recovers_a_known_slope():
    x = np.linspace(-5, 5, 80)
    y = 0.8 * x + np.random.default_rng(2).normal(0, 0.3, 80)
    s = ab.slope_significance(x, y)
    assert s['slope'] == pytest.approx(0.8, abs=0.1)
    assert s['p_slope_ne_0'] < 0.001          # clearly non-zero
    assert s['p_slope_ne_1'] < 0.001          # clearly not 1.0


def test_fuel_backtest_carries_significance_fields():
    """A correlation without a p-value is a point estimate, not a result."""
    bt = ab.backtest(_synthetic_panel(passthrough=0.8, n=60))
    assert 'correlation_significance' in bt and 'p_value' in bt['correlation_significance']
    assert 'dm_vs_naive' in bt and 'p_value' in bt['dm_vs_naive']
    assert 'slope_significance' in bt


def test_lagged_corr_and_scale_ratio():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert ab._lagged_corr(x, x, 0) == pytest.approx(1.0)
    # a perfectly-scaled predictor has scale ratio 1
    assert ab._scale_ratio(x, x) == pytest.approx(1.0)
    # half-magnitude predictor
    assert ab._scale_ratio(x * 0.5, x) == pytest.approx(0.5)


# ── Sector backtests run on committed PSA CSVs (headless, no network) ──────────

def test_electricity_backtest_reports_scale_and_predictiveness():
    r = ab.backtest_electricity()
    assert r['n_months'] > 100
    assert 'scale_ratio' in r and r['scale_ratio'] > 0
    assert isinstance(r['is_predictive'], bool)
    assert 'finding' in r


def test_food_finding_is_consistent_with_its_numbers():
    """The verdict must be derived from the data, not hardcoded — this test
    exists because an earlier version asserted a conclusion its own numbers
    contradicted."""
    r = ab.backtest_food()
    if r['indistinguishable'] and abs(r['persistence_correlation']) < 0.25 \
            and abs(r['oil_correlation']) < 0.25:
        assert 'magnitude guard' in r['finding']
    # never claim a winner when the two are within sampling noise
    if r['indistinguishable']:
        assert 'outpredicts' not in r['finding']
