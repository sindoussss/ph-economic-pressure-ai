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


# ── Staleness: the defect that let this artifact drift ────────────────────────

def test_the_electricity_result_records_the_levels_it_was_computed_from():
    """`scale_ratio` is a RATIO OF LEVELS, so it is only readable next to them.

    The committed artifact said 1.02 and "its magnitude is right" while it had
    been computed from a generation charge frozen at 5.50. When that constant was
    corrected to the published 9.28 the same measurement became 1.84 — the number
    moved 80% and nothing in the artifact recorded why, because the artifact
    stored the ratio without the inputs that produced it.

    That is `RSK-025` on the validation side: a number stored without recording
    what it is OF.
    """
    r = ab.backtest_electricity()
    assert 'generation_charge_php_kwh' in r, (
        'scale_ratio without the charge it used cannot be checked for staleness')
    assert 'base_rate_php_kwh' in r
    assert r['generation_charge_php_kwh'] > 0 and r['base_rate_php_kwh'] > 0


def test_the_recorded_charge_matches_the_one_the_anchor_actually_uses():
    """The regression guard. If `anchoring` starts reading a different level and
    this backtest keeps its own copy, the artifact silently describes a formula
    the app no longer runs — which is exactly what happened.
    """
    from ph_economic_ai.engine import anchoring
    r = ab.backtest_electricity()
    assert r['generation_charge_php_kwh'] == pytest.approx(
        anchoring._default_generation_charge()), (
        'the backtest and the live anchor disagree about the generation charge')


def test_the_scale_claim_admits_it_compares_across_eras():
    """A current price level applied to a 227-month panel does not measure
    whether the anchor was correctly sized in 2012.

    The ratio read ~1.0 only while the charge was stale-LOW; correcting it moved
    it to 1.84. Neither number supports "the magnitude is right" — the comparison
    is anachronistic, and the finding has to say so rather than let a reader take
    the ratio at face value.
    """
    r = ab.backtest_electricity()
    finding = r['finding'].lower()
    assert 'current' in finding or 'anachron' in finding or 'today' in finding, (
        f'the finding presents scale_ratio {r["scale_ratio"]} without noting it '
        f'applies a present-day charge level to a historical panel')


# ── The scale ratio depends on the window it is measured over ────────────────

def test_the_scale_ratio_is_reported_by_era_not_only_as_one_number():
    """A single ratio invites tuning a constant until it reads 1.0.

    Measured 2026-08-19: 1.11 over 2007-2011, 1.16 over 2012-2016, 2.02 over
    2017-2021 and 1.55 over 2022-2026. The anchor's own magnitude barely moves
    across those eras (median 1.78 to 2.14); what changed is electricity CPI
    volatility, which fell from 1.65 to 1.06 and partly recovered. The whole-panel
    1.40 is therefore a property of the comparison window, not of the anchor.
    """
    r = ab.backtest_electricity()
    assert 'scale_ratio_by_era' in r
    eras = r['scale_ratio_by_era']
    assert len(eras) >= 3
    assert all(v > 0 for v in eras.values())


def test_the_finding_says_the_ratio_moves_with_the_window():
    """Publishing 1.40 alone reads as "the anchor is 40 percent oversized", which
    is a claim about the anchor. The honest claim is narrower."""
    r = ab.backtest_electricity()
    finding = r['finding'].lower()
    assert 'window' in finding or 'era' in finding or 'period' in finding


def test_the_anchor_was_correctly_sized_in_the_early_panel():
    """The evidence against tuning. If a constant were wrong the ratio would be
    uniformly off; instead the first decade sits near 1."""
    r = ab.backtest_electricity()
    early = [v for k, v in r['scale_ratio_by_era'].items() if k.startswith('2007') or k.startswith('2012')]
    assert early, 'no early era reported'
    assert all(0.8 < v < 1.4 for v in early), (
        f'early eras {early} no longer near 1; a uniform miss would justify '
        f'revisiting the fuel share, which this test exists to distinguish')


def test_the_fuel_share_is_not_fitted_to_the_ratio():
    """`_GEN_FUEL_SHARE` is the fuel-indexed portion of the generation charge, a
    physical quantity. Forcing the whole-panel ratio to 1.0 would need 0.392, and
    that would make the early panel read about 0.79 -- worse, not better. The
    constant stays at its stated value and this test fails if it is quietly
    fitted."""
    from ph_economic_ai.engine import anchoring
    assert anchoring._GEN_FUEL_SHARE == pytest.approx(0.55)
