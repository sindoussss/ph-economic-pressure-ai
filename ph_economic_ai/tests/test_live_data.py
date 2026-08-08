import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.live_data import (
    LiveDataBrief, BSP_TARGET_LOW, BSP_TARGET_HIGH,
    _CURRENT_CPI_PCT, _CPI_AS_OF,
    _FUEL_BASKET_WEIGHT, _FOOD_BASKET_WEIGHT,
    _FUEL_PASSTHROUGH_PER_PHP, _FOOD_PASSTHROUGH_PER_PCT,
)


def test_cpi_baseline_is_not_the_stale_april_figure():
    """Regression guard: this app hit the owner's screen showing a near-target
    reading built from an April 2026 baseline (3.8%) while PSA's actual July
    2026 release was 6.2%, already 2.2 points above the BSP ceiling. Pins the
    corrected value rather than letting it silently drift back to a stale one."""
    assert _CURRENT_CPI_PCT == 6.2
    assert 'July 2026' in _CPI_AS_OF


def test_food_passthrough_is_exactly_the_basket_weight():
    """Not an assumption: a Laspeyres CPI's contribution from one sub-index is
    (% change in that sub-index) x (its basket weight), so this must hold by
    construction. If it ever doesn't, the two constants have drifted apart."""
    assert _FOOD_PASSTHROUGH_PER_PCT == _FOOD_BASKET_WEIGHT


def test_check_bsp_alert_reports_the_cpi_vintage():
    """The projection must never be shown without the reader also being able
    to see how old the baseline it was projected from is."""
    alert = LiveDataBrief.check_bsp_alert(
        gas_php_per_l=1.0, food_pct=0.5, elec_php_per_kwh=0.1)
    assert alert['cpi_as_of'] == _CPI_AS_OF
    assert alert['current_cpi'] == _CURRENT_CPI_PCT


def test_check_bsp_alert_projection_arithmetic():
    alert = LiveDataBrief.check_bsp_alert(
        gas_php_per_l=1.0, food_pct=None, elec_php_per_kwh=None,
        current_cpi=4.0, cpi_as_of='test vintage')
    expected_impact = 1.0 * _FUEL_PASSTHROUGH_PER_PHP
    assert alert['sector_cpi_impact'] == round(expected_impact, 3)
    assert alert['projected_cpi'] == round(4.0 + expected_impact, 2)
    assert alert['breakdown'] == {'fuel': round(expected_impact, 3)}
    assert 'food' not in alert['breakdown'] and 'electricity' not in alert['breakdown']


def test_check_bsp_alert_severity_bands():
    within = LiveDataBrief.check_bsp_alert(0, 0, 0, current_cpi=3.0)
    assert within['severity'] == 'STABLE' and within['within_target']

    watch = LiveDataBrief.check_bsp_alert(0, 0, 0, current_cpi=3.8)
    assert watch['severity'] == 'WATCH'

    alert_band = LiveDataBrief.check_bsp_alert(0, 0, 0, current_cpi=4.5)
    assert alert_band['severity'] == 'ALERT' and alert_band['breaches_upper']

    critical = LiveDataBrief.check_bsp_alert(0, 0, 0, current_cpi=5.5)
    assert critical['severity'] == 'CRITICAL'


def test_check_bsp_alert_with_todays_real_baseline_is_already_critical():
    """The concrete case that motivated the fix: with no sector move at all,
    the corrected baseline alone already sits well past BSP's target -- the
    reading the stale 3.8% baseline was hiding."""
    alert = LiveDataBrief.check_bsp_alert(None, None, None)
    assert alert['current_cpi'] == 6.2
    assert not alert['within_target']
    assert alert['severity'] in ('ALERT', 'CRITICAL')


def test_bsp_target_band_unchanged():
    assert BSP_TARGET_LOW == 2.0 and BSP_TARGET_HIGH == 4.0
