import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.ui.sector_forecast import sector_forecast_rows


def _by_key(**kw):
    return {r['key']: r for r in sector_forecast_rows(**kw)}


def test_bar_fraction_per_sector():
    r = _by_key(gas=-1.8, food=-2.6, elec=0.18)
    assert abs(r['gas']['bar'] - 0.36) < 1e-9     # 1.8 / 5.0
    assert abs(r['food']['bar'] - 0.52) < 1e-9    # 2.6 / 5.0
    assert abs(r['elec']['bar'] - 0.09) < 1e-9    # 0.18 / 2.0


def test_bar_clamps_and_none():
    r = _by_key(gas=100.0, food=None)
    assert r['gas']['bar'] == 1.0                  # clamped
    assert r['food']['bar'] == 0.0 and r['food']['direction'] == 'na'
    assert r['gas']['value_str'] == '+100.00 ₱/L'  # existing fields unchanged
