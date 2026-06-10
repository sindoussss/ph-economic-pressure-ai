# coding: utf-8
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.ui.sector_forecast import sector_forecast_rows


def test_values_units_directions():
    rows = sector_forecast_rows(-2.40, 0.50, 0.05)
    by = {r['key']: r for r in rows}
    assert by['gas']['direction'] == 'down'
    assert '-2.40' in by['gas']['value_str'] and '/L' in by['gas']['value_str']
    assert by['food']['direction'] == 'up'
    assert '+0.50' in by['food']['value_str'] and '%' in by['food']['value_str']
    assert by['elec']['direction'] == 'up'
    assert '+0.0500' in by['elec']['value_str'] and 'kWh' in by['elec']['value_str']


def test_none_and_flat():
    rows = sector_forecast_rows(None, 0.0, None)
    by = {r['key']: r for r in rows}
    assert by['gas']['value_str'] == '—' and by['gas']['direction'] == 'na'
    assert by['food']['direction'] == 'flat'
    assert by['elec']['value_str'] == '—' and by['elec']['direction'] == 'na'


def test_always_three_rows_in_order():
    assert [r['key'] for r in sector_forecast_rows()] == ['gas', 'food', 'elec']
