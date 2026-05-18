import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.utils.explanation import generate


def _gen(oil_delta=0.0, usd_delta=0.0, demand_norm=0.7,
         pressure_index=50.0, current=70.0, predicted=71.5):
    return generate(oil_delta, usd_delta, demand_norm, pressure_index, current, predicted)


def test_returns_required_keys():
    result = _gen()
    for key in ('drivers', 'risk_badge', 'risk_color', 'summary',
                'advisory', 'advisory_icon', 'expected_increase', 'price_direction'):
        assert key in result, f"Missing key: {key}"


def test_drivers_has_three_items():
    assert len(_gen()['drivers']) == 3


def test_driver_keys():
    driver = _gen()['drivers'][0]
    for key in ('icon', 'name', 'value', 'status', 'color'):
        assert key in driver


def test_high_pressure_advisory():
    result = _gen(pressure_index=75.0)
    assert 'Refuel' in result['advisory']
    assert result['advisory_icon'] == '⛽'


def test_rising_pressure_advisory():
    result = _gen(pressure_index=45.0)
    assert 'Monitor' in result['advisory']


def test_stable_advisory():
    result = _gen(pressure_index=15.0)
    assert 'stable' in result['advisory'].lower() or 'No action' in result['advisory']


def test_price_direction_increase():
    result = _gen(current=70.0, predicted=72.0)
    assert result['price_direction'] == 'increase'


def test_price_direction_decrease():
    result = _gen(current=72.0, predicted=70.0)
    assert result['price_direction'] == 'decrease'


def test_risk_colors():
    assert _gen(pressure_index=75.0)['risk_color'] == '#E07A4A'
    assert _gen(pressure_index=45.0)['risk_color'] == '#E0A84A'
    assert _gen(pressure_index=15.0)['risk_color'] == '#4A90E2'
