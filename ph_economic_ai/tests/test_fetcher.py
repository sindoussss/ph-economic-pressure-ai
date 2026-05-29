import json
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import patch, MagicMock

from ph_economic_ai.fetcher import (
    _compute_demand, _load_cache, _save_cache,
    _parse_world_bank_response, _forward_fill_annual,
    _fetch_open_meteo, _seasonal_weather_fallback,
    _fetch_fao_food, _derive_food_from_gas, _derive_electricity,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'date': ['2024-01', '2024-02'],
        'oil_price': [80.0, 82.0],
        'usd_php': [56.0, 56.5],
        'demand_index': [72.0, 68.0],
        'gas_price': [65.0, 66.0],
    })


def test_compute_demand_range():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    assert all(55.0 <= v <= 90.0 for v in values), f'Out of range: {values}'


def test_compute_demand_peaks():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    june = values[5]   # index 5 = June
    assert values[2] > june, f'March ({values[2]:.1f}) should be > June ({june:.1f})'
    assert values[11] > june, f'December ({values[11]:.1f}) should be > June ({june:.1f})'


def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / 'data.json'
    df = _sample_df()
    _save_cache(df, cache_path=cache_file)
    loaded_df, _ = _load_cache(cache_path=cache_file)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        loaded_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_fresh_cache_is_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    _save_cache(_sample_df(), cache_path=cache_file)
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert is_fresh


def test_stale_cache_is_not_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    payload = {
        'fetched_at': (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
        'data': _sample_df().to_dict(orient='records'),
    }
    cache_file.write_text(json.dumps(payload), encoding='utf-8')
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert not is_fresh


def test_forward_fill_annual_basic():
    annual = pd.Series({'2022': 5.0, '2023': 6.0})
    monthly = ['2022-06', '2022-12', '2023-03', '2023-09']
    result = _forward_fill_annual(annual, monthly)
    assert result['2022-06'] == 5.0
    assert result['2022-12'] == 5.0
    assert result['2023-03'] == 6.0
    assert result['2023-09'] == 6.0


def test_forward_fill_annual_carries_forward():
    annual = pd.Series({'2021': 3.0})
    monthly = ['2021-01', '2022-06']  # 2022 not in annual
    result = _forward_fill_annual(annual, monthly)
    assert result['2021-01'] == 3.0
    assert result['2022-06'] == 3.0  # carries last known value


def test_world_bank_parse_returns_series():
    sample_payload = [
        {'page': 1, 'pages': 1, 'per_page': 100, 'total': 2},
        [
            {'date': '2023', 'value': 5.97},
            {'date': '2022', 'value': 3.21},
            {'date': '2021', 'value': None},  # missing — should be dropped
        ]
    ]
    result = _parse_world_bank_response(sample_payload)
    assert isinstance(result, pd.Series)
    assert '2023' in result.index
    assert '2022' in result.index
    assert '2021' not in result.index  # None dropped
    assert abs(result['2023'] - 5.97) < 0.01


def _make_meteo_response(rain_val: float, temp_val: float):
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        'monthly': {
            'time': ['2024-01-01', '2024-07-01'],
            'precipitation_sum': [rain_val, rain_val],
            'temperature_2m_mean': [temp_val, temp_val],
        }
    }
    return mock


def test_fetch_open_meteo_makes_three_requests():
    with patch('ph_economic_ai.fetcher.requests.get',
               return_value=_make_meteo_response(100.0, 27.0)) as mock_get:
        _fetch_open_meteo()
    assert mock_get.call_count == 3


def test_fetch_open_meteo_weighted_sum():
    """Central Luzon 100mm×0.45 + Bicol 80mm×0.25 + Davao 60mm×0.30 = 83mm."""
    responses = [
        _make_meteo_response(100.0, 26.0),  # Central Luzon weight 0.45
        _make_meteo_response(80.0,  28.0),  # Bicol weight 0.25
        _make_meteo_response(60.0,  30.0),  # Davao weight 0.30
    ]
    with patch('ph_economic_ai.fetcher.requests.get', side_effect=responses):
        rain, temp = _fetch_open_meteo()
    assert abs(rain['2024-01'] - 83.0) < 0.5   # 45 + 20 + 18
    assert abs(temp['2024-01'] - 27.5) < 0.2   # 11.7 + 7.0 + 9.0


def test_seasonal_weather_fallback_july():
    """July is peak wet season — rainfall norm should be highest."""
    monthly_index = [f'2024-{m:02d}' for m in range(1, 13)]
    rain, temp = _seasonal_weather_fallback(monthly_index)
    assert rain['2024-07'] > rain['2024-01']   # July wetter than January
    assert rain['2024-07'] == 200.0
    assert temp['2024-01'] == 26.5


def test_seasonal_fallback_covers_all_months():
    monthly_index = [f'2024-{m:02d}' for m in range(1, 13)]
    rain, temp = _seasonal_weather_fallback(monthly_index)
    assert len(rain) == 12
    assert len(temp) == 12


def test_fetch_open_meteo_normalizes_partial_none():
    """When one zone returns None for a month, result is normalized to present zones only."""
    def make_partial(rain_val, temp_val, null_month=False):
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        rain = [None if null_month else rain_val, rain_val]
        temp = [None if null_month else temp_val, temp_val]
        mock.json.return_value = {
            'monthly': {
                'time': ['2024-01-01', '2024-07-01'],
                'precipitation_sum': rain,
                'temperature_2m_mean': temp,
            }
        }
        return mock

    # Zone 1 (weight 0.45) returns None for Jan; zones 2+3 return 80mm
    responses = [
        make_partial(100.0, 26.0, null_month=True),   # Central Luzon — None for Jan
        make_partial(80.0,  28.0),                      # Bicol
        make_partial(80.0,  28.0),                      # Davao
    ]
    with patch('ph_economic_ai.fetcher.requests.get', side_effect=responses):
        rain, temp = _fetch_open_meteo()

    # Jan: only zones 2+3 contribute (0.25 + 0.30 = 0.55 total weight)
    # Normalized: 80*0.25/0.55 + 80*0.30/0.55 = 80.0 (both zones same value)
    assert abs(rain['2024-01'] - 80.0) < 0.5
    # July: all three zones contribute
    assert abs(rain['2024-07'] - (100.0*0.45 + 80.0*0.25 + 80.0*0.30)) < 0.5


def test_fetch_fao_food_parses_response():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        'data': [
            {'Year': '2022', 'Value': 115.3},
            {'Year': '2023', 'Value': 118.7},
        ]
    }
    with patch('ph_economic_ai.fetcher.requests.get', return_value=mock):
        series = _fetch_fao_food()
    assert series['2022'] == pytest.approx(115.3)
    assert series['2023'] == pytest.approx(118.7)


def test_fetch_fao_food_raises_on_empty():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {'data': []}
    with patch('ph_economic_ai.fetcher.requests.get', return_value=mock):
        with pytest.raises(ValueError, match='no food price data'):
            _fetch_fao_food()


def test_derive_food_from_gas_increases_with_gas_spike():
    monthly_index = ['2024-01', '2024-02', '2024-03']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 70.0, '2024-03': 75.0})
    rain = pd.Series({'2024-01': 30.0, '2024-02': 25.0, '2024-03': 40.0})
    idx = _derive_food_from_gas(gas, rain, monthly_index, last_known_idx=100.0)
    # Gas rose 5 PHP/L from Jan→Feb, so food index should increase
    assert idx['2024-02'] > idx['2024-01']


def test_derive_food_clips_at_floor():
    """A large gas price drop should be clipped at 80.0 floor."""
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 30.0})   # big drop → -35 delta
    rain = pd.Series({'2024-01': 30.0, '2024-02': 30.0})   # at norm → deficit = 0
    # Raw result for Feb: 85.0 + (-35 * 0.22) + 0 = 85 - 7.7 = 77.3 → clipped to 80.0
    idx = _derive_food_from_gas(gas, rain, monthly_index, last_known_idx=85.0)
    assert idx['2024-02'] == pytest.approx(80.0)


def test_derive_electricity_base_rate():
    """With no gas change, electricity stays at base rate."""
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 65.0})
    rate = _derive_electricity(gas, monthly_index)
    assert rate['2024-02'] == pytest.approx(11.20, abs=0.05)


def test_derive_electricity_rises_with_gas():
    """A +5 PHP/L gas increase → +0.90 PHP/kWh electricity increase."""
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 65.0, '2024-02': 70.0})
    rate = _derive_electricity(gas, monthly_index)
    assert rate['2024-02'] > rate['2024-01']
    assert abs(rate['2024-02'] - rate['2024-01'] - 0.90) < 0.05  # 5 × 0.18


def test_derive_electricity_never_below_floor():
    monthly_index = ['2024-01', '2024-02']
    gas = pd.Series({'2024-01': 30.0, '2024-02': 10.0})  # extreme drop
    rate = _derive_electricity(gas, monthly_index)
    assert all(r >= 8.0 for r in rate.values)


