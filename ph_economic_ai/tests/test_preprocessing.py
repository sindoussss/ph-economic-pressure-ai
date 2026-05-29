import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import (
    build_features, compute_index, pressure_band,
    build_gas_features, build_food_features,
    build_electricity_features, build_all_features,
)


def _df():
    return generate_dataset()

def _full_df():
    return generate_dataset()


def test_build_features_shape():
    df = _df()
    X, y, cols, df_out = build_features(df)
    assert X.shape == (119, 4)
    assert y.shape == (119,)
    assert cols == ['oil_price', 'usd_php', 'demand_index', 'prev_gas_price']


def test_build_features_no_nan():
    df = _df()
    X, y, _, _ = build_features(df)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_compute_index_range():
    df = _df()
    last = df.iloc[-1]
    index, *_ = compute_index(last['oil_price'], last['usd_php'], last['demand_index'], df)
    assert 0.0 <= index <= 100.0


def test_compute_index_returns_deltas():
    df = _df()
    last = df.iloc[-1]
    index, oil_delta, usd_delta, demand_norm = compute_index(
        last['oil_price'], last['usd_php'], last['demand_index'], df
    )
    assert isinstance(oil_delta, float)
    assert isinstance(usd_delta, float)
    assert 0.0 <= demand_norm <= 1.0


def test_compute_index_high_oil_raises_pressure():
    df = _df()
    base_idx, *_ = compute_index(df['oil_price'].mean(), df['usd_php'].mean(), 70.0, df)
    high_idx, *_ = compute_index(df['oil_price'].max(), df['usd_php'].mean(), 70.0, df)
    assert high_idx > base_idx


def test_pressure_band_labels():
    assert pressure_band(15.0) == 'Stable'
    assert pressure_band(45.0) == 'Rising'
    assert pressure_band(70.0) == 'High'
    assert pressure_band(90.0) == 'Critical'
    assert pressure_band(0.0) == 'Stable'
    assert pressure_band(100.0) == 'Critical'


def test_build_gas_features_backward_compat():
    """build_gas_features must return same shape as old build_features on synthetic data."""
    df = _full_df()
    X, y, cols, _ = build_gas_features(df)
    assert 'prev_gas_price' in cols
    assert 'gas_price' not in cols   # target is not a feature
    assert X.shape[0] == len(df) - 1  # one row dropped for lag
    assert not np.isnan(X).any()


def test_build_food_features_shape():
    df = _full_df()
    gas_pred = df['gas_price'].values
    X, y, cols, _ = build_food_features(df, gas_pred)
    assert 'food_price_idx_lag1' in cols
    assert 'gas_pred' in cols
    assert 'food_price_idx' not in cols   # target only
    assert X.shape[1] == len(cols)
    assert not np.isnan(X).any()


def test_build_food_features_scalar_gas_pred():
    """Should also accept a scalar gas_pred for single-point inference."""
    df = _full_df()
    X, y, cols, _ = build_food_features(df, gas_pred=70.0)
    assert X.shape[1] == len(cols)
    assert not np.isnan(X).any()


def test_build_electricity_features_shape():
    df = _full_df()
    gas_pred = df['gas_price'].values
    X, y, cols, _ = build_electricity_features(df, gas_pred)
    assert 'electricity_rate_lag1' in cols
    assert 'gas_pred' in cols
    assert 'electricity_rate' not in cols
    assert not np.isnan(X).any()


def test_build_all_features_returns_three_sectors():
    df = _full_df()
    gas_pred = df['gas_price'].values
    result = build_all_features(df, gas_pred)
    assert set(result.keys()) == {'gas', 'food', 'electricity'}
    for sector, (X, y, cols, _) in result.items():
        assert X.shape[0] > 0, f'{sector} X is empty'
        assert not np.isnan(X).any(), f'{sector} X has NaN'


def test_build_features_alias_unchanged():
    """Existing build_features() must still work (backward compat)."""
    df = _full_df()
    X, y, cols, _ = build_features(df)
    assert 'prev_gas_price' in cols


def test_build_food_features_short_gas_pred_raises():
    df = generate_dataset()
    _, _, _, df_clean = build_food_features(df, gas_pred=1.0)  # scalar to get post-dropna length
    short_pred = np.ones(len(df_clean) - 1)  # one element shorter than post-dropna df
    with pytest.raises(ValueError, match="shorter than df after dropna"):
        build_food_features(df, gas_pred=short_pred)


def test_build_electricity_features_short_gas_pred_raises():
    df = generate_dataset()
    _, _, _, df_clean = build_electricity_features(df, gas_pred=1.0)
    short_pred = np.ones(len(df_clean) - 1)
    with pytest.raises(ValueError, match="shorter than df after dropna"):
        build_electricity_features(df, gas_pred=short_pred)
