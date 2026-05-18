import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features, compute_index, pressure_band


def _df():
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
