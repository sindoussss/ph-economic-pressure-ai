import numpy as np
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.ensemble import HistGradientBoostingRegressor

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai.model import (
    train, predict, predict_interval, get_training_predictions, simulate_scenarios,
    get_feature_importances, cross_val_rmse, forecast,
)


def _trained():
    df = generate_dataset()
    X, y, feature_cols, df_feat = build_features(df)
    regressor = train(X, y)
    last = df_feat.iloc[-1]
    last_features = np.array([last[c] if c != 'prev_gas_price' else last['gas_price']
                               for c in feature_cols])
    return regressor, X, y, df_feat, last_features, float(last['gas_price']), feature_cols


def test_train_returns_fitted_model():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    reg = train(X, y)
    assert isinstance(reg, HistGradientBoostingRegressor)
    assert hasattr(reg, 'feature_importances_')


def test_predict_returns_point_and_band():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    point, low, high = predict_interval(reg, last_features, qhat=1.5)
    assert 50.0 < point < 90.0
    assert low == pytest.approx(point - 1.5)
    assert high == pytest.approx(point + 1.5)


def test_predict_interval_zero_width_when_qhat_zero():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    point, low, high = predict_interval(reg, last_features, qhat=0.0)
    assert low == pytest.approx(point) == pytest.approx(high)


def test_get_training_predictions_shape():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    means, stds = get_training_predictions(reg, X)
    assert means.shape == (len(X),)
    assert stds.shape == (len(X),)
    assert (stds >= 0).all()


def test_get_training_predictions_stds_are_zeros():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    means, stds = get_training_predictions(reg, X)
    assert np.all(stds == 0.0), 'HGB has no per-tree variance — stds must be exactly zero'


def test_simulate_scenarios_keys():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert set(scenarios.keys()) == {'oil_shock', 'usd_shock', 'demand_drop'}


def test_simulate_oil_shock_raises_price():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['oil_shock'] > 0, 'Higher oil should raise price'


def test_simulate_demand_drop_lowers_price():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['demand_drop'] < 0, 'Lower demand should reduce price'


def test_feature_importances_shape():
    reg, X, y, df_feat, last_features, _, feature_cols = _trained()
    importances = get_feature_importances(reg, feature_cols)
    assert len(importances) == len(feature_cols)
    assert abs(sum(importances.values()) - 1.0) < 1e-6


def test_cross_val_rmse_positive():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    rmse = cross_val_rmse(X, y)
    assert rmse > 0.0


def test_forecast_returns_n_months():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    prices = forecast(reg, last_features, n_months=6)
    assert len(prices) == 6
    assert all(30.0 < p < 150.0 for p in prices)
