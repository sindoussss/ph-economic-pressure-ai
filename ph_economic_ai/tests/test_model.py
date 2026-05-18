import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai.model import train, predict, get_training_predictions, simulate_scenarios


def _trained():
    df = generate_dataset()
    X, y, _, df_feat = build_features(df)
    regressor = train(X, y)
    last = df_feat.iloc[-1]
    last_features = np.array([
        last['oil_price'], last['usd_php'], last['demand_index'], last['gas_price']
    ])
    return regressor, X, y, df_feat, last_features, last['gas_price']


def test_train_returns_fitted_model():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    reg = train(X, y)
    assert hasattr(reg, 'estimators_')
    assert len(reg.estimators_) == 100


def test_predict_returns_tuple():
    reg, X, y, df_feat, last_features, _ = _trained()
    result = predict(reg, last_features)
    assert len(result) == 3
    predicted_price, confidence, pred_std = result
    assert 50.0 < predicted_price < 90.0
    assert 0.0 <= confidence <= 100.0
    assert pred_std >= 0.0


def test_get_training_predictions_shape():
    reg, X, y, df_feat, last_features, _ = _trained()
    means, stds = get_training_predictions(reg, X)
    assert means.shape == (len(X),)
    assert stds.shape == (len(X),)
    assert (stds >= 0).all()


def test_simulate_scenarios_keys():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert set(scenarios.keys()) == {'oil_shock', 'usd_shock', 'demand_drop'}


def test_simulate_oil_shock_raises_price():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['oil_shock'] > 0, "Higher oil should raise price"


def test_simulate_demand_drop_lowers_price():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['demand_drop'] < 0, "Lower demand should reduce price"
