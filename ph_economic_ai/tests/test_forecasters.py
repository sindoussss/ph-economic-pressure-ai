import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.forecasters import make_forecaster, FORECASTERS


def _data(n=40):
    rng = np.random.default_rng(0)
    y = 50 + np.cumsum(rng.normal(0, 0.5, n))
    X = np.column_stack([y - 1, rng.normal(0, 1, n)])
    return X, y


def test_all_forecasters_return_finite_float():
    X, y = _data()
    xn = X[-1]
    for name in FORECASTERS:
        pred = make_forecaster(name)(X[:-1], y[:-1], xn)
        assert isinstance(pred, float) and np.isfinite(pred), name


def test_random_walk_returns_last():
    f = make_forecaster('random_walk')
    assert f(None, np.array([1.0, 2.0, 3.5]), None) == pytest.approx(3.5)


def test_drift_adds_mean_step():
    f = make_forecaster('drift')
    assert f(None, np.array([0.0, 1.0, 2.0, 3.0]), None) == pytest.approx(4.0)


def test_seasonal_naive_uses_season_lag():
    f = make_forecaster('seasonal_naive')
    y = np.arange(13, dtype=float)
    assert f(None, y, None) == pytest.approx(1.0)


def test_arima_falls_back_on_degenerate_series():
    f = make_forecaster('arima')
    y = np.zeros(5)
    assert f(None, y, None) == pytest.approx(0.0)
