"""Standard forecasters for the efficiency panel. Each is a
predict_fn(X_train, y_train, x_next) -> float so it plugs into backtest.walk_forward.

Univariate methods ignore X. ARIMA/ETS fall back to random walk on fit failure.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def _random_walk(X_train, y_train, x_next) -> float:
    return float(y_train[-1])


def _drift(X_train, y_train, x_next) -> float:
    step = float(np.mean(np.diff(y_train))) if len(y_train) > 1 else 0.0
    return float(y_train[-1] + step)


def _seasonal_naive(X_train, y_train, x_next, season: int = 12) -> float:
    return float(y_train[-season]) if len(y_train) > season else float(y_train[-1])


def _mean(X_train, y_train, x_next) -> float:
    """Historical-mean forecast: the expanding-window unconditional mean, using
    only past data (no leakage). It is the optimal constant predictor and the
    *strong* naive baseline for a mean-reverting series, where the random walk is
    weak. Omitting it lets a mean-reverting rate series look 'beatable' when the
    only thing beating the random walk is reversion to the mean. Ignores X."""
    return float(np.mean(y_train))


def _ridge(X_train, y_train, x_next) -> float:
    model = Ridge(alpha=1.0).fit(X_train, y_train)
    return float(model.predict(x_next.reshape(1, -1))[0])


def _hgb(X_train, y_train, x_next) -> float:
    model = HistGradientBoostingRegressor(
        random_state=42, min_samples_leaf=5, max_leaf_nodes=15).fit(X_train, y_train)
    return float(model.predict(x_next.reshape(1, -1))[0])


def _arima(X_train, y_train, x_next) -> float:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(np.asarray(y_train, dtype=float), order=(1, 1, 1)).fit()
        return float(np.asarray(fit.forecast(1)).ravel()[0])
    except Exception:
        return float(y_train[-1])


def _ets(X_train, y_train, x_next) -> float:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(np.asarray(y_train, dtype=float), trend='add').fit()
        return float(np.asarray(fit.forecast(1)).ravel()[0])
    except Exception:
        return float(y_train[-1])


FORECASTERS = {
    'random_walk':   _random_walk,
    'drift':         _drift,
    'seasonal_naive': _seasonal_naive,
    'mean':          _mean,
    'arima':         _arima,
    'ets':           _ets,
    'ridge':         _ridge,
    'hgb':           _hgb,
}


def make_forecaster(name: str):
    return FORECASTERS[name]
