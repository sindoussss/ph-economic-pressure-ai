"""Standard forecasters for the efficiency panel. Each is a
predict_fn(X_train, y_train, x_next) -> float so it plugs into backtest.walk_forward.

Univariate methods ignore X. ARIMA/ETS fall back to random walk on fit failure.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# Explicit optimizer-iteration ceilings for the statsmodels fits below. ARIMA's
# value matches statsmodels' own default (so this changes nothing on data that
# already converges) -- it just stops that default from being an implicit,
# upstream-controlled bound. ETS has no explicit default of its own: an
# unspecified `method` falls back to scipy's L-BFGS-B with *its* default of
# 15000 iterations, which is enough headroom that a pathological walk-forward
# window (checked against every candidate on every step) can run far longer
# than any of the ~30 iterations a converging fit here actually needs.
_ARIMA_MAXITER = 50
_ETS_MAXITER = 200


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
        fit = ARIMA(np.asarray(y_train, dtype=float), order=(1, 1, 1)).fit(
            method_kwargs={'maxiter': _ARIMA_MAXITER})
        return float(np.asarray(fit.forecast(1)).ravel()[0])
    except Exception:
        return float(y_train[-1])


def _ets(X_train, y_train, x_next) -> float:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        fit = ExponentialSmoothing(np.asarray(y_train, dtype=float), trend='add').fit(
            minimize_kwargs={'options': {'maxiter': _ETS_MAXITER}})
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
