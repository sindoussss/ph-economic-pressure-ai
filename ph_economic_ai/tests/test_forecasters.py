import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark import forecasters
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


# ── Optimizer bounds: a walk-forward loop re-fits these on every step of every
# candidate, so an unbounded optimizer call is a worst-case-runtime risk even
# though it never affects a converging fit's result. ─────────────────────────

def test_arima_passes_an_explicit_maxiter(monkeypatch):
    """_arima must bound the statespace optimizer explicitly, not rely on
    whatever statsmodels currently defaults to."""
    import statsmodels.tsa.arima.model as arima_mod
    captured = {}
    orig_fit = arima_mod.ARIMA.fit

    def spy_fit(self, *args, **kwargs):
        captured.update(kwargs)
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(arima_mod.ARIMA, 'fit', spy_fit)
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, 30)) + 100
    make_forecaster('arima')(None, y, None)
    assert captured.get('method_kwargs', {}).get('maxiter') == forecasters._ARIMA_MAXITER


def test_ets_passes_an_explicit_maxiter(monkeypatch):
    """_ets must bound the L-BFGS-B optimizer explicitly -- left unset it
    inherits scipy's own default of 15000 iterations."""
    import statsmodels.tsa.holtwinters as hw_mod
    captured = {}
    orig_fit = hw_mod.ExponentialSmoothing.fit

    def spy_fit(self, *args, **kwargs):
        captured.update(kwargs)
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(hw_mod.ExponentialSmoothing, 'fit', spy_fit)
    y = np.cumsum(np.random.default_rng(0).normal(0, 1, 30)) + 100
    make_forecaster('ets')(None, y, None)
    assert captured.get('minimize_kwargs', {}).get('options', {}).get('maxiter') \
        == forecasters._ETS_MAXITER


def test_arima_maxiter_is_actually_enforced(monkeypatch):
    """Proves the bound is wired to the real optimizer, not a dead kwarg: an
    artificially tiny ceiling can't converge on real data, so it must warn."""
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    monkeypatch.setattr(forecasters, '_ARIMA_MAXITER', 1)
    y = np.cumsum(np.random.default_rng(3).normal(0, 1, 40)) + 100
    with pytest.warns(ConvergenceWarning):
        make_forecaster('arima')(None, y, None)


def test_ets_maxiter_is_actually_enforced(monkeypatch):
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    monkeypatch.setattr(forecasters, '_ETS_MAXITER', 1)
    y = np.cumsum(np.random.default_rng(3).normal(0, 1, 40)) + 100
    with pytest.warns(ConvergenceWarning):
        make_forecaster('ets')(None, y, None)
