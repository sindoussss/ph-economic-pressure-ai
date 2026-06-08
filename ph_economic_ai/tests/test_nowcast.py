import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.nowcast import build_nowcast_frame


def _fake_sources(monkeypatch, n=40):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(0)
    infl = pd.Series(3.0 + rng.normal(0, 0.5, n), index=idx)
    feats = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': 60 + np.cumsum(rng.normal(0, 0.5, n)),
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, 'load_inflation', lambda: infl)
    monkeypatch.setattr(nc, '_features', lambda: feats)
    return idx, infl, feats


def test_frame_has_contemporaneous_drivers_and_lagged_target(monkeypatch):
    idx, infl, feats = _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_inflation', 'target']
    assert not f.isna().any().any()
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
    assert f.loc[t, 'fuel'] == pytest.approx(feats['gas_price'].iloc[pos])
    assert f.loc[t, 'prev_inflation'] == pytest.approx(infl.iloc[pos - 1])
    assert f.loc[t, 'target'] == pytest.approx(infl.iloc[pos])


def test_no_same_month_cpi_feature_leak(monkeypatch):
    _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    cpi_like = [c for c in f.columns if 'infl' in c.lower() or 'cpi' in c.lower()]
    assert cpi_like == ['prev_inflation']


from ph_economic_ai.benchmark.nowcast import run_nowcast


def test_run_nowcast_beats_naive_on_constructed_signal():
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(1)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, 90))
    target = 0.5 * (fuel - 60) + rng.normal(0, 0.05, 90)
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90),
        'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': fuel,
        'prev_inflation': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'beats_naive'
    assert res['best_method'] != 'random_walk'
    assert res['best_skill'] > 0


def test_run_nowcast_ties_naive_on_random_walk_target():
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    target = np.cumsum(rng.normal(0, 0.3, 90)) + 3.0
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90),
        'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': 60 + rng.normal(0, 1, 90),
        'prev_inflation': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'no_better_than_naive'
    assert res['best_method'] == 'random_walk'


def test_run_nowcast_insufficient_data():
    idx = pd.date_range('2020-01', periods=10, freq='MS').strftime('%Y-%m')
    frame = pd.DataFrame({'oil': range(10), 'fx': range(10), 'fuel': range(10),
                          'prev_inflation': range(10), 'target': range(10)},
                         index=idx).astype(float)
    res = run_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'insufficient_data'


def test_build_nowcast_frame_mom_variant(monkeypatch):
    idx = pd.date_range('2017-01', periods=40, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(4)
    mom = pd.Series(rng.normal(0.3, 0.4, 40), index=idx)
    feats = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, 40)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, 40)),
        'gas_price': 60 + np.cumsum(rng.normal(0, 0.5, 40)),
        'demand_index': 70 + rng.normal(0, 2, 40),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, '_features', lambda: feats)
    f = nc.build_nowcast_frame(target_loader=lambda: mom, prev_col='prev_mom')
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_mom', 'target']
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'prev_mom'] == pytest.approx(mom.iloc[pos - 1])
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
    cpi_like = [c for c in f.columns if 'mom' in c.lower() or 'infl' in c.lower() or 'cpi' in c.lower()]
    assert cpi_like == ['prev_mom']
