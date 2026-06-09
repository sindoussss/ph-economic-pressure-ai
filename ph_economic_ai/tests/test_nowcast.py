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


from ph_economic_ai.benchmark.nowcast import mom_verdict, run_mom_nowcast


def test_mom_verdict_beats_best_naive():
    rng = np.random.default_rng(0)
    n = 300
    base_loss = rng.uniform(0.8, 1.2, n)
    rmse = {'random_walk': 1.5, 'seasonal_naive': 1.0, 'drift': 1.3,
            'arima': 1.2, 'ets': 1.1, 'ridge': 0.7, 'hgb': 0.9}
    loss = {'random_walk': base_loss + 1.0, 'seasonal_naive': base_loss,
            'drift': base_loss + 0.6, 'arima': base_loss + 0.4, 'ets': base_loss + 0.2,
            'ridge': base_loss - 0.4, 'hgb': base_loss - 0.1}
    v = mom_verdict(rmse, loss)
    assert v['verdict'] == 'beats_best_naive'
    assert v['best_method'] == 'ridge'
    assert v['best_naive'] == 'seasonal_naive'


def test_mom_verdict_hollow_win_guard():
    rng = np.random.default_rng(1)
    n = 300
    seas = rng.uniform(0.4, 0.6, n)
    rmse = {'random_walk': 1.5, 'seasonal_naive': 0.5, 'drift': 1.2,
            'arima': 1.1, 'ets': 1.0, 'ridge': 0.9, 'hgb': 1.0}
    loss = {'random_walk': seas + 1.0, 'seasonal_naive': seas, 'drift': seas + 0.7,
            'arima': seas + 0.6, 'ets': seas + 0.5, 'ridge': seas + 0.4, 'hgb': seas + 0.5}
    v = mom_verdict(rmse, loss)
    assert v['verdict'] == 'no_better_than_naive'
    assert v['best_method'] == 'seasonal_naive'


def test_run_mom_nowcast_beats_on_constructed_signal():
    idx = pd.date_range('2016-01', periods=110, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, 110))
    dfuel = np.r_[0.0, np.diff(fuel)]
    target = 0.6 * dfuel + rng.normal(0, 0.05, 110)
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 110),
        'fx': 55 + rng.normal(0, 0.1, 110),
        'fuel': fuel,
        'prev_mom': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_mom_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'beats_best_naive'
    assert res['best_method'] in ('ridge', 'hgb', 'arima', 'ets')


def test_run_mom_nowcast_insufficient_data():
    idx = pd.date_range('2020-01', periods=12, freq='MS').strftime('%Y-%m')
    frame = pd.DataFrame({'oil': range(12), 'fx': range(12), 'fuel': range(12),
                          'prev_mom': range(12), 'target': range(12)},
                         index=idx).astype(float)
    res = run_mom_nowcast(min_train=24, frame=frame)
    assert res['verdict'] == 'insufficient_data'


from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation


def test_run_mom_nowcast_respects_methods_param():
    idx = pd.date_range('2016-01', periods=90, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    target = rng.normal(0.3, 0.4, 90)
    frame = pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, 90), 'fx': 55 + rng.normal(0, 0.1, 90),
        'fuel': 60 + rng.normal(0, 1, 90), 'prev_mom': np.r_[target[0], target[:-1]],
        'target': target,
    }, index=idx)
    res = run_mom_nowcast(min_train=24, frame=frame, methods=['random_walk', 'ridge'])
    assert set(res['rmse_by_method']) == {'random_walk', 'ridge'}


def _driver_signal_frame(n=110, seed=5):
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(seed)
    fuel = 60 + np.cumsum(rng.normal(0, 1.0, n))
    target = 0.5 * fuel + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        'oil': 70 + rng.normal(0, 1, n), 'fx': 55 + rng.normal(0, 0.1, n),
        'fuel': fuel, 'prev_mom': np.r_[target[0], target[:-1]], 'target': target,
    }, index=idx)


def test_driver_ablation_detects_driver_edge():
    res = run_driver_only_ablation(min_train=24, frame=_driver_signal_frame())
    assert res['driver_edge'] is True
    assert res['verdict'] == 'beats_best_naive'
    assert res['best_method'] in ('ridge', 'hgb')
    assert set(res['rmse_by_method']) == {'random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb'}


def test_driver_ablation_absent_when_pure_ar_noise_drivers():
    n = 130
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(6)
    target = np.empty(n); target[0] = 0.0
    for i in range(1, n):
        target[i] = 0.7 * target[i - 1] + rng.normal(0, 0.3)
    frame = pd.DataFrame({
        'oil': rng.normal(0, 1, n), 'fx': rng.normal(0, 1, n), 'fuel': rng.normal(0, 1, n),
        'prev_mom': np.r_[target[0], target[:-1]], 'target': target,
    }, index=idx)
    res = run_driver_only_ablation(min_train=24, frame=frame)
    assert res['driver_edge'] is False
    assert res['verdict'] == 'no_better_than_naive'


def test_driver_ablation_handles_frame_without_prev_mom():
    f = _driver_signal_frame().drop(columns=['prev_mom'])
    res = run_driver_only_ablation(min_train=24, frame=f)
    assert 'verdict' in res


def test_build_nowcast_frame_accepts_features_arg(monkeypatch):
    idx = pd.date_range('2010-01', periods=50, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(9)
    mom = pd.Series(rng.normal(0.3, 0.4, 50), index=idx)
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, 50)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, 50)),
        'gas_price': 45 + np.cumsum(rng.normal(0, 0.5, 50)),
        'demand_index': 70 + rng.normal(0, 2, 50),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, '_features', lambda: (_ for _ in ()).throw(AssertionError('used _features')))
    f = nc.build_nowcast_frame(target_loader=lambda: mom, prev_col='prev_mom', features=feats)
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_mom', 'target']
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
