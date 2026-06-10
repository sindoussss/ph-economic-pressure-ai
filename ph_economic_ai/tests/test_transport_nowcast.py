import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.transport_nowcast as tn


def test_run_transport_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2010-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(7)
    gas = 45 + np.cumsum(rng.normal(0, 0.5, n))
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    mom = pd.Series(0.5 * np.r_[0.0, np.diff(gas)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(tn, 'load_transport_mom', lambda: mom)
    res = tn.run_transport_nowcast(min_train=24, features=feats, prelim_months=6)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge'], bool)
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']                 # trailing window dropped
    assert 'driver_edge' in res['robust']
    assert 'panel' not in res['mom']
