import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.electricity_nowcast as en


def test_run_electricity_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2008-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    oil = 60 + np.cumsum(rng.normal(0, 1, n))
    feats = pd.DataFrame({
        'oil_price': oil,
        'natgas': 3 + np.cumsum(rng.normal(0, 0.1, n)),
        'usd_php': 50 + np.cumsum(rng.normal(0, 0.1, n)),
    }, index=idx)
    mom = pd.Series(0.3 * np.r_[0.0, np.diff(oil)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(en, 'load_electricity_mom', lambda: mom)
    res = en.run_electricity_nowcast(min_train=24, features=feats, prelim_months=6)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']
    assert 'panel' not in res['mom']
