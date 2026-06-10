import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.food_nowcast as fn


def test_run_food_nowcast_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2005-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(5)
    rice = 12 + np.cumsum(rng.normal(0, 0.2, n))
    feats = pd.DataFrame({
        'rice': rice,
        'wheat': 500 + np.cumsum(rng.normal(0, 5, n)),
        'corn': 400 + np.cumsum(rng.normal(0, 4, n)),
        'soybean': 1000 + np.cumsum(rng.normal(0, 8, n)),
        'oil_price': 60 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 50 + np.cumsum(rng.normal(0, 0.1, n)),
    }, index=idx)
    mom = pd.Series(0.4 * np.r_[0.0, np.diff(rice)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(fn, 'load_food_mom', lambda: mom)
    res = fn.run_food_nowcast(min_train=24, features=feats, prelim_months=6)
    assert set(res) >= {'n', 'mom', 'driver_ablation', 'driver_edge',
                        'robust', 'driver_edge_robust'}
    assert res['n'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert isinstance(res['driver_edge_robust'], bool)
    assert res['robust']['n'] < res['n']
    assert 'panel' not in res['mom']
