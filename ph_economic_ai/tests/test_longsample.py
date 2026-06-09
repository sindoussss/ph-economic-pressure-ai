import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

import ph_economic_ai.benchmark.longsample as ls


def test_run_mom_longsample_wires_through(monkeypatch):
    n = 130
    idx = pd.date_range('2010-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(11)
    gas = 45 + np.cumsum(rng.normal(0, 0.5, n))
    feats = pd.DataFrame({
        'oil_price': 40 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 48 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    mom = pd.Series(0.4 * np.r_[0.0, np.diff(gas)] + rng.normal(0, 0.05, n), index=idx)
    monkeypatch.setattr(ls, 'load_inflation_mom', lambda: mom)
    res = ls.run_mom_longsample(min_train=24, features=feats)
    assert set(res) == {'n_long', 'mom', 'driver_ablation'}
    assert res['n_long'] > 60
    assert 'verdict' in res['mom'] and 'verdict' in res['driver_ablation']
    assert 'driver_edge' in res['driver_ablation']
    assert 'panel' not in res['mom'] and 'calibration' not in res['mom']
