import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.passthrough import estimate_passthrough


def test_recovers_known_passthrough():
    rng = np.random.default_rng(0)
    n = 120
    idx = pd.date_range('2016-01', periods=n, freq='MS').strftime('%Y-%m')
    cost = 50 + np.cumsum(rng.normal(0, 1.0, n))
    dcost = np.diff(cost, prepend=cost[0])
    pump = np.empty(n); pump[0] = 60.0
    for i in range(1, n):
        pump[i] = pump[i - 1] + 0.5 * dcost[i]
    df = pd.DataFrame({'gas_price': cost, 'ron95': pump}, index=idx)
    r = estimate_passthrough(df)
    assert r['beta_total'] == pytest.approx(0.5, abs=0.05)
    assert r['r2'] > 0.95
    assert r['n'] > 100


def test_short_series_returns_none_coeffs():
    df = pd.DataFrame({'gas_price': [1.0, 2.0, 3.0], 'ron95': [1.0, 2.0, 3.0]},
                      index=['2020-01', '2020-02', '2020-03'])
    r = estimate_passthrough(df)
    assert r['beta_total'] is None
    assert r['n'] < 10
