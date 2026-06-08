import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.features import build_feature_frame, VARIANTS
from ph_economic_ai.benchmark.efficiency import run_panel


def _frame(n=70):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(2)
    gas = 50 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)
    return build_feature_frame(df)


def test_panel_one_row_per_method_with_keys():
    methods = ['random_walk', 'drift', 'ridge', 'hgb']
    rows = run_panel(_frame(), methods, min_train=24,
                     feature_cols=VARIANTS['passthrough_lags']['cols'])
    assert [r['method'] for r in rows] == methods
    for r in rows:
        assert set(r) >= {'method', 'rmse', 'mae', 'skill_vs_rw', 'dm_stat', 'dm_p', 'n'}


def test_random_walk_row_has_zero_skill_and_null_dm():
    rows = run_panel(_frame(), ['random_walk', 'ridge'], min_train=24,
                     feature_cols=VARIANTS['passthrough_lags']['cols'])
    rw = next(r for r in rows if r['method'] == 'random_walk')
    assert rw['skill_vs_rw'] == 0.0
    assert rw['dm_p'] is None


def test_run_panel_accepts_custom_target_col():
    idx = pd.date_range('2017-01', periods=70, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(5)
    y = 50 + np.cumsum(rng.normal(0, 0.4, 70))
    frame = pd.DataFrame({
        'prev_x': np.r_[y[0], y[:-1]],
        'drv_lag1': np.r_[0, np.diff(y)],
        'target': y,
    }, index=idx)
    rows = run_panel(frame, ['random_walk', 'ridge'], min_train=24,
                     feature_cols=['prev_x', 'drv_lag1'], target_col='target')
    assert [r['method'] for r in rows] == ['random_walk', 'ridge']
    assert rows[0]['n'] > 0
