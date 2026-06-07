import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.features import build_feature_frame
from ph_economic_ai.benchmark.ablation import run_variant, run_ablation, select_winner


def _frame(n=60):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(1)
    gas = 50 + np.cumsum(rng.normal(0, 0.4, n))
    df = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)
    return build_feature_frame(df)


def _mean_predict_fn(X_train, y_train, x_next):
    return float(np.mean(y_train[-3:]))


def test_run_variant_returns_metrics():
    row = run_variant('baseline', _frame(), _mean_predict_fn, min_train=12)
    assert set(row) >= {'name', 'rmse', 'skill_vs_rw', 'mae', 'band90', 'n'}
    assert row['name'] == 'baseline'
    assert row['n'] > 0


def test_structural_reconstruction_scores_in_ron95_space():
    row = run_variant('structural_hybrid', _frame(), _mean_predict_fn, min_train=12)
    assert row['rmse'] >= 0.0
    assert np.isfinite(row['skill_vs_rw'])


def test_run_ablation_one_row_per_variant():
    rows = run_ablation(_frame(), ['baseline', 'drop_demand', 'finished_gas'],
                        _mean_predict_fn, min_train=12)
    assert [r['name'] for r in rows] == ['baseline', 'drop_demand', 'finished_gas']


def test_select_winner_prefers_higher_skill_then_narrower_band():
    rows = [
        {'name': 'a', 'skill_vs_rw': -0.10, 'band90': 8.0},
        {'name': 'b', 'skill_vs_rw': 0.05, 'band90': 7.0},
        {'name': 'c', 'skill_vs_rw': 0.05, 'band90': 5.0},
    ]
    assert select_winner(rows)['name'] == 'c'


def test_select_winner_handles_all_negative():
    rows = [{'name': 'a', 'skill_vs_rw': -0.3, 'band90': 9.0},
            {'name': 'b', 'skill_vs_rw': -0.1, 'band90': 9.0}]
    assert select_winner(rows)['name'] == 'b'
