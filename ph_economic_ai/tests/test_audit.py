import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.targets import Target
from ph_economic_ai.benchmark.audit import verdict_from_panel, run_audit


def test_verdict_predictable_when_a_method_beats_rw():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'ridge', 'skill_vs_rw': 0.2, 'dm_p': 0.01},
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'predictable' and best['method'] == 'ridge'


def test_verdict_efficient_when_none_significantly_better():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'hgb', 'skill_vs_rw': -0.05, 'dm_p': 0.9},
        {'method': 'arima', 'skill_vs_rw': -0.2, 'dm_p': 0.01},
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'efficient' and best['method'] == 'random_walk'


def _predictable_target():
    idx = pd.date_range('2016-01', periods=80, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(7)
    drv = np.cumsum(rng.normal(0, 1, 80))
    y = np.r_[0, 0.9 * np.diff(drv)] + 50
    frame = pd.DataFrame({'prev_t': np.r_[y[0], y[:-1]],
                          'drv_lag1': np.r_[0, np.diff(drv)], 'target': y}, index=idx)
    return Target('synthetic', lambda: pd.Series(y, index=idx), lambda: frame)


def test_run_audit_reports_per_target_verdict():
    reg = {'synthetic': _predictable_target()}
    rows = run_audit(['synthetic'], min_train=24, registry=reg)
    assert rows[0]['target'] == 'synthetic'
    assert rows[0]['verdict'] in ('predictable', 'efficient')
    assert 'panel' in rows[0] and rows[0]['n'] > 0


def test_run_audit_insufficient_data():
    short = Target('short', lambda: pd.Series(dtype=float),
                   lambda: pd.DataFrame({'a': [1.0, 2.0], 'target': [1.0, 2.0]},
                                        index=['2020-01', '2020-02']))
    rows = run_audit(['short'], min_train=24, registry={'short': short})
    assert rows[0]['verdict'] == 'insufficient_data'
