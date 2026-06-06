import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json

from ph_economic_ai.benchmark.report import build_report, REQUIRED_KEYS


def test_build_report_has_required_keys():
    rep = build_report(
        date_range=('2016-01', '2024-12'), n_months=108,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}, 'seasonal_naive': {'rmse': 2.4}},
        skill={'vs_random_walk': 0.105, 'vs_seasonal_naive': 0.29},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 100},
        data_hash='abc123',
    )
    for key in REQUIRED_KEYS:
        assert key in rep, f'missing {key}'
    assert rep['headline_skill_vs_random_walk'] == 0.105


def test_report_roundtrips_to_json(tmp_path):
    rep = build_report(
        date_range=('2016-01', '2024-12'), n_months=108,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': 0.105},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 100},
        data_hash='abc123',
    )
    from ph_economic_ai.benchmark.report import write_report, load_report
    p = tmp_path / 'r.json'
    write_report(rep, p)
    assert load_report(p)['data_hash'] == 'abc123'
