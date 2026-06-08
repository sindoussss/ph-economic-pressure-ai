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


def test_report_includes_ablation_and_selected():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': 0.05},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        ablation=[{'name': 'baseline', 'skill_vs_rw': -0.18, 'band90': 21.5, 'rmse': 4.7, 'mae': 3.5, 'n': 79},
                  {'name': 'structural_hybrid', 'skill_vs_rw': 0.05, 'band90': 9.0, 'rmse': 1.7, 'mae': 1.2, 'n': 79}],
        selected_variant='structural_hybrid',
    )
    assert rep['selected_variant'] == 'structural_hybrid'
    assert len(rep['ablation']) == 2
    assert 'ablation' in REQUIRED_KEYS and 'selected_variant' in REQUIRED_KEYS


def test_report_includes_efficiency_and_passthrough():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        efficiency=[{'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None, 'rmse': 1.9, 'mae': 1.5, 'dm_stat': 0.0, 'n': 79},
                    {'method': 'hgb', 'skill_vs_rw': -0.18, 'dm_p': 0.21, 'rmse': 2.2, 'mae': 1.8, 'dm_stat': 1.2, 'n': 79}],
        passthrough={'beta_total': 0.83, 'beta0': 0.6, 'beta1': 0.23, 'r2': 0.74, 'driver_acf1': 0.03, 'n': 96, 'alpha': 0.1},
    )
    assert len(rep['efficiency']) == 2
    assert rep['passthrough']['beta_total'] == 0.83
    assert 'efficiency' in REQUIRED_KEYS and 'passthrough' in REQUIRED_KEYS


def test_report_includes_audit():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        audit=[{'target': 'fuel', 'verdict': 'efficient', 'best_method': 'random_walk',
                'best_skill': 0.0, 'best_dm_p': None, 'n': 79}],
    )
    assert rep['audit'][0]['verdict'] == 'efficient'
    assert 'audit' in REQUIRED_KEYS


def test_report_includes_nowcast():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        nowcast={'verdict': 'beats_naive', 'best_method': 'ridge', 'best_skill': 0.18,
                 'best_dm_p': 0.02, 'n': 70},
    )
    assert rep['nowcast']['verdict'] == 'beats_naive'
    assert 'nowcast' in REQUIRED_KEYS


def test_report_includes_nowcast_mom():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        nowcast_mom={'verdict': 'beats_best_naive', 'best_method': 'ridge',
                     'best_naive': 'seasonal_naive', 'best_skill_vs_naive': 0.15,
                     'dm_p': 0.03, 'n': 70},
    )
    assert rep['nowcast_mom']['verdict'] == 'beats_best_naive'
    assert 'nowcast_mom' in REQUIRED_KEYS


def test_report_includes_mom_driver_ablation():
    rep = build_report(
        date_range=('2017-03', '2025-03'), n_months=79,
        model_metrics={'mae': 1.2, 'rmse': 1.7, 'mape': 2.5, 'mase': 0.9},
        baseline_metrics={'random_walk': {'rmse': 1.9}},
        skill={'vs_random_walk': -0.01},
        calibration=[{'nominal': 0.9, 'qhat': 2.8, 'measured': 0.91}],
        proxy={'pearson_r': 0.97, 'bias_mean': 0.4, 'mae': 1.1, 'n': 79},
        data_hash='abc123',
        mom_driver_ablation={'verdict': 'no_better_than_naive', 'driver_edge': False,
                             'best_method': 'random_walk', 'best_naive': 'random_walk',
                             'best_skill_vs_naive': 0.0, 'dm_p': None, 'n': 61},
    )
    assert rep['mom_driver_ablation']['driver_edge'] is False
    assert 'mom_driver_ablation' in REQUIRED_KEYS
