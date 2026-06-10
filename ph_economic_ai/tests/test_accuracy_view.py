import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import pytest

pytest.importorskip('PyQt6')
from PyQt6.QtWidgets import QApplication

from ph_economic_ai.ui.accuracy_view import AccuracyView

_app = QApplication.instance() or QApplication(sys.argv)


def _report(tmp_path):
    rep = {
        'headline_skill_vs_random_walk': 0.12, 'n_months': 100,
        'date_range': ['2016-01', '2024-04'], 'horizon': '1_month',
        'model_metrics': {'mae': 1.1, 'rmse': 1.6, 'mape': 2.2, 'mase': 0.85},
        'baseline_metrics': {'random_walk': {'rmse': 1.82}},
        'skill': {'vs_random_walk': 0.12},
        'conformal_widths': {'0.9': 2.6},
        'calibration': [{'nominal': 0.9, 'qhat': 2.6, 'measured': 0.91}],
        'proxy_validation': {'pearson_r': 0.97, 'bias_mean': 0.3, 'mae': 1.0, 'n': 100},
        'data_hash': 'abc', 'limitations': ['food/elec are derived'],
        'ablation': [
            {'name': 'baseline', 'skill_vs_rw': -0.18, 'band90': 21.5, 'rmse': 4.73, 'mae': 3.54, 'n': 79},
            {'name': 'structural_hybrid', 'skill_vs_rw': 0.04, 'band90': 8.9, 'rmse': 3.8, 'mae': 2.9, 'n': 79},
        ],
        'selected_variant': 'structural_hybrid',
        'efficiency': [
            {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None, 'rmse': 4.0, 'mae': 3.1, 'dm_stat': 0.0, 'n': 79},
            {'method': 'hgb', 'skill_vs_rw': -0.18, 'dm_p': 0.21, 'rmse': 4.7, 'mae': 3.5, 'dm_stat': 1.2, 'n': 79},
        ],
        'passthrough': {'beta_total': 0.83, 'beta0': 0.6, 'beta1': 0.23, 'r2': 0.74, 'driver_acf1': 0.03, 'n': 96, 'alpha': 0.1},
        'audit': [
            {'target': 'fuel', 'verdict': 'efficient', 'best_method': 'random_walk', 'best_skill': 0.0, 'best_dm_p': None, 'n': 79},
            {'target': 'inflation', 'verdict': 'predictable', 'best_method': 'ridge', 'best_skill': 0.22, 'best_dm_p': 0.01, 'n': 60},
        ],
        'nowcast': {'verdict': 'beats_naive', 'best_method': 'ridge', 'best_skill': 0.18,
                    'best_dm_p': 0.02, 'n': 70},
        'nowcast_mom': {'verdict': 'beats_best_naive', 'best_method': 'arima',
                        'best_naive': 'random_walk', 'best_skill_vs_naive': 0.16,
                        'dm_p': 0.03, 'n': 61},
        'mom_driver_ablation': {'verdict': 'no_better_than_naive', 'driver_edge': False,
                                'best_method': 'random_walk', 'best_naive': 'random_walk',
                                'best_skill_vs_naive': 0.0, 'dm_p': None, 'n': 61},
        'mom_longsample': {'n_long': 143,
                           'mom': {'verdict': 'beats_best_naive', 'best_method': 'arima',
                                   'best_skill_vs_naive': 0.16, 'dm_p': 0.001},
                           'driver_ablation': {'verdict': 'no_better_than_naive',
                                               'driver_edge': False}},
        'transport_nowcast': {'n': 151, 'driver_edge': True, 'driver_edge_robust': False,
                              'mom': {'verdict': 'no_better_than_naive'},
                              'driver_ablation': {'verdict': 'beats_best_naive',
                                                  'driver_edge': True},
                              'robust': {'prelim_months_dropped': 6, 'n': 145,
                                         'driver_edge': False,
                                         'driver_ablation': {'verdict': 'no_better_than_naive'}}},
    }
    p = tmp_path / 'accuracy_report.json'
    p.write_text(json.dumps(rep), encoding='utf-8')
    return p


def test_view_shows_transport_nowcast_robust(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.transport_nowcast_summary()
    assert '151' in s
    assert 'driver_edge_robust=False' in s
    assert 'artifact' in s.lower()          # the full-sample True is flagged


def test_view_builds_and_shows_headline(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    text = view.headline_text()
    assert 'skill' in text.lower()
    assert '+0.12' in text or '0.12' in text


def test_view_handles_missing_report(tmp_path):
    view = AccuracyView(report_path=tmp_path / 'does_not_exist.json')
    # must not crash; shows an explanatory message
    assert 'run' in view.headline_text().lower()


def test_view_shows_ablation_when_present(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    t = view.ablation_summary()
    assert 'structural_hybrid' in t
    assert 'baseline' in t


def test_view_shows_efficiency_and_passthrough(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    eff = view.efficiency_summary()
    pt = view.passthrough_summary()
    assert 'random_walk' in eff and 'hgb' in eff
    assert 'pass-through' in pt.lower() or 'β' in pt or 'beta' in pt.lower()
    assert '0.83' in pt


def test_view_shows_audit(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    a = view.audit_summary()
    assert 'fuel' in a and 'inflation' in a
    assert 'efficient' in a and 'predictable' in a


def test_view_shows_nowcast(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.nowcast_summary()
    assert 'nowcast' in s.lower()
    assert 'ridge' in s
    assert 'beats' in s.lower()


def test_view_shows_nowcast_mom(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.nowcast_mom_summary()
    assert 'mom' in s.lower() or 'month-over-month' in s.lower()
    assert 'arima' in s
    assert 'random_walk' in s


def test_view_shows_mom_driver_ablation(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.mom_driver_ablation_summary()
    assert 'driver' in s.lower()
    assert 'random_walk' in s


def test_view_shows_mom_longsample(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    s = view.mom_longsample_summary()
    assert '143' in s
    assert 'arima' in s
