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
    }
    p = tmp_path / 'accuracy_report.json'
    p.write_text(json.dumps(rep), encoding='utf-8')
    return p


def test_view_builds_and_shows_headline(tmp_path):
    view = AccuracyView(report_path=_report(tmp_path))
    text = view.headline_text()
    assert 'skill' in text.lower()
    assert '+0.12' in text or '0.12' in text


def test_view_handles_missing_report(tmp_path):
    view = AccuracyView(report_path=tmp_path / 'does_not_exist.json')
    # must not crash; shows an explanatory message
    assert 'run' in view.headline_text().lower()
