import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.engine.track_record import TrackRecord


def test_predict_then_grade_two_phases(tmp_path):
    tr = TrackRecord(tmp_path / 'log.jsonl')
    rid = tr.record_prediction(target_month='2026-07', predicted=64.0,
                               low=62.0, high=66.0, model_version='hgb-1')
    # outcome arrives later, as a separate row
    tr.record_outcome(rid, actual=64.8)
    rows = tr.all_rows()
    assert rows[0]['kind'] == 'prediction' and rows[0]['target_month'] == '2026-07'
    assert rows[1]['kind'] == 'outcome'
    assert rows[1]['error'] == pytest.approx(64.8 - 64.0)


def test_chain_verifies_when_untouched(tmp_path):
    tr = TrackRecord(tmp_path / 'log.jsonl')
    tr.record_prediction('2026-07', 64.0, 62.0, 66.0, 'hgb-1')
    tr.record_prediction('2026-08', 65.0, 63.0, 67.0, 'hgb-1')
    assert tr.verify_chain() is True


def test_chain_detects_tampering(tmp_path):
    path = tmp_path / 'log.jsonl'
    tr = TrackRecord(path)
    tr.record_prediction('2026-07', 64.0, 62.0, 66.0, 'hgb-1')
    tr.record_prediction('2026-08', 65.0, 63.0, 67.0, 'hgb-1')
    # tamper: rewrite the first row's predicted value
    lines = path.read_text(encoding='utf-8').splitlines()
    lines[0] = lines[0].replace('64.0', '99.0')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert TrackRecord(path).verify_chain() is False


def test_scorecard_over_matured_predictions(tmp_path):
    tr = TrackRecord(tmp_path / 'log.jsonl')
    r1 = tr.record_prediction('2026-07', 64.0, 62.0, 66.0, 'hgb-1')
    tr.record_outcome(r1, actual=64.5)       # inside band, error 0.5
    r2 = tr.record_prediction('2026-08', 65.0, 63.0, 67.0, 'hgb-1')
    tr.record_outcome(r2, actual=70.0)       # outside band, error 5.0
    sc = tr.scorecard()
    assert sc['n_matured'] == 2
    assert sc['mae'] == pytest.approx((0.5 + 5.0) / 2)
    assert sc['coverage_90'] == pytest.approx(0.5)
