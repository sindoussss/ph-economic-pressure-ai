import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai import ui  # noqa
from ph_economic_ai.ui import honesty


def test_honesty_constants():
    assert honesty.EXPLORATORY == 'exploratory'
    assert honesty.VALIDATED == 'validated'
    assert 'varies per run' in honesty.AGREEMENT_NOTE
    # the composed consensus note carries both signals
    assert 'exploratory' in honesty.consensus_note()
    assert 'varies per run' in honesty.consensus_note()
    assert 'exploratory' in honesty.interact_caption().lower()


def test_tile_narrow_marker_fires_below_collapse_distinct():
    assert honesty.tile_narrow_marker(3, 2) == 'narrow room'
    assert honesty.tile_narrow_marker(3, honesty.COLLAPSE_DISTINCT) == ''


def test_tile_narrow_marker_silent_with_no_estimates():
    assert honesty.tile_narrow_marker(0, 0) == ''


def test_chain_integrity_line_empty_log_is_silent(tmp_path):
    from ph_economic_ai.engine.track_record import TrackRecord
    tr = TrackRecord(tmp_path / 'log.jsonl')
    assert honesty.chain_integrity_line(tr) == ''


def test_chain_integrity_line_reports_unmatured_predictions(tmp_path):
    from ph_economic_ai.engine.track_record import TrackRecord
    tr = TrackRecord(tmp_path / 'log.jsonl')
    tr.record_prediction('2026-08', 1.0, -1.0, 3.0, 'test-model')
    line = honesty.chain_integrity_line(tr)
    assert 'verified' in line
    assert '1 locked-in prediction' in line
    assert 'none matured' in line


def test_chain_integrity_line_reports_scorecard_once_matured(tmp_path):
    from ph_economic_ai.engine.track_record import TrackRecord
    tr = TrackRecord(tmp_path / 'log.jsonl')
    rid = tr.record_prediction('2026-08', 1.0, -1.0, 3.0, 'test-model')
    tr.record_outcome(rid, actual=1.5)
    line = honesty.chain_integrity_line(tr)
    assert 'verified' in line
    assert '1 matured' in line
    assert '0.50' in line  # mean absolute error


def test_chain_integrity_line_flags_a_broken_chain(tmp_path):
    from ph_economic_ai.engine.track_record import TrackRecord
    path = tmp_path / 'log.jsonl'
    tr = TrackRecord(path)
    tr.record_prediction('2026-08', 1.0, -1.0, 3.0, 'test-model')
    tr.record_prediction('2026-09', 2.0, 0.0, 4.0, 'test-model')
    lines = path.read_text(encoding='utf-8').splitlines()
    lines[0] = lines[0].replace('"predicted": 1.0', '"predicted": 99.0')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    line = honesty.chain_integrity_line(TrackRecord(path))
    assert 'FAILED' in line
    assert 'do not trust' in line
