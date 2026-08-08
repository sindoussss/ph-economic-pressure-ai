import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication
from ph_economic_ai.engine.store import AgentTrustStore


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_empty_states(app, tmp_path):
    from ph_economic_ai.ui.learning_view import LearningView
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    v = LearningView(store)
    assert 'Run a simulation' in v._revisions_lbl.text()         # block 2 empty
    assert '0 runs logged' in v._track_lbl.text()                # block 4 status
    assert 'grading waits' in v._track_lbl.text()                # block 4 empty


def test_revisions_and_grade(app, tmp_path):
    from ph_economic_ai.ui.learning_view import LearningView
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    rid = store.save_run(scenario={'x': 1}, final_estimate=-1.8, confidence_pct=77)
    base = dict(statement='s', citation_count=1, has_causal_chain=0,
                internal_score=0.5, model_used='m')
    store.save_agent_responses(rid, [
        {'agent_name': 'FCST-NCR', 'round_num': 1, 'estimate': -1.2, **base},
        {'agent_name': 'FCST-NCR', 'round_num': 2, 'estimate': -1.8, **base},
    ])
    v = LearningView(store)
    v.refresh(rid)
    txt = v._revisions_lbl.text()
    assert 'FCST-NCR' in txt and 'R1' in txt and 'R2' in txt      # within-run revision
    store.apply_ground_truth_grade(rid, actual_change=-1.5)       # grade it
    v.refresh(rid)
    assert '1 graded' in v._track_lbl.text()                      # store-derived scorecard


def test_graded_count_is_not_capped_by_the_200_row_window(app, tmp_path):
    """RSK-029 shape, recurring through a different call: get_recent_runs(200)
    is a window, not a census. A run graded outside the newest 200 must still
    count -- the earlier fix (count_graded_runs, a real COUNT(*)) covered
    get_graded_errors()'s callers but never this screen."""
    from ph_economic_ai.ui.learning_view import LearningView
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))

    old_rid = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=50)
    store.apply_ground_truth_grade(old_rid, actual_change=1.1)   # graded, will age out
    for _ in range(205):                                          # push it past limit=200
        store.save_run(scenario={}, final_estimate=0.5, confidence_pct=50)

    v = LearningView(store)
    v.refresh()
    assert '1 graded' in v._track_lbl.text()
    assert '0 graded' not in v._track_lbl.text()
