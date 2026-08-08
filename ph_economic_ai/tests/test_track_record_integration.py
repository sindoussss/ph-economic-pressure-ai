"""Every real run now files a phase-A/B row in the hash-chained track record,
not only in the mutable `trust.db`. These tests check the wiring between the
two stores, not the chain mechanics themselves (see test_track_record.py)."""
import pytest

from ph_economic_ai.engine.store import AgentTrustStore


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def test_track_record_defaults_next_to_the_db_it_was_given(tmp_path):
    """A store built on an isolated db_path must not fall back to the shared
    production cache file for its track record -- two isolated test stores
    writing into one real file would corrupt each other's hash chains."""
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    assert s.track_record.path.parent == tmp_path
    assert s.track_record.path.name == 'track_record.jsonl'


def test_save_run_files_a_prediction(store):
    run_id = store.save_run(
        scenario={'current_price': 98.82}, final_estimate=1.42, confidence_pct=78)
    rows = store.track_record.all_rows()
    preds = [r for r in rows if r['kind'] == 'prediction']
    assert len(preds) == 1
    assert preds[0]['run_id'] == str(run_id)
    assert preds[0]['predicted'] == pytest.approx(1.42)
    assert preds[0]['low'] <= 1.42 <= preds[0]['high']
    assert store.track_record.verify_chain() is True


def test_save_run_with_no_estimate_files_nothing(store):
    store.save_run(scenario={}, final_estimate=None, confidence_pct=0)
    assert store.track_record.all_rows() == []


def test_grading_files_a_matching_outcome(store):
    run_id = store.save_run(
        scenario={'current_price': 98.82}, final_estimate=1.42, confidence_pct=78)
    store.apply_ground_truth_grade(run_id, actual_change=1.20)

    rows = store.track_record.all_rows()
    outcomes = [r for r in rows if r['kind'] == 'outcome']
    assert len(outcomes) == 1
    assert outcomes[0]['run_id'] == str(run_id)
    assert outcomes[0]['actual'] == pytest.approx(1.20)
    assert outcomes[0]['error'] == pytest.approx(1.20 - 1.42)
    assert store.track_record.verify_chain() is True

    sc = store.track_record.scorecard()
    assert sc['n_matured'] == 1
    assert sc['mae'] == pytest.approx(abs(1.20 - 1.42))


def test_grading_twice_files_the_outcome_once(store):
    """The idempotency guard in apply_ground_truth_grade must also protect the
    chain: a second grade of the same run must not file a second outcome row
    for a run_id that already has one (record_outcome does not forbid it, the
    caller must not call it twice)."""
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=70)
    store.apply_ground_truth_grade(run_id, actual_change=1.05)
    store.apply_ground_truth_grade(run_id, actual_change=9.99)  # ignored, already graded
    outcomes = [r for r in store.track_record.all_rows() if r['kind'] == 'outcome']
    assert len(outcomes) == 1
    assert outcomes[0]['actual'] == pytest.approx(1.05)


def test_grading_a_pre_feature_run_does_not_crash(store):
    """A run graded through apply_ground_truth_grade but with no matching
    track_record prediction (e.g. a row from before this wiring existed) must
    still grade successfully in trust.db; the chain simply has nothing to say
    about it."""
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=70)
    # Simulate a pre-feature run: drop its prediction row from the chain file
    # directly, as if it had never been written.
    store.track_record.path.write_text('', encoding='utf-8')
    store.apply_ground_truth_grade(run_id, actual_change=1.05)  # must not raise
    assert store.track_record.all_rows() == []


def test_withdrawal_is_filed_and_excluded_from_the_scorecard(tmp_path):
    """RSK-023: a grade found to compare a run against the wrong pricing week
    is withdrawn, not deleted. The chain must record that too, and the
    scorecard must stop counting it."""
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    run_id = s.save_run(scenario={'current_price': 84.38}, final_estimate=-1.28,
                        confidence_pct=70, horizon_days=-1.0)
    s.apply_ground_truth_grade(run_id, actual_change=5.13)
    assert s.track_record.scorecard()['n_matured'] == 1

    import sqlite3
    con = sqlite3.connect(s._path)
    con.execute("UPDATE runs SET graded_against='2026-08-04T15:01:29+00:00 (gap 0.71d)' "
               "WHERE run_id=?", (run_id,))
    con.commit()
    con.close()

    withdrawn = s.withdraw_cross_cycle_grades()
    assert [w['run_id'] for w in withdrawn] == [run_id]

    rows = s.track_record.all_rows()
    assert any(r['kind'] == 'withdrawal' and r['run_id'] == str(run_id) for r in rows)
    assert s.track_record.verify_chain() is True
    # The prediction and outcome rows are still in the file, unedited -- only
    # a new row was appended, nothing rewritten.
    assert any(r['kind'] == 'prediction' and r['run_id'] == str(run_id) for r in rows)
    assert any(r['kind'] == 'outcome' and r['run_id'] == str(run_id) for r in rows)
    assert s.track_record.scorecard()['n_matured'] == 0
