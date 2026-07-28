import pytest
from unittest.mock import patch
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.ground_truth import (
    compute_accuracy_score,
    find_and_grade_runs,
)


@pytest.fixture
def store_with_run(tmp_path):
    """A run whose forecast period has already elapsed.

    `horizon_days=-1` puts the target date one day in the PAST, which is what makes
    the run due for grading. Since RSK-018 a run is graded when its period is over,
    not merely when the row is old, so a fixture that only aged the row would never
    be gradable.
    """
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    run_id = s.save_run(
        scenario={'current_price': 98.82},
        final_estimate=1.42,
        confidence_pct=78,
        horizon_days=-1.0,
    )
    s.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.42,
         'statement': 'Rising.', 'citation_count': 1, 'has_causal_chain': 1,
         'internal_score': 0.7, 'model_used': 'deepseek-r1:8b'},
    ])
    return s, run_id


def test_accuracy_score_perfect():
    assert compute_accuracy_score(estimate=1.42, actual=1.42) == 1.0


def test_accuracy_score_half_php_error():
    score = compute_accuracy_score(estimate=1.92, actual=1.42)
    assert abs(score - (1 - 0.5 / 3.0)) < 0.001


def test_accuracy_score_three_php_error():
    score = compute_accuracy_score(estimate=4.42, actual=1.42)
    assert score == 0.0


def test_find_and_grade_runs_skips_a_run_whose_period_has_not_elapsed(tmp_path):
    """A forecast for next week cannot be graded this week."""
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'current_price': 98.82}, final_estimate=1.42,
               confidence_pct=78, horizon_days=7.0)
    assert find_and_grade_runs(s, current_price=100.22) == 0


def test_find_and_grade_runs_grades_a_run_whose_period_has_elapsed(store_with_run):
    store, run_id = store_with_run
    assert find_and_grade_runs(store, current_price=100.22) == 1
    assert store.get_ungraded_runs(min_age_days=0.0) == []


def test_the_grade_records_which_observation_it_used(store_with_run):
    """A grade that cannot be traced to an observation cannot be audited."""
    store, run_id = store_with_run
    find_and_grade_runs(store, current_price=100.22)
    row = store.get_run(run_id) if hasattr(store, 'get_run') else None
    if row is None:
        import sqlite3
        conn = sqlite3.connect(store._path)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone())
    assert row['graded_against'], 'no observation recorded for this grade'
    assert 'gap' in row['graded_against']


def test_a_run_with_no_price_near_its_period_stays_ungraded(tmp_path):
    """The heart of RSK-018. A run whose week has no observation must NOT be
    scored against a price from a different week."""
    from datetime import datetime, timedelta, timezone
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    # Target date sits 60 days in the past; the only observation is today's.
    s.save_run(scenario={'current_price': 98.82}, final_estimate=1.42,
               confidence_pct=78, horizon_days=-60.0)
    assert find_and_grade_runs(s, current_price=100.22) == 0
    assert len(s.get_ungraded_runs(min_age_days=0.0)) == 1

    # Supply an observation for that run's actual period and it becomes gradable.
    s.record_price_observation(
        99.10, observed_at=datetime.now(timezone.utc) - timedelta(days=60))
    assert find_and_grade_runs(s, current_price=100.22) == 1


def test_trust_improves_after_accurate_grade(store_with_run):
    store, run_id = store_with_run
    trust_before = store.get_trust('Market Analyst')
    # actual_change = 100.22 - 98.82 = 1.40, estimate was 1.42, error ≈ ₱0.02.
    # Grade the run directly — this isolates "accurate grade -> trust rises" and
    # removes the wall-clock age filter (the source of the past intermittent
    # failure; see test_find_and_grade_runs_grades_old_run). find_and_grade_runs'
    # age behaviour is covered by the skips_recent / grades_old_run tests.
    store.apply_ground_truth_grade(run_id, actual_change=100.22 - 98.82)
    trust_after = store.get_trust('Market Analyst')
    assert trust_after > trust_before


def test_find_and_grade_runs_skips_missing_current_price(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'fuel_type': 'diesel'}, final_estimate=1.0, confidence_pct=60)
    graded = find_and_grade_runs(s, current_price=100.0)
    assert graded == 0


def test_accuracy_score_symmetric():
    assert compute_accuracy_score(1.92, 1.42) == compute_accuracy_score(1.42, 1.92)
