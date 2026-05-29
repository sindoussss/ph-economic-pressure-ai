import pytest
from unittest.mock import patch
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.ground_truth import (
    compute_accuracy_score,
    find_and_grade_runs,
)


@pytest.fixture
def store_with_run(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    run_id = s.save_run(
        scenario={'current_price': 98.82},
        final_estimate=1.42,
        confidence_pct=78,
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


def test_find_and_grade_runs_skips_recent(store_with_run):
    store, run_id = store_with_run
    # Run is just created — younger than 5 days
    graded = find_and_grade_runs(store, current_price=100.22, min_age_days=5.0)
    assert graded == 0


def test_find_and_grade_runs_grades_old_run(store_with_run):
    store, run_id = store_with_run
    # Grade with min_age_days=0 to bypass age check in tests
    graded = find_and_grade_runs(store, current_price=100.22, min_age_days=0.0)
    assert graded == 1
    # Confirm run is now graded
    ungraded = store.get_ungraded_runs(min_age_days=0.0)
    assert len(ungraded) == 0


def test_trust_improves_after_accurate_grade(store_with_run):
    store, _ = store_with_run
    trust_before = store.get_trust('Market Analyst')
    # actual_change = 100.22 - 98.82 = 1.40, estimate was 1.42, error ≈ ₱0.02
    find_and_grade_runs(store, current_price=100.22, min_age_days=0.0)
    trust_after = store.get_trust('Market Analyst')
    assert trust_after > trust_before


def test_find_and_grade_runs_skips_missing_current_price(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'fuel_type': 'diesel'}, final_estimate=1.0, confidence_pct=60)
    graded = find_and_grade_runs(s, current_price=100.0, min_age_days=0.0)
    assert graded == 0


def test_accuracy_score_symmetric():
    assert compute_accuracy_score(1.92, 1.42) == compute_accuracy_score(1.42, 1.92)
