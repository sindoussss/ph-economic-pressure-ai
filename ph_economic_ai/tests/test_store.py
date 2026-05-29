import pytest
from ph_economic_ai.engine.store import AgentTrustStore


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def test_save_and_get_run(store):
    run_id = store.save_run(
        scenario={'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72},
        final_estimate=1.42,
        confidence_pct=78,
    )
    assert run_id == 1
    runs = store.get_ungraded_runs(min_age_days=0)
    assert len(runs) == 1
    assert runs[0]['final_estimate'] == 1.42


def test_total_runs(store):
    assert store.total_runs() == 0
    store.save_run(scenario={}, final_estimate=1.0, confidence_pct=60)
    assert store.total_runs() == 1


def test_save_agent_responses(store):
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=60)
    store.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.2,
         'statement': 'Brent at $72.40 supports a ₱1.20 rise.', 'citation_count': 2,
         'has_causal_chain': 1, 'internal_score': 0.8, 'model_used': 'deepseek-r1:8b'},
    ])
    rows = store.get_agent_responses(run_id)
    assert len(rows) == 1
    assert rows[0]['agent_name'] == 'Market Analyst'


def test_trust_initialized_at_half(store):
    trust = store.get_trust('Market Analyst')
    assert trust == 0.5


def test_update_trust_internal_only(store):
    store.update_trust('Market Analyst', internal_score=0.9)
    trust = store.get_trust('Market Analyst')
    # EMA: 0.3 * 0.9 + 0.7 * 0.5 = 0.27 + 0.35 = 0.62
    assert abs(trust - 0.62) < 0.001


def test_update_trust_with_accuracy(store):
    store.update_trust('Market Analyst', internal_score=0.8, accuracy_score=1.0)
    trust = store.get_trust('Market Analyst')
    # raw = 0.4*0.8 + 0.6*1.0 = 0.32 + 0.60 = 0.92
    # EMA: 0.3 * 0.92 + 0.7 * 0.5 = 0.276 + 0.35 = 0.626
    assert abs(trust - 0.626) < 0.001


def test_trust_clamped(store):
    for _ in range(20):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    assert store.get_trust('Market Analyst') <= 0.95


def test_get_all_trust(store):
    store.update_trust('Agent A', internal_score=0.8)
    store.update_trust('Agent B', internal_score=0.2)
    all_trust = store.get_all_trust()
    assert 'Agent A' in all_trust
    assert 'Agent B' in all_trust


def test_apply_ground_truth_grade(store):
    run_id = store.save_run(scenario={'current_price': 98.82}, final_estimate=1.42, confidence_pct=78)
    store.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.42,
         'statement': 'Estimate.', 'citation_count': 1, 'has_causal_chain': 1,
         'internal_score': 0.7, 'model_used': 'deepseek-r1:8b'},
    ])
    store.apply_ground_truth_grade(run_id, actual_change=1.20)
    runs = store.get_ungraded_runs(min_age_days=0)
    assert len(runs) == 0  # run is now graded
    trust = store.get_trust('Market Analyst')
    assert trust > 0.5  # accurate prediction improved trust


def test_trust_floor_clamp(store):
    """Pushing trust down with 20 zero-score iterations must not drop below 0.05."""
    for _ in range(20):
        store.update_trust('Low Agent', internal_score=0.0, accuracy_score=0.0)
    assert store.get_trust('Low Agent') >= 0.05


def test_update_run_quality(store):
    """update_run_quality should persist internal_quality to the runs table."""
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=60)
    store.update_run_quality(run_id, internal_quality=0.77)
    # Verify via get_ungraded_runs (all columns returned)
    runs = store.get_ungraded_runs(min_age_days=0)
    assert len(runs) == 1
    assert abs(runs[0]['internal_quality'] - 0.77) < 0.001


def test_get_all_trust_rows_ordered(store):
    """get_all_trust_rows should return rows sorted by trust_score DESC."""
    store.update_trust('Agent Low', internal_score=0.1)
    store.update_trust('Agent High', internal_score=0.9)
    rows = store.get_all_trust_rows()
    assert len(rows) == 2
    assert rows[0]['trust_score'] >= rows[1]['trust_score']
    assert rows[0]['agent_name'] == 'Agent High'


def test_apply_ground_truth_grade_idempotent(store):
    """Calling apply_ground_truth_grade twice must not move trust on the second call."""
    run_id = store.save_run(scenario={}, final_estimate=1.0, confidence_pct=70)
    store.save_agent_responses(run_id, [
        {'agent_name': 'Idempotent Agent', 'round_num': 1, 'estimate': 1.0,
         'statement': 'Test.', 'citation_count': 0, 'has_causal_chain': 0,
         'internal_score': 0.6, 'model_used': 'test-model'},
    ])
    store.apply_ground_truth_grade(run_id, actual_change=1.05)
    trust_after_first = store.get_trust('Idempotent Agent')
    store.apply_ground_truth_grade(run_id, actual_change=1.05)
    trust_after_second = store.get_trust('Idempotent Agent')
    assert trust_after_first == trust_after_second
