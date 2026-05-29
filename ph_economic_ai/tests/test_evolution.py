import math
import pytest
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.evolution import get_evolved_debate_agents, get_evolved_swarm_agents
from ph_economic_ai.engine.debate import DEFAULT_AGENTS, Agent
from ph_economic_ai.engine.swarm import build_swarm_agents


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def test_cold_start_returns_base_agents(store):
    # Fewer than 3 runs → no evolution
    for _ in range(2):
        store.save_run({}, 1.0, 60)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    assert len(evolved) == len(DEFAULT_AGENTS)
    for orig, ev in zip(DEFAULT_AGENTS, evolved):
        assert ev.model == orig.model


def test_promoted_agent_gets_bigger_model(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Push Market Analyst above 0.70 trust
    for _ in range(8):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    market_analyst = next(a for a in evolved if a.name == 'Market Analyst')
    assert market_analyst.model == 'deepseek-r1:32b'


def test_demoted_agent_gets_smaller_model(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Push Risk Assessor below 0.30 trust
    for _ in range(8):
        store.update_trust('Risk Assessor', internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    risk_assessor = next(a for a in evolved if a.name == 'Risk Assessor')
    assert risk_assessor.model == 'qwen2.5:7b'


def test_promoted_agent_gets_confidence_suffix(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    for _ in range(8):
        store.update_trust('Market Analyst', internal_score=1.0, accuracy_score=1.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    market_analyst = next(a for a in evolved if a.name == 'Market Analyst')
    assert 'accurate' in market_analyst.system_prompt.lower()


def test_demoted_agent_gets_skeptic_suffix(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    for _ in range(8):
        store.update_trust('Risk Assessor', internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    risk_assessor = next(a for a in evolved if a.name == 'Risk Assessor')
    assert 'conservative' in risk_assessor.system_prompt.lower()


def test_diversity_guard_prevents_all_benched(store):
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    agents = build_swarm_agents()
    # Demote all agents in group 0 (NCR)
    ncr_agents = [a for a in agents if a.group_id == 0]
    for a in ncr_agents:
        for _ in range(8):
            store.update_trust(a.name, internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_swarm_agents(store, agents)
    ncr_evolved = [a for a in evolved if a.group_id == 0]
    # At least 60% of original NCR count must survive
    assert len(ncr_evolved) >= math.ceil(len(ncr_agents) * 0.6)


def test_swarm_cold_start(store):
    agents = build_swarm_agents()
    evolved = get_evolved_swarm_agents(store, agents)
    assert len(evolved) == len(agents)
    for orig, ev in zip(agents, evolved):
        assert ev.model == orig.model


def test_default_trust_no_evolution(store):
    """All agents at default trust (0.5) — no model changes, no prompt suffixes."""
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Don't update any trust — all agents stay at 0.5 default
    evolved = get_evolved_debate_agents(store, DEFAULT_AGENTS)
    for orig, ev in zip(DEFAULT_AGENTS, evolved):
        assert ev.model == orig.model
        assert ev.system_prompt == orig.system_prompt


def test_unknown_model_unchanged(store):
    """Agent with model not in _DEBATE_TIERS passes through unchanged."""
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    # Push a fake agent with unknown model to promoted tier
    store.update_trust('Unknown Agent', internal_score=1.0)
    for _ in range(8):
        store.update_trust('Unknown Agent', internal_score=1.0, accuracy_score=1.0)
    fake_agent = Agent(
        name='Unknown Agent', role='Test', model='llama3:8b',
        system_prompt='Test prompt.', rag_sources=[], is_mini=False,
    )
    evolved = get_evolved_debate_agents(store, [fake_agent])
    assert evolved[0].model == 'llama3:8b'  # unchanged — not in tier map


def test_swarm_diversity_bench_actually_occurs(store):
    """With enough agents demoted, some are actually benched (not just kept)."""
    for _ in range(3):
        store.save_run({}, 1.0, 60)
    agents = build_swarm_agents()
    ncr_agents = [a for a in agents if a.group_id == 0]
    # Demote all NCR agents
    for a in ncr_agents:
        for _ in range(8):
            store.update_trust(a.name, internal_score=0.0, accuracy_score=0.0)
    evolved = get_evolved_swarm_agents(store, agents)
    ncr_evolved = [a for a in evolved if a.group_id == 0]
    # Some should be benched (not all survive)
    if len(ncr_agents) > math.ceil(len(ncr_agents) * 0.6):
        assert len(ncr_evolved) < len(ncr_agents)  # at least one was benched
