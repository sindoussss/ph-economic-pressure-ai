"""The roster has to be able to recover, and the room has to stay a room.

Two defects with one symptom. Runs 21 through 28 recorded 18 agent responses
where run 20 recorded 32, and agreement fell from 94 percent to a 38 to 56 band.

1. Benching was permanent. Trust is an EMA that only moves when an agent
   produces a response, so a benched agent's score froze: benched because trust
   was low, trust stayed low because it was benched. Seven of twenty agents were
   stuck, all carrying `last_updated` of 2026-07-27T14:28 while the thirteen
   still running carried the current run's.
2. The bracket assumed five agents. With a group pruned to three, "remove 2 then
   remove 2" left the final round as one agent talking to itself, which produces
   one estimate and cannot be scored for agreement at all.

The benched seven included both remaining Critics and two ConfidenceScorers, the
roles whose output the elimination bracket reads.
"""
import math

import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.evolution import (
    _UNBENCHABLE_ROLES, get_evolved_swarm_agents,
)
from ph_economic_ai.engine.store import AgentTrustStore, trust_tier
from ph_economic_ai.engine.swarm import build_bracket, build_swarm_agents


@pytest.fixture
def store(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    for _ in range(3):                      # clear the cold-start window
        s.save_run({}, 1.0, 60)
    yield s
    s.close()


def _bury(store, name, times=8):
    """Drive an agent's trust below the 0.30 demotion threshold."""
    for _ in range(times):
        store.update_trust(name, internal_score=0.0, accuracy_score=0.0)


# ── Benching is temporary ─────────────────────────────────────────────────────

def test_a_benched_agent_recovers_toward_neutral(store):
    _bury(store, 'NCR Forecaster')
    buried = store.get_trust('NCR Forecaster')
    assert trust_tier(buried) == 'demoted'

    recovered = store.recover_benched('NCR Forecaster')
    assert recovered > buried
    assert recovered < 0.5, 'recovery is probation, not vindication'


def test_recovery_clears_the_demotion_threshold_in_a_few_runs(store):
    """The ratchet's real cost was that it never ended. Bounded is the point."""
    _bury(store, 'NCR Forecaster', times=20)
    runs = 0
    while trust_tier(store.get_trust('NCR Forecaster')) == 'demoted' and runs < 20:
        store.recover_benched('NCR Forecaster')
        runs += 1
    assert runs <= 5, f'took {runs} benched runs to become eligible again'


def test_recovery_does_not_credit_the_agent_with_a_run(store):
    _bury(store, 'NCR Forecaster')
    before = {r['agent_name']: r for r in store.get_all_trust_rows()}['NCR Forecaster']
    store.recover_benched('NCR Forecaster')
    after = {r['agent_name']: r for r in store.get_all_trust_rows()}['NCR Forecaster']
    assert after['runs_participated'] == before['runs_participated']
    assert after['avg_internal_score'] == before['avg_internal_score']


def test_recovering_an_unknown_agent_is_harmless(store):
    assert store.recover_benched('Nobody') == pytest.approx(0.5)


def test_benching_an_agent_starts_its_recovery(store):
    """The decay has to be wired to the bench, not merely available."""
    agents = build_swarm_agents()
    ncr = [a for a in agents if a.group_id == 0]
    for a in ncr:
        _bury(store, a.name)
    before = {a.name: store.get_trust(a.name) for a in ncr}

    evolved = get_evolved_swarm_agents(store, agents)
    active = {a.name for a in evolved if a.group_id == 0}
    benched = [a for a in ncr if a.name not in active]
    assert benched, 'the guard benched nobody, so this proves nothing'
    for a in benched:
        assert store.get_trust(a.name) > before[a.name]


def test_an_agent_that_ran_is_not_decayed(store):
    """Only absence decays. A low score earned in the room stands."""
    agents = build_swarm_agents()
    ncr = [a for a in agents if a.group_id == 0]
    for a in ncr:
        _bury(store, a.name)
    before = {a.name: store.get_trust(a.name) for a in ncr}

    evolved = get_evolved_swarm_agents(store, agents)
    for a in evolved:
        if a.group_id == 0:
            assert store.get_trust(a.name) == before[a.name]


# ── The scoring roles keep their seats ────────────────────────────────────────

def test_the_critic_and_confidence_scorer_are_never_benched(store):
    """They are the instruments. Without them every combined score is 0.5 and the
    bracket eliminates by hash tiebreak."""
    agents = build_swarm_agents()
    for a in agents:
        _bury(store, a.name)
    evolved = get_evolved_swarm_agents(store, agents)

    for group_id in {a.group_id for a in agents}:
        roles = {a.role for a in evolved if a.group_id == group_id}
        assert _UNBENCHABLE_ROLES <= roles, f'group {group_id} lost {_UNBENCHABLE_ROLES - roles}'


def test_a_protected_role_still_takes_its_demoted_tier_and_prompt(store):
    """Protected from benching is not protected from consequences."""
    agents = build_swarm_agents()
    for a in agents:
        _bury(store, a.name)
    evolved = get_evolved_swarm_agents(store, agents)
    critic = next(a for a in evolved if a.name == 'NCR Critic')
    assert 'conservative' in critic.system_prompt.lower()


def test_the_diversity_floor_still_benches_someone(store):
    """Protection must not become a way for nobody to ever be benched."""
    agents = build_swarm_agents()
    ncr = [a for a in agents if a.group_id == 0]
    for a in ncr:
        _bury(store, a.name)
    evolved = [a for a in get_evolved_swarm_agents(store, agents) if a.group_id == 0]
    assert len(evolved) < len(ncr)
    assert len(evolved) >= math.ceil(len(ncr) * 0.6)


# ── The final round stays a debate ────────────────────────────────────────────

@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_every_round_runs_with_more_than_one_agent(n):
    alive = n
    for _round_num, eliminate in build_bracket(n):
        assert alive >= 2, f'roster {n} reached a round of {alive}'
        alive -= eliminate
    assert alive == 1, f'roster {n} ended with {alive} winners'


@pytest.mark.parametrize('n', [3, 4, 5])
def test_the_final_round_keeps_a_scorable_quorum(n):
    """Below two estimates a group cannot be scored for agreement at all, which
    is how NCR dropped out of the run 28 headline."""
    rounds = build_bracket(n)
    alive = n
    for _round_num, eliminate in rounds[:-1]:
        alive -= eliminate
    assert alive >= min(n, swarm._FINAL_ROUND_AGENTS)


def test_the_full_roster_bracket_is_unchanged():
    """Five agents must still be remove 2, then remove 2."""
    assert build_bracket(5) == [(1, 2), (2, 2)]
    assert swarm._BRACKET == [(1, 2), (2, 2)]


def test_a_three_agent_group_no_longer_debates_alone():
    """The regression. Old bracket: 3 agents, remove 2, round 2 has one."""
    rounds = build_bracket(3)
    alive = 3
    sizes = []
    for _round_num, eliminate in rounds:
        sizes.append(alive)
        alive -= eliminate
    assert sizes == [3, 3]


def test_a_single_agent_group_does_not_crash():
    assert build_bracket(1) == [(1, 0)]


def test_the_ablation_override_is_still_honoured(monkeypatch):
    """`tools/swarm_ablation.py` reshapes `_BRACKET` to measure a one-round swarm."""
    monkeypatch.setattr(swarm, '_BRACKET', [(1, 4)])
    assert swarm._bracket_for([object()] * 5) == [(1, 4)]


def test_an_override_that_cannot_fit_the_roster_is_replaced():
    """Removing 4 from a group of 3 emptied the room and raised IndexError from a
    worker thread."""
    assert swarm._bracket_for([object()] * 3) == build_bracket(3)
