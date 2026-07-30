"""An agent must not reason one way and sign the other.

Measured, not assumed. Direction agreement across a live run was 55 percent
against an unambiguous negative oil shock, and reading the statements showed why:

    NCR Forecaster: "consumer sees a reduction in retail gasoline prices"
                    ESTIMATE: +PHP4.50/L

    Central Luzon Critic: "lower wholesale costs ... formula implies a pump
                    change of -2." ESTIMATE: +PHP0.50/L

The agents agree in prose and disagree in sign. That is an EMISSION failure, not
an analytical one, and no amount of extra debate rounds addresses it — they
already agree.

It also scales past individual agents. A later run had Western Visayas and Davao
return +1.00 against a -3.28 anchor at 97 percent internal agreement: a whole
half of the country inverted, and confident enough that the metric read it as
consensus. High agreement on a wrong sign is more dangerous than low agreement,
which is why the judges carry this contract too.

The fix is a commitment, not a correction. The agent states a direction in words,
and when the sign disagrees it is asked which it meant. If it still contradicts
itself the estimate is refused rather than guessed.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupArena, GroupSurvivor, RegionalJudge, build_swarm_agents,
    direction_contradicts, parse_direction,
)

SCENARIO = {'oil_pct': -8.0, 'usd_pct': -0.1, 'bsp_rate': 6.5,
            'demand_index': 75.0, 'current_price': 84.38}


def _rag():
    rag = MagicMock()
    rag.query.return_value = []
    return rag


# ── Parsing ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('text,expected', [
    ('DIRECTION: DOWN', -1),
    ('DIRECTION: UP', 1),
    ('DIRECTION: FLAT', 0),
    ('**DIRECTION:** decrease', -1),
    ('direction: rising', 1),
    ('DIRECTION: unchanged', 0),
    ('no such line', None),
])
def test_direction_parses(text, expected):
    assert parse_direction(text) == expected


def test_the_last_direction_wins():
    """Matches `parse_fuel_estimate`: a model that restates its answer means the
    final one."""
    assert parse_direction('DIRECTION: UP\n...revised...\nDIRECTION: DOWN') == -1


# ── Contradiction ─────────────────────────────────────────────────────────────

def test_the_run_29_defect_is_caught():
    """Says DOWN, signs +4.50."""
    assert direction_contradicts(-1, 4.50) is True


@pytest.mark.parametrize('direction,estimate', [(-1, -2.0), (1, 2.0)])
def test_consistent_pairs_pass(direction, estimate):
    assert direction_contradicts(direction, estimate) is False


def test_flat_is_permissive():
    """An agent that says flat and writes -0.05 is being consistent. Only a sign
    flip counts, or every near-zero estimate becomes a false positive."""
    assert direction_contradicts(0, -0.05) is False
    assert direction_contradicts(0, 3.0) is False


@pytest.mark.parametrize('direction,estimate', [(None, 4.5), (-1, None)])
def test_a_missing_half_is_not_a_contradiction(direction, estimate):
    """Nothing to contradict. Treating absence as conflict would refuse
    estimates from every agent that simply did not print the line."""
    assert direction_contradicts(direction, estimate) is False


# ── The prompt asks for it ────────────────────────────────────────────────────

def test_every_agent_role_is_asked_for_a_direction():
    for agent in build_swarm_agents():
        assert 'DIRECTION:' in agent.system_prompt, agent.role


def test_the_agent_prompt_requires_the_two_to_agree():
    """Asserts the CONTRACT, not the wording.

    This pinned the literal menu "DIRECTION: UP or DIRECTION: DOWN or DIRECTION:
    FLAT", which is a template a small model copies back — five of twenty agents
    did exactly that on a live run. A test that freezes the exact string a fix
    needs to change turns a defect into a passing assertion, so it now checks
    that the prompt asks for a direction, offers the three values, and requires
    them to agree with the estimate.
    """
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    arena = GroupArena(group_id=0, agents=agents, rag=_rag(), scenario=SCENARIO)
    prompt = ' '.join(m['content'] for m in arena._build_prompt(agents[0], 1, []))
    assert 'DIRECTION:' in prompt
    for value in ('UP', 'DOWN', 'FLAT'):
        assert value in prompt, value
    assert 'must agree' in prompt
    assert 'or DIRECTION:' not in prompt, 'the menu is back'


# ── Resolution ────────────────────────────────────────────────────────────────

class _Replies:
    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return [self._replies.pop(0) if self._replies else 'DIRECTION: DOWN\nESTIMATE: -₱1.00/L']


def _arena():
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    return GroupArena(group_id=0, agents=agents, rag=_rag(), scenario=SCENARIO)


def test_a_contradicting_agent_is_asked_which_it_meant():
    arena = _arena()
    stream = _Replies('DIRECTION: DOWN\nESTIMATE: +₱4.50/L',
                      'DIRECTION: DOWN\nESTIMATE: -₱2.10/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 2
    assert resp.price_estimate == pytest.approx(-2.10)


def test_a_consistent_agent_is_not_re_asked():
    """The check is a failure path. It must not double the bill on a good run."""
    arena = _arena()
    stream = _Replies('DIRECTION: DOWN\nESTIMATE: -₱2.10/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 1
    assert resp.price_estimate == pytest.approx(-2.10)


def test_an_unresolved_contradiction_refuses_the_estimate():
    """Refusing costs a data point. Keeping a contradicted number costs the
    metric its meaning: a sign flip is the difference between prices rising and
    falling, and a region can agree on it at 97 percent while inverted."""
    arena = _arena()
    stream = _Replies('DIRECTION: DOWN\nESTIMATE: +₱4.50/L',
                      'DIRECTION: DOWN\nESTIMATE: +₱4.50/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert resp.price_estimate is None


def test_an_agent_that_states_no_direction_is_left_alone():
    """The contract is new. An agent that ignores it must not lose its estimate
    for that alone, or the population collapses on the first run."""
    arena = _arena()
    stream = _Replies('ESTIMATE: +₱4.50/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 1
    assert resp.price_estimate == pytest.approx(4.50)


def test_a_failed_retry_does_not_keep_the_contradicted_number():
    def _boom(messages, **kwargs):
        if any(m['role'] == 'assistant' for m in messages):
            raise RuntimeError('provider down')
        return ['DIRECTION: DOWN\nESTIMATE: +₱4.50/L']

    arena = _arena()
    with patch('ph_economic_ai.engine.swarm.llm.stream', _boom):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert resp.price_estimate is None


# ── The judges carry it too ───────────────────────────────────────────────────

def _survivor(region, estimate):
    return GroupSurvivor(
        group_id=0, region_name=region,
        response=AgentResponse(f'{region} Forecaster', 1, '', 'x', estimate),
        combined_score=0.8, agent_role='Forecaster', agent_model='fast',
    )


def _judge():
    return RegionalJudge(
        judge_id=0,
        survivors=(_survivor('Western Visayas', -2.0),
                   _survivor('Davao Region', -2.2)),
        rag=_rag(), scenario=SCENARIO, agent_estimates=[-2.0, -2.2, -1.9],
        anchor=-3.28,
    )


def test_a_contradicting_judge_is_re_asked():
    """The +1.00 verdict at 97 percent agreement came from a judge, not an
    agent, so the contract has to reach this layer."""
    stream = _Replies('DIRECTION: DOWN\nESTIMATE: -₱2.00/L',
                      'DIRECTION: DOWN\nESTIMATE: -₱2.20/L',
                      'DIRECTION: DOWN\nESTIMATE: +₱1.00/L',
                      'DIRECTION: DOWN\nESTIMATE: -₱2.60/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        verdict = _judge().run()
    assert len(stream.calls) == 4          # 2 defences, synthesis, one resolution
    assert verdict.estimate == pytest.approx(-2.60)


def test_a_consistent_judge_is_not_re_asked():
    stream = _Replies('DIRECTION: DOWN\nESTIMATE: -₱2.00/L',
                      'DIRECTION: DOWN\nESTIMATE: -₱2.20/L',
                      'DIRECTION: DOWN\nESTIMATE: -₱2.10/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        verdict = _judge().run()
    assert len(stream.calls) == 3
    assert verdict.estimate == pytest.approx(-2.10)
