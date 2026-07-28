"""A response with no ESTIMATE line gets one follow-up before it is written off.

Run 28 lost 5 of 18 agent responses to an unparseable estimate, 28 percent. That
is not a quiet loss: the response scores 0.0 in `compute_combined_score`, drops
out of the agreement population, and when it belongs to the group's survivor it
removes that whole region from the master verdict. The missing NCR number is
exactly what dropped a region pair out of the run 28 headline.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupArena, GroupSurvivor, RegionalJudge, _reask_for_estimate,
    build_swarm_agents,
)

SCENARIO = {'oil_pct': -2.0, 'usd_pct': 0.0, 'current_price': 98.82,
            'bsp_rate': 6.5, 'demand_index': 72.0}
NO_NUMBER = 'Prices look likely to ease somewhat next week.'
WITH_NUMBER = 'ESTIMATE: -₱1.00/L'


def _rag():
    rag = MagicMock()
    rag.query.return_value = []
    return rag


def _arena():
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    return GroupArena(group_id=0, agents=agents, rag=_rag(), scenario=SCENARIO)


class _Replies:
    """Returns each scripted reply in turn, recording what it was asked."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        reply = self._replies.pop(0) if self._replies else WITH_NUMBER
        return [reply]


# ── The agent path ────────────────────────────────────────────────────────────

def test_an_agent_without_a_number_is_asked_again():
    arena = _arena()
    stream = _Replies(NO_NUMBER, WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 2
    assert resp.price_estimate == pytest.approx(-1.00)


def test_an_agent_that_answered_is_not_asked_again():
    """The retry is a failure path. It must not double the bill on a good run."""
    arena = _arena()
    stream = _Replies(WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 1


def test_the_retry_asks_only_for_the_missing_line():
    arena = _arena()
    stream = _Replies(NO_NUMBER, WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    followup = stream.calls[1]
    assert followup[-2]['role'] == 'assistant'
    assert followup[-2]['content'] == NO_NUMBER      # its own answer, for context
    assert 'Do not repeat your analysis' in followup[-1]['content']


def test_the_original_reasoning_survives_the_retry():
    """The Critic reads the statement and the report shows it. Replacing it with
    a bare number would trade a missing estimate for a missing argument."""
    arena = _arena()
    stream = _Replies(NO_NUMBER, WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert NO_NUMBER in resp.statement
    assert 'ESTIMATE' in resp.statement


def test_two_failures_give_up_rather_than_loop():
    arena = _arena()
    stream = _Replies(NO_NUMBER, NO_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 2
    assert resp.price_estimate is None
    assert resp.statement == NO_NUMBER      # nothing usable was appended


def test_a_retry_that_raises_keeps_the_original_answer():
    def _boom(messages, **kwargs):
        if any(m['role'] == 'assistant' for m in messages):
            raise RuntimeError('provider down')
        return [NO_NUMBER]

    arena = _arena()
    with patch('ph_economic_ai.engine.swarm.llm.stream', _boom):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert resp.statement == NO_NUMBER
    assert resp.price_estimate is None


def test_the_retry_uses_a_different_seed_than_the_first_ask():
    """Same seed, same prompt prefix, same answer. Retrying would be theatre."""
    seeds = []

    def _capture(messages, **kwargs):
        seeds.append(kwargs.get('seed'))
        return [NO_NUMBER if len(seeds) == 1 else WITH_NUMBER]

    arena = _arena()
    with patch('ph_economic_ai.engine.swarm.llm.stream', _capture):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert seeds[0] != seeds[1]


def test_the_retry_seed_is_reproducible():
    """ADR-002 still holds: the same run retries to the same number."""
    def _run():
        seeds = []

        def _capture(messages, **kwargs):
            seeds.append(kwargs.get('seed'))
            return [NO_NUMBER if len(seeds) == 1 else WITH_NUMBER]

        arena = _arena()
        with patch('ph_economic_ai.engine.swarm.llm.stream', _capture):
            arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
        return seeds

    assert _run() == _run()


# ── The regional judge path ───────────────────────────────────────────────────

def _survivor(region, estimate):
    return GroupSurvivor(
        group_id=0, region_name=region,
        response=AgentResponse(f'{region} Forecaster', 1, '', 'x', estimate),
        combined_score=0.8, agent_role='Forecaster', agent_model='fast',
    )


def _judge():
    return RegionalJudge(
        judge_id=0,
        survivors=(_survivor('NCR', -0.5), _survivor('Central Luzon', -0.6)),
        rag=_rag(), scenario=SCENARIO, agent_estimates=[-0.5, -0.6, -0.55],
    )


def test_a_judge_with_no_number_is_asked_again():
    """Three deep-tier calls end in "no estimate" on the card otherwise."""
    stream = _Replies(WITH_NUMBER, WITH_NUMBER, NO_NUMBER, WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        verdict = _judge().run()
    assert len(stream.calls) == 4          # 2 defences, synthesis, one retry
    assert verdict.estimate == pytest.approx(-1.00)


def test_a_judge_whose_number_was_rejected_is_not_asked_again():
    """`rejected` means it gave a number and the plausibility guard threw it out.
    That is a judgement to report, not a lost turn to retry."""
    stream = _Replies(WITH_NUMBER, WITH_NUMBER, 'ESTIMATE: -₱92.30/L')
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        verdict = _judge().run()
    assert len(stream.calls) == 3
    assert verdict.estimate is None
    assert verdict.rejected_estimate == pytest.approx(-92.30)


def test_a_judge_that_answered_is_not_asked_again():
    stream = _Replies(WITH_NUMBER, WITH_NUMBER, WITH_NUMBER)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        _judge().run()
    assert len(stream.calls) == 3


# ── The helper itself ─────────────────────────────────────────────────────────

def test_the_helper_reports_the_clean_path_cost_unchanged():
    """Retries are a failure path and are deliberately not in the quota estimate."""
    counts = swarm.expected_call_counts()
    assert counts['total'] == counts['fast'] + counts['deep']
    assert 'retr' in swarm.expected_call_counts.__doc__


def test_the_helper_returns_the_statement_untouched_on_failure():
    with patch('ph_economic_ai.engine.swarm.llm.stream', return_value=[NO_NUMBER]):
        statement, estimate = _reask_for_estimate(
            [{'role': 'user', 'content': 'x'}], 'original',
            tier='fast', max_tokens=100, seed=1,
        )
    assert statement == 'original'
    assert estimate is None
