"""An agent that returns the causal-chain TEMPLATE has not answered.

Measured on a live run, 2026-07-29: eleven of twenty agents opened with

    CAUSAL CHAIN: [scenario shock] -> [market effect] -> [retail mechanism

placeholders intact. `qwen2.5:3b` is small enough to treat the requested format
as the requested answer, and nothing downstream noticed. The ESTIMATE line still
parsed, so the agent stayed in the agreement population contributing a number it
had never reasoned to — and an agent that skipped the reasoning has nothing to
offer but the anchor it was handed, which is the shape of the echo problem the
same run measured at 68.8 percent.

It corrupts the diversity metric too. Eleven identical openings read as a room
herding, when it is really a room not answering. Two unrelated failures wearing
one number, which is why the caveat text had to stop naming a cause.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.swarm import (
    GroupArena, build_swarm_agents, unfilled_scaffold,
)

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'current_price': 98.82,
            'bsp_rate': 6.5, 'demand_index': 72.0}

TEMPLATE = ('CAUSAL CHAIN: [scenario shock] -> [market effect] -> '
            '[retail mechanism] -> [consumer impact]\n'
            'DIRECTION: UP\nESTIMATE: +₱2.21/L')
FILLED = ('CAUSAL CHAIN: Brent +5% -> import cost up -> DOE weekly pass-through '
          '-> jeepney fares rise\nDIRECTION: UP\nESTIMATE: +₱2.40/L')


def _arena():
    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    return GroupArena(group_id=0, agents=agents, rag=rag, scenario=SCENARIO)


class _Replies:
    def __init__(self, *replies):
        self._replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        return [self._replies.pop(0) if self._replies else FILLED]


# ── The detector ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('statement', [
    'CAUSAL CHAIN: [scenario shock] -> [market effect] -> [retail mechanism]',
    'CAUSAL CHAIN: [trigger] -> [market effect] -> [price mechanism]',
    'CAUSAL CHAIN: <trigger> -> <effect> -> <household impact>',
    'causal chain: [consumer impact] only',
])
def test_an_unfilled_chain_is_detected(statement):
    assert unfilled_scaffold(statement) is True


@pytest.mark.parametrize('statement', [
    FILLED,
    # The agent's OWN bracketed content. Brackets are not the offence; this
    # codebase's placeholder words inside them are.
    'CAUSAL CHAIN: [Brent +5%] -> [freight] -> [pump] -> [jeepney fares]',
    # A placeholder word outside the chain line says nothing about the chain.
    'The [market effect] is large. CAUSAL CHAIN: oil up -> pump up -> fares up',
    '',
])
def test_a_filled_chain_is_left_alone(statement):
    assert unfilled_scaffold(statement) is False


# ── The retry ────────────────────────────────────────────────────────────────

def test_an_agent_that_returned_the_template_is_asked_again():
    arena = _arena()
    stream = _Replies(TEMPLATE, FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 2
    assert not unfilled_scaffold(resp.statement)


def test_an_agent_that_answered_is_not_asked_again():
    """The retry is a failure path and must not double the bill on a good run."""
    arena = _arena()
    stream = _Replies(FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(stream.calls) == 1


def test_the_template_line_is_replaced_not_appended():
    """`_reask_for_estimate` appends because the original reasoning is worth
    keeping. Here the offending line is this codebase's own template read back,
    so appending would leave both the placeholder text and the identical opening
    that misleads the diversity metric."""
    arena = _arena()
    stream = _Replies(TEMPLATE, FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert '[scenario shock]' not in resp.statement
    assert 'Brent +5%' in resp.statement


def test_prose_written_around_the_template_survives():
    arena = _arena()
    with_prose = 'NCR demand is inelastic near payday.\n' + TEMPLATE
    stream = _Replies(with_prose, FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert 'inelastic near payday' in resp.statement
    assert '[scenario shock]' not in resp.statement


def test_a_retry_that_returns_the_template_again_keeps_the_original():
    """A failure that cannot be repaired must stay visible. Swapping in a second
    copy of the template would hide it behind a number that looks answered."""
    arena = _arena()
    stream = _Replies(TEMPLATE, TEMPLATE)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert unfilled_scaffold(resp.statement)
    assert resp.price_estimate == pytest.approx(2.21)


def test_a_failed_retry_call_does_not_take_the_answer_down():
    arena = _arena()

    def _boom(messages, **kwargs):
        if _boom.n:
            raise RuntimeError('provider down')
        _boom.n += 1
        return [TEMPLATE]
    _boom.n = 0

    with patch('ph_economic_ai.engine.swarm.llm.stream', _boom):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert resp.price_estimate == pytest.approx(2.21)


def test_the_retry_asks_for_the_chain_and_not_the_analysis():
    arena = _arena()
    stream = _Replies(TEMPLATE, FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    ask = stream.calls[1][-1]['content']
    assert 'Do not repeat your analysis' in ask
    assert 'placeholders' in ask


def test_the_retry_runs_before_the_estimate_retry():
    """Order matters. An agent that returned the template has not reasoned, so
    asking it for a number first would bank the anchor it was handed."""
    arena = _arena()
    no_number = ('CAUSAL CHAIN: [scenario shock] -> [market effect]\n'
                 'Prices should rise somewhat.')
    stream = _Replies(no_number, FILLED)
    with patch('ph_economic_ai.engine.swarm.llm.stream', stream):
        resp = arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert 'placeholders' in stream.calls[1][-1]['content']
    assert resp.price_estimate == pytest.approx(2.40)


def test_the_prompt_no_longer_ships_a_copyable_template():
    """The retry is the backstop. The fix is the prompt.

    `engine.forum` found this first and wrote the remedy down at `_EST_LINE`: a
    small model copies a template verbatim, so the line has to read as an
    instruction — name the quantity, show a worked example with real content,
    and say outright not to echo it. That remedy was never carried across to the
    swarm's chain line, which is why eleven of twenty agents copied it.

    If a future edit reinstates a bracketed template here, that is the defect
    returning and this test is what catches it.
    """
    arena = _arena()
    prompt = arena._build_prompt(arena._agents[0], 1, [])[-1]['content']
    chain = [ln for ln in prompt.splitlines() if 'CAUSAL CHAIN' in ln]
    assert chain, 'the agent prompt no longer shows a causal-chain line'
    assert not unfilled_scaffold(chain[0]), (
        'the swarm prompt ships a chain line an agent can copy verbatim')
    assert 'worked example' in chain[0]
    assert 'X.XX' not in prompt.split('ESTIMATE:')[1].split('worked example')[0]


def test_the_detector_still_catches_the_wording_that_caused_this():
    """The prompts were reworded, but models still emit the old shape — it is
    what they were trained on, and it is what the live run produced."""
    assert unfilled_scaffold(
        'CAUSAL CHAIN: [scenario shock] -> [market effect] -> '
        '[retail mechanism] -> [consumer impact]')
    assert unfilled_scaffold(
        'CAUSAL CHAIN: [trigger] → [market effect] → [price mechanism] → '
        '[consumer impact]')
    assert unfilled_scaffold('CAUSAL CHAIN: <trigger> → <effect> → '
                             '<household impact>')
