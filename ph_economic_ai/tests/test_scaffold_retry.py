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
import re
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


def test_no_output_line_is_something_an_agent_can_copy():
    """Three lines, three separate discoveries of the same failure.

    The chain line and the estimate line were fixed first. The DIRECTION line
    was left as a menu — "DIRECTION: UP or DIRECTION: DOWN or DIRECTION: FLAT" —
    and the very next live run had five of twenty agents open by copying the
    whole menu back. A menu is a template. So was the instruction sentence
    around them: agents echoed the phrase "ALL THREE LINES" as though it were
    part of the answer.

    This checks the class, not the three instances, because the pattern has now
    recurred every time a line was written as something to reproduce rather than
    something to do.
    """
    arena = _arena()
    prompt = arena._build_prompt(arena._agents[0], 1, [])[-1]['content']

    direction = next(ln for ln in prompt.splitlines()
                     if ln.startswith('DIRECTION:'))
    assert 'or DIRECTION:' not in direction, (
        'the direction line offers a menu an agent can copy verbatim')
    assert direction.count('UP') == 1

    assert 'ALL THREE' not in prompt.upper(), (
        'agents echoed this phrase back as if it were part of the answer')
    # Every slot the agent must fill names what belongs there and is followed by
    # a worked example containing a real value.
    for line in prompt.splitlines():
        if line.startswith(('CAUSAL CHAIN:', 'ESTIMATE:')):
            assert 'worked example' in line, f'no worked example on: {line[:40]}'


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


# ── Enumerate EVERY prompt, not one of them ──────────────────────────────────

def _every_prompt() -> dict:
    """Every prompt string this engine can hand a model, by name.

    `test_no_output_line_is_something_an_agent_can_copy` checked
    `arena._build_prompt`, the agent USER turn, and passed while six role SYSTEM
    prompts, two regional-judge prompts, the master prompt, two retry prompts and
    twenty-three personas in `engine.debate` still shipped a copyable template.
    Testing the class means enumerating the population, not picking a member.
    """
    from unittest.mock import MagicMock
    from ph_economic_ai.engine import debate, swarm

    rag = MagicMock()
    rag.query.return_value = []
    scenario = {'oil_pct': 5.0, 'usd_pct': 2.0, 'current_price': 98.82,
                'bsp_rate': 6.5, 'demand_index': 72.0}
    agents = swarm.build_swarm_agents(98.82, models=[])
    out = {f'swarm role system: {a.name}': a.system_prompt for a in agents}

    arena = swarm.GroupArena(group_id=0,
                             agents=[a for a in agents if a.group_id == 0],
                             rag=rag, scenario=scenario)
    out['swarm agent user turn'] = arena._build_prompt(agents[0], 1, [])[-1]['content']

    for roster in ('DEFAULT_AGENTS', 'FOOD_AGENTS', 'ELECTRICITY_AGENTS'):
        for a in getattr(debate, roster):
            out[f'debate {roster}: {a.name}'] = a.system_prompt

    out['swarm estimate retry'] = swarm._ESTIMATE_RETRY_PROMPT
    out['swarm scaffold retry'] = debate._SCAFFOLD_RETRY_PROMPT
    return out


#: Wordings a small model copies back verbatim instead of instantiating. Each was
#: found in a live run, not imagined.
_COPYABLE = ('X.XX/L or ESTIMATE', 'X.X% or ESTIMATE', 'X.XX/kWh or ESTIMATE',
             'DIRECTION: UP or DIRECTION', '[scenario shock]', '[trigger]',
             'ALL THREE LINES')


def test_no_prompt_anywhere_ships_a_copyable_template():
    offenders = {name: pattern
                 for name, prompt in _every_prompt().items()
                 for pattern in _COPYABLE
                 if pattern in prompt}
    assert not offenders, f'{len(offenders)} prompts still copyable: {offenders}'


def test_the_enumeration_actually_covers_the_layers_it_claims():
    """A guard over an empty or truncated population passes for the wrong reason,
    which is how the narrow version of this test stayed green."""
    prompts = _every_prompt()
    assert sum(1 for k in prompts if k.startswith('swarm role system')) == 20
    assert sum(1 for k in prompts if k.startswith('debate ')) == 23
    assert 'swarm agent user turn' in prompts
    assert len(prompts) >= 45


def test_every_estimate_line_names_its_unit_and_shows_a_real_number():
    from ph_economic_ai.engine.debate import ESTIMATE_LINE
    for sector, unit in (('gas', '/L'), ('food', '%'), ('electricity', '/kWh')):
        line = ESTIMATE_LINE[sector]
        assert 'worked example' in line, sector
        assert unit in line, sector
        assert 'never write' in line, sector
        assert not unfilled_scaffold(f'CAUSAL CHAIN: {line}')


def test_the_direction_instruction_is_not_a_menu():
    from ph_economic_ai.engine.debate import DIRECTION_INSTRUCTION
    assert 'or DIRECTION:' not in DIRECTION_INSTRUCTION
    assert DIRECTION_INSTRUCTION.count('UP') == 1


def test_the_output_lines_have_exactly_one_definition():
    """`forum._EST_LINE` was the ORIGINAL fix, and because it lived only in the
    Forum the same remedy had to be rediscovered three more times: the swarm's
    chain line, its estimate line, and its DIRECTION menu. One definition, aliased
    where call sites want a local name."""
    from ph_economic_ai.engine import debate, forum
    assert forum._EST_LINE is debate.ESTIMATE_LINE


def test_no_module_defines_its_own_copy_of_the_estimate_lines():
    """A copy drifts. The point of centralising is that a fix reaches every
    caller, so a second dict literal anywhere defeats it."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / 'engine'
    definers = [f.name for f in root.glob('*.py')
                if 'ESTIMATE_LINE = {' in f.read_text(encoding='utf-8')
                or "_EST_LINE = {" in f.read_text(encoding='utf-8')]
    assert definers == ['debate.py'], f'defined in {definers}'


# ── The guard was a blacklist, which can only catch what was already found ────

#: An output-spec: an ALL-CAPS label the model is told to emit, wherever it
#: appears. Anchored to the start of a LINE first, which missed the case this
#: test was written for: "SCORE: <agent_name>: X" sits mid-sentence inside the
#: Critic's system prompt, so a line-start match never saw it and the guard was
#: decorative for its own motivating example.
_SPEC_LINE = re.compile(r'\b([A-Z][A-Z ]{2,}):')
#: Placeholder shapes. `<...>` and `[...]` around a descriptive word, or a token
#: made only of X, x, dots and digits, which is how every X.XX in this project
#: was written.
_PLACEHOLDER = re.compile(r'[<\[][a-zA-Z][^\]>\n]{0,40}[>\]]|\b[Xx]\.?[Xx]{1,2}\b|\b0\.[Xx]{2}\b')


def test_every_output_spec_carries_a_worked_example():
    """The property, not a list of known-bad strings.

    `test_no_prompt_anywhere_ships_a_copyable_template` enumerated the patterns
    already discovered, so it passed while "SCORE: <agent_name>: X" and
    "CONFIDENCE: <agent_name>: 0.XX" shipped in the Critic and ConfidenceScorer
    prompts. A blacklist can only ever catch what has already been found, which
    makes it a sampling guard wearing a class-level name.

    The remedy `engine.forum` discovered is the property worth asserting: a line
    that names a slot must show a worked example containing real content. A
    descriptive placeholder is fine WITH one and a trap without.
    """
    offenders = []
    for name, prompt in _every_prompt().items():
        # Grouped by LABEL rather than by occurrence. A label appears twice when
        # its own worked example quotes it back — "ESTIMATE: +0.85/L" contains
        # the string ESTIMATE: — and scanning forward from that second match
        # found no worked example ahead of it and flagged a line that was
        # correct. The spec is satisfied when any line carrying the label shows
        # an example.
        by_label: dict = {}
        for line in prompt.splitlines():
            for m in _SPEC_LINE.finditer(line):
                by_label.setdefault(m.group(1), []).append(line)
        for label, lines in by_label.items():
            if not any(_PLACEHOLDER.search(l) for l in lines):
                continue
            if not any('worked example' in l for l in lines):
                offenders.append(f'{name}: {label}')
    assert not offenders, (
        f'{len(offenders)} output specs use a placeholder with no worked '
        f'example: {sorted(set(offenders))}')


def test_the_scoring_lines_are_the_case_that_blacklist_missed():
    """Named directly so the regression is unambiguous, and because these are the
    likely cause of Q-ENG-013: an echoed scorer template means `_parse_scores`
    finds no real name, every agent defaults to 0.5, and the elimination stops
    measuring anything."""
    from ph_economic_ai.engine.debate import CONFIDENCE_LINE, SCORE_LINE
    for line in (SCORE_LINE, CONFIDENCE_LINE):
        assert 'worked example' in line
        assert 'agent_name' in line and 'never write agent_name' in line
        assert 'NCR Forecaster' in line, 'the example must name a real agent'
    assert '0.XX' not in CONFIDENCE_LINE.split('never write')[0]


def test_the_worked_examples_actually_parse():
    """A worked example the parser rejects would teach the model a format that
    scores nothing, which is worse than the template it replaced."""
    from ph_economic_ai.engine.swarm import _parse_confidence, _parse_scores
    names = ['NCR Forecaster', 'NCR Critic']
    assert _parse_scores('SCORE: NCR Forecaster: 7', names)['NCR Forecaster'] == 0.7
    assert _parse_confidence('CONFIDENCE: NCR Forecaster: 0.75',
                             names)['NCR Forecaster'] == 0.75
