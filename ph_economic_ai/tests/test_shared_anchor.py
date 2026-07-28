"""Everyone who produces a number sees the physical baseline.

The mechanical pass-through lived inside MasterJudge alone. Twenty agents and
two regional judges estimated without it, and the one call that had it then
clamped their work back to physics at the end. Measuring how far agents landed
from each other after withholding the input that sets the scale measures the
absence of information, not disagreement about the economy.

The cost was not theoretical: in run 28 the entire Davao group answered +₱2.00/L
against a −₱1.02/L pass-through — the wrong sign — and the room spanned ₱7.00.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupArena, GroupSurvivor, MasterJudge, RegionalJudge, RegionalVerdict,
    anchor_prompt_block, build_swarm_agents, compute_physical_anchor,
)

SCENARIO = {'oil_pct': 6.8, 'usd_pct': 0.0, 'current_price': 98.0,
            'bsp_rate': 6.5, 'demand_index': 72.0}
ANCHOR = compute_physical_anchor(SCENARIO)


def _rag():
    rag = MagicMock()
    rag.query.return_value = []
    return rag


def _arena(anchor=ANCHOR, ml_baseline=''):
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    return GroupArena(group_id=0, agents=agents, rag=_rag(), scenario=SCENARIO,
                      ml_baseline=ml_baseline, anchor=anchor)


def _agent_prompt(arena):
    msgs = arena._build_prompt(arena._agents[0], 1, [])
    return ' '.join(m['content'] for m in msgs)


def _survivor(region, estimate):
    return GroupSurvivor(
        group_id=0, region_name=region,
        response=AgentResponse(f'{region} Forecaster', 1, '', 'x', estimate),
        combined_score=0.8, agent_role='Forecaster', agent_model='fast',
    )


def _regional_prompts(anchor=ANCHOR):
    judge = RegionalJudge(
        judge_id=0, survivors=(_survivor('NCR', -0.5),
                               _survivor('Davao Region', 2.0)),
        rag=_rag(), scenario=SCENARIO, anchor=anchor,
    )
    defense = judge._defense_prompt(judge._s1, judge._s2)
    synthesis = judge._synthesis_prompt('a', 'b')
    return (' '.join(m['content'] for m in defense),
            ' '.join(m['content'] for m in synthesis))


# ── The anchor reaches every estimator ────────────────────────────────────────

def test_the_agents_are_given_the_pass_through():
    prompt = _agent_prompt(_arena())
    assert 'MECHANICAL PASS-THROUGH' in prompt
    assert f'{ANCHOR:+.2f}' in prompt


def test_the_regional_judge_is_given_the_pass_through():
    defense, synthesis = _regional_prompts()
    for prompt in (defense, synthesis):
        assert 'MECHANICAL PASS-THROUGH' in prompt
        assert f'{ANCHOR:+.2f}' in prompt


def test_the_master_still_has_it():
    judge = MasterJudge(
        verdicts=[RegionalVerdict(0, ('NCR', 'Davao Region'), 2.5, 0.7, 'x',
                                  ('a', 'b'))],
        rag=_rag(), scenario=SCENARIO, survivors=[],
    )
    prompt = ' '.join(m['content'] for m in judge._build_prompt())
    assert 'MECHANICAL PASS-THROUGH' in prompt
    assert f'{judge._anchor:+.2f}' in prompt


def test_all_three_are_given_the_same_number():
    """Three different wordings of the baseline is three baselines."""
    block = anchor_prompt_block(ANCHOR)
    agent = _agent_prompt(_arena())
    defense, synthesis = _regional_prompts()
    assert block in agent
    assert block in defense
    assert block in synthesis


def test_the_anchor_is_computed_once_and_shared(monkeypatch):
    """Each estimator recomputing it is three chances to drift apart.

    Asserted by running the orchestrator and watching what the three estimators
    were handed, rather than by reading the source. The previous version used
    `inspect.getsource` and grepped for `anchor=anchor`, which failed in a way
    worth recording: `getsource` finds a function by line offset in the file on
    disk, so editing the module while a long suite ran made it return a
    DIFFERENT function's body and the assertion failed against `MasterJudge.run`.

    A source-text assertion is also the wrong shape for this guarantee. It passes
    if the string is present and the wiring is broken, and it fails if the wiring
    is right and the code was reformatted. What matters is that all three
    estimators receive the same number, which is what this now checks.
    """
    seen: dict[str, float] = {}
    sentinel = -3.14159

    monkeypatch.setattr(swarm, 'compute_physical_anchor',
                        lambda scenario, brief=None: sentinel)
    monkeypatch.setattr(swarm, 'fetch_live_retail_price', lambda: 84.38)

    for name, cls in (('arena', swarm.GroupArena),
                      ('regional', swarm.RegionalJudge),
                      ('master', swarm.MasterJudge)):
        original = cls.__init__

        def spy(self, *args, __orig=original, __name=name, **kwargs):
            if kwargs.get('anchor') is not None:
                seen[__name] = kwargs['anchor']
            __orig(self, *args, **kwargs)

        monkeypatch.setattr(cls, '__init__', spy)

    with patch('ph_economic_ai.engine.swarm.llm.stream',
               side_effect=lambda messages, **kw: iter(['ESTIMATE: -₱1.50/L'])):
        swarm.SwarmOrchestrator(rag=_rag(), scenario=SCENARIO, parallel_n=2).run()

    assert set(seen) == {'arena', 'regional', 'master'}, (
        f'an estimator was never handed an anchor: got {sorted(seen)}')
    assert set(seen.values()) == {sentinel}, (
        f'the estimators disagree about the anchor: {seen}')


# ── It is a baseline, not a script ────────────────────────────────────────────

def test_the_block_invites_departure_with_a_reason():
    """An anchor a model is told to restate produces agreement by obedience,
    which is not the thing the card claims to measure."""
    block = anchor_prompt_block(-1.02)
    assert 'not a target' in block
    assert 'freight' in block          # regional divergence stays legitimate
    assert 'name the factor' in block


def test_regional_freight_is_named_so_regions_may_still_differ():
    assert 'freight premium' in _agent_prompt(_arena())


# ── Backwards compatibility ───────────────────────────────────────────────────

def test_without_an_anchor_the_agent_prompt_is_unchanged():
    """Ablations and tests construct arenas with no anchor at all."""
    prompt = _agent_prompt(_arena(anchor=None))
    assert 'MECHANICAL PASS-THROUGH' not in prompt
    assert 'Output only the next price CHANGE' in prompt


def test_without_an_anchor_the_regional_judge_prompt_is_unchanged():
    defense, synthesis = _regional_prompts(anchor=None)
    assert 'MECHANICAL PASS-THROUGH' not in defense
    assert 'MECHANICAL PASS-THROUGH' not in synthesis


def test_the_ml_baseline_stops_competing_with_the_physics_anchor():
    """Two centres of gravity in one prompt is how a 3b model splits the
    difference between them and calls it analysis."""
    rule = _arena(ml_baseline='-1.12 PHP/L')._calibration_rule()
    assert '-1.12 PHP/L' in rule                 # still stated
    assert 'corroboration' in rule               # as a second opinion
    assert 'center of gravity' not in rule       # not as a second target


def test_the_ml_baseline_keeps_its_old_role_when_there_is_no_anchor():
    rule = _arena(anchor=None, ml_baseline='-1.12 PHP/L')._calibration_rule()
    assert 'center of gravity' in rule


@pytest.mark.parametrize('oil,usd', [(6.8, 0.0), (-4.0, 1.5), (0.0, 0.0)])
def test_the_anchor_matches_the_master_that_used_to_own_it(oil, usd):
    scenario = {**SCENARIO, 'oil_pct': oil, 'usd_pct': usd}
    judge = MasterJudge(verdicts=[], rag=_rag(), scenario=scenario, survivors=[])
    assert judge._anchor == compute_physical_anchor(scenario)
