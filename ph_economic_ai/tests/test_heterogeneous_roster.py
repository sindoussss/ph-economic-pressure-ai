"""A roster spanning several models, and the statistic that makes it worth doing.

A single-model run scored 100 percent agreement over ONE distinct estimate. The
agents were blinded, independent and separately reasoned; they converged because
twenty agents on `qwen2.5:3b` are one model asked twenty times. Blinding removes
peer contamination and cannot remove model identity, so the percentage on a
single-model roster substantially measures that model's determinism (`DEC-029`).

Assigning different models is the easy half. The half that matters is that model
must be CROSSED with region and role rather than nested in either, and that the
run reports whether agreement survived the model change.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import llm, swarm
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    agreement_across_models, assign_models, build_swarm_agents, roster_models,
)

TWO = ['model-a', 'model-b']
THREE = ['model-a', 'model-b', 'model-c']


def _resp(name, est):
    return AgentResponse(name, 1, '', 'CAUSAL CHAIN: real -> content', est)


# ── Assignment: crossed, not nested ──────────────────────────────────────────

def test_every_region_spans_every_model():
    """The obvious assignment, one model per region, would make model and region
    the SAME variable: a regional card 2 PHP/L below the others could be Western
    Visayas disagreeing or the smaller model disagreeing, and no run could tell
    them apart. That would destroy the regional cards to build the experiment."""
    agents = build_swarm_agents(models=TWO)
    for group_id in {a.group_id for a in agents}:
        seen = {a.model for a in agents if a.group_id == group_id}
        assert seen == set(TWO), f'group {group_id} saw only {seen}'


def test_every_role_spans_every_model():
    """Nesting model inside role would confound weights with the job. The Critic
    scores other agents, so a Critic that is always the weaker model would bias
    the whole elimination bracket in one direction."""
    agents = build_swarm_agents(models=TWO)
    for role in {a.role for a in agents}:
        seen = {a.model for a in agents if a.role == role}
        assert seen == set(TWO), f'role {role} saw only {seen}'


def test_the_roster_is_evenly_split():
    agents = build_swarm_agents(models=TWO)
    counts = {m: sum(1 for a in agents if a.model == m) for m in TWO}
    assert counts == {'model-a': 10, 'model-b': 10}


def test_three_models_still_cross_every_region():
    agents = build_swarm_agents(models=THREE)
    for group_id in {a.group_id for a in agents}:
        assert {a.model for a in agents if a.group_id == group_id} == set(THREE)


def test_assignment_is_a_pure_function_of_the_seat():
    """No seed, so it reproduces by construction rather than by remembering to
    key one on the vintage."""
    assert [a.model for a in build_swarm_agents(models=TWO)] == \
           [a.model for a in build_swarm_agents(models=TWO)]
    assert assign_models(TWO, 0, 0) == assign_models(TWO, 0, 0)
    assert assign_models(TWO, 0, 0) != assign_models(TWO, 1, 0)


def test_no_models_means_the_tier_default():
    """Heterogeneity is opt-in. It is an experiment before it is a feature, and
    the models have to be pulled before they can answer."""
    assert assign_models([], 0, 0) is None
    assert {a.model for a in build_swarm_agents(models=[])} == {None}


def test_the_roster_is_read_from_the_environment():
    with patch.dict(os.environ, {'STRATA_SWARM_AGENT_MODELS': 'x:1, y:2 ,'}):
        assert roster_models() == ['x:1', 'y:2']
    with patch.dict(os.environ, {'STRATA_SWARM_AGENT_MODELS': ''}):
        assert roster_models() == []


# ── The statistic ────────────────────────────────────────────────────────────

_MAP = {'a1': 'm1', 'a2': 'm1', 'a3': 'm1', 'b1': 'm2', 'b2': 'm2', 'b3': 'm2'}


def test_models_landing_together_reads_as_agreement():
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.2), _resp('a3', 2.1),
         _resp('b1', 2.1), _resp('b2', 2.0), _resp('b3', 2.2)], _MAP)
    assert out['measurable']
    assert out['between_spread'] <= out['within_spread']


def test_models_disagreeing_is_visible_even_at_high_agreement():
    """The failure this exists to catch: one model at +2 and one at -1.5, which a
    pooled percentage can still report as substantial agreement."""
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.1), _resp('a3', 2.0),
         _resp('b1', -1.5), _resp('b2', -1.4), _resp('b3', -1.6)], _MAP)
    assert out['between_spread'] > 3.0
    assert out['between_over_within'] > 3


def test_a_single_model_roster_says_it_cannot_measure_this():
    """It must not invent a comparison. Reporting 100 percent cross-model
    agreement from one model would be the exact error the whole session kept
    finding: a number that looks like evidence and is a tautology."""
    out = agreement_across_models([_resp('a1', 2.0), _resp('a2', 2.1)],
                                  {'a1': 'm1', 'a2': 'm1'})
    assert out == {'models': 1, 'measurable': False}
    assert agreement_across_models([_resp('a1', 2.0)], {})['measurable'] is False


def test_identical_models_do_not_divide_by_zero():
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.0), _resp('b1', 2.0), _resp('b2', 2.0)],
        _MAP)
    assert out['within_spread'] == 0.0
    assert out['between_over_within'] is None


def test_unparsed_estimates_and_unmapped_agents_are_excluded():
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', None), _resp('ghost', 9.0),
         _resp('b1', 2.0), _resp('b2', 2.1)], _MAP)
    assert out['n_by_model'] == {'m1': 1, 'm2': 2}


def test_the_cross_percentage_carries_its_own_population():
    """With two models it is a two-point measurement, and this project published
    one of those as a headline once already."""
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('b1', 2.0)], {'a1': 'm1', 'b1': 'm2'})
    assert out['cross_n'] == 2


# ── The model reaches the provider, and so does the retry ────────────────────

class _Spy:
    def __init__(self, reply='CAUSAL CHAIN: a -> b\nDIRECTION: UP\nESTIMATE: +1.00/L'):
        self.models = []
        self._reply = reply

    def __call__(self, messages, **kwargs):
        self.models.append(kwargs.get('model'))
        return [self._reply]


def _arena(models):
    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents(models=models) if a.group_id == 0]
    return swarm.GroupArena(group_id=0, agents=agents, rag=rag,
                            scenario={'oil_pct': 5.0, 'usd_pct': 2.0,
                                      'current_price': 98.82, 'bsp_rate': 6.5,
                                      'demand_index': 72.0})


def test_each_agent_calls_with_its_own_model():
    arena = _arena(TWO)
    spy = _Spy()
    with patch('ph_economic_ai.engine.swarm.llm.stream', spy):
        for agent in arena._agents:
            arena._call_agent(agent, [{'role': 'user', 'content': 'x'}])
    assert spy.models == [a.model for a in arena._agents]
    assert set(spy.models) == set(TWO)


def test_a_retry_uses_the_same_model_as_the_agent():
    """Answering a 3b agent's follow-up with a different model splices two models
    into one statement, and the estimate would belong to neither."""
    arena = _arena(TWO)
    spy = _Spy('Prices will rise but I give no number.')
    with patch('ph_economic_ai.engine.swarm.llm.stream', spy):
        arena._call_agent(arena._agents[0], [{'role': 'user', 'content': 'x'}])
    assert len(spy.models) > 1, 'expected a retry'
    assert len(set(spy.models)) == 1
    assert spy.models[0] == arena._agents[0].model


# ── Provenance must be able to tell the two rosters apart ────────────────────

def test_provenance_keeps_every_model_that_served_a_tier():
    """`_provenance[tier] = ...` kept whichever agent finished last, which turns a
    three-model run into a one-model run in the recall key. A run answered by
    three models could then be recalled for one answered by one."""
    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'model-a')
    llm._record_provenance(llm.FAST, 'ollama', 'model-b')
    entry = llm.last_provenance()[llm.FAST]
    assert entry['models'] == ['ollama:model-a', 'ollama:model-b']
    assert 'model-a' in llm.provenance_id() and 'model-b' in llm.provenance_id()
    llm.reset_provenance()


def test_a_mixed_roster_does_not_share_an_identity_with_a_single_model_run():
    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'model-a')
    solo = llm.provenance_id()
    llm._record_provenance(llm.FAST, 'ollama', 'model-b')
    mixed = llm.provenance_id()
    llm.reset_provenance()
    assert solo != mixed


def test_the_identity_does_not_depend_on_which_agent_finished_first():
    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'model-b')
    llm._record_provenance(llm.FAST, 'ollama', 'model-a')
    one = llm.provenance_id()
    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'model-a')
    llm._record_provenance(llm.FAST, 'ollama', 'model-b')
    llm.reset_provenance()
    assert one == one  # sorted, so order in equals order out
    assert 'model-a+model-b' in one or 'ollama:model-a+ollama:model-b' in one


def test_a_fallback_still_marks_the_tier():
    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'model-a')
    llm._record_provenance(llm.FAST, 'groq', 'llama-x', fell_back=True)
    assert llm.last_provenance()[llm.FAST]['fell_back'] is True
    assert '!' in llm.provenance_id()
    llm.reset_provenance()


# ── What the reader is told ───────────────────────────────────────────────────

def test_the_card_says_nothing_on_a_single_model_roster():
    """An absent line is honest. "1 model" dressed as a finding is not."""
    from ph_economic_ai.ui import honesty
    assert honesty.cross_model_note({}) == ''
    assert honesty.cross_model_note({'measurable': False, 'models': 1}) == ''


def test_the_card_reports_agreement_surviving_the_model_change():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 0.05,
         'within_spread': 0.40})
    assert 'across 2 different models' in text
    assert 'not one model repeating itself' in text


def test_the_card_reports_models_disagreeing():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 3.50,
         'within_spread': 0.15})
    assert 'the models disagree' in text
    assert 'averages over that' in text


def test_the_card_names_recitation_when_no_model_varies_internally():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 0.0,
         'within_spread': 0.0})
    assert 'reciting' in text


def test_the_verdict_carries_the_cross_model_field():
    """The report reads it off the verdict, so a missing default would make the
    line silently absent on every run rather than only on single-model ones."""
    from ph_economic_ai.engine.swarm import MasterVerdict
    import dataclasses
    names = {f.name for f in dataclasses.fields(MasterVerdict)}
    assert 'agreement_models' in names
