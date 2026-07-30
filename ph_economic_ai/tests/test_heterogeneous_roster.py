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
    """An explicit list is parsed and wins outright.

    The empty case moved on 2026-07-30. It used to mean "one model"; it now means
    "fall through to the cross-family default", which depends on what is
    installed, so that path is covered by the default tests below rather than
    asserted as a constant here.
    """
    with patch.dict(os.environ, {'STRATA_SWARM_AGENT_MODELS': 'x:1, y:2 ,'}):
        assert roster_models() == ['x:1', 'y:2']
    with patch.dict(os.environ, {'STRATA_SWARM_AGENT_MODELS': '   '}):
        with patch.object(llm, 'provider_for', lambda tier: 'groq'):
            assert roster_models() == [], 'blank should fall through, not parse'


# ── The statistic ────────────────────────────────────────────────────────────

_MAP = {'a1': 'm1', 'a2': 'm1', 'a3': 'm1', 'b1': 'm2', 'b2': 'm2', 'b3': 'm2'}


def test_models_landing_together_reads_as_agreement():
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.2), _resp('a3', 2.1),
         _resp('b1', 2.1), _resp('b2', 2.0), _resp('b3', 2.2)], _MAP)
    assert out['measurable']
    assert out['models_agree'] is True
    assert out['between_spread'] <= out['band']


def test_models_disagreeing_is_visible_even_at_high_agreement():
    """The failure this exists to catch: one model at +2 and one at -1.5, which a
    pooled percentage can still report as substantial agreement."""
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.1), _resp('a3', 2.0),
         _resp('b1', -1.5), _resp('b2', -1.4), _resp('b3', -1.6)], _MAP)
    assert out['between_spread'] > 3.0
    assert out['models_agree'] is False


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
    assert out['within_mad'] == 0.0
    assert out['between_over_within'] is None
    assert out['models_agree'] is True


def test_one_outlier_cannot_hide_a_real_disagreement():
    """The failure this replaced. The verdict compared `between` against the mean
    within-model RANGE, and on the 2026-07-30 live run llama3.2 sat at median
    +2.50 against qwen2.5:3b at +1.20, a factor of 2.08, yet printed "agreement
    survives the model change" because 1.30 fell under a 3.25 range that one wild
    estimate had set. Range is the least robust dispersion measure there is."""
    llama = [2.5, 2.5, 2.4, 2.6, 2.5, 8.0]      # 8.0 is the outlier
    qwen = [1.2, 1.2, 1.1, 1.3, 1.2, -1.0]
    resp = ([_resp(f'l{i}', v) for i, v in enumerate(llama)]
            + [_resp(f'q{i}', v) for i, v in enumerate(qwen)])
    mapping = {**{f'l{i}': 'llama' for i in range(len(llama))},
               **{f'q{i}': 'qwen' for i in range(len(qwen))}}
    out = agreement_across_models(resp, mapping)

    assert out['median_by_model'] == {'llama': 2.5, 'qwen': 1.2}
    assert out['models_agree'] is False, 'the old range test said these agreed'
    # The range is what misled it, and is still reported so the outlier is visible.
    assert out['within_range'] > 3.0
    assert out['within_mad'] < 0.2, 'MAD must ignore the outlier the range chased'


def test_the_verdict_uses_the_same_band_that_judges_agents():
    """No invented threshold: two models agree by the standard already applied to
    two agents, which ADR-009's control study justified at 0.50."""
    from ph_economic_ai.engine.swarm import _AGREEMENT_BAND
    out = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.0), _resp('b1', 2.4), _resp('b2', 2.4)],
        _MAP)
    assert out['band'] == _AGREEMENT_BAND
    assert out['between_spread'] == pytest.approx(0.4)
    assert out['models_agree'] is True

    wider = agreement_across_models(
        [_resp('a1', 2.0), _resp('a2', 2.0), _resp('b1', 2.6), _resp('b2', 2.6)],
        _MAP)
    assert wider['between_spread'] == pytest.approx(0.6)
    assert wider['models_agree'] is False


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
         'models_agree': True, 'band': 0.50})
    assert 'across 2 different models' in text
    assert 'not one model repeating itself' in text


def test_the_card_reports_models_disagreeing():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 3.50,
         'models_agree': False, 'band': 0.50})
    assert 'the models disagree' in text
    assert 'averages over that' in text


def test_the_card_does_not_reassure_when_one_outlier_widened_the_range():
    """The live case. Medians 2.08x apart must never read as corroboration just
    because some model had a wild estimate."""
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 1.30,
         'models_agree': False, 'band': 0.50, 'within_range': 3.25})
    assert 'disagree' in text
    assert 'not one model repeating itself' not in text


def test_identical_medians_read_as_agreement_not_recitation():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 0.0,
         'models_agree': True, 'band': 0.50})
    assert 'not one model repeating itself' in text


def test_the_verdict_carries_the_cross_model_field():
    """The report reads it off the verdict, so a missing default would make the
    line silently absent on every run rather than only on single-model ones."""
    from ph_economic_ai.engine.swarm import MasterVerdict
    import dataclasses
    names = {f.name for f in dataclasses.fields(MasterVerdict)}
    assert 'agreement_models' in names


# ── Naming the winner, and the single-model synthesis ─────────────────────────

_ACROSS = {'measurable': True, 'models': 2, 'between_spread': 1.30,
           'models_agree': False, 'band': 0.50,
           'median_by_model': {'llama3.2:latest': 2.5, 'qwen2.5:3b': 1.2},
           'nearest_model': 'llama3.2:latest',
           'synthesis_model': 'qwen2.5:7b'}


def test_the_published_number_is_attributed_to_a_model():
    """Reporting that the models disagree without naming which one won leaves the
    reader with the disagreement and none of the consequence. Measured: medians
    +2.50 and +1.20, published +2.39."""
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(_ACROSS)
    assert 'disagree' in text
    assert 'llama3.2:latest' in text


def test_nearest_model_ignores_population_weight():
    """The published number must be attributed by DISTANCE, not by how many
    responses each model contributed. In the one run where the models genuinely
    split, qwen supplied 19 of 32 responses and the estimate still landed on
    llama's median, so a count-based winner would have named the wrong model."""
    from ph_economic_ai.engine.swarm import nearest_model
    assert nearest_model(2.39, _ACROSS) == 'llama3.2:latest'
    assert nearest_model(1.15, _ACROSS) == 'qwen2.5:3b'
    assert nearest_model(None, _ACROSS) is None
    assert nearest_model(2.39, {'measurable': False}) is None


def test_the_card_says_the_synthesis_is_a_single_model():
    """A heterogeneous roster diversifies the debate, not the synthesis: the
    survivors feed the judges and the master, all on the deep tier, which
    `assign_models` does not reach."""
    from ph_economic_ai.ui import honesty
    text = honesty.synthesis_note(_ACROSS)
    assert 'qwen2.5:7b' in text
    assert 'one model' in text
    assert 'rather than a vote across it' in text


def test_the_synthesis_note_is_silent_on_a_single_model_roster():
    from ph_economic_ai.ui import honesty
    assert honesty.synthesis_note({'measurable': False}) == ''
    assert honesty.synthesis_note({}) == ''


def test_survivors_are_counted_by_model():
    """A bracket that eliminates one family is selecting on prose style, since the
    Critic and ConfidenceScorer score peers by name in prose. Nothing measured it."""
    from ph_economic_ai.engine.swarm import survivors_by_model

    class _R:
        def __init__(self, name):
            self.agent_name = name

    class _S:
        def __init__(self, name):
            self.response = _R(name)

    mapping = {'NCR Critic': 'qwen', 'Davao Region Critic': 'qwen',
               'Central Luzon Critic': 'llama'}
    counts = survivors_by_model(
        [_S('NCR Critic'), _S('Davao Region Critic'), _S('Central Luzon Critic')],
        mapping)
    assert counts == {'qwen': 2, 'llama': 1}
    assert survivors_by_model([], mapping) == {}
    assert survivors_by_model([_S('unknown agent')], mapping) == {}


# ── The bracket split, surfaced ───────────────────────────────────────────────

def test_a_model_shut_out_of_the_synthesis_is_named():
    """The survivors are the only agents the regional judges read, so a model with
    no survivors contributed nothing to the published number however many agents
    it fielded. That is checkable, so it gets a verdict."""
    from ph_economic_ai.ui import honesty
    text = honesty.bracket_note(
        {'survivors_by_model': {'qwen': 4},
         'n_by_model': {'qwen': 19, 'llama': 13}})
    assert 'none from llama' in text
    assert 'through no one' in text


def test_a_balanced_bracket_refuses_to_call_it_bias():
    """Four survivors over two models is well within chance, and calling 3-to-1
    bias would be the kind of unearned conclusion this project keeps retracting."""
    from ph_economic_ai.ui import honesty
    text = honesty.bracket_note(
        {'survivors_by_model': {'qwen': 2, 'llama': 2},
         'n_by_model': {'qwen': 16, 'llama': 16}})
    assert 'too few to read as bias' in text
    assert 'through no one' not in text


def test_the_bracket_note_is_silent_without_survivor_data():
    from ph_economic_ai.ui import honesty
    assert honesty.bracket_note({}) == ''
    assert honesty.bracket_note({'survivors_by_model': {}}) == ''
    assert honesty.bracket_note(None) == ''


def test_no_winner_is_named_when_the_models_agree():
    """A second live run put the medians 0.005 PHP/L apart and a winner was still
    named, which is picking noise: with no gap there is nothing for the estimate
    to be nearer to. A winner means one model's answer was published and
    another's was not."""
    from ph_economic_ai.engine.swarm import nearest_model
    agreed = {'measurable': True, 'models_agree': True,
              'median_by_model': {'llama': 1.500, 'qwen': 1.505}}
    assert nearest_model(0.90, agreed) is None

    split = {'measurable': True, 'models_agree': False,
             'median_by_model': {'llama': 2.5, 'qwen': 1.2}}
    assert nearest_model(2.39, split) == 'llama'


def test_the_card_names_no_winner_on_an_agreeing_run():
    from ph_economic_ai.ui import honesty
    text = honesty.cross_model_note(
        {'measurable': True, 'models': 2, 'between_spread': 0.005,
         'models_agree': True, 'band': 0.50, 'nearest_model': None})
    assert 'published number is' not in text
    assert 'not one model repeating itself' in text


# ── The cross-family roster is now the DEFAULT ───────────────────────────────

def test_the_default_roster_spans_two_families(monkeypatch):
    """Changed 2026-07-30. A single-model roster's agreement figure substantially
    measures one model's determinism: runs scored 89 to 98 percent over as few as
    three distinct values. Five paired runs put the between-model spread at a
    median 0.250 PHP/L against a 0.50 band, so the two families agree on the
    answer while the percentage falls a median 15 points."""
    from ph_economic_ai.engine.swarm import DEFAULT_AGENT_MODELS
    monkeypatch.delenv('STRATA_SWARM_AGENT_MODELS', raising=False)
    monkeypatch.setattr(llm, 'provider_for', lambda tier: 'ollama')
    monkeypatch.setattr(llm, 'installed_models',
                        lambda refresh=False: frozenset(DEFAULT_AGENT_MODELS))
    assert roster_models() == list(DEFAULT_AGENT_MODELS)
    assert len({m.split(':')[0] for m in DEFAULT_AGENT_MODELS}) == 2, \
        'the default must span two FAMILIES; two sizes of one prove nothing'


def test_a_fresh_install_without_the_second_model_still_runs(monkeypatch):
    """Defaulting to a model the user never pulled would fail every call. Both of
    the pair must be present, because one of them is not a cross-family roster,
    it is the old behaviour with extra steps."""
    monkeypatch.delenv('STRATA_SWARM_AGENT_MODELS', raising=False)
    monkeypatch.setattr(llm, 'provider_for', lambda tier: 'ollama')
    monkeypatch.setattr(llm, 'installed_models',
                        lambda refresh=False: frozenset({'qwen2.5:3b'}))
    assert roster_models() == []


def test_an_unreachable_daemon_falls_back_rather_than_guessing(monkeypatch):
    """`installed_models` returns empty when it CANNOT CONFIRM, which is
    indistinguishable from nothing installed, and only one of those is safe to act
    on."""
    monkeypatch.delenv('STRATA_SWARM_AGENT_MODELS', raising=False)
    monkeypatch.setattr(llm, 'provider_for', lambda tier: 'ollama')
    monkeypatch.setattr(llm, 'installed_models', lambda refresh=False: frozenset())
    assert roster_models() == []


def test_a_hosted_provider_does_not_get_ollama_tags(monkeypatch):
    """Groq and Gemini name their models differently, so the local pair is
    meaningless there and would fail every call."""
    monkeypatch.delenv('STRATA_SWARM_AGENT_MODELS', raising=False)
    monkeypatch.setattr(llm, 'provider_for', lambda tier: 'groq')
    assert roster_models() == []


def test_the_environment_still_overrides_the_default(monkeypatch):
    """An experiment must be able to pin an arm to one model."""
    monkeypatch.setenv('STRATA_SWARM_AGENT_MODELS', 'qwen2.5:3b')
    assert roster_models() == ['qwen2.5:3b']


def test_installed_models_never_raises_and_caches(monkeypatch):
    """It is consulted while building the roster, so on every run and in most
    tests. A network call each time, or an exception, would be intolerable there."""
    calls = []

    def _boom(*a, **k):
        calls.append(1)
        raise OSError('no daemon')

    monkeypatch.setattr(llm, '_installed_cache', None, raising=False)
    monkeypatch.setattr(llm.requests, 'get', _boom)
    assert llm.installed_models() == frozenset()
    assert llm.installed_models() == frozenset()
    assert len(calls) == 1, 'the failure must be cached, not retried per call'
    monkeypatch.setattr(llm, '_installed_cache', None, raising=False)


# ── Q-ENG-011: the synthesis model is now testable ───────────────────────────

def test_the_judge_model_is_overridable(monkeypatch):
    """The synthesis is one model whatever the agent roster does, and the card
    said so while nothing could CHECK whether that reading is model-dependent.
    An override makes the question answerable without changing the default."""
    from ph_economic_ai.engine.swarm import judge_model
    monkeypatch.delenv('STRATA_SWARM_JUDGE_MODEL', raising=False)
    assert judge_model() is None, 'the default must stay the tier default'
    monkeypatch.setenv('STRATA_SWARM_JUDGE_MODEL', 'llama3.2')
    assert judge_model() == 'llama3.2'
    monkeypatch.setenv('STRATA_SWARM_JUDGE_MODEL', '   ')
    assert judge_model() is None, 'blank must not become a model name'


def test_every_judge_call_site_honours_the_override(monkeypatch):
    """Four sites: the regional judge's three prompts share `_call`, plus its two
    retries and the master. A site left on the tier default would splice two
    models into one synthesis."""
    import inspect
    from ph_economic_ai.engine import swarm as sw
    src = inspect.getsource(sw)
    assert src.count('model=judge_model()') == 4


def test_the_card_names_the_model_that_actually_synthesised(monkeypatch):
    """Not the tier default. An overridden judge that still reported the default
    would make the experiment's own artifact wrong about which model produced the
    number it records."""
    from ph_economic_ai.engine.swarm import judge_model
    monkeypatch.setenv('STRATA_SWARM_JUDGE_MODEL', 'llama3.2')
    assert (judge_model() or llm.describe_model(llm.DEEP)) == 'llama3.2'
    monkeypatch.delenv('STRATA_SWARM_JUDGE_MODEL', raising=False)
    assert (judge_model() or llm.describe_model(llm.DEEP)) == llm.describe_model(llm.DEEP)


def test_the_judge_override_does_not_touch_the_agent_roster(monkeypatch):
    """The whole point is holding the room fixed while the synthesis changes."""
    monkeypatch.setenv('STRATA_SWARM_JUDGE_MODEL', 'llama3.2')
    monkeypatch.setenv('STRATA_SWARM_AGENT_MODELS', 'model-a,model-b')
    agents = build_swarm_agents(models=None)
    assert {a.model for a in agents} == {'model-a', 'model-b'}
