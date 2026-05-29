import pytest
from ph_economic_ai.engine.swarm import (
    SwarmAgent, GroupSurvivor, RegionalVerdict, MasterVerdict, REGIONS
)


def test_regions_has_4_entries():
    assert len(REGIONS) == 4


def test_swarm_agent_defaults():
    agent = SwarmAgent(
        name='NCR Forecaster-1', role='Forecaster',
        model='deepseek-r1:8b', group_id=0, region_name='NCR',
        system_prompt='You are...', rag_sources=['YahooFinanceCrude'],
    )
    assert agent.is_alive is True
    assert agent.combined_score == 0.0


def test_regional_verdict_fields():
    rv = RegionalVerdict(
        judge_id=0, region_pair=('NCR', 'CAR'),
        estimate=1.5, confidence=0.8,
        reasoning='test', survivor_names=('NCR Forecaster-1', 'CAR Synthesizer-1'),
    )
    assert rv.estimate == 1.5


def test_master_verdict_fields():
    mv = MasterVerdict(
        final_estimate=2.0, confidence_pct=80,
        dissenting_regions=['BARMM'], reasoning='test', regional_verdicts=[],
    )
    assert mv.confidence_pct == 80


from ph_economic_ai.engine.swarm import build_swarm_agents


def test_build_swarm_agents_count():
    agents = build_swarm_agents()
    assert len(agents) == 20  # 4 groups × 5 agents


def test_build_swarm_agents_group_composition():
    agents = build_swarm_agents()
    group_0 = [a for a in agents if a.group_id == 0]
    assert len(group_0) == 5
    roles = [a.role for a in group_0]
    assert roles.count('Forecaster') == 1
    assert roles.count('DataExtractor') == 1
    assert roles.count('Synthesizer') == 1
    assert roles.count('Critic') == 1
    assert roles.count('ConfidenceScorer') == 1


def test_build_swarm_agents_models():
    agents = build_swarm_agents()
    forecasters = [a for a in agents if a.role == 'Forecaster']
    assert all(a.model == 'qwen2.5:7b' for a in forecasters)
    critics = [a for a in agents if a.role == 'Critic']
    assert all(a.model == 'qwen2.5:7b' for a in critics)


def test_build_swarm_agents_names_unique():
    agents = build_swarm_agents()
    names = [a.name for a in agents]
    assert len(names) == len(set(names))


from ph_economic_ai.engine.swarm import (
    _extract_fuel_change, _parse_scores, _parse_confidence,
    _robust_confidence_pct, compute_combined_score, eliminate_bottom_n
)
from ph_economic_ai.engine.debate import AgentResponse


def _make_response(name, estimate, round_num=1):
    return AgentResponse(
        agent_name=name, round_num=round_num,
        thinking='', statement=f'Analysis. ESTIMATE: +₱{estimate:.2f}/L',
        price_estimate=float(estimate),
    )


def test_parse_scores_extracts_lines():
    text = "Good work.\nSCORE: Alice: 8\nSCORE: Bob: 4\nSCORE: Charlie: 10"
    scores = _parse_scores(text, ['Alice', 'Bob', 'Charlie'])
    assert scores == {'Alice': 0.8, 'Bob': 0.4, 'Charlie': 1.0}


def test_parse_scores_missing_agent_defaults_to_half():
    scores = _parse_scores("SCORE: Alice: 7", ['Alice', 'Bob'])
    assert scores['Bob'] == 0.5


def test_parse_confidence_extracts_lines():
    text = "CONFIDENCE: Alice: 0.85\nCONFIDENCE: Bob: 0.40"
    confs = _parse_confidence(text, ['Alice', 'Bob'])
    assert confs == {'Alice': 0.85, 'Bob': 0.40}


def test_parse_confidence_missing_defaults_to_half():
    confs = _parse_confidence("", ['Alice'])
    assert confs['Alice'] == 0.5


def test_extract_fuel_change_prefers_estimate_line_over_absolute_price():
    text = 'Retail pump could fall to P92.30/L.\nESTIMATE: -P1.25/L'
    assert _extract_fuel_change(text) == pytest.approx(-1.25)


def test_extract_fuel_change_rejects_impossible_absolute_price_parse():
    assert _extract_fuel_change('ESTIMATE: -P92.30/L') is None


def test_robust_confidence_reaches_80_when_estimates_cluster():
    estimates = [1.9, 2.1, 2.2, 2.0, 2.3, 9.5, -92.3]
    assert _robust_confidence_pct(estimates, final_estimate=2.1) >= 80


def test_compute_combined_score_none_estimate_is_zero():
    resp = AgentResponse('x', 1, '', 'no estimate here', None)
    score = compute_combined_score(resp, critic_score=0.9, confidence=0.9,
                                   group_estimates=[1.0, 2.0, 3.0])
    assert score == 0.0


def test_compute_combined_score_at_median_gives_high_score():
    resp = _make_response('x', 2.0)
    score = compute_combined_score(resp, critic_score=0.8, confidence=0.9,
                                   group_estimates=[1.0, 2.0, 3.0])
    # deviation_normalized = 0.0 → confidence component = 0.9
    assert score == pytest.approx(0.4 * 0.8 + 0.6 * 0.9, rel=1e-3)


def test_eliminate_bottom_n_removes_lowest_scorers():
    agents = [
        SwarmAgent('A', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.9),
        SwarmAgent('B', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.1),
        SwarmAgent('C', 'Forecaster', 'm', 0, 'R', '', [], combined_score=0.5),
    ]
    survivors, eliminated = eliminate_bottom_n(agents, n=1)
    assert len(survivors) == 2
    assert eliminated[0].name == 'B'


from unittest.mock import patch, MagicMock
from ph_economic_ai.engine.swarm import GroupArena, GroupSurvivor


def _stream(text: str):
    """Returns a one-chunk ollama stream."""
    return [{'message': {'content': text}}]


def _make_rag():
    rag = MagicMock()
    rag.query.return_value = []
    return rag


SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}


def _build_arena(group_id=0):
    all_agents = build_swarm_agents()
    group_agents = [a for a in all_agents if a.group_id == group_id]
    return GroupArena(
        group_id=group_id,
        agents=group_agents,
        rag=_make_rag(),
        scenario=SCENARIO,
    )


def test_group_arena_run_returns_one_survivor():
    arena = _build_arena()
    critic_text = (
        "\n".join(f"SCORE: {a.name}: 7" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    conf_text = (
        "\n".join(f"CONFIDENCE: {a.name}: 0.70" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    normal_text = "Analysis here. ESTIMATE: +₱1.50/L"

    def fake_chat(model, messages, stream, **kwargs):
        role_hint = messages[0]['content'] if messages else ''
        if 'Challenge' in role_hint:     # Critic system prompt
            return _stream(critic_text)
        if 'Evaluate confidence' in role_hint:  # ConfidenceScorer system prompt
            return _stream(conf_text)
        return _stream(normal_text)

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        survivor = arena.run()

    assert isinstance(survivor, GroupSurvivor)
    assert survivor.group_id == 0


def test_group_arena_elimination_events_fired():
    arena = _build_arena()
    events = []

    def on_event(event_type, *args):
        events.append((event_type, *args))

    arena._on_event = on_event

    critic_text = (
        "\n".join(f"SCORE: {a.name}: 5" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    conf_text = (
        "\n".join(f"CONFIDENCE: {a.name}: 0.50" for a in arena._agents)
        + "\nESTIMATE: +₱1.50/L"
    )
    normal_text = "ESTIMATE: +₱2.00/L"

    def fake_chat(model, messages, stream, **kwargs):
        role_hint = messages[0]['content'] if messages else ''
        if 'Challenge' in role_hint:
            return _stream(critic_text)
        if 'Evaluate confidence' in role_hint:
            return _stream(conf_text)
        return _stream(normal_text)

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        arena.run()

    eliminated_events = [e for e in events if e[0] == 'eliminated']
    assert len(eliminated_events) == 4  # 2 + 2 total eliminations (bracket: [(1,2),(2,2)])


def test_group_prompt_includes_project_calibration_rule():
    arena = _build_arena()
    arena._ml_baseline = '+1.20 P/L (+/-0.40 uncertainty)'
    agent = arena._agents[0]

    messages = arena._build_prompt(agent, round_num=1, round_responses=[])
    prompt = messages[1]['content']

    assert 'CALIBRATION RULE' in prompt
    assert 'center of gravity' in prompt
    assert '+1.20 P/L' in prompt
    assert 'Do not output absolute pump prices' in prompt


def test_group_prompt_includes_reconciliation_rule_after_prior_estimates():
    arena = _build_arena()
    arena._history = [
        _make_response('A', 1.0, round_num=1),
        _make_response('B', 1.4, round_num=1),
        _make_response('C', 3.5, round_num=1),
    ]
    agent = arena._agents[0]

    messages = arena._build_prompt(agent, round_num=2, round_responses=[])
    prompt = messages[1]['content']

    assert 'RECONCILIATION RULE' in prompt
    assert 'group median' in prompt
    assert 'revise toward the median' in prompt


from ph_economic_ai.engine.swarm import RegionalJudge, MasterJudge


def _make_survivor(group_id, region, estimate_val):
    resp = AgentResponse(
        agent_name=f'{region} Forecaster', round_num=2,
        thinking='', statement=f'ESTIMATE: +₱{estimate_val:.2f}/L',
        price_estimate=float(estimate_val),
    )
    return GroupSurvivor(
        group_id=group_id, region_name=region, response=resp,
        combined_score=0.8, agent_role='Forecaster', agent_model='qwen2.5:7b',
    )


def test_regional_judge_returns_verdict():
    s1 = _make_survivor(0, 'NCR', 1.5)
    s2 = _make_survivor(1, 'CAR', 2.0)
    judge = RegionalJudge(judge_id=0, survivors=(s1, s2), rag=_make_rag(),
                          scenario=SCENARIO)

    with patch('ph_economic_ai.engine.swarm.ollama.chat',
               return_value=_stream('Good analysis. ESTIMATE: +₱1.75/L')):
        verdict = judge.run()

    assert isinstance(verdict, RegionalVerdict)
    assert verdict.judge_id == 0
    assert verdict.estimate == pytest.approx(1.75)
    assert verdict.region_pair == ('NCR', 'CAR')


def test_master_judge_returns_master_verdict():
    verdicts = [
        RegionalVerdict(i, (f'R{2*i}', f'R{2*i+1}'), 1.5 + i * 0.1,
                        0.8, 'ok', (f'a{i}', f'b{i}'))
        for i in range(2)
    ]
    master = MasterJudge(verdicts=verdicts, rag=_make_rag(), scenario=SCENARIO)

    with patch('ph_economic_ai.engine.swarm.ollama.chat',
               return_value=_stream(
                   'Final analysis. ESTIMATE: +₱1.80/L\n'
                   'Dissenting: Region IX — Zamboanga Peninsula'
               )):
        mv = master.run()

    assert isinstance(mv, MasterVerdict)
    assert mv.final_estimate == pytest.approx(1.80)
    assert mv.confidence_pct >= 0


from ph_economic_ai.engine.swarm import SwarmOrchestrator


def test_swarm_orchestrator_returns_master_verdict():
    def fake_chat(model, messages, stream, **kwargs):
        if 'mistral' in model:
            all_agents = build_swarm_agents()
            names = [a.name for a in all_agents[:10]]
            text = '\n'.join(f'SCORE: {n}: 7' for n in names) + '\nESTIMATE: +₱1.50/L'
            return _stream(text)
        if 'phi4' in model:
            all_agents = build_swarm_agents()
            names = [a.name for a in all_agents[:10]]
            text = '\n'.join(f'CONFIDENCE: {n}: 0.70' for n in names) + '\nESTIMATE: +₱1.50/L'
            return _stream(text)
        return _stream('ESTIMATE: +₱1.50/L')

    with patch('ph_economic_ai.engine.swarm.ollama.chat', side_effect=fake_chat):
        orch = SwarmOrchestrator(rag=_make_rag(), scenario=SCENARIO, parallel_n=2)
        mv = orch.run()

    assert isinstance(mv, MasterVerdict)
    assert mv.final_estimate is not None


def test_master_verdict_has_all_responses_field():
    from ph_economic_ai.engine.swarm import MasterVerdict
    import dataclasses
    fields = {f.name for f in dataclasses.fields(MasterVerdict)}
    assert 'all_responses' in fields


def test_swarm_orchestrator_accepts_evolved_agents():
    from ph_economic_ai.engine.swarm import SwarmOrchestrator
    import inspect
    sig = inspect.signature(SwarmOrchestrator.__init__)
    assert 'evolved_agents' in sig.parameters
