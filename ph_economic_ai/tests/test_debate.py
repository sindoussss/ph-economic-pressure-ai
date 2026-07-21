import pytest
from unittest.mock import MagicMock, patch
from ph_economic_ai.engine.debate import (
    _parse_think, _extract_price,
    Agent, AgentResponse, DebateEngine, DEFAULT_AGENTS,
)
from ph_economic_ai.engine.debate import FOOD_AGENTS, ELECTRICITY_AGENTS


def test_food_agents_count():
    assert len(FOOD_AGENTS) == 4


def test_food_agents_have_estimate_format():
    for agent in FOOD_AGENTS:
        assert 'ESTIMATE:' in agent.system_prompt


def test_electricity_agents_count():
    assert len(ELECTRICITY_AGENTS) == 4


def test_electricity_agents_have_estimate_format():
    for agent in ELECTRICITY_AGENTS:
        assert 'ESTIMATE:' in agent.system_prompt


def test_food_agents_use_main_model():
    from ph_economic_ai.engine.debate import _MAIN_TIER
    for agent in FOOD_AGENTS:
        assert agent.tier == _MAIN_TIER


def test_electricity_agents_use_main_model():
    from ph_economic_ai.engine.debate import _MAIN_TIER
    for agent in ELECTRICITY_AGENTS:
        assert agent.tier == _MAIN_TIER


def test_parse_think_splits_tag():
    thinking, statement = _parse_think(
        '<think>I must consider OPEC signals.</think>My estimate is +₱2.50/L.'
    )
    assert thinking == 'I must consider OPEC signals.'
    assert statement == 'My estimate is +₱2.50/L.'


def test_parse_think_no_tag():
    thinking, statement = _parse_think('My estimate is +₱2.50/L.')
    assert thinking == ''
    assert statement == 'My estimate is +₱2.50/L.'


def test_extract_price_positive_delta():
    assert _extract_price('price will rise by +₱2.50/L') == pytest.approx(2.50)


def test_extract_price_negative_delta():
    assert _extract_price('downward pressure of -₱1.20') == pytest.approx(-1.20)


def test_extract_price_absolute_unsigned():
    assert _extract_price('forecast ₱73.20 per liter') is None


def test_extract_price_none():
    assert _extract_price('no price mentioned here') is None


def test_default_agents_count():
    assert len(DEFAULT_AGENTS) == 15
    names = {a.name for a in DEFAULT_AGENTS}
    assert 'Market Analyst' in names
    assert 'Policy Expert' in names
    assert 'Risk Assessor' in names


def _make_mock_rag():
    rag = MagicMock()
    rag.query.return_value = [
        {'text': 'Fuel prices rising due to oil shock.', 'source': 'DOE', 'score': 0.9}
    ]
    return rag


def test_build_prompt_contains_scenario():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS, rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    messages = engine._build_prompt(DEFAULT_AGENTS[0], round_num=1)
    combined = ' '.join(m['content'] for m in messages)
    assert '+5.0' in combined or '5.0' in combined
    assert '6.5' in combined


def test_run_calls_the_provider_per_agent_per_round():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:2], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})

    fake_stream = [tok for tok in
                   ['<think>', 'thinking', '</think>', '+₱2.50/L']]

    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)) as mock_chat:
        responses = engine.run(rounds=2)

    assert mock_chat.call_count == 4  # 2 agents × 2 rounds
    assert len(responses) == 4
    assert all(isinstance(r, AgentResponse) for r in responses)


def test_run_extracts_price_estimate():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream = [tok
                   for tok in ['Pump price estimate is ', '+₱2.50', '/L']]
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)):
        responses = engine.run(rounds=1)
    assert responses[0].price_estimate == pytest.approx(2.50)


def test_parse_think_unclosed_tag():
    thinking, statement = _parse_think('<think>reasoning starts but never ends')
    assert statement == ''
    assert 'reasoning starts' in thinking


def test_parse_think_multiple_blocks():
    thinking, statement = _parse_think(
        '<think>first thought</think>middle<think>second thought</think>final answer'
    )
    assert 'first thought' in thinking
    assert 'second thought' in thinking
    assert statement == 'final answer'


def test_extract_price_space_before_sign():
    assert _extract_price('downward by - ₱1.20') == pytest.approx(-1.20)


def test_extract_price_integer():
    assert _extract_price('ESTIMATE: +₱73/L') == pytest.approx(73.0)


def test_ask_unknown_agent_returns_empty():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    result = engine.ask('Nonexistent Agent', 'What do you think?')
    assert result == ''


def test_ask_calls_the_provider():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream = [tok
                   for tok in ['The rate cut would push ', '+₱0.30/L', ' higher.']]
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)):
        result = engine.ask('Market Analyst', 'What about a rate cut?')
    assert '+₱0.30' in result or '0.30' in result


def test_consensus_empty_history():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    result = engine.consensus()
    assert result['weighted_avg'] is None
    assert result['verdicts'] == []


def test_consensus_final_round_only():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream_r1 = ['Round 1: +₱1.00/L']
    fake_stream_r2 = ['Round 2: +₱2.00/L']
    with patch('ph_economic_ai.engine.debate.llm.stream',
               side_effect=[iter(fake_stream_r1), iter(fake_stream_r2)]):
        engine.run(rounds=2)
    result = engine.consensus()
    # Only round 2 should be used
    assert result['weighted_avg'] == pytest.approx(2.00)


def test_run_clears_history_on_rerun():
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream = ['+₱2.50/L']
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)):
        engine.run(rounds=1)
    fake_stream2 = ['+₱3.00/L']
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream2)):
        responses = engine.run(rounds=1)
    assert len(responses) == 1
    assert responses[0].price_estimate == pytest.approx(3.00)


from ph_economic_ai.engine.debate import SynthesizerThread


def _make_chunk(text: str):
    return text


def test_synthesizer_emits_tokens():
    thread = SynthesizerThread(
        gas_verdict='Gas up ₱2.50/L.',
        food_verdict='Food index rising 3%.',
        elec_verdict='Electricity up ₱0.45/kWh.',
    )
    tokens = []
    thread.token_ready.connect(tokens.append)

    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=[_make_chunk('Summary'), _make_chunk(' text.')]):
        thread.run()

    assert ''.join(tokens) == 'Summary text.'


def test_synthesizer_finished_signal():
    thread = SynthesizerThread(
        gas_verdict='Gas verdict.',
        food_verdict='Food verdict.',
        elec_verdict='Electricity verdict.',
    )
    results = []
    thread.finished.connect(results.append)

    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=[_make_chunk('Done.')]):
        thread.run()

    assert results == ['Done.']
