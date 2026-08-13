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


def test_food_inflation_anchor_is_not_the_stale_april_figure():
    """RSK-041 lineage: baked into every food agent's system prompt at import
    time, with no live-refresh path (unlike gas, which fetches a real price
    every run). PSA's July 2026 release (2026-08-05) put food inflation at
    5.3%, not the stale 6.1% April figure this agent anchor used to state."""
    from ph_economic_ai.engine.debate import _FOOD_INFLATION_YOY_PCT, _FOOD_ANCHOR
    assert _FOOD_INFLATION_YOY_PCT == 5.3
    assert '6.1' not in _FOOD_ANCHOR
    assert 'April 2026' not in _FOOD_ANCHOR


def test_meralco_rate_anchor_is_not_the_stale_may_figure():
    """Meralco's own July 2026 rate announcement put the residential rate at
    14.8261/kWh, not the stale 14.3345 May figure this agent anchor used to
    state."""
    from ph_economic_ai.engine.debate import _MERALCO_RATE_PHP_KWH, _ELEC_ANCHOR
    assert _MERALCO_RATE_PHP_KWH == 14.8261
    assert '14.3345' not in _ELEC_ANCHOR
    assert 'May 2026' not in _ELEC_ANCHOR


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


def test_run_rejects_an_absolute_price_parsed_as_a_change():
    """Classic-mode gas debate's default price_extractor must apply the same
    plausibility ceiling forum.py and swarm.py apply at their own call sites --
    an agent quoting an absolute pump price (e.g. '+150.00/L') must not reach
    consensus() as if it were a real weekly change."""
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag,
                          {'oil_pct': 5.0, 'usd_pct': 2.0,
                           'bsp_rate': 6.5, 'demand_index': 72})
    fake_stream = ['ESTIMATE: +₱150.00/L']
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)):
        responses = engine.run(rounds=1)
    assert responses[0].price_estimate is None


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


def test_consensus_uses_electricity_band_not_the_gas_sized_default():
    """The engine used one 0.20 band for every sector -- double electricity's
    validated 0.10 (forum._BAND), so classic-mode electricity confidence read
    higher than the same room would on the Forum or the swarm."""
    rag = _make_mock_rag()
    engine = DebateEngine(ELECTRICITY_AGENTS[:3], rag, {}, sector='electricity')
    engine._history = [
        AgentResponse('A', 1, '', 's', 0.00),
        AgentResponse('B', 1, '', 's', 0.15),
        AgentResponse('C', 1, '', 's', 0.30),
    ]
    result = engine.consensus()
    # avg=0.15; only B sits within the 0.10 electricity band -> 1 of 3
    assert result['confidence_pct'] == 33


def test_consensus_uses_gas_band_not_the_narrower_default():
    """Gas's validated band is 0.50 (swarm._AGREEMENT_BAND, forum._BAND); the
    engine's old 0.20 understated classic-mode gas agreement relative to the
    swarm's measurement of the identical unit."""
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:3], rag, {}, sector='gas')
    engine._history = [
        AgentResponse('A', 1, '', 's', 0.00),
        AgentResponse('B', 1, '', 's', 0.30),
        AgentResponse('C', 1, '', 's', 0.60),
    ]
    result = engine.consensus()
    # avg=0.30; all three sit within the 0.50 gas band -> 3 of 3
    assert result['confidence_pct'] == 100


def test_consensus_always_carries_a_subcategories_key():
    """Every consensus() return path -- including the empty-history early
    return -- must carry 'subcategories', so main_window.py can rely on the
    key existing rather than needing a defensive .get() at every call site."""
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag, {}, sector='gas')
    assert engine.consensus()['subcategories'] == {}


def test_consensus_extracts_food_subcategories_from_agent_statements():
    """A food agent's own statement, not just the food judge's, can carry
    category lines now that FOOD_AGENTS' prompt asks for them -- consensus()
    must parse and average them the same way forum.py's judge does."""
    rag = _make_mock_rag()
    engine = DebateEngine(FOOD_AGENTS[:2], rag, {}, sector='food')
    engine._history = [
        AgentResponse('A', 1, '', 'ESTIMATE: +0.4%\nRICE: +0.2%\nMEAT: -0.3%', 0.4),
        AgentResponse('B', 1, '', 'ESTIMATE: +0.2%\nRICE: +0.4%\nFISH: +0.8%', 0.2),
    ]
    result = engine.consensus()
    assert result['subcategories'] == pytest.approx(
        {'rice': 0.3, 'meat': -0.3, 'fish': 0.8})


def test_consensus_subcategories_only_computed_for_food():
    """Gas/electricity agents never carry category lines; even if a gas
    statement happened to contain text matching a category label, it must
    not be reported as a food sub-category read."""
    rag = _make_mock_rag()
    engine = DebateEngine(DEFAULT_AGENTS[:1], rag, {}, sector='gas')
    engine._history = [
        AgentResponse('A', 1, '', 'ESTIMATE: +0.85/L\nRICE: +0.2%', 0.85),
    ]
    assert engine.consensus()['subcategories'] == {}


def test_consensus_subcategory_missing_from_every_agent_is_simply_absent():
    """A category no agent mentioned must not appear as 0.0 or any other
    stand-in value -- absence from the dict IS the honest answer, matching
    forum.py's own _extract_category_percents convention."""
    rag = _make_mock_rag()
    engine = DebateEngine(FOOD_AGENTS[:1], rag, {}, sector='food')
    engine._history = [
        AgentResponse('A', 1, '', 'ESTIMATE: +0.4%\nRICE: +0.2%', 0.4),
    ]
    result = engine.consensus()
    assert 'sugar' not in result['subcategories']
    assert result['subcategories'] == {'rice': 0.2}


def test_food_agent_category_lines_do_not_leak_into_its_own_price_estimate():
    """The exact leak forum.py's judge prompt already had to guard against:
    _extract_percent's prose fallback grabs the first signed percent
    anywhere in the text whenever the anchored ESTIMATE: line fails to
    parse. A food agent whose ESTIMATE line is unparseable, but whose
    category lines ARE parseable, must report price_estimate=None -- not
    silently adopt RICE's value as its own headline estimate."""
    rag = _make_mock_rag()
    engine = DebateEngine(FOOD_AGENTS[:1], rag, {}, sector='food')
    fake_stream = ['ESTIMATE: broadly unchanged\nRICE: +0.2%\nMEAT: -0.3%']
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=iter(fake_stream)):
        engine.run(rounds=1)
    assert engine._history[0].price_estimate is None


def test_food_agents_prompt_requests_all_six_category_lines():
    from ph_economic_ai.engine.debate import _CATEGORY_LABELS
    for agent in FOOD_AGENTS:
        for label in _CATEGORY_LABELS.values():
            assert f'{label}:' in agent.system_prompt


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
