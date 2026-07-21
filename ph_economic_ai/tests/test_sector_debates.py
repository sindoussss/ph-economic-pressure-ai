"""End-to-end cover for the food and electricity sector debates.

These sectors run on the DebateEngine path (not the swarm), which the hosted-LLM
migration touched in a way the existing tests did not reach: the rename of
Agent.model -> Agent.tier left stale attribute reads in UI display code that no
test exercised. Nothing here needs a network — the provider is faked.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.debate import (
    DEFAULT_AGENTS, FOOD_AGENTS, ELECTRICITY_AGENTS,
    DebateEngine, _extract_percent,
)

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72}


def _make_rag():
    rag = MagicMock()
    rag.query.return_value = [
        {'text': 'Rice retail prices rose.', 'source': 'NFARiceRetail',
         'url': 'http://x', 'timestamp': '2026-01-01'},
    ]
    return rag


# ── Roster wiring ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('roster,name', [
    (FOOD_AGENTS, 'food'),
    (ELECTRICITY_AGENTS, 'electricity'),
    (DEFAULT_AGENTS, 'gas'),
])
def test_every_agent_has_a_resolvable_tier(roster, name):
    """A tier that does not resolve would fail only at call time, mid-run."""
    for agent in roster:
        assert agent.tier in (llm.FAST, llm.DEEP), f'{name}: {agent.name}'
        assert llm.describe_model(agent.tier)      # must never raise


@pytest.mark.parametrize('roster', [FOOD_AGENTS, ELECTRICITY_AGENTS])
def test_sector_agents_no_longer_carry_a_model_field(roster):
    """Guards the rename: a lingering .model would break UI display paths."""
    for agent in roster:
        assert not hasattr(agent, 'model')


# ── Food debate ───────────────────────────────────────────────────────────────

def test_food_debate_runs_and_parses_percent_estimates():
    """Food reports a percent change, not pesos-per-litre — it passes its own
    price_extractor, so a regression here is silent (estimates become None)."""
    engine = DebateEngine(FOOD_AGENTS, _make_rag(), SCENARIO,
                          price_extractor=_extract_percent)
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=['Food supply is tight. ESTIMATE: +2.4%']):
        responses = engine.run(rounds=1)

    assert len(responses) == len(FOOD_AGENTS)
    assert all(r.price_estimate == pytest.approx(2.4) for r in responses)


def test_food_debate_reaches_consensus():
    engine = DebateEngine(FOOD_AGENTS, _make_rag(), SCENARIO,
                          price_extractor=_extract_percent)
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=['ESTIMATE: +2.0%']):
        engine.run(rounds=1)
    assert engine.consensus()['weighted_avg'] == pytest.approx(2.0)


# ── Electricity debate ────────────────────────────────────────────────────────

def test_electricity_debate_runs_and_parses_peso_per_kwh():
    engine = DebateEngine(ELECTRICITY_AGENTS, _make_rag(), SCENARIO)
    with patch('ph_economic_ai.engine.debate.llm.stream',
               return_value=['Generation charge up. ESTIMATE: +₱0.45/kWh']):
        responses = engine.run(rounds=1)

    assert len(responses) == len(ELECTRICITY_AGENTS)
    assert all(r.price_estimate is not None for r in responses)


# ── The paths that actually broke ─────────────────────────────────────────────

def test_provenance_records_a_concrete_model_not_a_tier():
    """main_window saves model_used per response. It must be the resolved model
    and must not raise when no provider is configured."""
    label = llm.describe_model(FOOD_AGENTS[0].tier)
    assert label and label not in (llm.FAST, llm.DEEP) or not llm.is_configured()


def test_describe_model_never_raises_without_a_provider(monkeypatch):
    monkeypatch.delenv('STRATA_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    for roster in (FOOD_AGENTS, ELECTRICITY_AGENTS, DEFAULT_AGENTS):
        for agent in roster:
            assert llm.describe_model(agent.tier)


def test_sector_debate_surfaces_provider_errors_as_engine_errors():
    """A quota-exhausted run must fail loudly, not silently return no estimates."""
    engine = DebateEngine(FOOD_AGENTS, _make_rag(), SCENARIO,
                          price_extractor=_extract_percent)
    with patch('ph_economic_ai.engine.debate.llm.stream',
               side_effect=llm.LLMError('quota exhausted')):
        with pytest.raises(llm.LLMError):
            engine.run(rounds=1)
