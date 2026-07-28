"""A re-run on unchanged inputs must ask the model the same question.

Recall stops the app re-answering a question it has already answered. This is
the other half: when a run genuinely does happen twice, it should reproduce.
ADR-002 claimed that and could not deliver it, for two reasons pinned here.

1. Every agent, judge and master prompt was prefixed with a DATA BRIEF whose
   header printed `fetched_at` at minute resolution. A seed only reproduces a
   call when the prompt is identical, so two runs a minute apart could not agree
   even in principle. Runs 21, 22 and 23 had identical stored scenarios, and
   therefore identical seeds, and returned -0.60, -1.12 and -2.14.

2. The seeds were keyed on the raw scenario dict, whose `oil_pct`, `usd_pct` and
   `demand_index` are recomputed from live Yahoo and Open-Meteo values on every
   run. Five of the eight runs on 2026-07-27 had a distinct scenario and so a
   distinct seed for every agent in the swarm.

And the gap nobody had noticed: the food and electricity debates passed no seed
at all.
"""
from unittest.mock import MagicMock

import pytest

from ph_economic_ai.engine import swarm, vintage
from ph_economic_ai.engine.debate import DebateEngine
from ph_economic_ai.engine.live_data import LiveDataBrief

SCENARIO = {'oil_pct': -3.31, 'usd_pct': -0.06, 'bsp_rate': 6.5,
            'demand_index': 74.2, 'current_price': 98.82}
# Same market, later in the day: the ticks moved a little, nothing happened.
DRIFTED = {'oil_pct': -3.38, 'usd_pct': -0.05, 'bsp_rate': 6.5,
           'demand_index': 74.3, 'current_price': 98.83}
MOVED = {'oil_pct': -1.10, 'usd_pct': -0.05, 'bsp_rate': 6.5,
         'demand_index': 74.2, 'current_price': 98.82}


# ── The clock is out of the prompt ────────────────────────────────────────────

def _brief(fetched_at):
    b = LiveDataBrief()
    b.brent, b.wti, b.usd_php = 68.4, 64.1, 58.2
    b.fetched_at = fetched_at
    b._ok = True
    return b


def test_the_data_brief_no_longer_carries_a_clock_into_the_prompt():
    """The decisive one. This block prefixes every prompt in the run."""
    first = _brief('2026-07-28 04:11 UTC').as_prompt_block(SCENARIO)
    later = _brief('2026-07-28 04:57 UTC').as_prompt_block(SCENARIO)
    assert first == later
    assert '04:11' not in first


def test_the_brief_still_names_the_day_it_describes():
    """Removing the timestamp must not leave the agents undated."""
    import datetime as dt
    block = _brief('2026-07-28 04:11 UTC').as_prompt_block(SCENARIO)
    today = dt.datetime.now(vintage.price_calendar.PH_TZ).date().isoformat()
    assert today in block


def test_fetched_at_is_still_recorded_for_the_report():
    """It was only ever wrong in the prompt. A human still wants to know when the
    data was pulled."""
    assert _brief('2026-07-28 04:11 UTC').fetched_at == '2026-07-28 04:11 UTC'


# ── Seeds hold still while the market does ────────────────────────────────────

def test_the_swarm_seed_holds_still_for_the_day():
    """Two runs an hour apart must ask the sampler the same question.

    The seed keys on the window, not the scenario. Quantising the scenario onto
    a grid was tried and rejected: runs 27 and 28 sat 0.07 percentage points
    apart across a grid line and still got different seeds."""
    assert (swarm._vintage_seed(0, 'NCR Forecaster')
            == swarm._vintage_seed(0, 'NCR Forecaster'))


def test_the_market_signal_lives_in_the_prompt_not_the_seed():
    """A stable seed is not a frozen answer. Moving the market still moves the
    number, because the market numbers are in the prompt."""
    quiet = _brief('2026-07-28 04:11 UTC')
    loud = _brief('2026-07-28 04:11 UTC')
    loud.brent = 82.0
    assert quiet.as_prompt_block(SCENARIO) != loud.as_prompt_block(SCENARIO)
    assert '82.00' in loud.as_prompt_block(SCENARIO)


def test_agents_still_get_different_seeds_from_each_other():
    """Stability must not collapse the roster into one voice."""
    seeds = {swarm._vintage_seed(g, name)
             for g in range(4)
             for name in ('Forecaster', 'Critic', 'Synthesizer')}
    assert len(seeds) == 12


def test_a_retry_does_not_resample_the_same_answer():
    assert (swarm._vintage_seed(0, 'NCR Forecaster')
            != swarm._vintage_seed(0, 'NCR Forecaster', 'retry'))


def test_the_judges_do_not_answer_themselves_identically():
    tags = {swarm._vintage_seed(0, t) for t in ('defense1', 'defense2', 'synthesis')}
    assert len(tags) == 3
    assert swarm._vintage_seed('master') not in tags


def test_a_new_pricing_week_reseeds(monkeypatch):
    """A stable seed must not outlive the window it is stable within."""
    first = swarm._vintage_seed('master')
    monkeypatch.setattr(swarm.vintage, 'vintage',
                        lambda now=None: {'fuel_cycle': '2026-08-04',
                                          'day': '2026-08-05'})
    assert swarm._vintage_seed('master') != first


def test_the_sector_debates_are_seeded_at_all():
    """`engine/debate.py` passed no seed on any call, so Ollama drew a fresh
    random one per call at temperature 0.2. Food and electricity were
    unreproducible even when the gas number on the same run row was seeded."""
    engine = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    assert isinstance(engine._seed('agent', 1), int)


def test_a_sector_seed_holds_still_across_market_noise():
    """Same window, same sampler question, whatever the ticks did."""
    food = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    later = DebateEngine([], MagicMock(), DRIFTED, sector='food')
    assert food._seed('Rice Analyst', 1) == later._seed('Rice Analyst', 1)


def test_food_and_electricity_do_not_answer_in_unison():
    """Two DebateEngines run in one process on the same scenario. Without the
    sector name they would request identical seeds."""
    food = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    elec = DebateEngine([], MagicMock(), SCENARIO, sector='electricity')
    assert food._seed('Analyst', 1) != elec._seed('Analyst', 1)


def test_a_sector_agent_gets_a_different_seed_each_round():
    food = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    assert food._seed('Analyst', 1) != food._seed('Analyst', 2)


def test_two_follow_up_questions_are_not_the_same_call():
    food = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    assert (food._seed('ask', 'A', 'why rice?')
            != food._seed('ask', 'A', 'why not rice?'))


@pytest.mark.parametrize('scenario', [SCENARIO, DRIFTED, MOVED, {}])
def test_the_seed_no_longer_depends_on_the_scenario_at_all(scenario):
    """ADR-002's actual requirement, now met: the sampler asks the same question
    all day. What the run is ABOUT still varies, through the prompt."""
    engine = DebateEngine([], MagicMock(), scenario, sector='food')
    baseline = DebateEngine([], MagicMock(), SCENARIO, sector='food')
    assert engine._seed('Analyst', 1) == baseline._seed('Analyst', 1)
