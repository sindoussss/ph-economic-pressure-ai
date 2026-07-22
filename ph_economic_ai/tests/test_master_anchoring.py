"""The physics anchor, wired through MasterJudge end to end.

Proves the experiment does its job on the real path: a hallucinated headline is
pulled back to physics, a blank one is filled, and a sane one is left alone —
all without a live model.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupSurvivor, RegionalVerdict, MasterJudge,
)

SCENARIO = {'oil_pct': 6.8, 'usd_pct': 0.0, 'current_price': 98.0}


def _verdict(estimate):
    return RegionalVerdict(
        judge_id=0, region_pair=('NCR', 'Central Luzon'),
        estimate=estimate, confidence=0.7, reasoning='x',
        survivor_names=('a', 'b'),
    )


def _master(statement):
    judge = MasterJudge(
        verdicts=[_verdict(2.5), _verdict(2.8)],
        rag=MagicMock(),
        scenario=SCENARIO,
        survivors=[],
    )
    with patch('ph_economic_ai.engine.swarm.llm.stream', return_value=[statement]):
        return judge.run()


def test_anchor_is_computed_for_the_scenario():
    mv = _master('ESTIMATE: +₱2.60/L')
    assert mv.physical_anchor == pytest.approx(2.72, abs=0.2)


def test_a_sane_headline_is_left_alone():
    mv = _master('ESTIMATE: +₱2.60/L')
    assert mv.estimate_source == 'agent'
    assert mv.final_estimate == pytest.approx(2.60)


def test_a_hallucinated_headline_is_clamped_to_physics():
    """The exact failure: the judge says +₱12.93/L, the report must not."""
    mv = _master('After analysis. ESTIMATE: +₱12.93/L')
    assert mv.estimate_source == 'clamped'
    assert mv.final_estimate < 6.0           # near the ~2.7 anchor, not 12.93
    assert mv.final_estimate > 2.72          # kept the upward direction


def test_a_blank_headline_falls_back_to_physics_not_none():
    """Previously this produced a bare em dash; now it is the anchor."""
    mv = _master('The regions broadly agree on an increase.')
    assert mv.estimate_source == 'anchor'
    assert mv.final_estimate == pytest.approx(mv.physical_anchor)
    assert mv.final_estimate is not None


def test_the_anchor_is_injected_into_the_prompt():
    """The model must see the physical baseline, not just be corrected after."""
    judge = MasterJudge(verdicts=[_verdict(2.5)], rag=MagicMock(),
                        scenario=SCENARIO, survivors=[])
    prompt = judge._build_prompt()
    joined = ' '.join(m['content'] for m in prompt)
    assert 'MECHANICAL PASS-THROUGH' in joined
    assert '2.7' in joined                   # the computed anchor, in the text
