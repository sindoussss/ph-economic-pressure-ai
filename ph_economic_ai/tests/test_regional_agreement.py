"""The regional 'agent agreement' figure must be measured, not assumed.

It was a hardcoded 0.75 whenever the estimate merely parsed, which the report
displayed as "Agent agreement: 75%" — identical on every card. For a project
whose stated value is honesty about its limits, a constant dressed as a
measurement is the worst kind of number to ship.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupSurvivor, RegionalJudge, _robust_confidence_pct,
)

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}


def _survivor(group_id: int, region: str, estimate: float) -> GroupSurvivor:
    return GroupSurvivor(
        group_id=group_id,
        region_name=region,
        response=AgentResponse(f'{region} Forecaster', 1, '', 'x', estimate),
        combined_score=0.8,
        agent_role='Forecaster',
        agent_model='qwen2.5:3b',
    )


def _judge(agent_estimates):
    return RegionalJudge(
        judge_id=0,
        survivors=(_survivor(0, 'NCR', 1.5), _survivor(1, 'Davao Region', 1.6)),
        rag=MagicMock(),
        scenario=SCENARIO,
        agent_estimates=agent_estimates,
    )


def _run(judge, statement='ESTIMATE: +₱1.50/L'):
    with patch('ph_economic_ai.engine.swarm.llm.stream', return_value=[statement]):
        return judge.run()


def test_tight_agreement_scores_higher_than_scattered():
    """The whole point: the number has to respond to the actual spread."""
    tight = _run(_judge([1.45, 1.50, 1.52, 1.55, 1.48])).confidence
    scattered = _run(_judge([-3.0, 0.2, 1.5, 4.0, 7.5])).confidence
    assert tight > scattered


def test_agreement_is_not_the_old_hardcoded_value():
    """0.75 used to be returned for any parseable estimate."""
    verdict = _run(_judge([1.45, 1.50, 1.52, 1.55, 1.48]))
    assert verdict.confidence != 0.75


def test_two_judges_with_different_spreads_do_not_report_the_same_number():
    """Both cards previously read 75% because both were the same constant."""
    a = _run(_judge([1.50, 1.51, 1.49])).confidence
    b = _run(_judge([-2.0, 1.5, 6.0])).confidence
    assert a != b


def test_no_agent_estimates_yields_no_confidence_rather_than_a_default():
    """Nothing to measure must not silently become a plausible-looking number."""
    assert _run(_judge([])).confidence == 0.0


def test_confidence_is_a_fraction_for_the_report():
    """stage4_report formats this with :.0% — an int would render as 15000%."""
    verdict = _run(_judge([1.5, 1.6]))
    assert 0.0 <= verdict.confidence <= 1.0


def test_matches_the_master_verdict_metric():
    """Regional and master agreement should mean the same thing."""
    estimates = [1.45, 1.50, 1.52]
    verdict = _run(_judge(estimates))
    assert verdict.confidence == pytest.approx(
        _robust_confidence_pct(estimates, verdict.estimate) / 100
    )


def test_unparseable_verdict_still_reports_measured_agreement():
    """A judge that fails to state an estimate should not also lose the
    agreement signal from its agents."""
    verdict = _run(_judge([1.5, 1.5, 1.5]), statement='no number here')
    assert verdict.estimate is None
    assert verdict.confidence > 0
