"""A discarded estimate must be visibly discarded.

The plausibility guard silently returned None, so the report rendered a bare
em dash. A reader could not tell "the judge said something impossible and we
threw it away" from "the judge crashed" — and in a run where both regional
judges were filtered, the headline consensus stood alone with no explanation.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupSurvivor, RegionalJudge, parse_fuel_estimate, _extract_fuel_change,
    _MAX_REALISTIC_FUEL_CHANGE,
)
from ph_economic_ai.ui.stage4_report import _missing_estimate_note


# ── parse_fuel_estimate ───────────────────────────────────────────────────────

def test_plausible_value_is_accepted():
    accepted, rejected = parse_fuel_estimate('ESTIMATE: +₱2.54/L')
    assert accepted == pytest.approx(2.54)
    assert rejected is None


def test_implausible_value_is_reported_not_silently_dropped():
    """This is the whole point — the number survives as evidence."""
    over = _MAX_REALISTIC_FUEL_CHANGE + 5
    accepted, rejected = parse_fuel_estimate(f'ESTIMATE: +₱{over:.2f}/L')
    assert accepted is None
    assert rejected == pytest.approx(over)


def test_absent_number_is_distinguishable_from_a_rejected_one():
    accepted, rejected = parse_fuel_estimate('The regions broadly agree.')
    assert accepted is None and rejected is None


def test_negative_implausible_value_is_also_captured():
    accepted, rejected = parse_fuel_estimate('ESTIMATE: -₱92.30/L')
    assert accepted is None
    assert rejected == pytest.approx(-92.30)


def test_legacy_extractor_still_returns_only_the_accepted_value():
    """Existing callers must be unaffected by the richer return type."""
    assert _extract_fuel_change('ESTIMATE: +₱2.54/L') == pytest.approx(2.54)
    assert _extract_fuel_change('ESTIMATE: +₱99.00/L') is None


def test_rejection_is_logged_with_the_value(caplog):
    """Without this the only record of what the judge said is gone."""
    with caplog.at_level('INFO'):
        parse_fuel_estimate('ESTIMATE: +₱12.93/L')
    assert '12.93' in caplog.text


# ── The verdict carries it ────────────────────────────────────────────────────

def _judge():
    def surv(gid, region, est):
        return GroupSurvivor(gid, region, AgentResponse(f'{region} F', 1, '', 'x', est),
                             0.8, 'Forecaster', 'qwen2.5:3b')
    return RegionalJudge(
        judge_id=0,
        survivors=(surv(0, 'NCR', 1.5), surv(1, 'Davao Region', 1.6)),
        rag=MagicMock(),
        scenario={'oil_pct': 5.0},
        agent_estimates=[1.5, 1.6],
    )


def test_verdict_records_the_rejected_value():
    with patch('ph_economic_ai.engine.swarm.llm.stream',
               return_value=['ESTIMATE: +₱12.93/L']):
        verdict = _judge().run()
    assert verdict.estimate is None
    assert verdict.rejected_estimate == pytest.approx(12.93)


def test_verdict_leaves_it_unset_on_a_good_estimate():
    with patch('ph_economic_ai.engine.swarm.llm.stream',
               return_value=['ESTIMATE: +₱2.54/L']):
        verdict = _judge().run()
    assert verdict.estimate == pytest.approx(2.54)
    assert verdict.rejected_estimate is None


# ── What the report says ──────────────────────────────────────────────────────

def test_note_is_empty_when_there_is_an_estimate():
    assert _missing_estimate_note(2.54, None) == ''


def test_note_quotes_the_discarded_value_and_the_bound():
    note = _missing_estimate_note(None, 12.93)
    assert '12.93' in note
    assert str(int(_MAX_REALISTIC_FUEL_CHANGE)) in note
    assert 'consensus' in note


def test_note_distinguishes_no_answer_from_a_bad_answer():
    assert _missing_estimate_note(None, None) != _missing_estimate_note(None, 12.93)
    assert 'no parseable estimate' in _missing_estimate_note(None, None)
