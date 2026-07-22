"""Food and electricity anchoring, wired through the debate engine.

Proves the sector experiment on the real path: the anchor is injected into the
agent prompt, and the sector estimate is reconciled against it — including the
fallback that keeps food and electricity from ever going blank.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.debate import DebateEngine, FOOD_AGENTS, _extract_percent


def _rag():
    r = MagicMock()
    r.query.return_value = [{'source': 'PSA', 'text': 'food prices', 'url': 'u', 'timestamp': 't'}]
    return r


# ── Prompt injection ──────────────────────────────────────────────────────────

def test_anchor_note_reaches_the_agent_prompt():
    engine = DebateEngine(
        FOOD_AGENTS[:1], _rag(), {'oil_pct': 5.0},
        price_extractor=_extract_percent,
        anchor_note='BASELINE: food runs about +0.60% per month.',
    )
    prompt = engine._build_prompt(FOOD_AGENTS[0], round_num=1)
    joined = ' '.join(m['content'] for m in prompt)
    assert 'BASELINE: food runs about +0.60%' in joined


def test_no_anchor_note_leaves_the_prompt_clean():
    engine = DebateEngine(FOOD_AGENTS[:1], _rag(), {'oil_pct': 5.0},
                          price_extractor=_extract_percent)
    prompt = engine._build_prompt(FOOD_AGENTS[0], round_num=1)
    assert 'BASELINE' not in ' '.join(m['content'] for m in prompt)


# ── Reconciliation behaviour, sector units ────────────────────────────────────

def test_food_wild_estimate_is_clamped_to_its_trend():
    """A +7.6% food reading (the old bug's magnitude) is pulled to the trend."""
    rec = anchoring.reconcile_estimate(
        7.6, anchor=0.8, tolerance=anchoring.FOOD_TOLERANCE_PCT)
    assert rec.source == 'clamped'
    assert rec.value == pytest.approx(2.3)      # 0.8 + 1.5, not 7.6


def test_food_blank_falls_back_to_the_trend_not_none():
    rec = anchoring.reconcile_estimate(
        None, anchor=0.8, tolerance=anchoring.FOOD_TOLERANCE_PCT)
    assert rec.value == pytest.approx(0.8)
    assert rec.source == 'anchor'


def test_electricity_wild_estimate_is_clamped():
    rec = anchoring.reconcile_estimate(
        2.5, anchor=0.20, tolerance=anchoring.ELECTRICITY_TOLERANCE_PHP_KWH)
    assert rec.source == 'clamped'
    assert rec.value == pytest.approx(0.60)     # 0.20 + 0.40


def test_explain_names_the_food_anchor_as_persistence():
    rec = anchoring.reconcile_estimate(0.9, anchor=0.8, tolerance=1.5)
    msg = anchoring.explain(rec, unit='%', anchor_label='own-trend persistence')
    assert '%' in msg
    assert 'persistence' in msg


def test_explain_names_the_electricity_anchor_as_pass_through():
    rec = anchoring.reconcile_estimate(0.25, anchor=0.20, tolerance=0.40)
    msg = anchoring.explain(rec, unit='₱/kWh', anchor_label='fuel pass-through')
    assert '₱/kWh' in msg
    assert 'pass-through' in msg
