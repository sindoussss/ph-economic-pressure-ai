"""Physics-anchored estimation — the core of the weak-LLM experiment.

The claim being tested: a deterministic pass-through formula supplies the
magnitude small models get wrong, and reconciliation keeps every headline
physically grounded. All pure, no model, no network.
"""
import pytest

from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.anchoring import (
    fuel_passthrough_anchor, electricity_passthrough_anchor,
    food_persistence_anchor, reconcile_estimate, explain,
)


# ── The pass-through formula ──────────────────────────────────────────────────

def test_the_shock_that_broke_the_swarm_anchors_near_two_pesos():
    """+6.8% oil, no FX move: the swarm answered +₱12.93/L; physics says ~₱2.7."""
    anchor = fuel_passthrough_anchor(oil_pct=6.8, usd_pct=0.0,
                                     brent_usd=98.0, fx_php_per_usd=58.0)
    assert anchor == pytest.approx(2.72, abs=0.1)


def test_no_shock_means_no_change():
    assert fuel_passthrough_anchor(0.0, 0.0) == pytest.approx(0.0)


def test_oil_and_fx_shocks_add():
    """Both act on the same crude-cost base, so equal shocks contribute equally."""
    oil_only = fuel_passthrough_anchor(5.0, 0.0)
    fx_only = fuel_passthrough_anchor(0.0, 5.0)
    both = fuel_passthrough_anchor(5.0, 5.0)
    assert oil_only == pytest.approx(fx_only)
    assert both == pytest.approx(oil_only + fx_only)


def test_a_price_fall_gives_a_negative_anchor():
    assert fuel_passthrough_anchor(-4.0, 0.0) < 0


def test_vat_is_included():
    """The pump number carries 12% VAT over the bare landed-cost change."""
    with_vat = fuel_passthrough_anchor(10.0, 0.0, brent_usd=100.0, fx_php_per_usd=60.0)
    bare = 100.0 * 60.0 / anchoring._LITRES_PER_BARREL * 0.10
    assert with_vat == pytest.approx(bare * 1.12)


def test_scale_is_robust_to_input_noise():
    """A ±$5 error in Brent must not move the anchor by more than pennies —
    this is why stale reference inputs are acceptable."""
    a = fuel_passthrough_anchor(6.8, 0.0, brent_usd=98.0, fx_php_per_usd=58.0)
    b = fuel_passthrough_anchor(6.8, 0.0, brent_usd=103.0, fx_php_per_usd=58.0)
    assert abs(a - b) < 0.20


# ── Electricity: physical fuel pass-through (a validated signal) ──────────────

def test_electricity_anchor_scales_with_the_fuel_shock():
    a = electricity_passthrough_anchor(oil_pct=6.8, usd_pct=0.0)
    assert a == pytest.approx(5.50 * 0.55 * 0.068, abs=0.01)
    assert 0.0 < a < 0.5              # sane ₱/kWh magnitude for a moderate shock


def test_electricity_oil_and_fx_add():
    oil = electricity_passthrough_anchor(5.0, 0.0)
    fx = electricity_passthrough_anchor(0.0, 5.0)
    assert electricity_passthrough_anchor(5.0, 5.0) == pytest.approx(oil + fx)


def test_electricity_fall_is_negative():
    assert electricity_passthrough_anchor(-5.0, 0.0) < 0


# ── Food: own-persistence, NOT a commodity pass-through ───────────────────────

def test_food_anchor_is_the_trailing_trend():
    """The benchmark says food is a null on commodities but predictable from its
    own dynamics — so the anchor is the recent mean, not a fuel formula."""
    a = food_persistence_anchor([0.4, 0.6, 0.8], oil_pct=0.0)
    assert a == pytest.approx(0.6)


def test_food_barely_moves_with_oil():
    """Fuel must be a weak driver for food — anchoring it to oil would be
    anchoring to what the backtest proved is noise."""
    flat = food_persistence_anchor([0.5, 0.5], oil_pct=0.0)
    shocked = food_persistence_anchor([0.5, 0.5], oil_pct=10.0)
    assert abs(shocked - flat) < 0.5     # a +10% oil move nudges food <0.5ppt


def test_food_anchor_survives_no_history():
    a = food_persistence_anchor([], oil_pct=0.0)
    assert a == pytest.approx(anchoring._FOOD_DEFAULT_MOM_PCT)


def test_each_sector_anchor_is_a_different_kind():
    """The experiment's core: fuel/electricity are pass-throughs, food is not."""
    # electricity responds strongly to oil; food barely does
    elec_sensitivity = abs(electricity_passthrough_anchor(10, 0)
                           - electricity_passthrough_anchor(0, 0))
    food_sensitivity = abs(food_persistence_anchor([0.5], 10)
                           - food_persistence_anchor([0.5], 0))
    assert elec_sensitivity > food_sensitivity


# ── Reconciliation ────────────────────────────────────────────────────────────

def test_model_estimate_near_physics_is_trusted():
    r = reconcile_estimate(llm_estimate=3.0, anchor=2.7)
    assert r.source == 'agent'
    assert r.value == 3.0
    assert not r.used_physics


def test_wild_overestimate_is_clamped_toward_physics():
    """The +₱12.93 case: keep the (correct) upward direction, drop the magnitude."""
    r = reconcile_estimate(llm_estimate=12.93, anchor=2.72, tolerance=2.0)
    assert r.source == 'clamped'
    assert r.value == pytest.approx(4.72)      # anchor + tolerance, not 12.93
    assert r.used_physics


def test_clamp_preserves_direction_downward():
    r = reconcile_estimate(llm_estimate=-9.0, anchor=-1.0, tolerance=2.0)
    assert r.value == pytest.approx(-3.0)      # anchor - tolerance
    assert r.value < 0


def test_missing_estimate_falls_back_to_the_anchor():
    """The blank-report case: physics stands in instead of showing nothing."""
    r = reconcile_estimate(llm_estimate=None, anchor=2.72)
    assert r.source == 'anchor'
    assert r.value == pytest.approx(2.72)
    assert r.used_physics


def test_reconciliation_never_returns_none():
    """Every path yields a number — that is the point of the fallback."""
    for est in (None, 2.5, 50.0, -50.0):
        assert reconcile_estimate(est, anchor=2.72).value is not None


def test_explain_covers_each_branch():
    for est, expect in [(2.8, 'consistent'), (12.9, 'diverged'), (None, 'No usable')]:
        msg = explain(reconcile_estimate(est, anchor=2.72))
        assert expect.lower() in msg.lower()
