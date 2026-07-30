"""Every number the app shows a user, re-derived independently.

Written after an audit found that the most consequential figure in the store was
fiction. Every graded run recorded an "actual price change" of exactly
-14.44 PHP/L: the observed price of 84.38 minus the 98.82 fallback constant that
the stored scenario carried instead of the price the run actually reasoned from.
Nothing rejected it, so `compute_accuracy_score` floored at zero for every agent,
`_update_trust_no_commit` folded that in at 60 percent weight, and trust
collapsed to 0.4 x internal. Seven of twenty agents fell below the 0.30 demotion
threshold and were benched on an outcome that never happened.

The formulas themselves were correct. What was missing was any check that their
INPUTS were real, so these tests assert properties and plausibility rather than
just re-running the arithmetic.
"""
import math
import random

import pytest

from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.ground_truth import compute_accuracy_score
from ph_economic_ai.engine.interval import conformal_halfwidth
from ph_economic_ai.engine.store import (
    _EMA_ALPHA, _TRUST_INIT, AgentTrustStore, trust_tier)
from ph_economic_ai.engine.swarm import (
    _robust_confidence_pct, compute_combined_score, derive_regional_estimates)

_LITRES_PER_BARREL = 158.987
_VAT = 0.12
_CALIBRATION = 0.79


# ── The physics anchor ────────────────────────────────────────────────────────

def test_the_fuel_anchor_matches_an_independent_derivation():
    """Crude cost per litre, shocked, VAT-ed, calibrated — computed here from
    first principles rather than by calling the same code twice."""
    brent, fx, oil, usd = 87.14, 61.61, -8.0, -0.1
    crude_per_litre = brent * fx / _LITRES_PER_BARREL
    expected = crude_per_litre * (oil + usd) / 100.0 * (1 + _VAT) * _CALIBRATION
    got = anchoring.fuel_passthrough_anchor(oil, usd, brent_usd=brent,
                                            fx_php_per_usd=fx)
    assert got == pytest.approx(expected, abs=1e-9)
    assert got == pytest.approx(-2.42, abs=0.01)


def test_a_weaker_peso_raises_the_pump_price():
    """The sign convention that would be silently catastrophic if inverted.
    `usd_pct` is the change in PHP per USD, so positive means a weaker peso,
    which makes dollar-priced crude cost more in pesos."""
    weaker = anchoring.fuel_passthrough_anchor(0.0, +5.0)
    stronger = anchoring.fuel_passthrough_anchor(0.0, -5.0)
    assert weaker > 0 > stronger
    assert weaker == pytest.approx(-stronger)


def test_oil_and_fx_enter_on_equal_footing():
    """Both act on the same peso-denominated crude cost, so to first order a
    point of each is worth the same."""
    assert (anchoring.fuel_passthrough_anchor(3.0, 0.0)
            == pytest.approx(anchoring.fuel_passthrough_anchor(0.0, 3.0)))


def test_the_linearisation_error_stays_under_the_quoting_resolution():
    """The anchor ADDS the two shocks; the exact form compounds them.

    The approximation is deliberate: `_FUEL_PASSTHROUGH_CALIBRATION` (0.79) was
    fitted by OLS against this linear form in `tools/anchor_backtest.py`, so
    swapping in exact compounding without re-fitting would change one half of a
    fitted product and is worse math rather than better.

    What must hold is that the drift stays below the resolution the answer is
    quoted at. Models emit on a rough half-peso grid, so anything under ~₱0.05 is
    invisible. This bounds it over the shock range the scenarios actually
    produce; the drift grows with shock size, so if the app ever accepts moves
    beyond about ±10 percent, the anchor should move to exact compounding and the
    calibration should be re-fitted alongside it.
    """
    brent, fx = 87.14, 61.61
    base = brent * fx / _LITRES_PER_BARREL

    def drift(oil, usd):
        exact = (1 + oil / 100) * (1 + usd / 100) - 1
        return abs(base * (exact - (oil + usd) / 100) * (1 + _VAT) * _CALIBRATION)

    # Shocks the live brief actually derives from five-day Brent and FX history.
    for oil, usd in ((-8.0, -0.1), (5.0, 2.0), (-3.3, -0.06)):
        assert drift(oil, usd) < 0.05, f'{oil}/{usd} drifts {drift(oil, usd):.4f}'


def test_the_linearisation_limit_is_documented_not_hidden():
    """The drift grows with the product of the two shocks, and past roughly
    ±10 percent it exceeds the half-peso grid the models quote on.

    Pinned so the boundary is a known property rather than a surprise. Crossing
    it is not a bug to patch in isolation: the fix is exact compounding AND a
    re-fit of `_FUEL_PASSTHROUGH_CALIBRATION`, because the constant was fitted
    against the linear form and the two are a matched pair.
    """
    brent, fx = 87.14, 61.61
    base = brent * fx / _LITRES_PER_BARREL

    def drift(oil, usd):
        exact = (1 + oil / 100) * (1 + usd / 100) - 1
        return abs(base * (exact - (oil + usd) / 100) * (1 + _VAT) * _CALIBRATION)

    assert drift(10.0, 3.0) > 0.05, 'the documented limit moved; re-check the note'
    assert drift(20.0, 5.0) > 0.20
    assert drift(3.0, 1.0) < 0.02


def test_a_zero_shock_moves_nothing():
    assert anchoring.fuel_passthrough_anchor(0.0, 0.0) == pytest.approx(0.0)
    assert anchoring.electricity_passthrough_anchor(0.0, 0.0) == pytest.approx(0.0)


def test_reconciliation_keeps_direction_when_it_clamps():
    """Clamping must bound the magnitude without inventing a direction."""
    anchor = -2.42
    for wild in (-12.0, 12.0):
        r = anchoring.reconcile_estimate(wild, anchor, tolerance=2.0)
        assert r.source == 'clamped'
        assert abs(r.value - anchor) == pytest.approx(2.0)
        assert (r.value > anchor) == (wild > anchor)


# ── The conformal band ────────────────────────────────────────────────────────

def test_conformal_uses_the_finite_sample_index():
    """ceil((n+1)*level), not a plain percentile — that is what carries the
    small-sample coverage guarantee."""
    errors = [float(i) for i in range(1, 11)]        # 1..10
    assert conformal_halfwidth(errors, 0.9) == errors[math.ceil(11 * 0.9) - 1]


def test_conformal_covers_at_least_its_stated_level():
    """The property that makes the band worth showing at all."""
    random.seed(11)
    for level in (0.5, 0.9):
        hits = trials = 0
        for _ in range(3000):
            calib = [abs(random.gauss(0, 1)) for _ in range(30)]
            fresh = abs(random.gauss(0, 1))
            hits += fresh <= conformal_halfwidth(calib, level)
            trials += 1
        assert hits / trials >= level - 0.02, f'level {level} under-covered'


def test_too_few_points_widen_rather_than_narrow():
    """Below the rank needed for a level, the band must not silently report a
    tighter number than the data supports."""
    assert conformal_halfwidth([1.0, 2.0], 0.95) == 2.0


# ── The agreement metric ──────────────────────────────────────────────────────

def test_agreement_is_bounded():
    random.seed(3)
    for _ in range(400):
        vals = [random.uniform(-8, 8) for _ in range(random.randint(2, 20))]
        assert 0 <= _robust_confidence_pct(vals, None) <= 100


def test_agreement_measures_spread_not_level():
    """Shift-invariant WITHIN the plausibility band.

    It is not invariant across it, and that is the metric working rather than
    failing: `_is_realistic_fuel_change` drops anything beyond ±₱8 before
    scoring, so a shift that pushes estimates over the edge changes the
    population being measured. Worth pinning explicitly, because the naive
    version of this test shifts a room from inside the band to across it and
    reports a real filter as a broken formula."""
    random.seed(3)
    for _ in range(400):
        vals = [random.uniform(-4, 4) for _ in range(random.randint(2, 20))]
        score = _robust_confidence_pct(vals, None)
        shifted = _robust_confidence_pct([v + 2.0 for v in vals], None)
        assert score == shifted


def test_the_plausibility_filter_is_what_breaks_invariance():
    """The control for the test above: name the cause rather than assume it."""
    inside = [-3.0, -3.5, -4.0]
    assert (_robust_confidence_pct(inside, None)
            == _robust_confidence_pct([v + 1.0 for v in inside], None))
    # +6 pushes -3.0 to +3.0 but -4.0 to +2.0; all still inside, so still equal.
    # +12 pushes every value past +8, leaving nothing to measure.
    assert _robust_confidence_pct([v + 12.0 for v in inside], None) == 0


def test_agreement_never_rises_as_the_room_spreads_out():
    scores = [_robust_confidence_pct([-2.0, -2.0 + gap], None)
              for gap in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)]
    assert scores == sorted(scores, reverse=True)


def test_zero_stays_reserved_for_not_measurable():
    """A real room cannot reach 0, because the centre always agrees with itself.
    That is what lets 0 mean "nothing to measure" without ambiguity."""
    assert _robust_confidence_pct([-8.0, 8.0], None) > 0
    assert _robust_confidence_pct([-8.0, 0.0, 8.0], None) > 0
    assert _robust_confidence_pct([-1.0], None) == 0
    assert _robust_confidence_pct([], None) == 0


# ── The elimination score ─────────────────────────────────────────────────────

def _score(estimate, critic=0.5, conf=0.5, group=(0.0, 1.0)):
    return compute_combined_score(
        AgentResponse('a', 1, '', 'x', estimate), critic, conf, list(group))


def test_the_combined_score_cannot_go_negative():
    """A negative would sort BELOW the 0.0 given to an agent that produced no
    estimate, inverting the elimination and cutting the agents who answered."""
    random.seed(5)
    for _ in range(2000):
        s = _score(random.uniform(-9, 9),
                   random.random(), random.random(),
                   [random.uniform(-5, 5) for _ in range(4)])
        assert 0.0 <= s <= 1.0


def test_an_estimate_outside_its_own_group_is_clamped_not_negative():
    assert _score(50.0, group=(0.0, 1.0)) >= 0.0


def test_no_estimate_scores_zero():
    assert _score(None) == 0.0


# ── Grading: the inputs, not just the arithmetic ──────────────────────────────

def test_an_implausible_outcome_is_refused(tmp_path):
    """The audit's central finding. A stored baseline of 98.82 against an
    observed 84.38 implies a -14.44 weekly move, which is not a market event."""
    from ph_economic_ai.engine import ground_truth as gt
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    run_id = store.save_run(scenario={'current_price': 98.82}, final_estimate=-1.0,
                            confidence_pct=50, horizon_days=-1.0)
    assert gt.find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 0
    assert store.get_run(run_id)['actual_price_change'] is None
    store.close()


def test_a_real_outcome_is_still_graded(tmp_path):
    """The guard must not stop the app grading itself."""
    from ph_economic_ai.engine import ground_truth as gt
    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    run_id = store.save_run(scenario={'current_price': 85.00}, final_estimate=-0.5,
                            confidence_pct=50, horizon_days=-1.0)
    assert gt.find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 1
    row = store.get_run(run_id)
    assert row['actual_price_change'] == pytest.approx(-0.62)
    assert row['accuracy_error'] == pytest.approx(0.12, abs=0.01)
    store.close()


def test_the_swarm_reports_the_price_it_reasoned_from():
    """The orchestrator scrapes the live price into its own copy of the scenario.
    Without carrying it back, the caller stores a baseline the run never saw."""
    from ph_economic_ai.engine.swarm import MasterVerdict
    mv = MasterVerdict(final_estimate=-1.0, confidence_pct=50,
                       dissenting_regions=[], reasoning='', regional_verdicts=[])
    assert hasattr(mv, 'current_price')


def test_accuracy_saturates_and_the_docstring_says_so():
    """Not a defect, but a limit a reader of a trust score must know: beyond a
    ₱3 error every miss scores the same."""
    assert compute_accuracy_score(0, 3.0) == 0.0
    assert compute_accuracy_score(0, 20.0) == 0.0
    assert compute_accuracy_score(0, 0.0) == 1.0
    assert 'floor' in compute_accuracy_score.__doc__


# ── Trust ─────────────────────────────────────────────────────────────────────

def test_the_ema_converges_to_its_input():
    trust = _TRUST_INIT
    for _ in range(40):
        trust = _EMA_ALPHA * 0.60 + (1 - _EMA_ALPHA) * trust
    assert trust == pytest.approx(0.60, abs=1e-3)


def test_a_zero_accuracy_benches_a_competent_agent():
    """Pins the mechanism the fake grades exploited: with accuracy at 60 percent
    weight, a fictional 0.0 drags a 0.60-internal agent to 0.24, under the 0.30
    demotion threshold. The guard above is what stops the 0.0 being fictional."""
    trust = _TRUST_INIT
    for _ in range(40):
        raw = 0.4 * 0.60 + 0.6 * 0.0
        trust = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * trust
    assert trust_tier(trust) == 'demoted'
    assert trust == pytest.approx(0.24, abs=0.01)


# ── Regional derivation ───────────────────────────────────────────────────────

def test_regional_scaling_preserves_direction():
    """The freight multiplier scales magnitude; it must never flip a fall into a
    rise for a remote region."""
    for base in (+2.42, -2.42):
        derived = derive_regional_estimates(base)
        assert all((v > 0) == (base > 0) for v in derived.values() if v is not None)


def test_ncr_is_the_unscaled_reference():
    assert derive_regional_estimates(-2.42)['NCR'] == pytest.approx(-2.42)


def test_a_missing_base_yields_no_invented_regional_numbers():
    assert all(v is None for v in derive_regional_estimates(None).values())


# ── The second grading failure mode, which the magnitude bound missed ─────────

def test_a_run_with_no_baseline_is_never_graded(tmp_path):
    """A run whose price could not be fetched stores no baseline, so there is
    nothing to measure a change against.

    This replaced a magic-value check that compared the baseline against the
    fallback constant. That could not tell "this IS the fallback" from "the real
    price happens to equal it", so a week landing on that number would have
    stopped grading for good, and it broke three existing tests that used the
    constant as ordinary data. Liveness is recorded at the source instead.
    """
    from ph_economic_ai.engine import ground_truth as gt

    store = AgentTrustStore(db_path=str(tmp_path / 't.db'))
    run_id = store.save_run(scenario={'oil_pct': -3.0}, final_estimate=-1.0,
                            confidence_pct=50, horizon_days=-1.0)
    assert gt.find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 0
    assert store.get_run(run_id)['actual_price_change'] is None
    store.close()


def test_a_failed_fetch_stores_no_baseline():
    """The orchestrator reports None rather than the fallback, so the caller has
    something unambiguous to act on."""
    from ph_economic_ai.engine import swarm

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(swarm, 'fetch_live_retail_price_checked',
                   lambda: (swarm._FALLBACK_RETAIL_PRICE_PHP, False))
        assert swarm.fetch_live_retail_price_checked()[1] is False


def test_the_price_fetcher_reports_whether_its_value_is_real():
    """Silently returning a constant is fine for a prompt and dangerous for
    grading, where it becomes an observed outcome."""
    from ph_economic_ai.engine import swarm

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(swarm.requests, 'get',
                   lambda *a, **k: (_ for _ in ()).throw(RuntimeError('offline')))
        price, is_live = swarm.fetch_live_retail_price_checked()
        assert is_live is False
        assert price == swarm._FALLBACK_RETAIL_PRICE_PHP
        # The unchecked helper keeps its old shape for prompt callers.
        assert swarm.fetch_live_retail_price() == swarm._FALLBACK_RETAIL_PRICE_PHP


# ── The freight premium was charged twice ─────────────────────────────────────

def test_an_anchor_region_is_not_charged_its_own_freight_twice():
    """`anchor_prompt_block` tells each agent to adjust for "your region's freight
    premium", so a Davao survivor's estimate already carries Davao freight.
    Multiplying it by Davao's 1.05 again billed the same premium twice."""
    from ph_economic_ai.engine.swarm import derive_regional_estimates
    survivors = {0: 2.42, 1: 2.42, 2: 2.42, 3: 2.42}
    derived = derive_regional_estimates(2.42, survivors)
    for region in ('NCR', 'Central Luzon', 'Western Visayas', 'Davao Region'):
        assert derived[region] == pytest.approx(2.42), (
            f'{region} supplied the estimate, so it must be returned unscaled')


def test_a_region_is_charged_only_the_premium_over_its_anchor():
    """Zamboanga is 1.08 over NCR and its estimate comes from Davao at 1.05, so
    only 1.08/1.05 is outstanding."""
    from ph_economic_ai.engine.swarm import derive_regional_estimates
    derived = derive_regional_estimates(2.42, {3: 2.42})
    assert derived['Zamboanga'] == pytest.approx(round(2.42 * (1.08 / 1.05), 2))


def test_ncr_anchored_regions_are_unchanged_by_the_fix():
    """These divide by 1.00 and were always correct, which is exactly why the bug
    survived: the regions anyone would spot-check were the right ones."""
    from ph_economic_ai.engine.swarm import derive_regional_estimates
    derived = derive_regional_estimates(2.42, {0: 2.42})
    for region, multiplier in (('Ilocos Region', 1.05), ('Cagayan Valley', 1.06),
                               ('CAR', 1.08)):
        assert derived[region] == pytest.approx(round(2.42 * multiplier, 2))


def test_a_relative_premium_below_one_is_legitimate():
    """Written first as "no region is ever cheaper than the region that priced
    it", which is false. Central Visayas carries 1.04 while its estimate comes
    from Western Visayas at 1.05, so its relative premium is 0.990 and its number
    lands BELOW the anchor's. Cebu being cheaper to serve than Iloilo is a fact
    about the table, not a defect in the derivation, and an assertion of
    monotonicity would have forced a wrong fix."""
    from ph_economic_ai.engine.swarm import derive_regional_estimates
    derived = derive_regional_estimates(2.42, {2: 2.42})
    assert derived['Central Visayas'] == pytest.approx(round(2.42 * (1.04 / 1.05), 2))
    assert derived['Central Visayas'] < derived['Western Visayas']


def test_every_region_is_scaled_by_its_premium_over_its_anchor():
    """The invariant that actually holds, stated over all seventeen."""
    from ph_economic_ai.engine.swarm import (
        ALL_REGIONS, REGIONS, derive_regional_estimates)
    anchor_mult = {r['anchor']: r['multiplier']
                   for r in ALL_REGIONS if r['name'] in REGIONS}
    derived = derive_regional_estimates(2.42, {g: 2.42 for g in anchor_mult})
    for reg in ALL_REGIONS:
        expected = round(2.42 * reg['multiplier'] / anchor_mult[reg['anchor']], 2)
        assert derived[reg['name']] == pytest.approx(expected), reg['name']


def test_the_fix_still_preserves_direction():
    from ph_economic_ai.engine.swarm import derive_regional_estimates
    for base in (+2.42, -2.42):
        derived = derive_regional_estimates(base, {0: base, 3: base})
        assert all((v > 0) == (base > 0) for v in derived.values() if v is not None)
