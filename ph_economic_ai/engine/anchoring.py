"""Physics-anchored estimation — making weak local LLMs produce coherent numbers.

The problem this solves is specific and was seen repeatedly in real runs: small
local models (qwen2.5:3b/7b) reason about *direction* fine but get the
*magnitude* badly wrong. Asked for the pump-price effect of a +6.8% oil shock
they answered +₱12.93/L; the mechanical pass-through is about +₱2.72/L. The
±₱8/L plausibility guard then discarded the estimate, leaving the report blank.

The idea here is not to make the model smarter. It is to stop asking it for the
thing it cannot do. The magnitude of an oil→pump pass-through is not an opinion,
it is accounting: crude cost per litre, revalued at the exchange rate, plus VAT.
That number is computed deterministically and used three ways:

1. **as a prior** — injected into the prompt so the model reasons from the right
   scale instead of inventing one;
2. **as a leash** — the model may refine within a band where qualitative factors
   (tax holidays, subsidies, competition, timing) plausibly live, and is clamped
   back toward physics when it drifts outside it;
3. **as a fallback** — when the model produces nothing usable, the physical
   anchor stands in, so the pipeline always yields a grounded number rather than
   a blank.

This is the "program-aided" pattern (LLM for structure, a solver for the math)
applied to macro pass-through. It does not claim to beat the random-walk
baseline — the benchmark shows nothing does at one month — only to make the
exploratory swarm's numbers physically coherent on hardware that otherwise
can't manage it.

Everything here is pure and deterministic, so it is fully testable without a
model or a network.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional, Sequence

# Litres per barrel of crude (42 US gal). Not an approximation — a definition.
_LITRES_PER_BARREL = 158.987

# VAT applied to the ex-refinery component of the pump price in the Philippines.
_VAT = 0.12

# Empirical pass-through calibration. The pure mechanical anchor is accounting,
# but real PH pump prices pass through only part of an oil move within a month —
# subsidy buffers, the DOE weekly-averaging lag, and competitive absorption damp
# it. Backtested on 78 months of World Bank RON95 vs monthly Brent/FX
# (tools/anchor_backtest.py): correlation 0.60, directional accuracy 74%, and an
# OLS slope of 0.79 of the mechanical value, which beats a no-change baseline
# (MAE ₱2.21 vs ₱2.64). That 0.79 is applied here so the anchor reflects the
# observed pass-through, not just the textbook one. It is a fit to one window
# and may drift; re-run the backtest to refresh it.
_FUEL_PASSTHROUGH_CALIBRATION = 0.79

# Reference inputs used when live values are unavailable. The anchor's job is
# scale, not precision — a ±$5 error in Brent moves a ₱2.7 anchor by cents — so
# stale-but-reasonable defaults are fine, and callers pass live values when they
# have them.
_BRENT_DEFAULT_USD = 98.0        # Brent, USD/bbl
_FX_DEFAULT_PHP_USD = 58.0       # USD/PHP

# How far a model estimate may sit from the anchor before we stop believing it.
# ₱2/L is roughly the room genuine qualitative factors (a fuel subsidy release,
# an excise suspension, a refinery outage) can move a monthly pump change beyond
# the pure pass-through. Past that, the model has misjudged scale, not spotted a
# real effect.
_DEFAULT_TOLERANCE_PHP = 2.0


def fuel_passthrough_anchor(
    oil_pct: float,
    usd_pct: float,
    brent_usd: float = _BRENT_DEFAULT_USD,
    fx_php_per_usd: float = _FX_DEFAULT_PHP_USD,
    calibrated: bool = True,
) -> float:
    """Pump-price change in ₱/L implied by an oil and FX shock.

    The crude cost embedded in one litre of fuel is ``brent / litres_per_barrel``
    dollars, or ``brent * fx / litres_per_barrel`` pesos. Both shocks act on that
    same base, so to first order they add:

        Δpump ≈ (brent · fx / L_bbl) · (oil% + usd%)/100 · (1 + VAT)

    A weaker peso raises the peso cost of the crude already in the fuel, which is
    why the FX shock enters on equal footing with the oil shock rather than as an
    afterthought. By default the result is scaled by the empirically-fitted
    `_FUEL_PASSTHROUGH_CALIBRATION`; pass ``calibrated=False`` for the raw
    textbook value (used by the backtest that fits the calibration).
    """
    base_landed_php_per_l = brent_usd * fx_php_per_usd / _LITRES_PER_BARREL
    delta_landed = base_landed_php_per_l * (oil_pct + usd_pct) / 100.0
    mechanical = delta_landed * (1 + _VAT)
    return mechanical * _FUEL_PASSTHROUGH_CALIBRATION if calibrated else mechanical


# ── Electricity: a fuel pass-through anchor (a magnitude guard) ───────────────
# The generation charge is a formulaic fuel pass-through, and the benchmark found
# electricity-CPI predictable within the month via that formula (Ridge +28%,
# DM p ≈ 0.001). This anchor is a simpler proxy: a fuel-cost shock scaled by the
# fuel share of the generation charge. Regressed against 175 months of real PSA
# electricity CPI (tools/anchor_backtest.py) it does NOT predict the monthly move
# (corr ~0.03–0.13 — the benchmark's edge needs the actual formula, not raw oil),
# but its magnitude is right (scale ratio ~1.0). That is the anchor's job here:
# keep a weak model's estimate physically sized, not forecast the series.

# Meralco's generation charge, the fuel-driven slice of a ~₱11–14/kWh total bill.
_GEN_CHARGE_PHP_KWH = 5.50
# Share of the generation charge that tracks fuel prices (natural gas via
# Malampaya/LNG, imported coal, oil peaking plants).
_GEN_FUEL_SHARE = 0.55


def electricity_passthrough_anchor(
    oil_pct: float,
    usd_pct: float,
    generation_charge_php_kwh: float = _GEN_CHARGE_PHP_KWH,
    fuel_share: float = _GEN_FUEL_SHARE,
) -> float:
    """Mechanical ₱/kWh change in the generation charge from a fuel/FX shock.

    The fuel-indexed slice of the generation charge is
    ``generation_charge · fuel_share``; a fuel-cost move passes through it about
    one-for-one, and a weaker peso raises the cost of imported coal and LNG, so
    the oil and FX shocks enter together as they do for pump prices.
    """
    fuel_indexed = generation_charge_php_kwh * fuel_share
    return fuel_indexed * (oil_pct + usd_pct) / 100.0


# ── Food: a *persistence* anchor, deliberately not a commodity one ─────────────
# The benchmark found food-CPI a clean null on commodity drivers, so anchoring
# food to oil would be anchoring it to noise; what it found predictable is food's
# own dynamics. Anchoring to the recent trend follows that. Regressed against 172
# months of real PSA food CPI (tools/anchor_backtest.py), persistence and oil are
# both weak and within sampling noise of each other (corr ~0.18 vs ~0.21), and a
# plain mean is competitive — monthly food CPI is close to unpredictable here. So
# like electricity this anchor is a magnitude guard (scale ratio ~0.9), not a
# predictor; it still keeps a weak model from claiming +7% when the trend is <1%.

_FOOD_DEFAULT_MOM_PCT = 0.4      # fallback trend when no history is available
_FOOD_TRANSPORT_FUEL_BETA = 0.03  # ppt of monthly food inflation per 1% oil move


def food_persistence_anchor(
    recent_mom_pcts: Sequence[float],
    oil_pct: float = 0.0,
) -> float:
    """% month-on-month food-inflation anchor from own persistence.

    The base is the trailing mean of recent monthly food inflation — the
    own-dynamics the benchmark found predictable — plus a small transport term
    so a large fuel shock can nudge it. The transport coefficient is small by
    design: the benchmark rejected commodity drivers for food, so fuel must not
    dominate this number.
    """
    usable = [p for p in recent_mom_pcts if p is not None]
    base = statistics.fmean(usable) if usable else _FOOD_DEFAULT_MOM_PCT
    return base + _FOOD_TRANSPORT_FUEL_BETA * oil_pct


# Sector-appropriate reconciliation bands. Each is the room genuine
# sector-specific factors have to move the number beyond its anchor before the
# estimate is more likely a model error than a real signal.
FUEL_TOLERANCE_PHP_L = _DEFAULT_TOLERANCE_PHP
ELECTRICITY_TOLERANCE_PHP_KWH = 0.40
FOOD_TOLERANCE_PCT = 1.5


@dataclass
class Reconciled:
    """Outcome of blending a model estimate with the physical anchor."""
    value: float
    source: str          # 'agent' | 'clamped' | 'anchor'
    anchor: float
    llm_estimate: Optional[float]

    @property
    def used_physics(self) -> bool:
        return self.source in ('clamped', 'anchor')


def reconcile_estimate(
    llm_estimate: Optional[float],
    anchor: float,
    tolerance: float = _DEFAULT_TOLERANCE_PHP,
) -> Reconciled:
    """Combine a model's pump-change estimate with the physical anchor.

    * within tolerance of the anchor → trust the model (it refined the number
      using qualitative signal the pure formula cannot see);
    * outside tolerance → clamp to the nearest edge of the band, keeping the
      model's *direction* but not its implausible magnitude;
    * missing → fall back to the anchor outright.

    Every path returns a physically grounded number, which is the point: the
    report can no longer show a blank or a wild figure.
    """
    if llm_estimate is None:
        return Reconciled(anchor, 'anchor', anchor, None)
    if abs(llm_estimate - anchor) <= tolerance:
        return Reconciled(llm_estimate, 'agent', anchor, llm_estimate)
    edge = anchor + (tolerance if llm_estimate > anchor else -tolerance)
    return Reconciled(edge, 'clamped', anchor, llm_estimate)


def explain(
    reconciled: Reconciled,
    unit: str = '₱/L',
    anchor_label: str = 'mechanical pass-through',
) -> str:
    """One line describing what the reconciliation did, for the report.

    `anchor_label` names the *kind* of anchor, which differs by sector: a
    mechanical pass-through for fuel and electricity, own-trend persistence for
    food. That distinction is the point of the experiment, so it is surfaced.
    """
    a = reconciled.anchor
    if reconciled.source == 'agent':
        return (
            f'Agent estimate {reconciled.value:+.2f} {unit} is consistent with the '
            f'{a:+.2f} {unit} {anchor_label}.'
        )
    if reconciled.source == 'clamped':
        return (
            f'Agent estimate {reconciled.llm_estimate:+.2f} {unit} diverged from the '
            f'{a:+.2f} {unit} {anchor_label}; clamped to {reconciled.value:+.2f} {unit}.'
        )
    return (
        f'No usable agent estimate; using the {a:+.2f} {unit} {anchor_label} '
        f'as the anchor.'
    )
