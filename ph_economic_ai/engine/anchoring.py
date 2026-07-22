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

from dataclasses import dataclass
from typing import Optional

# Litres per barrel of crude (42 US gal). Not an approximation — a definition.
_LITRES_PER_BARREL = 158.987

# VAT applied to the ex-refinery component of the pump price in the Philippines.
_VAT = 0.12

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
) -> float:
    """Mechanical pump-price change in ₱/L implied by an oil and FX shock.

    The crude cost embedded in one litre of fuel is ``brent / litres_per_barrel``
    dollars, or ``brent * fx / litres_per_barrel`` pesos. Both shocks act on that
    same base, so to first order they add:

        Δpump ≈ (brent · fx / L_bbl) · (oil% + usd%)/100 · (1 + VAT)

    A weaker peso raises the peso cost of the crude already in the fuel, which is
    why the FX shock enters on equal footing with the oil shock rather than as an
    afterthought.
    """
    base_landed_php_per_l = brent_usd * fx_php_per_usd / _LITRES_PER_BARREL
    delta_landed = base_landed_php_per_l * (oil_pct + usd_pct) / 100.0
    return delta_landed * (1 + _VAT)


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


def explain(reconciled: Reconciled) -> str:
    """One line describing what the reconciliation did, for the report."""
    a = reconciled.anchor
    if reconciled.source == 'agent':
        return (
            f'Agent estimate {reconciled.value:+.2f} ₱/L is consistent with the '
            f'{a:+.2f} ₱/L mechanical pass-through.'
        )
    if reconciled.source == 'clamped':
        return (
            f'Agent estimate {reconciled.llm_estimate:+.2f} ₱/L diverged from the '
            f'{a:+.2f} ₱/L mechanical pass-through; clamped to '
            f'{reconciled.value:+.2f} ₱/L.'
        )
    return (
        f'No usable agent estimate; using the {a:+.2f} ₱/L mechanical '
        f'pass-through as the physical anchor.'
    )
