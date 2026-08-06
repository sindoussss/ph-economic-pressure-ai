"""A calibrated band around the app's estimate.

The card showed a single number, which reads as more precise than the system can
justify. The fix is not to show three point estimates: three draws from one model
share its biases, so the spread would understate real uncertainty while looking
like independent confirmation. Worse, before the sampling fix (ADR-002) the
run-to-run range on identical inputs was 1.08 PHP/L of pure sampler noise, and
surfacing that as "three estimates" would put the random number generator on
screen and call it analysis.

What is defensible is a split-conformal band: one central estimate plus a low and
a high whose coverage is measured rather than assumed. The half-width is the
(1 - alpha) quantile of past absolute errors, so the band widens exactly when the
app has been wrong before.

The band is built from the app's OWN graded history when there is enough of it.
That matters more than it looks: since the horizon-matching fix (RSK-018) those
errors compare each run against a price observed near its own target date, so
they measure the question the app actually answers. Before that fix the error
history mixed one-week forecasts graded against two-month-old prices, and a band
built from it would have been calibrated to the wrong question.
"""
from __future__ import annotations

import math

# Below this many graded runs the empirical quantile is too unstable to trust, and
# the band falls back to a stated prior rather than pretending to be calibrated.
MIN_GRADED_FOR_CALIBRATION = 12

# Fallback half-widths, used only until a sector has its own graded history.
#
# These are PER SECTOR because the sectors are not in the same units, and mixing
# them is a real hazard rather than a theoretical one: the store grades only fuel,
# against DOE peso-per-litre prices, so feeding that error history into a food
# band would put PHP/L widths around a percentage and label the result
# "calibrated". A band in the wrong unit is worse than no band.
#
# Magnitudes come from the ranges the prompts already describe as typical, not
# from the monthly benchmark backtest, whose bands answer a different horizon.
FALLBACK_HALFWIDTH = {
    'gas':         {0.5: 0.60, 0.8: 1.40, 0.9: 2.00, 0.95: 2.60},   # PHP/L, weekly
    'food':        {0.5: 0.25, 0.8: 0.55, 0.9: 0.80, 0.95: 1.00},   # % MoM
    'electricity': {0.5: 0.20, 0.8: 0.50, 0.9: 0.75, 0.95: 1.00},   # PHP/kWh, monthly
}
_DEFAULT_SECTOR = 'gas'

# Only this sector is graded against an observed outcome, so only this one can
# ever produce a calibrated band. The others stay on the stated prior until a
# real outcome series exists for them.
GRADED_SECTORS = frozenset({'gas'})

DEFAULT_LEVEL = 0.5
EXPANDED_LEVEL = 0.9


def conformal_halfwidth(abs_errors, level: float) -> float:
    """Split-conformal half-width: the (1 - alpha) quantile of |error|.

    Uses the finite-sample index ceil((n + 1) * level) rather than a plain
    percentile, which is what gives conformal its coverage guarantee at small n.
    """
    errors = sorted(float(e) for e in abs_errors if e is not None and not math.isnan(float(e)))
    if not errors:
        raise ValueError('no errors to calibrate on')
    if not 0 < level < 1:
        raise ValueError(f'level must be in (0, 1), got {level}')
    n = len(errors)
    rank = math.ceil((n + 1) * level)
    if rank > n:                      # too few points to reach this level
        return errors[-1]
    return errors[rank - 1]


def fallback_halfwidth(sector: str, level: float) -> float:
    """The stated prior for a sector, in that sector's own unit.

    Refuses an unknown sector rather than substituting the default's numbers.
    It used to fall back to `gas`, so a typo or a sector nobody added to the
    table received +/-0.60 PHP/L and a card rendered it beside a percent sign.

    `band` already guards the mirror image of this -- it drops graded errors for
    a sector the app does not grade, "so that is refused rather than trusted to
    call sites" -- and the same hazard through the fallback table was left open.
    A sector is a closed set defined in this module, so an unknown one is a
    caller's bug and should say so rather than render a plausible wrong number.
    """
    table = FALLBACK_HALFWIDTH.get(sector)
    if table is None:
        raise ValueError(
            f'no stated prior for sector {sector!r}; known sectors are '
            f'{sorted(FALLBACK_HALFWIDTH)}. Falling back to '
            f'{_DEFAULT_SECTOR!r} would return a band in the wrong unit.')
    return table.get(level, table[DEFAULT_LEVEL])


def band(point_estimate: float, abs_errors=None, level: float = DEFAULT_LEVEL,
         sector: str = _DEFAULT_SECTOR) -> dict:
    """Low / central / high around `point_estimate`.

    `calibrated` says plainly whether the band came from measured errors or from
    the stated prior. The UI must not present the two identically.

    Errors are accepted only for a sector the app actually grades. Passing fuel
    errors for a percentage sector would silently produce a band in the wrong
    unit, so that is refused rather than trusted to call sites.
    """
    errors = [e for e in (abs_errors or []) if e is not None]
    if sector not in GRADED_SECTORS:
        errors = []
    calibrated = len(errors) >= MIN_GRADED_FOR_CALIBRATION
    if calibrated:
        half = conformal_halfwidth(errors, level)
        source = f'calibrated on {len(errors)} graded runs'
    elif sector not in GRADED_SECTORS:
        half = fallback_halfwidth(sector, level)
        source = 'stated prior; this sector has no graded outcome series yet'
    else:
        half = fallback_halfwidth(sector, level)
        source = (f'uncalibrated prior; {len(errors)} of '
                  f'{MIN_GRADED_FOR_CALIBRATION} graded runs needed')
    return {
        'central': round(float(point_estimate), 2),
        'low': round(float(point_estimate) - half, 2),
        'high': round(float(point_estimate) + half, 2),
        'half_width': round(float(half), 2),
        'level': level,
        'calibrated': calibrated,
        # Whether this sector CAN ever be calibrated, as opposed to not being
        # calibrated yet. The screen conflated them and told a reader that food
        # had "0 of 12 graded runs", which implies 12 are reachable. Only fuel
        # is graded against an observed price, so for every other sector the
        # honest sentence is that no outcome series exists at all.
        'gradable': sector in GRADED_SECTORS,
        'n_graded': len(errors),
        'source': source,
    }


def bands(point_estimate: float, abs_errors=None,
          levels=(DEFAULT_LEVEL, EXPANDED_LEVEL),
          sector: str = _DEFAULT_SECTOR) -> dict:
    """The lead band plus the wider one held behind an expand control.

    The 50 percent band leads because "half the time it lands in here" is both
    intuitive and narrow enough to act on. The 90 percent band is the honest full
    picture and stays one interaction away rather than being hidden.
    """
    return {str(level): band(point_estimate, abs_errors, level, sector)
            for level in levels}


def format_band(b: dict, unit: str = 'PHP/L') -> str:
    """One line a card can show without further formatting logic."""
    sign = '+' if b['central'] > 0 else ''
    return (f"{sign}{b['central']:.2f} {unit} "
            f"({b['low']:+.2f} to {b['high']:+.2f}, {int(b['level'] * 100)}% band)")
