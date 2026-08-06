"""Grade a per-region forecast against DOE's published regional prices.

**The gap this closes.** `ground_truth` scores the national figure and nothing
else: `find_and_grade_runs` compares one estimate to one price. So the seventeen
regional numbers on the map have never been compared to anything, which is why
`03 Thesis/Unsafe Claims` forbids saying the regional forecasts are validated,
and why `ui.honesty.regional_basis` has to call them derived rather than forecast.

That was unavoidable until Phase 1: there was no regional actual in the project.
There is now, `benchmark/data/doe_regional_ron95.csv`, so the comparison is
possible for the first time.

## What this does and does not license

It grades a region's forecast against a published price for that region. It does
NOT settle whether a level premium should scale a change, which is `Q-ENG-009`
and closed as infeasible (`ADR-016`). A region can be graded well and the model
producing it still be the wrong shape.

**Eleven of seventeen regions can be graded, and six cannot at any date.** Four
have no DOE series at all, BARMM's is too thin, and NCR is the base. A grader
that quietly skipped them would report an accuracy figure covering two thirds of
the map as if it covered the map, so `grade_regional` returns the ungradable
regions by name rather than dropping them.

## Deliberately reusing the national contract

`compute_accuracy_score` and the plausibility bound are imported, not
re-implemented. A regional grade computed on a different scale could not be read
beside the national one, and an outcome the app refuses to accept as an ESTIMATE
must not be accepted as the TRUTH it is graded against -- the rule `ground_truth`
already applies nationally, and the defect it was written for (`RSK-018`, every
run scored against a hardcoded fallback) is exactly the shape that reappears when
a second grader invents its own guards.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import statistics
from typing import Optional

from ph_economic_ai.benchmark.paths import data
from ph_economic_ai.engine.ground_truth import (
    _MAX_PLAUSIBLE_CHANGE, compute_accuracy_score,
)

PANEL = data('doe_regional_ron95.csv')

#: A region-week needs this many reporting cities before its median is a regional
#: level rather than one city's price. Matches the Phase 2 pre-registration.
MIN_CITIES = 3

#: How far from the target date a pricing week may sit and still be the outcome
#: being graded. One cycle: the same rule `ground_truth` applies nationally, where
#: a run is graded against a price observed near ITS OWN target date rather than
#: against whatever is current.
MAX_WEEK_GAP = dt.timedelta(days=7)

#: A published city price outside this band is a parse artifact, not a market.
#: The same sanity bound `swarm.fetch_live_retail_price_checked` applies to the
#: live scrape, so a value the app would refuse to READ is not accepted from the
#: archive either. Deliberately wide: DOE's RON 95 spans about 51 to 66 by
#: region while the live aggregator reads near 90, and a narrow band here would
#: silently discard whichever of the two turns out to be right.
PLAUSIBLE_LEVEL = (20.0, 200.0)


def _price(row: dict):
    """DOE's own common price, else the midpoint of the range it published.

    Returns `(price, basis)` where basis is `'common'` or `'midpoint'`, or
    `(None, None)`.

    **The basis has to travel with the number.** These are two different
    quantities and the panel is 56 percent one and 39 percent the other: 20,933
    rows carry a common price and 14,715 carry only a range. Where both exist
    they differ by more than 0.50 PHP/L in 73.5 percent of rows, mean -0.758,
    and the midpoint sits systematically higher because it is pulled up by the
    top of the range while the common price is the prevailing one.

    Merging them into one series is the `RSK-025` defect on the regional side: a
    number stored without recording what it is OF.
    """
    def parsed(text):
        try:
            return float(text) if text else None
        except ValueError:
            return None

    def plausible(value):
        return value is not None and PLAUSIBLE_LEVEL[0] <= value <= PLAUSIBLE_LEVEL[1]

    common = parsed(row.get('common'))
    low, high = parsed(row.get('low')), parsed(row.get('high'))
    midpoint = (low + high) / 2 if low is not None and high is not None else None

    # `common` is preferred, but only when it is a possible retail price. The
    # panel has eleven rows where it is not, and eight of those sit beside a
    # perfectly sane range: Tacloban City on 2026-05-12 reads common 8.00 with a
    # published range of 81.25 to 82.27. Preferring `common` unconditionally
    # discarded the good number in favour of the broken one.
    #
    # The remaining three have no usable range either -- `457.0` to `49.0`
    # transposed, `48.5` to `495.0` digit-shifted, `0.27` with no low -- and
    # yield no price at all rather than a guess.
    if plausible(common):
        return common, 'common'
    if plausible(midpoint):
        return midpoint, 'midpoint'
    return None, None


def regional_levels(panel_path=PANEL, with_bases: bool = False):
    """{region: {cycle: median city price}} from the committed panel.

    Only `grain='city'` rows. A metro-wide aggregate is not a city and averaging
    a whole market with its own components would move the level without any price
    moving (`DEC-062`).
    """
    from ph_economic_ai.tools.doe_price_series import (
        REGION_PROVINCES, region_of_south_luzon_file)

    # One priced row per city-week, later filename winning. The Phase 2
    # pre-registered tie-break, and NOT optional: without it a city that appears
    # in two documents contributes twice to its own median. Found by comparing
    # this function against `regional_level_premiums`, which had it -- two
    # constructions of the same quantity that must agree and did not, on 22 of
    # 1798 week-values.
    latest: dict = {}
    with open(panel_path, encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            if row.get('grain', 'city') != 'city' or not row['city']:
                continue
            price, basis = _price(row)
            if price is None:
                continue
            key = (row['area'], row['province'], row['city'], row['cycle'])
            if key not in latest or row['source_file'] > latest[key]['source_file']:
                latest[key] = {**row, '_price': price, '_basis': basis}

    buckets: dict = {}
    for row in latest.values():
        # South Luzon sheets are published per REGION GROUP and name it in the
        # FILENAME, carrying no province. Omitting this cost CALABARZON,
        # MIMAROPA and Bicol about 183 weeks EACH in the accuracy test.
        if row['area'] == 'South Luzon' and not row['province']:
            named = region_of_south_luzon_file(row['source_file'])
            if named:
                buckets.setdefault(named, {}).setdefault(row['cycle'], []).append(
                    (row['_price'], row['_basis']))
            continue
        for region, provinces in REGION_PROVINCES.items():
            if region == 'NCR':
                if row['area'] != 'NCR':
                    continue
            elif not provinces or row['province'] not in provinces:
                continue
            buckets.setdefault(region, {}).setdefault(row['cycle'], []).append(
                (row['_price'], row['_basis']))

    levels, bases = {}, {}
    for region, weeks in buckets.items():
        for cycle, priced in weeks.items():
            if len(priced) < MIN_CITIES:
                continue
            day = dt.date.fromisoformat(cycle)
            levels.setdefault(region, {})[day] = statistics.median(
                p for p, _ in priced)
            # The basis the majority of this week's reporting cities used. A
            # week whose cities are split carries the majority label; where a
            # mixed basis does damage is a CHANGE measured across two different
            # labels, which `regional_actual` refuses.
            bases.setdefault(region, {})[day] = collections.Counter(
                b for _, b in priced).most_common(1)[0][0]
    return (levels, bases) if with_bases else levels


def regional_actual(region: str, target: dt.date,
                    levels: Optional[dict] = None,
                    bases: Optional[dict] = None) -> Optional[float]:
    """The published price CHANGE for `region` over the week containing `target`.

    None when the region has no series, when the target week is not covered, or
    when the week before it is missing. **A change is only computed between
    CONSECUTIVE pricing weeks**: differencing across a gap labels a three-week
    move as a one-week move, which is `ADR-003`'s defect and the reason the Phase
    2 panel refuses to bridge one.
    """
    if levels is None:
        levels, bases = regional_levels(with_bases=True)
    weeks = levels.get(region) or {}
    if not weeks:
        return None
    near = [d for d in weeks if abs(d - target) <= MAX_WEEK_GAP]
    if not near:
        return None
    day = min(near, key=lambda d: abs(d - target))
    prev = day - dt.timedelta(days=7)
    if prev not in weeks:
        return None
    # The basis is consulted only for the series this module built. A caller
    # that supplied its own levels carries no basis, and absence is not
    # contradiction -- the same rule `ground_truth` applies to a run whose own
    # week has no observation.
    own = (bases or {}).get(region, {})
    now_basis, then_basis = own.get(day), own.get(prev)
    if now_basis is not None and then_basis is not None and now_basis != then_basis:
        # A change measured from a midpoint to a common price, or the reverse,
        # is a change of DEFINITION carrying a change of price. Measured on the
        # panel: the two differ by a mean 0.758 PHP/L, which is larger than
        # every effect size the accuracy test compares, and 4.8 percent of
        # consecutive region-week pairs switch between them.
        #
        # The same rule `ground_truth` applies to a run whose baseline came from
        # another price series: refuse rather than measure a relabelling.
        return None
    change = weeks[day] - weeks[prev]
    # An outcome the app would refuse as an estimate cannot be the truth it is
    # graded against. `ground_truth` applies this nationally; a regional grader
    # that skipped it would grade against exactly the outliers the national path
    # rejects.
    return change if abs(change) <= _MAX_PLAUSIBLE_CHANGE else None


def grade_regional(estimates: dict[str, float], target: dt.date,
                   levels: Optional[dict] = None,
                   bases: Optional[dict] = None) -> dict:
    """Score each region's forecast against its published change.

    Returns the graded scores AND the regions that could not be graded, by name
    and reason. Reporting a mean over whatever happened to be gradable would
    describe two thirds of the map as if it were the map.
    """
    from ph_economic_ai.engine.swarm import ASSUMED_MULTIPLIERS

    if levels is None:
        levels, bases = regional_levels(with_bases=True)
    scored: dict[str, dict] = {}
    ungraded: dict[str, str] = {}

    for region, estimate in estimates.items():
        actual = regional_actual(region, target, levels, bases)
        if actual is None:
            ungraded[region] = ('no DOE series for this region'
                                if not levels.get(region)
                                else 'no consecutive pricing weeks at the target')
            continue
        scored[region] = {
            'estimate': estimate, 'actual': actual,
            'error': estimate - actual,
            'score': compute_accuracy_score(estimate, actual),
            # A graded region whose own premium was never measured is graded on a
            # number partly produced by a guess. The score is real; what it says
            # about the model is weaker.
            'premium_assumed': region in ASSUMED_MULTIPLIERS,
        }

    return {
        'target': target.isoformat(),
        'graded': scored,
        'ungraded': ungraded,
        'n_graded': len(scored),
        'n_ungraded': len(ungraded),
        'mean_score': (statistics.mean(v['score'] for v in scored.values())
                       if scored else None),
        'mean_abs_error': (statistics.mean(abs(v['error']) for v in scored.values())
                           if scored else None),
    }
