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


def _price(row: dict) -> Optional[float]:
    """DOE's own common price, else the midpoint of the range it published."""
    if row.get('common'):
        try:
            return float(row['common'])
        except ValueError:
            pass
    low, high = row.get('low'), row.get('high')
    try:
        return (float(low) + float(high)) / 2 if low and high else None
    except ValueError:
        return None


def regional_levels(panel_path=PANEL) -> dict[str, dict[dt.date, float]]:
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
            price = _price(row)
            if price is None:
                continue
            key = (row['area'], row['province'], row['city'], row['cycle'])
            if key not in latest or row['source_file'] > latest[key]['source_file']:
                latest[key] = {**row, '_price': price}

    buckets: dict = {}
    for row in latest.values():
        # South Luzon sheets are published per REGION GROUP and name it in the
        # FILENAME, carrying no province. Omitting this cost CALABARZON,
        # MIMAROPA and Bicol about 183 weeks EACH in the accuracy test.
        if row['area'] == 'South Luzon' and not row['province']:
            named = region_of_south_luzon_file(row['source_file'])
            if named:
                buckets.setdefault(named, {}).setdefault(row['cycle'], []).append(
                    row['_price'])
            continue
        for region, provinces in REGION_PROVINCES.items():
            if region == 'NCR':
                if row['area'] != 'NCR':
                    continue
            elif not provinces or row['province'] not in provinces:
                continue
            buckets.setdefault(region, {}).setdefault(row['cycle'], []).append(
                row['_price'])

    return {region: {dt.date.fromisoformat(c): statistics.median(v)
                     for c, v in weeks.items() if len(v) >= MIN_CITIES}
            for region, weeks in buckets.items()}


def regional_actual(region: str, target: dt.date,
                    levels: Optional[dict] = None) -> Optional[float]:
    """The published price CHANGE for `region` over the week containing `target`.

    None when the region has no series, when the target week is not covered, or
    when the week before it is missing. **A change is only computed between
    CONSECUTIVE pricing weeks**: differencing across a gap labels a three-week
    move as a one-week move, which is `ADR-003`'s defect and the reason the Phase
    2 panel refuses to bridge one.
    """
    levels = levels if levels is not None else regional_levels()
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
    change = weeks[day] - weeks[prev]
    # An outcome the app would refuse as an estimate cannot be the truth it is
    # graded against. `ground_truth` applies this nationally; a regional grader
    # that skipped it would grade against exactly the outliers the national path
    # rejects.
    return change if abs(change) <= _MAX_PLAUSIBLE_CHANGE else None


def grade_regional(estimates: dict[str, float], target: dt.date,
                   levels: Optional[dict] = None) -> dict:
    """Score each region's forecast against its published change.

    Returns the graded scores AND the regions that could not be graded, by name
    and reason. Reporting a mean over whatever happened to be gradable would
    describe two thirds of the map as if it were the map.
    """
    from ph_economic_ai.engine.swarm import ASSUMED_MULTIPLIERS

    levels = levels if levels is not None else regional_levels()
    scored: dict[str, dict] = {}
    ungraded: dict[str, str] = {}

    for region, estimate in estimates.items():
        actual = regional_actual(region, target, levels)
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
