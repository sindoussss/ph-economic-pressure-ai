"""Does a regional freight premium scale a price CHANGE? `Q-ENG-009` Phase 2.

**Every choice in this file was fixed in
`docs/preregistration/2026-08-01-regional-multiplier-test.md` before it was run.**
Read that first. This module is the execution of a decision rule, not the place
where one is made, and the separation is the whole point: `DEC-010` exists
because a hypothesis family was once assembled after seeing which results were
interesting.

## What is being tested

`derive_regional_estimates` multiplies a price CHANGE by a region's freight
multiplier. Those multipliers are LEVEL premiums, so the two competing models
predict different values of the same coefficient in

    delta_R(t) = a + b * delta_NCR(t) + e(t)

`b = multiplier` supports the current model. `b = 1` supports freight sitting in
the level and says the multiplication should go. **Both testable regions carry
1.05 against NCR's 1.00, so the two hypotheses live 0.05 apart** and the
pre-registration records that the interval may well be wider than the gap. That
outcome is `CANNOT DISTINGUISH`, it is declared in advance, and its action is to
change nothing.

## Construction rules, all pre-registered

* Window from 2023-01-03, the common coverage of all three regions.
* Regional level is the MEDIAN of city prices that week, DOE's own `common`
  column, falling back to the midpoint of the published range.
* At least 3 reporting cities, else the week is dropped.
* Changes ONLY between consecutive pricing weeks exactly 7 days apart. The panel
  has gaps and differencing across one would label a three-week move as a
  one-week move, which is the defect `ADR-003` fixed nationally.
* Where two documents cover a city-week, the later filename wins. Arbitrary, and
  declared arbitrary in advance so it cannot be chosen later to suit an outcome.

    python -m ph_economic_ai.tools.regional_multiplier_backtest
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Optional

from ph_economic_ai.benchmark.paths import artifact, data
from ph_economic_ai.tools.doe_price_series import DEBATED_REGIONS

PANEL = data('doe_regional_ron95.csv')
ARTIFACT = artifact('regional_multiplier_backtest.json')
PREREGISTRATION = 'docs/preregistration/2026-08-01-regional-multiplier-test.md'

#: Pre-registered. The common coverage of all three regions and nothing else.
WINDOW_START = dt.date(2023, 1, 3)

#: Pre-registered: a "regional" median over fewer cities than this is not one.
MIN_CITIES = 3

#: Pre-registered: weekly changes are autocorrelated because one crude move
#: propagates over several weeks, so plain OLS intervals would be too narrow.
HAC_LAGS = 4

#: Pre-registered invalidation thresholds. Checked and reported whatever they say.
MIN_CHANGES = 60

#: Added in Amendment 1, AFTER the first run, and it withdraws a result rather
#: than producing one. Philippine retail adjustments run about 0.20 to 3.00 PHP/L
#: a week. A larger move is not a price change, it is a measurement change: the
#: 2026 NCR level tracks which document TEMPLATE DOE used, 56 to 61 on the
#: 9-city 3-brand sheet against 72 to 96 on the 12-city 10-brand one, smooth
#: within each and jumping at every switch.
#:
#: The three pre-registered invalidation conditions all PASSED. None of them
#: asked whether the series measured a constant thing over time, which is the
#: one that failed.
IMPLAUSIBLE_WEEKLY_MOVE = 5.0

REFERENCE = 'NCR'

#: From `swarm.REGION_GROUPS`. Read here rather than imported so the test does not
#: depend on the engine importing cleanly, and asserted against it in the tests.
MULTIPLIERS = {'NCR': 1.00, 'Western Visayas': 1.05, 'Davao Region': 1.05}


def _price(row: dict) -> Optional[float]:
    """DOE's own common price, else the midpoint of the range it published."""
    for key in ('common',):
        if row.get(key):
            try:
                return float(row[key])
            except ValueError:
                pass
    low, high = row.get('low'), row.get('high')
    try:
        return (float(low) + float(high)) / 2 if low and high else None
    except ValueError:
        return None


def regional_levels(rows: list[dict]) -> dict[str, dict[dt.date, dict]]:
    """{region: {cycle: {level, cities}}}, the median city price that week."""
    # Pre-registered tie-break: where two documents cover one city-week, the
    # later filename wins. Declared arbitrary in advance.
    latest: dict[tuple, dict] = {}
    for row in rows:
        price = _price(row)
        if price is None or not row['city']:
            continue
        key = (row['area'], row['province'], row['city'], row['cycle'])
        if key not in latest or row['source_file'] > latest[key]['source_file']:
            latest[key] = {**row, '_price': price}

    buckets: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in latest.values():
        for region, provinces in DEBATED_REGIONS.items():
            if region == REFERENCE:
                if row['area'] != REFERENCE:
                    continue
            elif not provinces or row['province'] not in provinces:
                continue
            buckets[region][row['cycle']].append(row['_price'])

    out: dict = {}
    for region, weeks in buckets.items():
        out[region] = {}
        for cycle, prices in weeks.items():
            if len(prices) < MIN_CITIES:
                continue
            out[region][dt.date.fromisoformat(cycle)] = {
                'level': statistics.median(prices), 'cities': len(prices)}
    return out


def weekly_changes(levels: dict[dt.date, dict]) -> dict[dt.date, float]:
    """Change from the previous week, for CONSECUTIVE weeks only.

    A gap ends a run rather than being bridged. Differencing across one would
    label a three-week move as a one-week move: `ADR-003`'s defect, which passed
    a green suite nationally because nothing asserted the index was contiguous.
    """
    out: dict[dt.date, float] = {}
    for day in sorted(levels):
        prev = day - dt.timedelta(days=7)
        if prev in levels:
            out[day] = levels[day]['level'] - levels[prev]['level']
    return out


def regress(y: list[float], x: list[float]) -> dict:
    """OLS with Newey-West standard errors at the pre-registered lag."""
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    model = OLS(np.array(y), add_constant(np.array(x))).fit(
        cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    lo, hi = model.conf_int()[1]
    resid = model.resid
    worst = int(np.argmax(np.abs(resid)))
    return {
        'n': int(model.nobs),
        'intercept': float(model.params[0]),
        'b': float(model.params[1]),
        'se': float(model.bse[1]),
        'ci_low': float(lo), 'ci_high': float(hi),
        'r_squared': float(model.rsquared),
        'reference_sd': float(np.std(x, ddof=1)),
        'residual_sd': float(np.std(resid, ddof=1)),
        'largest_residual_share': float(resid[worst] ** 2 / np.sum(resid ** 2)),
        'reference_zero_share': float(np.mean(np.abs(np.array(x)) < 0.005)),
    }


def verdict(fit: dict, multiplier: float) -> tuple[str, str]:
    """The pre-registered decision rule. No branch is added here after the fact."""
    lo, hi = fit['ci_low'], fit['ci_high']
    holds_one = lo <= 1.00 <= hi
    holds_mult = lo <= multiplier <= hi
    if holds_one and holds_mult:
        return ('CANNOT DISTINGUISH',
                'The interval covers both hypotheses. Change nothing; the '
                'multipliers remain unvalidated rather than validated.')
    if holds_one:
        return ('DELTA-EQUAL',
                'Freight sits in the level. Remove the multiplication, and refit '
                'the 0.79 pass-through in the same pass (DEC-021).')
    if holds_mult:
        return ('CURRENT MODEL',
                'Keep the multiplication and refit the multipliers to the '
                'estimated slopes; they have never been fitted at all.')
    return ('NEITHER',
            'Report a national change with a per-region uncertainty band, and '
            'open a new question.')


def implausible_weeks(changes: dict[dt.date, float]) -> list[tuple[dt.date, float]]:
    """Weeks whose move is too large to be a retail price adjustment."""
    return sorted(((d, c) for d, c in changes.items()
                   if abs(c) > IMPLAUSIBLE_WEEKLY_MOVE), key=lambda kv: -abs(kv[1]))


def invalidations(fit: dict) -> list[str]:
    """Pre-registered conditions that would make the test uninformative."""
    out = []
    if fit['n'] < MIN_CHANGES:
        out.append(f'only {fit["n"]} usable weekly changes, fewer than {MIN_CHANGES}')
    if fit['reference_zero_share'] > 0.5:
        out.append(f'{fit["reference_zero_share"]:.0%} of reference changes are '
                   f'zero, so b rests on a handful of moves')
    if fit['largest_residual_share'] > 0.5:
        out.append(f'one week supplies {fit["largest_residual_share"]:.0%} of the '
                   f'residual variance: an event, not a relationship')
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', type=Path, default=ARTIFACT)
    args = ap.parse_args()

    with open(PANEL, encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh)
                if dt.date.fromisoformat(r['cycle']) >= WINDOW_START]

    levels = regional_levels(rows)
    changes = {r: weekly_changes(v) for r, v in levels.items()}
    ref = changes.get(REFERENCE, {})

    print(f'Pre-registered at {PREREGISTRATION}')
    print(f'Window from {WINDOW_START}, reference {REFERENCE}\n')
    print(f'{"region":<18}{"weeks":>7}{"changes":>9}{"median cities":>15}')
    for region in DEBATED_REGIONS:
        lv, ch = levels.get(region, {}), changes.get(region, {})
        cities = (f'{statistics.median(v["cities"] for v in lv.values()):.0f}'
                  if lv else '-')
        print(f'{region:<18}{len(lv):>7}{len(ch):>9}{cities:>15}')

    results = {}
    for region, multiplier in MULTIPLIERS.items():
        if region == REFERENCE:
            continue
        paired = sorted(set(changes.get(region, {})) & set(ref))
        if len(paired) < 2:
            print(f'\n{region}: no paired weeks')
            continue
        fit = regress([changes[region][d] for d in paired],
                      [ref[d] for d in paired])
        call, action = verdict(fit, multiplier)
        bad = invalidations(fit)

        # Amendment 1. The series has to be measuring a constant thing before the
        # decision rule means anything, so this gate runs BEFORE the verdict is
        # allowed to stand, and it can only withdraw a result, never create one.
        wild = {'reference': implausible_weeks({d: ref[d] for d in paired}),
                region: implausible_weeks({d: changes[region][d] for d in paired})}
        withdrawn = any(wild.values())

        results[region] = {
            **fit, 'multiplier': multiplier,
            'verdict': 'WITHDRAWN' if withdrawn else call,
            'verdict_under_rule': call, 'action': action, 'invalidations': bad,
            'implausible_weeks': {k: [[d.isoformat(), round(c, 2)] for d, c in v]
                                  for k, v in wild.items() if v}}

        print(f'\n── {region}  (multiplier {multiplier}) ' + '─' * 28)
        print(f'   b = {fit["b"]:.4f}   SE {fit["se"]:.4f}   '
              f'95% CI [{fit["ci_low"]:.4f}, {fit["ci_high"]:.4f}]')
        print(f'   n = {fit["n"]} weekly changes,  R2 {fit["r_squared"]:.3f},  '
              f'residual sd {fit["residual_sd"]:.3f}')
        print(f'   CI width {fit["ci_high"] - fit["ci_low"]:.4f} against a '
              f'{multiplier - 1.00:.2f} gap between the hypotheses')
        print(f'   contains 1.00: {fit["ci_low"] <= 1.00 <= fit["ci_high"]}   '
              f'contains {multiplier}: {fit["ci_low"] <= multiplier <= fit["ci_high"]}')
        for note in bad:
            print(f'   PRE-REGISTERED INVALIDATION: {note}')
        if withdrawn:
            print(f'   VERDICT WITHDRAWN. Under the rule it would read {call}.')
            for series_name, weeks in wild.items():
                if weeks:
                    shown = ', '.join(f'{d} {c:+.1f}' for d, c in weeks[:4])
                    print(f'      {series_name}: {len(weeks)} weeks move more than '
                          f'{IMPLAUSIBLE_WEEKLY_MOVE} PHP/L  [{shown}]')
        else:
            print(f'   VERDICT: {call}')
            print(f'   {action}')

    withdrawn = any(r['verdict'] == 'WITHDRAWN' for r in results.values())
    print('\n' + '=' * 72)
    if withdrawn:
        print('NO VERDICT. The series do not measure a constant quantity over '
              'time: the level tracks which document template DOE used, so a\n'
              'regression of one region on another is partly a regression of one '
              'sheet layout on another. See Amendment 1 in the pre-registration.\n'
              'The decision rule stands as written and is not renegotiated on the '
              'strength of having seen the coefficient.')
    else:
        calls = {r['verdict'] for r in results.values()}
        if len(calls) > 1:
            print('The two regions DISAGREE. Pre-registered action: report both, '
                  'change nothing, and do not pick the region that supports a '
                  'preferred model.')
        elif calls:
            print(f'Both regions: {calls.pop()}')

    # Written whatever the outcome. A withdrawn result is the finding here, and
    # an artifact that only exists when the test succeeded is a file drawer.
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        'preregistration': PREREGISTRATION,
        'window_start': WINDOW_START.isoformat(),
        'reference': REFERENCE,
        'hac_lags': HAC_LAGS,
        'min_cities': MIN_CITIES,
        'implausible_weekly_move': IMPLAUSIBLE_WEEKLY_MOVE,
        'family_size': len(MULTIPLIERS) - 1,
        'withdrawn': withdrawn,
        'results': results,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nartifact {args.json}')
    return 1 if withdrawn else 0


if __name__ == '__main__':
    raise SystemExit(main())
