"""Are the freight multipliers right as LEVEL premiums? `Q-ENG-009` Phase 2b.

**Pre-registered at `docs/preregistration/2026-08-01-regional-level-premiums.md`,
committed before these intervals were computed.** That document also discloses
what was already seen: two of the region ratios, Western Visayas and Davao, were
computed as an exploratory follow-up in Phase 2 and are flagged `seen` in the
output. The rest were not looked at in any form.

## Why this runs when Phase 2 could not

Phase 2 asked whether a freight premium should scale a price CHANGE and closed as
infeasible: the two hypotheses differ by 5 percent of a slope, and resolving that
from weekly retail changes needs 527 to 779 years.

`swarm.ALL_REGIONS` documents each multiplier as a LEVEL premium, "freight and
logistics premium over NCR". A level is about 70 PHP/L, so a 5 percent premium is
about 3.5 PHP/L against weekly changes under 1. **The signal is roughly seventy
times larger**, which is the whole reason this is answerable and that was not.

This does NOT settle whether to multiply a change by these constants. That stays
closed. It settles whether the constants are the right numbers for what they are
documented to be.

    python -m ph_economic_ai.tools.regional_level_premiums
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import random
import statistics
from pathlib import Path
from typing import Optional

from ph_economic_ai.benchmark.paths import artifact, data
from ph_economic_ai.tools.doe_price_series import (
    REGION_PROVINCES, region_of_south_luzon_file)
from ph_economic_ai.tools.regional_multiplier_backtest import (
    MIN_CITIES, WINDOW_START, _deduplicated,
)

PANEL = data('doe_regional_ron95.csv')
ARTIFACT = artifact('regional_level_premiums.json')
PREREGISTRATION = 'docs/preregistration/2026-08-01-regional-level-premiums.md'

BASE = 'NCR'

#: Pre-registered: fewer paired weeks than this is reported, never estimated.
MIN_WEEKS = 52

#: Pre-registered: 10,000 resamples of WHOLE WEEKS, because prices within a week
#: are not independent. Seed fixed in the document so the interval is reproducible.
BOOTSTRAP = 10_000
SEED = 20260801

#: Pre-registered: an interval wider than twice the largest premium in the table
#: cannot speak to a 3 to 10 percent claim.
UNINFORMATIVE_WIDTH = 0.10

#: Ratios already computed and seen before this test was written. Flagged in the
#: output rather than dropped, because dropping them would hide the exposure.
SEEN_BEFORE = ('Western Visayas', 'Davao Region')


def multipliers() -> dict[str, float]:
    """The table under test, read from the engine so it cannot drift from it."""
    from ph_economic_ai.engine import swarm
    return {g['name']: g['multiplier'] for g in swarm.ALL_REGIONS}


def regional_levels_by_region(rows: list[dict]) -> dict[str, dict[dt.date, float]]:
    """{region: {cycle: median city price}} for every region the panel reaches."""
    buckets: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for row in _deduplicated(rows):
        if row['area'] == 'South Luzon' and not row['province']:
            named = region_of_south_luzon_file(row['source_file'])
            if named:
                buckets[named][row['cycle']].append(row['_price'])
            continue
        for region, provinces in REGION_PROVINCES.items():
            if region == BASE:
                if row['area'] == BASE:
                    buckets[region][row['cycle']].append(row['_price'])
            elif provinces and row['province'] in provinces:
                buckets[region][row['cycle']].append(row['_price'])

    out: dict = {}
    for region, weeks in buckets.items():
        priced = {dt.date.fromisoformat(c): statistics.median(v)
                  for c, v in weeks.items() if len(v) >= MIN_CITIES}
        # A region whose every week fell below `MIN_CITIES` is not a region the
        # panel reaches, and emitting `{}` for it made this construction disagree
        # with `engine.regional_grading`, which omits the key. Inert for the
        # callers here, which all use `.get(region, {})` -- but `len(levels)` and
        # `region in levels` are different answers, and the two constructions of
        # one quantity have diverged once before, on 22 of 1798 week-values.
        if priced:
            out[region] = priced
    return out


def bootstrap_median(values: list[float], level: float) -> tuple[float, float]:
    """Percentile interval for the median, resampling whole observations.

    `level` is the two-sided coverage, so 0.95 gives the 2.5th and 97.5th
    percentiles. The Bonferroni-corrected interval is obtained by passing a
    higher coverage, not by rescaling a computed one.
    """
    rng = random.Random(SEED)
    n = len(values)
    draws = sorted(statistics.median([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(BOOTSTRAP))
    lo_i = int((1 - level) / 2 * BOOTSTRAP)
    hi_i = min(BOOTSTRAP - 1, int((1 + level) / 2 * BOOTSTRAP))
    return draws[lo_i], draws[hi_i]


def assess(ratios: list[float], multiplier: float, family: int) -> dict:
    """The pre-registered decision rule for one region."""
    median = statistics.median(ratios)
    lo, hi = bootstrap_median(ratios, 0.95)
    corrected = 1 - 0.05 / max(family, 1)
    clo, chi = bootstrap_median(ratios, corrected)

    holds_mult = clo <= multiplier <= chi
    holds_one = clo <= 1.00 <= chi
    if holds_mult:
        call, action = 'CONSISTENT', 'leave this multiplier alone'
    elif holds_one:
        call, action = ('NO PREMIUM DETECTABLE',
                        f'replace {multiplier} with {round(median, 2)}')
    else:
        call, action = ('WRONG SIZE',
                        f'replace {multiplier} with {round(median, 2)}')
    return {
        'n_weeks': len(ratios), 'median_ratio': median,
        'ci_low': lo, 'ci_high': hi,
        'corrected_ci_low': clo, 'corrected_ci_high': chi,
        'corrected_coverage': corrected,
        'multiplier': multiplier, 'verdict': call, 'action': action,
        'proposed': round(median, 2) if not holds_mult else multiplier,
        'uninformative': (chi - clo) > UNINFORMATIVE_WIDTH,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--json', type=Path, default=ARTIFACT)
    args = ap.parse_args()

    with open(PANEL, encoding='utf-8') as fh:
        rows = [r for r in csv.DictReader(fh)
                if dt.date.fromisoformat(r['cycle']) >= WINDOW_START]

    levels = regional_levels_by_region(rows)
    base = levels.get(BASE, {})
    table = multipliers()

    paired = {}
    for region in REGION_PROVINCES:
        if region == BASE or not REGION_PROVINCES[region]:
            continue
        weeks = sorted(set(levels.get(region, {})) & set(base))
        if weeks:
            paired[region] = [(d, levels[region][d] / base[d]) for d in weeks]

    testable = {r: v for r, v in paired.items() if len(v) >= MIN_WEEKS}
    family = len(testable)

    print(f'Pre-registered at {PREREGISTRATION}')
    print(f'Base {BASE}, window from {WINDOW_START}, family of {family}, '
          f'Bonferroni coverage {1 - 0.05 / max(family, 1):.4f}\n')

    results, corrections = {}, {}
    print(f'{"region":<19}{"claims":>7}{"median":>8}{"corrected 95% CI":>22}'
          f'  verdict')
    for region in REGION_PROVINCES:
        if region == BASE:
            continue
        mult = table.get(region)
        if not REGION_PROVINCES[region]:
            print(f'{region:<19}{mult:>7.2f}{"-":>8}{"no DOE series at all":>22}'
                  f'  UNTESTABLE')
            results[region] = {'verdict': 'UNTESTABLE', 'multiplier': mult,
                               'reason': 'DOE publishes nothing for this region'}
            continue
        obs = paired.get(region, [])
        if len(obs) < MIN_WEEKS:
            print(f'{region:<19}{mult:>7.2f}{"-":>8}'
                  f'{f"{len(obs)} weeks < {MIN_WEEKS}":>22}  INSUFFICIENT')
            results[region] = {'verdict': 'INSUFFICIENT', 'multiplier': mult,
                               'n_weeks': len(obs)}
            continue

        res = assess([r for _d, r in obs], mult, family)
        res['first_week'] = obs[0][0].isoformat()
        res['last_week'] = obs[-1][0].isoformat()
        res['seen_before_preregistration'] = region in SEEN_BEFORE
        results[region] = res
        flag = ' *seen' if region in SEEN_BEFORE else ''
        span = f'[{res["corrected_ci_low"]:.3f}, {res["corrected_ci_high"]:.3f}]'
        print(f'{region:<19}{mult:>7.2f}{res["median_ratio"]:>8.3f}{span:>22}'
              f'  {res["verdict"]}{flag}')
        if res['verdict'] != 'CONSISTENT' and not res['uninformative']:
            corrections[region] = res['proposed']

    print(f'\n* seen before the pre-registration was written, per its disclosure')
    if corrections:
        print(f'\n{len(corrections)} multipliers the rule says to correct:')
        for region, value in corrections.items():
            print(f'   {region:<19} {table[region]:.2f} -> {value:.2f}')
    else:
        print('\nNo multiplier is refuted by the corrected interval.')

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        'preregistration': PREREGISTRATION,
        'base': BASE, 'window_start': WINDOW_START.isoformat(),
        'min_weeks': MIN_WEEKS, 'bootstrap': BOOTSTRAP, 'seed': SEED,
        'family_size': family,
        'seen_before_preregistration': list(SEEN_BEFORE),
        'corrections': corrections,
        'results': results,
        # RSK-034: this script imports its price function from
        # `regional_multiplier_backtest`, which now delegates to the fixed
        # `regional_grading._price` (RSK-036), so every run from that commit
        # onward reads through the basis guard. `honesty.regional_basis` clears
        # its "predates the guard" caveat on this field alone -- derived from the
        # artifact, not typed into prose.
        'basis_guard': True,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nartifact {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
