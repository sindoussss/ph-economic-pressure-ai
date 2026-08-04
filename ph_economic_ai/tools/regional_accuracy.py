"""Does the regional derivation beat doing nothing?

**Pre-registered at `docs/preregistration/2026-08-01-regional-accuracy.md`,
committed before this file was written.** That document also discloses that the
pooled 2024-2026 figure had already been seen, so this is confirmatory in
procedure rather than blind; what it fixes is the naive pool, the paired test,
and what the app does with each outcome.

Not `Q-ENG-009`. That asks whether a level premium SHOULD scale a change and
closed as infeasible. This asks whether the thing being shipped performs.

    python -m ph_economic_ai.tools.regional_accuracy
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import random
import statistics
from pathlib import Path

from ph_economic_ai.benchmark.paths import artifact
from ph_economic_ai.engine.regional_grading import regional_levels
from ph_economic_ai.engine.swarm import (
    ASSUMED_MULTIPLIERS, MEASURED_MULTIPLIERS, derive_regional_estimates,
)

ARTIFACT = artifact('regional_accuracy.json')
PREREGISTRATION = 'docs/preregistration/2026-08-01-regional-accuracy.md'

REFERENCE = 'NCR'
BOOTSTRAP = 10_000
SEED = 20260801
ALPHA = 0.05
MIN_WEEKS = 60
#: The bound the estimate parser applies, so an implausible actual is not truth.
MAX_MOVE = 8.0


def build_arms(levels: dict) -> dict[dt.date, dict[str, dict[str, float]]]:
    """{week: {arm: {region: error}}}, every arm seeing identical actuals."""
    ref = levels[REFERENCE]
    weeks = [d for d in sorted(ref) if d - dt.timedelta(days=7) in ref]
    prev_actual: dict[str, float] = {}
    out: dict = {}

    for day in weeks:
        national = ref[day] - ref[day - dt.timedelta(days=7)]
        if abs(national) > MAX_MOVE:
            continue
        derived = derive_regional_estimates(national, {0: national})

        actuals = {}
        for region, weekly in levels.items():
            if region == REFERENCE:
                continue
            before = day - dt.timedelta(days=7)
            if day in weekly and before in weekly:
                change = weekly[day] - weekly[before]
                if abs(change) <= MAX_MOVE:
                    actuals[region] = change
        if not actuals:
            continue

        arms = {
            'DERIVATION': {r: derived[r] for r in actuals if r in derived},
            'DELTA-EQUAL': {r: national for r in actuals},
            'ZERO': {r: 0.0 for r in actuals},
            'PERSISTENCE': {r: prev_actual[r] for r in actuals if r in prev_actual},
        }
        out[day] = {name: {r: abs(v - actuals[r]) for r, v in arm.items()}
                    for name, arm in arms.items()}
        out[day]['_actual'] = actuals
        prev_actual = actuals
    return out


def paired_bootstrap(weeks: list, a: str, b: str) -> tuple:
    """CI on MAE(a) - MAE(b), resampling WHOLE WEEKS.

    Regions inside a week share the national move, so resampling observations
    would treat 11 correlated errors as 11 independent ones and report an
    interval far too narrow.
    """
    rng = random.Random(SEED)

    def mae_diff(sample) -> float:
        ea = [e for w in sample for e in w[a].values()]
        eb = [e for w in sample for e in w[b].values()]
        return statistics.mean(ea) - statistics.mean(eb) if ea and eb else 0.0

    usable = [w for w in weeks if w.get(a) and w.get(b)]
    point = mae_diff(usable)
    n = len(usable)
    draws = sorted(mae_diff([usable[rng.randrange(n)] for _ in range(n)])
                   for _ in range(BOOTSTRAP))
    corrected = 1 - ALPHA / 3          # Bonferroni across the three comparisons
    lo = draws[int((1 - corrected) / 2 * BOOTSTRAP)]
    hi = draws[min(BOOTSTRAP - 1, int((1 + corrected) / 2 * BOOTSTRAP))]
    return point, lo, hi, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', type=Path, default=ARTIFACT)
    args = ap.parse_args()

    levels = regional_levels()
    by_week = build_arms(levels)
    weeks = list(by_week.values())
    print(f'Pre-registered at {PREREGISTRATION}')
    print(f'{len(weeks)} paired weeks, reference {REFERENCE}\n')

    arms = ('DERIVATION', 'DELTA-EQUAL', 'ZERO', 'PERSISTENCE')
    print(f'{"arm":<14}{"MAE":>8}{"n errors":>10}')
    maes = {}
    for arm in arms:
        errs = [e for w in weeks for e in w.get(arm, {}).values()]
        maes[arm] = statistics.mean(errs) if errs else None
        print(f'{arm:<14}{maes[arm]:>8.3f}{len(errs):>10}')

    best_naive = min(('DELTA-EQUAL', 'ZERO', 'PERSISTENCE'), key=lambda a: maes[a])
    print(f'\nbest naive arm: {best_naive}\n')

    comparisons = {}
    for naive in ('DELTA-EQUAL', 'ZERO', 'PERSISTENCE'):
        point, lo, hi, n = paired_bootstrap(weeks, 'DERIVATION', naive)
        comparisons[naive] = {'diff': point, 'ci_low': lo, 'ci_high': hi, 'n_weeks': n}
        verdict = ('derivation BETTER' if hi < 0 else
                   'derivation WORSE' if lo > 0 else 'no measurable difference')
        print(f'  DERIVATION - {naive:<12} {point:+.3f}  '
              f'CI [{lo:+.3f}, {hi:+.3f}]  {verdict}')

    d = comparisons[best_naive]
    if d['ci_high'] < 0:
        call = 'DERIVATION WINS'
        action = 'keep the regional figures as they are'
    elif d['ci_low'] > 0:
        call = 'DERIVATION IS WORSE THAN DOING NOTHING'
        action = ('label every derived figure as no better than assuming no '
                  'regional change, AND stop showing a number for the four '
                  'regions no source can check')
    else:
        call = 'NO MEASURABLE DIFFERENCE'
        action = ('label every derived figure as no better than assuming no '
                  'regional change; the four unmeasurable regions keep their '
                  'number under that label')

    print(f'\n{"=" * 72}\nVERDICT: {call}\nACTION:  {action}')

    if all(comparisons[a]['ci_low'] <= 0 <= comparisons[a]['ci_high']
           for a in comparisons):
        print('\nPRE-REGISTERED CAVEAT: every interval contains 0. The honest '
              'reading is that this data cannot rank the arms, not that they\n'
              'are equivalent. A noisy target systematically flatters ZERO.')

    per_region = collections.defaultdict(list)
    for w in weeks:
        for region, err in w.get('DERIVATION', {}).items():
            per_region[region].append(err)
    print(f'\n{"region":<20}{"n":>5}{"MAE":>8}  premium')
    for region in sorted(per_region, key=lambda r: -statistics.mean(per_region[r])):
        basis = ('measured' if region in MEASURED_MULTIPLIERS
                 else 'assumed' if region in ASSUMED_MULTIPLIERS else 'base')
        print(f'{region:<20}{len(per_region[region]):>5}'
              f'{statistics.mean(per_region[region]):>8.2f}  {basis}')

    ungradable = sorted(set(MEASURED_MULTIPLIERS | ASSUMED_MULTIPLIERS)
                        - set(per_region) - {REFERENCE})
    print(f'\nungradable at any date ({len(ungradable)}): {", ".join(ungradable)}')

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        'preregistration': PREREGISTRATION, 'n_weeks': len(weeks),
        'mae': maes, 'best_naive': best_naive, 'comparisons': comparisons,
        'verdict': call, 'action': action,
        'per_region_mae': {r: statistics.mean(v) for r, v in per_region.items()},
        'ungradable': ungradable,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nartifact {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
