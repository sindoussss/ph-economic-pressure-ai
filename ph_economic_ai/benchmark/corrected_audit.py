"""Corrected predictability audit: the paper's map re-run with the historical
MEAN added to the naive baseline pool.

Motivation
----------
The nowcast verdict (`nowcast.mom_verdict`) chooses the best simple baseline from
the pool {random_walk, seasonal_naive, drift} and requires a candidate to beat it
by a Diebold–Mariano test. That pool omits the *historical mean*. For a persistent
*level* series (fuel, FX, YoY inflation) the mean is a useless predictor, so the
efficiency verdicts are unaffected. But every MoM inflation target is a
*mean-reverting rate*, for which the mean — not the random walk — is the strong
naive. Against the mean, none of the paper's MoM 'positives' (headline, food,
electricity) remain significant, and the electricity 'driver edge' is reproduced
by Ridge on pure noise (see docs/defense/mean-baseline-finding.md). This module
re-derives the whole map with the mean in the pool.

Non-destructive: it writes a NEW artifact, `corrected_predictability_map.json`,
and does not touch any frozen artifact or change the shipping pipeline. To make
the correction canonical, add 'mean' to `nowcast.BASELINE_POOL` (and to
`PANEL_METHODS`) and regenerate — but that rewrites the numbers the current
manuscript cites and several unit-test expectations, so it is left as a
deliberate, reviewed step.

Reproduce:  python -m ph_economic_ai.benchmark.corrected_audit
"""
from __future__ import annotations

import json
from pathlib import Path

from ph_economic_ai.benchmark.audit import verdict_from_panel
from ph_economic_ai.benchmark.efficiency import run_panel
from ph_economic_ai.benchmark.electricity_nowcast import (
    _build_electricity_frame, load_electricity_features)
from ph_economic_ai.benchmark.food_nowcast import _build_food_frame, load_food_features
from ph_economic_ai.benchmark.longsample import load_long_features
from ph_economic_ai.benchmark.nowcast import (
    PANEL_METHODS, build_nowcast_frame, run_mom_nowcast)
from ph_economic_ai.benchmark.psa_cpi import load_transport_mom
from ph_economic_ai.benchmark.targets import TARGETS, load_inflation_mom

MIN_TRAIN = 24
_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
_OUT = _ARTIFACTS / 'corrected_predictability_map.json'

POOL_OLD = ('random_walk', 'seasonal_naive', 'drift')
POOL_NEW = ('random_walk', 'seasonal_naive', 'drift', 'mean')

FULL_OLD = list(PANEL_METHODS)                                   # paper's panel
FULL_NEW = list(PANEL_METHODS) + ['mean']
DRV_OLD = ['random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb']
DRV_NEW = DRV_OLD + ['mean']


def _slim(v: dict) -> dict:
    return {k: v.get(k) for k in ('verdict', 'best_method', 'best_naive',
                                  'best_skill_vs_naive', 'dm_p', 'n')}


def _compare(frame, methods_old, methods_new) -> dict:
    """Old (paper) verdict vs corrected (mean-in-pool) verdict for one frame."""
    old = run_mom_nowcast(MIN_TRAIN, baseline_pool=POOL_OLD, frame=frame, methods=methods_old)
    new = run_mom_nowcast(MIN_TRAIN, baseline_pool=POOL_NEW, frame=frame, methods=methods_new)
    return {'old': _slim(old), 'corrected': _slim(new)}


def _nowcast_rows() -> list[dict]:
    """Every MoM nowcast target: full nowcast and (where the paper runs one) the
    driver-only ablation, each with old vs corrected verdict."""
    headline = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    headline_long = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom',
                                        features=load_long_features())
    food = _build_food_frame(load_food_features())
    elec = _build_electricity_frame(load_electricity_features())
    transport = build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom',
                                    features=load_long_features())

    def drv(frame):
        return frame.drop(columns=['prev_mom'], errors='ignore')

    specs = [
        ('Headline MoM inflation (short)', 'full nowcast', headline, FULL_OLD, FULL_NEW),
        ('Headline MoM inflation (long, 2007-2026)', 'full nowcast', headline_long, FULL_OLD, FULL_NEW),
        ('Food MoM inflation', 'full nowcast', food, FULL_OLD, FULL_NEW),
        ('Food MoM inflation', 'driver-only', drv(food), DRV_OLD, DRV_NEW),
        ('Electricity MoM inflation', 'full nowcast', elec, FULL_OLD, FULL_NEW),
        ('Electricity MoM inflation', 'driver-only (flagship edge)', drv(elec), DRV_OLD, DRV_NEW),
        ('Transport MoM inflation', 'full nowcast', transport, FULL_OLD, FULL_NEW),
        ('Transport MoM inflation', 'driver-only', drv(transport), DRV_OLD, DRV_NEW),
    ]
    rows = []
    for name, setup, frame, mo, mn in specs:
        cmp = _compare(frame, mo, mn)
        rows.append({'target': name, 'setup': setup, **cmp,
                     'flipped_to_null': (cmp['old']['verdict'] == 'beats_best_naive'
                                         and cmp['corrected']['verdict'] != 'beats_best_naive')})
    return rows


def _forecast_rows() -> list[dict]:
    """The one-month forecast efficiency verdicts (fuel/FX/YoY), confirming the
    mean does NOT create a false positive on persistent level series: its skill
    vs the random walk is strongly negative there, so the nulls stand."""
    rows = []
    for name in ('fuel', 'fx', 'inflation'):
        frame = TARGETS[name].build_frame()
        feature_cols = [c for c in frame.columns if c != 'target']
        panel = run_panel(frame, FULL_NEW, MIN_TRAIN, feature_cols, target_col='target')
        verdict, best = verdict_from_panel(panel)
        mean_row = next(r for r in panel if r['method'] == 'mean')
        rows.append({'target': name, 'verdict_with_mean': verdict,
                     'best_method': best['method'], 'best_skill_vs_rw': best['skill_vs_rw'],
                     'mean_skill_vs_rw': mean_row['skill_vs_rw'], 'n': int(panel[0]['n'])})
    return rows


def run() -> dict:
    result = {
        'note': 'Corrected map: historical MEAN added to the naive baseline pool.',
        'pool_old': list(POOL_OLD), 'pool_new': list(POOL_NEW),
        'forecast_efficiency': _forecast_rows(),
        'nowcast': _nowcast_rows(),
    }
    _OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def _main() -> int:
    r = run()

    print('FORECAST efficiency (mean added as a candidate — nulls must stand):')
    print(f"  {'target':10} {'verdict':11} {'best skill vs RW':>17} {'mean skill vs RW':>17}")
    for f in r['forecast_efficiency']:
        print(f"  {f['target']:10} {f['verdict_with_mean']:11} "
              f"{f['best_skill_vs_rw']:>+17.3f} {f['mean_skill_vs_rw']:>+17.3f}")

    print('\nNOWCAST verdicts — paper pool  vs  mean-in-pool:')
    print(f"  {'target':42} {'setup':28} {'OLD':>18}   {'CORRECTED':>20}")
    for row in r['nowcast']:
        o, c = row['old'], row['corrected']
        old = f"{o['verdict']}({o['best_skill_vs_naive']:+.2f})"
        cor = f"{c['verdict']}"
        flag = '   <-- FLIPS' if row['flipped_to_null'] else ''
        print(f"  {row['target']:42} {row['setup']:28} {old:>18}   {cor:>20}{flag}")

    n_flip = sum(1 for row in r['nowcast'] if row['flipped_to_null'])
    print(f"\n{n_flip} of {len(r['nowcast'])} nowcast verdicts flip from "
          f"'beats_best_naive' to null once the mean is in the pool.")
    print(f"Wrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
