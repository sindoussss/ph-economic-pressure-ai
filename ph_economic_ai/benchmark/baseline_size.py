"""How often does the audit protocol declare a false positive when the baseline
pool omits the mean? A size-and-power study.

`baseline_theory.py` gives the *expected* spurious skill, S(rho). This module
answers the question a methods referee actually asks: **what is the test's
false-positive rate?** A Diebold-Mariano test at alpha = 0.05 is supposed to
reject a true null 5% of the time. That guarantee is about sampling noise; it
assumes the null being tested is the right one. When the baseline pool omits the
mean on a mean-reverting target, the hypothesis "beats the strongest naive"
is false in a way no amount of significance testing can detect, because the
comparison itself is misspecified.

Design (a paired experiment — this is the point):
  * simulate a target with NO exploitable structure: a stationary AR(1) plus
    features that are pure noise, so any "edge" found is definitionally spurious;
  * run the project's real `walk_forward` once and score every method;
  * then evaluate `mom_verdict` twice on the SAME fitted losses, changing only
    the baseline pool.

Because the data, the models and the losses are identical across the two arms,
every difference in the verdict is attributable to the pool and nothing else.

The power arm repeats this with a genuine, recoverable driver, to confirm the
corrected pool has not simply made the audit unable to find anything.

Reproduce:  python -m ph_economic_ai.benchmark.baseline_size
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.paths import MIN_TRAIN, artifact

_OUT = artifact('baseline_size.json')

POOL_OLD = ('random_walk', 'seasonal_naive', 'drift')
POOL_NEW = ('random_walk', 'seasonal_naive', 'drift', 'mean')

#: Baselines plus the one ML candidate that produces the artifact. HGB is ~100x
#: slower for an identical qualitative result, so it is reserved for a single
#: confirmatory cell rather than the whole grid.
_METHODS = ['random_walk', 'seasonal_naive', 'drift', 'mean', 'ridge']

ALPHA = 0.05


def _losses(y: np.ndarray, X: np.ndarray, methods, min_train: int) -> tuple[dict, dict]:
    """Per-method squared-error series and RMSE from the project's own backtest."""
    from ph_economic_ai.benchmark.backtest import walk_forward
    from ph_economic_ai.benchmark.forecasters import make_forecaster
    from ph_economic_ai.benchmark.metrics import rmse

    loss, rm = {}, {}
    for m in methods:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        loss[m] = (bt['y_true'] - bt['y_pred']) ** 2
        rm[m] = rmse(bt['y_true'], bt['y_pred'])
    return rm, loss


def _verdict_under(pool, rm: dict, loss: dict) -> str:
    """`mom_verdict` restricted to the methods that pool would have evaluated.

    Critical detail: under the OLD pool the mean was never *run*, so it must be
    removed from the candidate set too. Leaving it in would let the mean itself
    be crowned the winner and would simulate a protocol nobody ever used.
    """
    from ph_economic_ai.benchmark.nowcast import mom_verdict

    keep = [m for m in rm if m in pool or m not in POOL_NEW]
    rm_s = {m: rm[m] for m in keep}
    loss_s = {m: loss[m] for m in keep}
    return mom_verdict(rm_s, loss_s, baseline_pool=pool)['verdict']


def _sim_frame(rho: float, n: int, rng, beta: float = 0.0):
    """Stationary AR(1) target with `n_feat` noise features.

    With beta = 0 the features are pure noise and NO model can legitimately win —
    every positive is a false positive. With beta > 0 the first feature is a
    genuine, contemporaneously-observable driver, so a competent model should win.
    """
    driver = rng.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + rng.normal(0, 1)
    y = y + beta * driver
    X = np.column_stack([driver, rng.normal(0, 1, n), rng.normal(0, 1, n)])
    return y, X


def cell(rho: float, n: int, reps: int, beta: float = 0.0, seed: int = 0,
         methods=None, min_train: int = MIN_TRAIN) -> dict:
    """Rejection rate under each pool for one (rho, n) cell."""
    methods = list(methods or _METHODS)
    old_hits = new_hits = 0
    for r in range(reps):
        rng = np.random.default_rng(seed * 100_003 + r)
        y, X = _sim_frame(rho, n, rng, beta=beta)
        rm, loss = _losses(y, X, methods, min_train)
        old_hits += _verdict_under(POOL_OLD, rm, loss) == 'beats_best_naive'
        new_hits += _verdict_under(POOL_NEW, rm, loss) == 'beats_best_naive'
    return {
        'rho': rho, 'n': n, 'reps': reps, 'beta': beta,
        'rate_pool_without_mean': round(old_hits / reps, 4),
        'rate_pool_with_mean': round(new_hits / reps, 4),
    }


def run(reps: int = 300, rhos=(0.0, 0.2, 0.35, 0.5, 0.7), ns=(61, 151),
        power_beta: float = 0.6) -> dict:
    """Size grid (beta = 0) plus a power arm (beta > 0) at the real sample sizes."""
    size = [cell(rho, n, reps, beta=0.0, seed=i)
            for i, (rho, n) in enumerate((r, n) for r in rhos for n in ns)]
    power = [cell(rho, n, reps, beta=power_beta, seed=500 + i)
             for i, (rho, n) in enumerate((r, n) for r in (0.0, 0.35) for n in ns)]
    worst = max(size, key=lambda c: c['rate_pool_without_mean'])

    # Does the distortion die where S(rho) says it should? baseline_theory derives
    # the crossover at rho = 1/2 analytically; this checks it from the other side.
    above = [c['rate_pool_without_mean'] for c in size if c['rho'] >= 0.5]
    below = [c['rate_pool_without_mean'] for c in size if c['rho'] <= 0.2]

    # Does MORE data help? It does not — and that is the sharpest single fact
    # here, because "re-run it on a longer sample" is the standard robustness move.
    grows = [
        {'rho': r,
         'n_small': min(ns), 'rate_small': next(c['rate_pool_without_mean']
                                                for c in size if c['rho'] == r and c['n'] == min(ns)),
         'n_large': max(ns), 'rate_large': next(c['rate_pool_without_mean']
                                                for c in size if c['rho'] == r and c['n'] == max(ns))}
        for r in rhos
    ]
    worsens_with_n = all(g['rate_large'] >= g['rate_small'] for g in grows)

    return {
        'alpha': ALPHA,
        'note': ('Size: fraction of PURE-NOISE targets declared beats_best_naive '
                 '(every rejection is a false positive; nominal rate is alpha). '
                 'Power: fraction of targets with a genuine driver correctly '
                 'detected. Both arms score the identical fitted losses under two '
                 'baseline pools, so the pool is the only difference.'),
        'power_caveat': ('The without-mean column of the power arm is NOT power. On '
                         'these targets the same pool has a 43-100% false-positive '
                         'rate, so its detections are power and size confounded. '
                         'Only the with-mean column measures detection of real '
                         'signal.'),
        'methods': _METHODS,
        'size': size,
        'power': power,
        'worst_case_false_positive_rate': worst['rate_pool_without_mean'],
        'worst_case_cell': {'rho': worst['rho'], 'n': worst['n']},
        'max_rate_at_or_above_crossover': max(above) if above else None,
        'max_rate_well_below_crossover': max(below) if below else None,
        'distortion_confined_below_crossover': bool(above and max(above) <= 0.05),
        'false_positive_rate_by_n': grows,
        'worsens_with_more_data': worsens_with_n,
        'max_size_with_mean': max(c['rate_pool_with_mean'] for c in size),
    }


def _main() -> int:
    r = run()
    _OUT.write_text(json.dumps(r, indent=2), encoding='utf-8')

    print(f'SIZE - false-positive rate on pure noise (nominal alpha = {ALPHA})')
    print(f"  {'rho':>5} {'n':>5} {'without mean':>14} {'with mean':>11}")
    for c in r['size']:
        print(f"  {c['rho']:>5.2f} {c['n']:>5} {c['rate_pool_without_mean']:>13.1%} "
              f"{c['rate_pool_with_mean']:>10.1%}")
    print(f"\n  Worst case: {r['worst_case_false_positive_rate']:.1%} at "
          f"rho={r['worst_case_cell']['rho']}, n={r['worst_case_cell']['n']} "
          f"-- vs a nominal {ALPHA:.0%}.")

    print(f"\n  Distortion confined to rho < 0.5 (the S(rho) crossover): "
          f"{r['distortion_confined_below_crossover']} "
          f"(max rate at rho>=0.5 is {r['max_rate_at_or_above_crossover']:.1%})")
    print(f"  With the mean in the pool, size is correct everywhere: max "
          f"{r['max_size_with_mean']:.1%}")

    print('\n  MORE DATA MAKES IT WORSE (the standard robustness move backfires):')
    for g in r['false_positive_rate_by_n']:
        print(f"    rho={g['rho']:.2f}:  n={g['n_small']} -> {g['rate_small']:.1%}"
              f"   |   n={g['n_large']} -> {g['rate_large']:.1%}")

    print('\nPOWER - detection rate when a genuine driver IS present')
    print(f"  {'rho':>5} {'n':>5} {'without mean':>14} {'with mean':>11}")
    for c in r['power']:
        print(f"  {c['rho']:>5.2f} {c['n']:>5} {c['rate_pool_without_mean']:>13.1%} "
              f"{c['rate_pool_with_mean']:>10.1%}")
    print(f"\n  {r['power_caveat']}")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
