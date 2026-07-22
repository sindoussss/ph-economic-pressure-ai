"""Statistical power / minimum-detectable-effect for the efficiency nulls.

An efficiency finding *accepts* the null of equal predictive accuracy against a
random walk. On small samples that is only informative if the test *could* have
detected a meaningful edge — "we found nothing" means little if the test was too
weak to find anything. This module quantifies that: given the observed
loss-differential variance and the sample size, what is the smallest forecast
skill (RMSE improvement over the random walk) the Diebold–Mariano test could
detect at a target power?

Reads the committed `backtest_predictions.csv` (the one-month fuel forecast, the
flagship RQ1 null), reconstructs the random-walk error as the first difference of
the actuals, and reports the minimum detectable skill. Pure numpy/scipy — stays
inside the validated benchmark. Reproduce with
`python -m ph_economic_ai.benchmark.power`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
_PRED = _ARTIFACTS / 'backtest_predictions.csv'
_OUT = _ARTIFACTS / 'power.json'


def min_detectable_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Minimum RMSE-skill vs a random walk detectable by a DM test at `power`.

    The random-walk one-step error is the first difference of the actuals. The
    DM test is on the squared-error differential; the minimum detectable mean
    differential at two-sided α and power 1−β is (t_{1−α/2} + t_{power})·SE, and
    that is translated back into a skill (1 − RMSE_model/RMSE_rw) equivalent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e_model = (y_true - y_pred)[1:]          # align to months with a predecessor
    e_rw = np.diff(y_true)                    # RW error = actual monthly change
    n = len(e_rw)
    d = e_model ** 2 - e_rw ** 2             # loss differential (model − RW)
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')

    mse_rw = float(np.mean(e_rw ** 2))
    rmse_rw = float(np.sqrt(mse_rw))
    rmse_model = float(np.sqrt(np.mean(e_model ** 2)))
    observed_skill = 1.0 - rmse_model / rmse_rw if rmse_rw else 0.0

    df = n - 1
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    t_pow = float(stats.t.ppf(power, df))
    mde_loss_diff = (t_crit + t_pow) * se                 # in squared-error units
    mde_rmse = float(np.sqrt(max(mse_rw - mde_loss_diff, 0.0)))
    mde_skill = 1.0 - mde_rmse / rmse_rw if rmse_rw else 0.0

    return {
        'n': n,
        'alpha': alpha,
        'power': power,
        'rmse_random_walk': round(rmse_rw, 3),
        'rmse_model': round(rmse_model, 3),
        'observed_skill': round(observed_skill, 4),
        'min_detectable_skill': round(mde_skill, 4),
        'min_detectable_skill_pct': round(mde_skill * 100, 1),
        'interpretation': (
            f'At n={n} the test can detect a skill of ~{mde_skill * 100:.0f}% over '
            f'the random walk at {power:.0%} power; the observed skill is '
            f'{observed_skill * 100:+.1f}%. The efficiency finding therefore rules '
            f'out an edge of roughly {mde_skill * 100:.0f}% or larger, not smaller '
            f'edges — "no detectable edge at this power", not proven efficiency.'
        ),
    }


def run() -> dict:
    df = pd.read_csv(_PRED)
    result = {'fuel_one_month_forecast': min_detectable_skill(
        df['y_true'].values, df['y_pred'].values)}
    _OUT.write_text(json.dumps(result, indent=2))
    return result


def _main() -> int:
    r = run()['fuel_one_month_forecast']
    print('Minimum-detectable-effect — fuel one-month forecast (flagship RQ1 null):')
    print(f"  n = {r['n']}, RMSE random walk = {r['rmse_random_walk']}")
    print(f"  observed skill      : {r['observed_skill'] * 100:+.1f}%")
    print(f"  minimum detectable  : {r['min_detectable_skill_pct']}% "
          f"(at {r['power']:.0%} power, α = {r['alpha']})")
    print(f"\n  {r['interpretation']}")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
