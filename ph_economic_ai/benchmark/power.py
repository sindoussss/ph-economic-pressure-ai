"""Statistical power / minimum-detectable-effect for the efficiency nulls.

An efficiency finding *accepts* the null of equal predictive accuracy against a
random walk. On small samples that is only informative if the test *could* have
detected a meaningful edge — "we found nothing" means little if the test was too
weak to find anything. This module quantifies that: given the observed
loss-differential variance and the sample size, what is the smallest forecast
skill (RMSE improvement over the random walk) the Diebold–Mariano test could
detect at a target power?

Every null in the map is bounded here, not just the flagship. Two families:

  * the **fuel one-month forecast** (RQ1), measured against the random walk from
    the committed `backtest_predictions.csv`; and
  * every **month-on-month nowcast null** (headline, food, electricity,
    transport), measured against the baseline that actually binds them — the
    historical **mean** (§4.7), not the random walk. Bounding a nowcast null
    against the random walk would overstate the test's power, because the mean is
    the harder baseline on a mean-reverting rate.

Reporting an MDE per target is what makes "no detectable edge" a claim rather
than a shrug: it says how large an edge would have had to be to show up.

Pure numpy/scipy — stays inside the validated benchmark. Reproduce with
`python -m ph_economic_ai.benchmark.power`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from ph_economic_ai.benchmark.paths import ARTIFACTS_DIR, MIN_TRAIN, artifact

_ARTIFACTS = ARTIFACTS_DIR
_PRED = _ARTIFACTS / 'backtest_predictions.csv'
_OUT = artifact('power.json')


def mde_from_errors(
    e_model: np.ndarray,
    e_base: np.ndarray,
    baseline_name: str = 'random walk',
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Minimum RMSE-skill over `e_base` that a DM test could detect at `power`.

    The DM test is on the squared-error differential d = e_model² − e_base². The
    smallest mean differential detectable at two-sided α and power 1−β is
    (t_{1−α/2} + t_{power})·SE(d̄); translating that back through the baseline MSE
    gives the equivalent skill (1 − RMSE_model/RMSE_base).

    Works against *any* baseline, because which baseline binds is target-specific:
    the random walk for the level forecasts, the historical mean for the
    mean-reverting rate nowcasts.
    """
    e_model = np.asarray(e_model, dtype=float)
    e_base = np.asarray(e_base, dtype=float)
    n = len(e_base)
    d = e_model ** 2 - e_base ** 2
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float('nan')

    mse_base = float(np.mean(e_base ** 2))
    rmse_base = float(np.sqrt(mse_base))
    rmse_model = float(np.sqrt(np.mean(e_model ** 2)))
    observed_skill = 1.0 - rmse_model / rmse_base if rmse_base else 0.0

    df = n - 1
    t_crit = float(stats.t.ppf(1 - alpha / 2, df))
    t_pow = float(stats.t.ppf(power, df))
    mde_loss_diff = (t_crit + t_pow) * se                 # in squared-error units
    mde_rmse = float(np.sqrt(max(mse_base - mde_loss_diff, 0.0)))
    mde_skill = 1.0 - mde_rmse / rmse_base if rmse_base else 0.0

    return {
        'n': n,
        'alpha': alpha,
        'power': power,
        'baseline': baseline_name,
        'rmse_baseline': round(rmse_base, 4),
        'rmse_model': round(rmse_model, 4),
        'observed_skill': round(observed_skill, 4),
        'min_detectable_skill': round(mde_skill, 4),
        'min_detectable_skill_pct': round(mde_skill * 100, 1),
        'interpretation': (
            f'At n={n} the test can detect a skill of ~{mde_skill * 100:.0f}% over '
            f'the {baseline_name} at {power:.0%} power; the observed skill is '
            f'{observed_skill * 100:+.1f}%. The null therefore rules out an edge of '
            f'roughly {mde_skill * 100:.0f}% or larger, not smaller ones — "no '
            f'detectable edge at this power", not proven efficiency.'
        ),
    }


def min_detectable_skill(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """MDE for a *level* forecast against the random walk, whose one-step error is
    the first difference of the actuals. Thin wrapper over `mde_from_errors`."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e_model = (y_true - y_pred)[1:]           # align to months with a predecessor
    e_rw = np.diff(y_true)                    # RW error = actual monthly change
    out = mde_from_errors(e_model, e_rw, baseline_name='random walk',
                          alpha=alpha, power=power)
    out['rmse_random_walk'] = out['rmse_baseline']   # back-compat key
    return out


# ── Nowcast nulls: bound each against the baseline that actually binds it ──────

def _nowcast_frames() -> dict:
    """The MoM nowcast targets, reusing the same builders the audit runs on."""
    from ph_economic_ai.benchmark.nowcast import build_nowcast_frame
    from ph_economic_ai.benchmark.targets import load_inflation_mom
    from ph_economic_ai.benchmark.longsample import load_long_features
    from ph_economic_ai.benchmark.psa_cpi import load_transport_mom
    from ph_economic_ai.benchmark.food_nowcast import _build_food_frame, load_food_features
    from ph_economic_ai.benchmark.electricity_nowcast import (
        _build_electricity_frame, load_electricity_features)

    headline = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    long = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom',
                               features=load_long_features())
    food = _build_food_frame(load_food_features())
    elec = _build_electricity_frame(load_electricity_features())
    transport = build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom',
                                    features=load_long_features())

    def drivers_only(f):
        return f.drop(columns=['prev_mom'], errors='ignore')

    return {
        'headline_mom': headline,
        'headline_mom_long': long,
        'food_mom': food,
        'electricity_mom': elec,
        'electricity_driver_only': drivers_only(elec),
        'transport_driver_only': drivers_only(transport),
    }


def _best_candidate_errors(frame, min_train: int):
    """Errors of the strongest *candidate* (non-baseline) and of the mean baseline.

    The strongest candidate is used deliberately: the MDE should describe the most
    favourable case the audit actually gave a model, so the bound is not flattered
    by picking a weak one.
    """
    from ph_economic_ai.benchmark.audit import BASELINES
    from ph_economic_ai.benchmark.backtest import walk_forward
    from ph_economic_ai.benchmark.forecasters import FORECASTERS, make_forecaster
    from ph_economic_ai.benchmark.metrics import rmse

    y = frame['target'].to_numpy(dtype=float)
    X = frame[[c for c in frame.columns if c != 'target']].to_numpy(dtype=float)
    candidates = [m for m in FORECASTERS if m not in BASELINES]

    best_name, best_err, best_rmse = None, None, float('inf')
    for m in candidates:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        r = rmse(bt['y_true'], bt['y_pred'])
        if r < best_rmse:
            best_name, best_err, best_rmse = m, bt['y_true'] - bt['y_pred'], r
    bm = walk_forward(y, X, make_forecaster('mean'), min_train)
    return best_name, best_err, bm['y_true'] - bm['y_pred']


def run_nowcast_power(min_train: int = MIN_TRAIN) -> dict:
    """MDE for every MoM nowcast null, measured against the mean baseline."""
    out = {}
    for label, frame in _nowcast_frames().items():
        best, e_model, e_mean = _best_candidate_errors(frame, min_train)
        row = mde_from_errors(e_model, e_mean, baseline_name='mean')
        row['best_candidate'] = best
        out[label] = row
    return out


def run() -> dict:
    df = pd.read_csv(_PRED)
    result = {
        'fuel_one_month_forecast': min_detectable_skill(
            df['y_true'].values, df['y_pred'].values),
        'nowcast_nulls_vs_mean': run_nowcast_power(),
    }
    _OUT.write_text(json.dumps(result, indent=2))
    return result


def _main() -> int:
    r = run()
    f = r['fuel_one_month_forecast']
    print('Minimum-detectable-effect - fuel one-month forecast (flagship RQ1 null):')
    print(f"  n = {f['n']}, RMSE random walk = {f['rmse_baseline']}")
    print(f"  observed skill      : {f['observed_skill'] * 100:+.1f}%")
    print(f"  minimum detectable  : {f['min_detectable_skill_pct']}% "
          f"(at {f['power']:.0%} power, alpha = {f['alpha']})")
    print(f"\n  {f['interpretation']}\n")

    print('MDE for every MoM nowcast null (vs the MEAN, the baseline that binds):')
    print(f"  {'target':26} {'n':>4} {'best cand':>10} {'observed':>10} {'detectable':>11}")
    for label, row in r['nowcast_nulls_vs_mean'].items():
        print(f"  {label:26} {row['n']:>4} {row['best_candidate']:>10} "
              f"{row['observed_skill'] * 100:>+9.1f}% "
              f"{row['min_detectable_skill_pct']:>10.1f}%")
    print('\n  Each null rules out an edge at or above its detectable column, not '
          'smaller ones.')
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
