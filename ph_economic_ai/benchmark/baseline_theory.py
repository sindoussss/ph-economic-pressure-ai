"""When does a useless model appear to beat the random walk? A closed form.

Motivation
----------
Adding the historical mean to the naive pool (§4.7) turned every positive in this
audit into a null. That the mean *belongs* in the pool is not new — it is implicit
in Hyndman & Koehler (2006) and in the Atkeson–Ohanian (2001) benchmark debate.
What is not standard is a statement of **when the omission bites and by how much**.
This module supplies it, so the correction is a general result rather than an
anecdote about one dataset.

The result
----------
Let the target be covariance-stationary with variance sigma^2 and lag-1
autocorrelation rho. Then

    RMSE(random walk) = sigma * sqrt(2 * (1 - rho))        # E[(y_t - y_{t-1})^2]
    RMSE(mean)        = sigma                              # E[(y_t - mu)^2]

so a forecaster that carries NO information beyond the unconditional mean shows an
apparent skill over the random walk of

    S(rho) = 1 - 1 / sqrt(2 * (1 - rho))

which is **positive if and only if rho < 1/2**, and grows without bound as rho
approaches -1. The crossover at rho = 1/2 is exact.

Consequences:
  * On a mean-reverting rate (rho < 1/2) the random walk is not a neutral baseline;
    it is a *weak* one, by a margin the target's own autocorrelation determines.
  * A model can therefore be reported as significantly beating the random walk
    while being no better than — or worse than — a constant.
  * The effect requires no data mining, no leakage, and no overfitting. It is a
    property of the target, so it is stable across sub-samples, survives
    robustness re-tests, and is unaffected by multiple-comparison correction.
    Every guard that does not interrogate the baseline will pass it.

Only rho enters, so the result is not specific to an AR(1) generating process; it
holds for any stationary target.

Reproduce:  python -m ph_economic_ai.benchmark.baseline_theory
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
_OUT = _ARTIFACTS / 'baseline_theory.json'

#: Lag-1 autocorrelation below which a mean-predictor out-scores the random walk.
CROSSOVER_RHO = 0.5


def spurious_skill(rho: float) -> float:
    """Apparent skill vs the random walk of a forecaster carrying no information
    beyond the unconditional mean, for a stationary target with lag-1
    autocorrelation `rho`. Positive iff rho < 0.5."""
    if rho >= 1.0:
        raise ValueError('rho must be < 1 for a stationary target')
    return 1.0 - 1.0 / np.sqrt(2.0 * (1.0 - rho))


def implied_rho(skill: float) -> float:
    """Invert `spurious_skill`: the lag-1 autocorrelation at which a mean-predictor
    would show exactly this apparent skill. Lets a reported 'edge' be checked
    against the target's measured autocorrelation."""
    if skill >= 1.0:
        raise ValueError('skill must be < 1')
    return 1.0 - 1.0 / (2.0 * (1.0 - skill) ** 2)


def lag1_autocorr(y) -> float:
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])


def simulate(rho: float, n: int = 400, reps: int = 30, min_train: int = 24) -> float:
    """Empirical skill of the mean over the random walk on a simulated AR(1) with
    autocorrelation `rho`, measured through the project's own walk-forward
    backtest. Validates the closed form against the actual estimator (which uses
    an expanding-window mean, not the true mu)."""
    from ph_economic_ai.benchmark.backtest import walk_forward
    from ph_economic_ai.benchmark.forecasters import make_forecaster
    from ph_economic_ai.benchmark.metrics import rmse

    out = []
    for seed in range(reps):
        rng = np.random.default_rng(seed)
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = rho * y[t - 1] + rng.normal(0, 1)
        X = rng.normal(0, 1, (n, 2))          # deliberately irrelevant features
        bm = walk_forward(y, X, make_forecaster('mean'), min_train)
        br = walk_forward(y, X, make_forecaster('random_walk'), min_train)
        out.append(1 - rmse(bm['y_true'], bm['y_pred']) / rmse(br['y_true'], br['y_pred']))
    return float(np.mean(out))


def _target_frames() -> dict:
    from ph_economic_ai.benchmark.nowcast import build_nowcast_frame
    from ph_economic_ai.benchmark.targets import load_inflation_mom
    from ph_economic_ai.benchmark.longsample import load_long_features
    from ph_economic_ai.benchmark.psa_cpi import load_transport_mom
    from ph_economic_ai.benchmark.food_nowcast import _build_food_frame, load_food_features
    from ph_economic_ai.benchmark.electricity_nowcast import (
        _build_electricity_frame, load_electricity_features)
    return {
        'headline MoM': build_nowcast_frame(
            target_loader=load_inflation_mom, prev_col='prev_mom'),
        'headline MoM (long)': build_nowcast_frame(
            target_loader=load_inflation_mom, prev_col='prev_mom',
            features=load_long_features()),
        'food MoM': _build_food_frame(load_food_features()),
        'electricity MoM': _build_electricity_frame(load_electricity_features()),
        'transport MoM': build_nowcast_frame(
            target_loader=load_transport_mom, prev_col='prev_mom',
            features=load_long_features()),
    }


def validate_on_targets() -> list[dict]:
    """For each real target: measured rho, the skill the closed form predicts a
    mean-predictor would show, and the skill actually measured by the backtest."""
    from ph_economic_ai.benchmark.backtest import walk_forward
    from ph_economic_ai.benchmark.forecasters import make_forecaster
    from ph_economic_ai.benchmark.metrics import rmse

    rows = []
    for label, frame in _target_frames().items():
        y = frame['target'].to_numpy(dtype=float)
        X = frame[[c for c in frame.columns if c != 'target']].to_numpy(dtype=float)
        rho = lag1_autocorr(y)
        bm = walk_forward(y, X, make_forecaster('mean'), 24)
        br = walk_forward(y, X, make_forecaster('random_walk'), 24)
        observed = 1 - rmse(bm['y_true'], bm['y_pred']) / rmse(br['y_true'], br['y_pred'])
        rows.append({
            'target': label,
            'rho': round(rho, 4),
            'predicted_spurious_skill': round(spurious_skill(rho), 4),
            'observed_mean_vs_rw_skill': round(float(observed), 4),
            'abs_error': round(abs(float(observed) - spurious_skill(rho)), 4),
        })
    return rows


def run(simulate_grid=(-0.2, 0.0, 0.2, 0.4, 0.5, 0.6, 0.8)) -> dict:
    sims = [{'rho': r, 'predicted': round(spurious_skill(r), 4),
             'simulated': round(simulate(r), 4)} for r in simulate_grid]
    for s in sims:
        s['abs_error'] = round(abs(s['simulated'] - s['predicted']), 4)
    targets = validate_on_targets()
    return {
        'formula': 'skill_vs_random_walk = 1 - 1/sqrt(2*(1-rho))',
        'crossover_rho': CROSSOVER_RHO,
        'note': ('A forecaster carrying no information beyond the unconditional '
                 'mean out-scores the random walk whenever the target\'s lag-1 '
                 'autocorrelation is below 0.5. The effect is a property of the '
                 'target, so it is stable across sub-samples and survives '
                 'robustness and multiple-comparison checks.'),
        'simulation': sims,
        'targets': targets,
        'max_abs_error_targets': max(t['abs_error'] for t in targets),
    }


def _main() -> int:
    r = run()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(r, indent=2), encoding='utf-8')

    print(f"Spurious skill of a mean-predictor vs the random walk: {r['formula']}")
    print(f"Positive iff rho < {r['crossover_rho']}\n")
    print('Simulation (project walk-forward on synthetic AR(1)):')
    print(f"  {'rho':>6} {'predicted':>10} {'simulated':>10} {'|err|':>7}")
    for s in r['simulation']:
        print(f"  {s['rho']:>6.2f} {s['predicted']:>+10.3f} {s['simulated']:>+10.3f} "
              f"{s['abs_error']:>7.3f}")
    print('\nReal targets — does the closed form explain the observed gap?')
    print(f"  {'target':22} {'rho':>7} {'predicted':>10} {'observed':>9} {'|err|':>7}")
    for t in r['targets']:
        print(f"  {t['target']:22} {t['rho']:>+7.3f} "
              f"{t['predicted_spurious_skill']:>+10.3f} "
              f"{t['observed_mean_vs_rw_skill']:>+9.3f} {t['abs_error']:>7.3f}")
    print(f"\nMax absolute error on real targets: {r['max_abs_error_targets']:.3f}")
    print(f"Wrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
