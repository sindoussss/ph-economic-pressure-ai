"""Select on one segment, evaluate on another that selection never saw.

`RSK-004`: best-model p-values may be optimistic. The audit runs a panel of
candidate methods, picks whichever wins, and reports that winner's
Diebold-Mariano p-value. That p-value answers "would this model look this good by
chance", but the model was chosen *because* it looked good, so the question it
actually needs to answer is "would the BEST OF K models look this good by chance".
Those differ, and the gap grows with K.

Two defences are provided here and they are not interchangeable.

`selection_corrected_p` is the cheap one: a Bonferroni step over the number of
candidates actually compared. It is honest but conservative, and it still reuses
the same data for choosing and for testing.

`run_selection_holdout` is the real one. The model class is chosen using only the
early segment; the reported number is that fixed model's performance over a
holdout the selection never touched. The walk-forward remains strictly causal
throughout, so holdout predictions are still trained only on their own past.

One subtlety worth stating because getting it wrong would silently reintroduce the
bug: the naive baseline is also fixed during selection. Re-choosing the best naive
on the holdout would be a second selection performed on the evaluation data, which
is the very thing this module exists to prevent.
"""
from __future__ import annotations

import numpy as np

from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.forecasters import make_forecaster
from ph_economic_ai.benchmark.metrics import rmse as _rmse
from ph_economic_ai.benchmark.significance import diebold_mariano

DEFAULT_HOLDOUT_FRAC = 0.30
MIN_HOLDOUT_PREDICTIONS = 12


def selection_corrected_p(p_value: float, n_candidates: int) -> float:
    """Bonferroni step over the candidates a winner was chosen from."""
    if n_candidates < 1:
        raise ValueError('n_candidates must be at least 1')
    return float(min(1.0, p_value * n_candidates))


def split_point(n: int, min_train: int, holdout_frac: float = DEFAULT_HOLDOUT_FRAC) -> int:
    """Row index where the holdout begins.

    Both sides must be able to produce predictions: the selection segment needs
    more than `min_train` rows, and the holdout needs enough predictions for a DM
    test to mean anything.
    """
    cut = int(round(n * (1.0 - holdout_frac)))
    cut = min(cut, n - MIN_HOLDOUT_PREDICTIONS)
    return max(cut, min_train + MIN_HOLDOUT_PREDICTIONS)


def _losses(y, X, method, min_train):
    bt = walk_forward(y, X, make_forecaster(method), min_train)
    return bt['index'], bt['y_true'], bt['y_pred']


def run_selection_holdout(frame, methods, baseline_pool, min_train: int = 24,
                          holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
                          target_col: str = 'target') -> dict:
    """Choose a model on the early segment, then report it on an untouched holdout.

    Returns both stages so the shrinkage between them is visible. A large drop from
    selection skill to holdout skill is the signature of selection bias, and is
    exactly what a single-stage p-value hides.
    """
    n = len(frame)
    feature_cols = [c for c in frame.columns if c != target_col]
    y = frame[target_col].to_numpy(dtype=float)
    X = frame[feature_cols].to_numpy(dtype=float)

    cut = split_point(n, min_train, holdout_frac)
    if cut >= n - 1 or cut <= min_train:
        return {'verdict': 'insufficient_data', 'n': int(n), 'cut': int(cut)}

    candidates = [m for m in methods if m not in baseline_pool]
    pool = [m for m in baseline_pool]

    # ---- stage 1: selection, on rows [0, cut) only -------------------------
    sel_rmse = {}
    for method in candidates + pool:
        _, yt, yp = _losses(y[:cut], X[:cut], method, min_train)
        sel_rmse[method] = _rmse(yt, yp)

    available_pool = [m for m in pool if m in sel_rmse] or ['random_walk']
    best_naive = min(available_pool, key=lambda m: sel_rmse[m])
    beat = [m for m in candidates if sel_rmse[m] < sel_rmse[best_naive]]
    if not beat:
        selected = min(candidates, key=lambda m: sel_rmse[m]) if candidates else best_naive
        selection_beat_naive = False
    else:
        selected = min(beat, key=lambda m: sel_rmse[m])
        selection_beat_naive = True
    selection_skill = 1 - sel_rmse[selected] / sel_rmse[best_naive]

    # ---- stage 2: evaluation, on rows [cut, n) only ------------------------
    # The full walk-forward is causal, so a prediction at row i used only rows < i.
    # Restricting to i >= cut gives predictions whose MODEL CHOICE also predates
    # them.
    idx_m, yt_m, yp_m = _losses(y, X, selected, min_train)
    idx_b, yt_b, yp_b = _losses(y, X, best_naive, min_train)
    keep_m, keep_b = idx_m >= cut, idx_b >= cut
    if keep_m.sum() < 3:
        return {'verdict': 'insufficient_data', 'n': int(n), 'cut': int(cut),
                'n_holdout_predictions': int(keep_m.sum())}

    model_loss = (yt_m[keep_m] - yp_m[keep_m]) ** 2
    naive_loss = (yt_b[keep_b] - yp_b[keep_b]) ** 2
    holdout_model_rmse = _rmse(yt_m[keep_m], yp_m[keep_m])
    holdout_naive_rmse = _rmse(yt_b[keep_b], yp_b[keep_b])
    holdout_skill = 1 - holdout_model_rmse / holdout_naive_rmse
    dm = diebold_mariano(model_loss, naive_loss)
    holdout_p = float(dm['p_value'])
    confirmed = bool(holdout_skill > 0 and holdout_p < 0.05 and dm['dm_stat'] < 0)

    return {
        'verdict': 'confirmed_on_holdout' if confirmed else 'not_confirmed_on_holdout',
        'n': int(n),
        'cut': int(cut),
        'holdout_frac': round(1 - cut / n, 3),
        'n_candidates': len(candidates),
        'candidates': candidates,
        'selected_method': selected,
        'best_naive': best_naive,
        'selection_beat_naive': selection_beat_naive,
        'selection_skill': round(float(selection_skill), 4),
        'n_holdout_predictions': int(keep_m.sum()),
        'holdout_skill': round(float(holdout_skill), 4),
        'holdout_dm_p': round(holdout_p, 4),
        'holdout_model_rmse': round(float(holdout_model_rmse), 4),
        'holdout_naive_rmse': round(float(holdout_naive_rmse), 4),
        'shrinkage': round(float(selection_skill - holdout_skill), 4),
        'interpretation': (
            f'{selected} was chosen on the first {cut} rows and evaluated on the '
            f'{int(keep_m.sum())} predictions after them, against {best_naive} fixed at '
            f'selection time. Skill {selection_skill:+.1%} in selection, '
            f'{holdout_skill:+.1%} on the holdout (DM p={holdout_p:.4f}). '
            + ('The edge survives selection.' if confirmed else
               'The edge does not survive selection and should not be reported as a positive.')
        ),
    }


def run() -> dict:
    """Selection-honest re-test of the targets whose verdicts depend on a choice."""
    from ph_economic_ai.benchmark.nowcast import (
        BASELINE_POOL, PANEL_METHODS, build_nowcast_frame)
    from ph_economic_ai.benchmark.targets import TARGETS, load_inflation_mom

    out = {}
    frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    out['headline_mom'] = run_selection_holdout(frame, PANEL_METHODS, BASELINE_POOL)

    # The fuel audit positive is the one claim that currently depends on picking a
    # winner from a panel, so it is the one that most needs this.
    fuel = TARGETS['fuel'].build_frame()
    out['fuel_audit'] = run_selection_holdout(
        fuel, PANEL_METHODS, ('random_walk', 'drift', 'seasonal_naive', 'mean'))
    return out


def main() -> int:
    for name, r in run().items():
        print(f'{name}:')
        if r.get('verdict') == 'insufficient_data':
            print(f'  insufficient data (n={r.get("n")})')
            continue
        print(f'  selected {r["selected_method"]} from {r["n_candidates"]} candidates '
              f'vs {r["best_naive"]}')
        print(f'  selection skill {r["selection_skill"]:+.4f}  ->  '
              f'holdout skill {r["holdout_skill"]:+.4f} '
              f'(shrinkage {r["shrinkage"]:+.4f}, DM p={r["holdout_dm_p"]:.4f}, '
              f'n_holdout={r["n_holdout_predictions"]})')
        print(f'  {r["verdict"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
