"""Phase-2 ablation: run each feature variant through the same causal backtest and
score it in RON95 space, so levers are compared apples-to-apples and the winner is
chosen by an explicit, auditable gate.
"""
import numpy as np

from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.conformal import conformal_quantile
from ph_economic_ai.benchmark.features import make_variant
from ph_economic_ai.benchmark.metrics import mae, rmse, skill_score


def run_variant(name, frame, predict_fn, min_train: int) -> dict:
    """Backtest one variant; reconstruct to RON95 space; score vs random walk."""
    v = make_variant(name, frame)
    bt = walk_forward(v.y_model, v.X, predict_fn, min_train)
    idx = bt['index']
    final_pred = bt['y_pred'] + v.structural[idx]      # reconstruct (0 for plain)
    final_true = v.y_actual[idx]

    rw = walk_forward(v.y_actual, None,
                      lambda Xt, yt, xn: float(yt[-1]), min_train)
    rmse_model = rmse(final_true, final_pred)
    rmse_rw = rmse(rw['y_true'], rw['y_pred'])
    qhat90 = conformal_quantile(final_true - final_pred, 0.9)
    return {
        'name': name,
        'rmse': round(rmse_model, 4),
        'mae': round(mae(final_true, final_pred), 4),
        'skill_vs_rw': round(skill_score(rmse_model, rmse_rw), 4),
        'band90': round(2 * qhat90, 4),
        'n': int(len(final_true)),
    }


def run_ablation(frame, names, predict_fn, min_train: int) -> list:
    return [run_variant(n, frame, predict_fn, min_train) for n in names]


def select_winner(rows: list) -> dict:
    """Gate: highest skill_vs_rw; tie-break (within 1e-9) by narrower band90."""
    return sorted(rows, key=lambda r: (-r['skill_vs_rw'], r['band90']))[0]
