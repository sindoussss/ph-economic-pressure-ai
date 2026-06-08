"""Run a panel of forecasters through the same causal backtest and attach a
Diebold-Mariano p-value vs random walk. The efficiency result: no method's
accuracy is statistically better than naive persistence (DM p > 0.05).
"""
from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.forecasters import make_forecaster
from ph_economic_ai.benchmark.metrics import mae, rmse, skill_score
from ph_economic_ai.benchmark.significance import diebold_mariano


def run_panel(frame, methods, min_train: int, feature_cols) -> list:
    """frame: build_feature_frame output. Returns one row per method:
    {method, rmse, mae, skill_vs_rw, dm_stat, dm_p, n}. Reference = random walk."""
    y = frame['ron95'].to_numpy(dtype=float)
    X = frame[list(feature_cols)].to_numpy(dtype=float)

    rw_bt = walk_forward(y, None, make_forecaster('random_walk'), min_train)
    rw_loss = (rw_bt['y_true'] - rw_bt['y_pred']) ** 2
    rmse_rw = rmse(rw_bt['y_true'], rw_bt['y_pred'])

    rows = []
    for m in methods:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        loss = (bt['y_true'] - bt['y_pred']) ** 2
        r = rmse(bt['y_true'], bt['y_pred'])
        if m == 'random_walk':
            dm_stat, dm_p = 0.0, None
        else:
            dm = diebold_mariano(loss, rw_loss, h=1)
            dm_stat, dm_p = round(dm['dm_stat'], 4), round(dm['p_value'], 4)
        rows.append({
            'method': m,
            'rmse': round(r, 4),
            'mae': round(mae(bt['y_true'], bt['y_pred']), 4),
            'skill_vs_rw': round(skill_score(r, rmse_rw), 4),
            'dm_stat': dm_stat,
            'dm_p': dm_p,
            'n': int(len(bt['y_true'])),
        })
    return rows
