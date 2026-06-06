"""One-command benchmark: load data -> backtest -> conformal -> report + figures.

    python -m ph_economic_ai.benchmark.run
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ph_economic_ai.benchmark import baselines, conformal, figures, report
from ph_economic_ai.benchmark.ground_truth import load_world_bank_ron95
from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.metrics import mae, rmse, mape, mase, skill_score
from ph_economic_ai.benchmark.proxy_validation import proxy_vs_gold

FEATURES_CSV = Path(__file__).parent / 'data' / 'features_monthly.csv'
MIN_TRAIN = 24
CONFORMAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def _hgb_predict_fn(X_train, y_train, x_next):
    model = HistGradientBoostingRegressor(
        random_state=42, min_samples_leaf=5, max_leaf_nodes=15)
    model.fit(X_train, y_train)
    return float(model.predict(x_next.reshape(1, -1))[0])


def main():
    gold = load_world_bank_ron95()
    feats = pd.read_csv(FEATURES_CSV, dtype={'date': str}).set_index('date')
    df = feats.join(gold.rename('ron95'), how='inner').dropna().sort_index()
    dates = df.index.tolist()
    y = df['ron95'].to_numpy()
    X = df.drop(columns=['ron95']).to_numpy()

    model_bt = walk_forward(y, X, _hgb_predict_fn, MIN_TRAIN)
    rw_bt = walk_forward(y, None, lambda Xt, yt, xn: baselines.random_walk_next(yt), MIN_TRAIN)
    sn_bt = walk_forward(y, None, lambda Xt, yt, xn: baselines.seasonal_naive_next(yt, 12), MIN_TRAIN)

    yt, yp = model_bt['y_true'], model_bt['y_pred']
    rmse_model = rmse(yt, yp)
    rmse_rw = rmse(rw_bt['y_true'], rw_bt['y_pred'])
    rmse_sn = rmse(sn_bt['y_true'], sn_bt['y_pred'])

    res = model_bt['residuals']
    half = len(res) // 2
    cal_res, val_true, val_pred = res[:half], yt[half:], yp[half:]
    calib = conformal.build_calibration_table(cal_res, val_true, val_pred, CONFORMAL_LEVELS)
    qhat90 = conformal.conformal_quantile(cal_res, 0.9)

    proxy = (df['gas_price'] if 'gas_price' in df.columns else df.iloc[:, 0])
    proxy_stats = proxy_vs_gold(proxy.rename('p'), df['ron95'].rename('g'))

    data_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:16]

    rep = report.build_report(
        date_range=(dates[0], dates[-1]), n_months=len(df),
        model_metrics={'mae': round(mae(yt, yp), 4), 'rmse': round(rmse_model, 4),
                       'mape': round(mape(yt, yp), 4), 'mase': round(mase(yt, yp, y[:MIN_TRAIN]), 4)},
        baseline_metrics={'random_walk': {'rmse': round(rmse_rw, 4)},
                          'seasonal_naive': {'rmse': round(rmse_sn, 4)}},
        skill={'vs_random_walk': round(skill_score(rmse_model, rmse_rw), 4),
               'vs_seasonal_naive': round(skill_score(rmse_model, rmse_sn), 4)},
        calibration=calib, proxy=proxy_stats, data_hash=data_hash,
    )
    report.write_report(rep)

    bt_dates = [dates[i] for i in model_bt['index']]
    figures.plot_pred_vs_actual(bt_dates, yt, yp, yp - qhat90, yp + qhat90)
    figures.plot_baseline_bars(rmse_model, rmse_rw, rmse_sn)
    figures.plot_proxy_scatter(proxy.values, df['ron95'].values)

    pd.DataFrame({'date': bt_dates, 'y_true': yt, 'y_pred': yp,
                  'low90': yp - qhat90, 'high90': yp + qhat90}
                 ).to_csv(report.ARTIFACTS / 'backtest_predictions.csv', index=False)
    print(f"Skill vs random walk: {rep['headline_skill_vs_random_walk']:+.3f} "
          f"over {rep['n_months']} months")


if __name__ == '__main__':
    main()
