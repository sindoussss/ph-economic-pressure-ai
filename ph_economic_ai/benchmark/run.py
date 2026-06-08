"""One-command benchmark: load data -> backtest -> conformal -> report + figures.

    python -m ph_economic_ai.benchmark.run
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ph_economic_ai.benchmark import baselines, conformal, figures, report
from ph_economic_ai.benchmark import ablation as ablation_mod
from ph_economic_ai.benchmark import efficiency as efficiency_mod
from ph_economic_ai.benchmark import passthrough as passthrough_mod
from ph_economic_ai.benchmark.features import build_feature_frame, make_variant, VARIANTS
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

    # ── Phase-2: ablation over feature variants, pick the winner by the gate ──
    frame = build_feature_frame(df)
    ablation_rows = ablation_mod.run_ablation(
        frame, list(VARIANTS.keys()), _hgb_predict_fn, MIN_TRAIN)
    winner = ablation_mod.select_winner(ablation_rows)
    selected = winner['name']
    print('Ablation (skill vs random walk):')
    for r in sorted(ablation_rows, key=lambda x: -x['skill_vs_rw']):
        mark = ' <= selected' if r['name'] == selected else ''
        print(f"  {r['name']:<18} skill={r['skill_vs_rw']:+.3f} "
              f"band90=P{r['band90']:.2f} rmse=P{r['rmse']:.2f}{mark}")

    # -- Efficiency panel + pass-through mechanism --
    panel_methods = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']
    efficiency_rows = efficiency_mod.run_panel(
        frame, panel_methods, MIN_TRAIN, VARIANTS['passthrough_lags']['cols'])
    passthrough_stats = passthrough_mod.estimate_passthrough(df)
    print('Efficiency panel (skill vs RW | DM p):')
    for r in sorted(efficiency_rows, key=lambda x: -x['skill_vs_rw']):
        p = 'n/a' if r['dm_p'] is None else f"{r['dm_p']:.3f}"
        print(f"  {r['method']:<14} skill={r['skill_vs_rw']:+.3f}  DM p={p}")
    print(f"Pass-through: beta_total={passthrough_stats['beta_total']} "
          f"R2={passthrough_stats['r2']} driver_acf1={passthrough_stats['driver_acf1']}")

    # Re-run the winning variant to get its reconstructed predictions for the report.
    v = make_variant(selected, frame)
    bt = walk_forward(v.y_model, v.X, _hgb_predict_fn, MIN_TRAIN)
    idx = bt['index']
    dates = [frame.index[i] for i in idx]
    yp = bt['y_pred'] + v.structural[idx]      # reconstruct to RON95 space
    yt = v.y_actual[idx]

    rw_bt = walk_forward(v.y_actual, None,
                         lambda Xt, ytr, xn: float(ytr[-1]), MIN_TRAIN)
    sn_bt = walk_forward(v.y_actual, None,
                         lambda Xt, ytr, xn: baselines.seasonal_naive_next(ytr, 12), MIN_TRAIN)
    rmse_model = rmse(yt, yp)
    rmse_rw = rmse(rw_bt['y_true'], rw_bt['y_pred'])
    rmse_sn = rmse(sn_bt['y_true'], sn_bt['y_pred'])

    res = yt - yp
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
                       'mape': round(mape(yt, yp), 4), 'mase': round(mase(yt, yp, v.y_actual[:MIN_TRAIN]), 4)},
        baseline_metrics={'random_walk': {'rmse': round(rmse_rw, 4)},
                          'seasonal_naive': {'rmse': round(rmse_sn, 4)}},
        skill={'vs_random_walk': round(skill_score(rmse_model, rmse_rw), 4),
               'vs_seasonal_naive': round(skill_score(rmse_model, rmse_sn), 4)},
        calibration=calib, proxy=proxy_stats, data_hash=data_hash,
        ablation=ablation_rows, selected_variant=selected,
        efficiency=efficiency_rows, passthrough=passthrough_stats,
    )
    report.write_report(rep)

    bt_dates = dates
    figures.plot_pred_vs_actual(bt_dates, yt, yp, yp - qhat90, yp + qhat90)
    figures.plot_baseline_bars(rmse_model, rmse_rw, rmse_sn)
    figures.plot_proxy_scatter(proxy.values, df['ron95'].values)
    figures.plot_method_skill_bar(efficiency_rows)
    _dc = df['gas_price'].diff().dropna()
    _dp = df['ron95'].diff().reindex(_dc.index)
    _mask = _dp.notna()
    if passthrough_stats['beta_total'] is not None:
        figures.plot_passthrough(_dc[_mask].to_numpy(), _dp[_mask].to_numpy(),
                                 passthrough_stats['beta_total'])

    pd.DataFrame({'date': bt_dates, 'y_true': yt, 'y_pred': yp,
                  'low90': yp - qhat90, 'high90': yp + qhat90}
                 ).to_csv(report.ARTIFACTS / 'backtest_predictions.csv', index=False)

    import json as _json
    (report.ARTIFACTS / 'ablation_table.json').write_text(
        _json.dumps({'selected': selected, 'rows': ablation_rows}, indent=2),
        encoding='utf-8')

    print(f"Selected variant: {selected} | "
          f"skill vs random walk: {rep['headline_skill_vs_random_walk']:+.3f} "
          f"over {rep['n_months']} months")


if __name__ == '__main__':
    main()
