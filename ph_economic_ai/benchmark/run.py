"""One-command benchmark: load data -> backtest -> conformal -> report + figures.

    python -m ph_economic_ai.benchmark.run
"""
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ph_economic_ai.benchmark import baselines, conformal, figures, report
from ph_economic_ai.benchmark import ablation as ablation_mod
from ph_economic_ai.benchmark import efficiency as efficiency_mod
from ph_economic_ai.benchmark import passthrough as passthrough_mod
from ph_economic_ai.benchmark import audit as audit_mod
from ph_economic_ai.benchmark.features import build_feature_frame, make_variant, VARIANTS
from ph_economic_ai.benchmark.ground_truth import load_world_bank_ron95
from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.forecasters import make_forecaster
from ph_economic_ai.benchmark.metrics import mae, rmse, mape, mase, skill_score
from ph_economic_ai.benchmark.proxy_validation import proxy_vs_gold
from ph_economic_ai.benchmark import nowcast as nowcast_mod

from ph_economic_ai.benchmark.paths import MIN_TRAIN, artifact, data

FEATURES_CSV = data('features_monthly.csv')
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
    # Use the canonical list rather than a local copy: a duplicated panel silently
    # drifts from the audit's (it did — this one omitted the historical mean).
    efficiency_rows = efficiency_mod.run_panel(
        frame, audit_mod.PANEL_METHODS, MIN_TRAIN, VARIANTS['passthrough_lags']['cols'])
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

    # -- Cross-target predictability audit --
    audit_rows = audit_mod.run_audit(['fuel', 'fx', 'inflation'], MIN_TRAIN)
    print('Predictability audit:')
    for a in audit_rows:
        if a['verdict'] == 'insufficient_data':
            print(f"  {a['target']:<10} insufficient_data (n={a.get('n', 0)})")
        else:
            print(f"  {a['target']:<10} {a['verdict']:<12} best={a['best_method']} "
                  f"skill={a['best_skill']:+.3f}")

    # -- CPI nowcast (estimate inflation before official release) --
    nowcast_res = nowcast_mod.run_nowcast(MIN_TRAIN)
    if nowcast_res['verdict'] == 'insufficient_data':
        print(f"CPI nowcast: insufficient_data (n={nowcast_res.get('n', 0)})")
    else:
        print(f"CPI nowcast: {nowcast_res['verdict']} | best={nowcast_res['best_method']} "
              f"skill_vs_naive={nowcast_res['best_skill']:+.3f} DM p={nowcast_res['best_dm_p']}")

    # -- Month-over-month CPI nowcast (vs best simple baseline) --
    mom_res = nowcast_mod.run_mom_nowcast(MIN_TRAIN)
    if mom_res['verdict'] == 'insufficient_data':
        print(f"MoM CPI nowcast: insufficient_data (n={mom_res.get('n', 0)})")
    else:
        print(f"MoM CPI nowcast: {mom_res['verdict']} | best={mom_res['best_method']} "
              f"vs {mom_res['best_naive']} | skill={mom_res['best_skill_vs_naive']:+.3f} "
              f"DM p={mom_res['dm_p']}")

    # -- Driver-only ablation of the MoM nowcast (isolate the within-month edge) --
    mom_abl = nowcast_mod.run_driver_only_ablation(MIN_TRAIN)
    if mom_abl['verdict'] == 'insufficient_data':
        print(f"MoM driver-only ablation: insufficient_data (n={mom_abl.get('n', 0)})")
    else:
        print(f"MoM driver-only ablation: driver_edge={mom_abl['driver_edge']} | "
              f"best={mom_abl['best_method']} vs {mom_abl['best_naive']} | "
              f"skill={mom_abl['best_skill_vs_naive']:+.3f} DM p={mom_abl['dm_p']}")

    # -- MoM nowcast longer-sample confirmation (isolated long feature history) --
    try:
        from ph_economic_ai.benchmark import longsample as longsample_mod
        mom_long = longsample_mod.run_mom_longsample(MIN_TRAIN)
        _lm, _la = mom_long['mom'], mom_long['driver_ablation']
        print(f"MoM long-sample (n={mom_long['n_long']}): mom={_lm['verdict']} "
              f"best={_lm.get('best_method')} skill={_lm.get('best_skill_vs_naive')} "
              f"DM p={_lm.get('dm_p')} | driver_edge={_la.get('driver_edge')}")
    except FileNotFoundError:
        mom_long = {'verdict': 'not_run', 'reason': 'features_monthly_long.csv missing'}
        print('MoM long-sample: not_run (features_monthly_long.csv missing)')

    # -- MoM Transport-CPI nowcast (fuel -> inflation pass-through) --
    try:
        from ph_economic_ai.benchmark import transport_nowcast as transport_mod
        transport_res = transport_mod.run_transport_nowcast(MIN_TRAIN)
        _tm = transport_res['mom']
        print(f"Transport nowcast (n={transport_res['n']}): mom={_tm['verdict']} "
              f"best={_tm.get('best_method')} skill={_tm.get('best_skill_vs_naive')} "
              f"DM p={_tm.get('dm_p')} | driver_edge_robust={transport_res['driver_edge_robust']}")
    except FileNotFoundError:
        transport_res = {'verdict': 'not_run', 'reason': 'transport gold missing'}
        print('Transport nowcast: not_run (psa_transport_cpi_monthly.csv missing)')

    # -- MoM Food-CPI nowcast (global food-commodity pass-through) --
    try:
        from ph_economic_ai.benchmark import food_nowcast as food_mod
        food_res = food_mod.run_food_nowcast(MIN_TRAIN)
        _fm = food_res['mom']
        print(f"Food nowcast (n={food_res['n']}): mom={_fm['verdict']} "
              f"best={_fm.get('best_method')} skill={_fm.get('best_skill_vs_naive')} "
              f"DM p={_fm.get('dm_p')} | driver_edge_robust={food_res['driver_edge_robust']}")
    except FileNotFoundError:
        food_res = {'verdict': 'not_run', 'reason': 'food gold/features missing'}
        print('Food nowcast: not_run (gold or features CSV missing)')

    # -- MoM Electricity-CPI nowcast (energy pass-through) --
    try:
        from ph_economic_ai.benchmark import electricity_nowcast as elec_mod
        elec_res = elec_mod.run_electricity_nowcast(MIN_TRAIN)
        _em = elec_res['mom']
        print(f"Electricity nowcast (n={elec_res['n']}): mom={_em['verdict']} "
              f"best={_em.get('best_method')} skill={_em.get('best_skill_vs_naive')} "
              f"DM p={_em.get('dm_p')} | driver_edge_robust={elec_res['driver_edge_robust']}")
    except FileNotFoundError:
        elec_res = {'verdict': 'not_run', 'reason': 'electricity gold/features missing'}
        print('Electricity nowcast: not_run (gold or features CSV missing)')

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
        audit=[{k: v for k, v in a.items() if k != 'panel'} for a in audit_rows],
        nowcast={k: v for k, v in nowcast_res.items() if k != 'panel'},
        nowcast_mom={k: v for k, v in mom_res.items() if k != 'calibration'},
        mom_driver_ablation={k: v for k, v in mom_abl.items() if k != 'calibration'},
        mom_longsample=mom_long,
        transport_nowcast=transport_res,
        food_nowcast=food_res,
        electricity_nowcast=elec_res,
    )
    report.write_report(rep)

    # Publication predictability-map figure (manuscript Fig. 3) — regenerated here
    # from the report just written, so a single `benchmark.run` reproduces every
    # figure the manuscript cites (not only the diagnostic ones below).
    try:
        from ph_economic_ai.benchmark import render_pub_figures
        render_pub_figures.main()
    except Exception as _fig_err:
        print(f'predictability_map figure skipped: {type(_fig_err).__name__}: {_fig_err}')

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
                 ).to_csv(report.ARTIFACTS / 'backtest_predictions.csv', index=False,
                          lineterminator='\n')

    import json as _json
    (report.ARTIFACTS / 'ablation_table.json').write_text(
        _json.dumps({'selected': selected, 'rows': ablation_rows}, indent=2),
        encoding='utf-8')

    import json as _json2
    (report.ARTIFACTS / 'audit_table.json').write_text(
        _json2.dumps(audit_rows, indent=2), encoding='utf-8')
    figures.plot_audit_verdicts(audit_rows)

    import json as _json3
    (report.ARTIFACTS / 'nowcast_table.json').write_text(
        _json3.dumps(nowcast_res, indent=2), encoding='utf-8')
    if nowcast_res['verdict'] != 'insufficient_data':
        _nf = nowcast_mod.build_nowcast_frame()
        _y = _nf['target'].to_numpy(float)
        _X = _nf[nowcast_mod.FEATURE_COLS].to_numpy(float)
        _bt = walk_forward(_y, _X, make_forecaster(nowcast_res['best_method']), MIN_TRAIN)
        _nbt = walk_forward(_y, None, make_forecaster('random_walk'), MIN_TRAIN)
        _nd = [_nf.index[i] for i in _bt['index']]
        figures.plot_nowcast(_nd, _bt['y_true'], _bt['y_pred'], _nbt['y_pred'])
        import os as _os_yoy
        _os_yoy.replace(figures.FIG_DIR / 'nowcast.png', figures.FIG_DIR / 'nowcast_yoy.png')

    import json as _json4
    (report.ARTIFACTS / 'nowcast_mom_table.json').write_text(
        _json4.dumps(mom_res, indent=2), encoding='utf-8')

    import json as _json5
    (report.ARTIFACTS / 'mom_driver_ablation_table.json').write_text(
        _json5.dumps(mom_abl, indent=2), encoding='utf-8')

    import json as _json6
    (report.ARTIFACTS / 'mom_longsample_table.json').write_text(
        _json6.dumps(mom_long, indent=2), encoding='utf-8')

    import json as _json7
    (report.ARTIFACTS / 'transport_nowcast_table.json').write_text(
        _json7.dumps(transport_res, indent=2), encoding='utf-8')

    import json as _json8
    (report.ARTIFACTS / 'food_nowcast_table.json').write_text(
        _json8.dumps(food_res, indent=2), encoding='utf-8')

    import json as _json9
    (report.ARTIFACTS / 'electricity_nowcast_table.json').write_text(
        _json9.dumps(elec_res, indent=2), encoding='utf-8')

    if mom_res['verdict'] != 'insufficient_data':
        _mf = nowcast_mod.build_nowcast_frame(
            target_loader=nowcast_mod.load_inflation_mom, prev_col='prev_mom')
        _mcols = [c for c in _mf.columns if c != 'target']
        _my = _mf['target'].to_numpy(float); _mX = _mf[_mcols].to_numpy(float)
        _mbt = walk_forward(_my, _mX, make_forecaster(mom_res['best_method']), MIN_TRAIN)
        _mnbt = walk_forward(_my, _mX, make_forecaster(mom_res['best_naive']), MIN_TRAIN)
        _md = [_mf.index[i] for i in _mbt['index']]
        figures.plot_nowcast(_md, _mbt['y_true'], _mbt['y_pred'], _mnbt['y_pred'])
        import os as _os
        _os.replace(figures.FIG_DIR / 'nowcast.png', figures.FIG_DIR / 'nowcast_mom.png')
        # Restore the YoY nowcast figure as nowcast.png (was temporarily saved as nowcast_yoy.png)
        _yoy_tmp = figures.FIG_DIR / 'nowcast_yoy.png'
        if _yoy_tmp.exists():
            _os.replace(_yoy_tmp, figures.FIG_DIR / 'nowcast.png')

    print(f"Selected variant: {selected} | "
          f"skill vs random walk: {rep['headline_skill_vs_random_walk']:+.3f} "
          f"over {rep['n_months']} months")

    # Multiple-comparison correction over the confirmatory DM tests — reads the
    # accuracy_report just written, so it always reflects the current run.
    from ph_economic_ai.benchmark import multiple_testing
    _mt = multiple_testing.run()
    print(f"Multiple testing: {len(_mt['survive_bonferroni'])}/{_mt['n_tests']} "
          f"confirmatory tests survive Bonferroni (FWER): {_mt['survive_bonferroni']}")

    # Power / minimum-detectable-effect for the flagship efficiency null.
    from ph_economic_ai.benchmark import power
    _pw = power.run()['fuel_one_month_forecast']
    print(f"Power: fuel forecast min detectable skill = "
          f"{_pw['min_detectable_skill_pct']}% at {_pw['power']:.0%} power "
          f"(observed {_pw['observed_skill'] * 100:+.1f}%)")

    # Closed form for the spurious skill a mean-predictor shows over the random
    # walk, plus its validation against every real target. This is what makes the
    # baseline correction a general result rather than a one-dataset anecdote.
    from ph_economic_ai.benchmark import baseline_theory
    _bl = baseline_theory.run()
    artifact('baseline_theory.json').write_text(
        json.dumps(_bl, indent=2), encoding='utf-8')
    print(f"Baseline theory: mean-predictor beats RW iff rho<{_bl['crossover_rho']}; "
          f"closed form matches all targets to {_bl['max_abs_error_targets']:.3f}")

    # Selection-honest re-test (RSK-004). The panel verdicts above choose the best
    # of K methods and then report that winner's p-value, which is optimistic
    # because the winner was chosen for looking good. Here the model is fixed on an
    # early segment and scored on a holdout the choice never saw, with the naive
    # baseline also frozen at selection time.
    from ph_economic_ai.benchmark import selection as _selection
    _sel = _selection.run()
    artifact('selection_holdout.json').write_text(
        json.dumps(_sel, indent=2), encoding='utf-8')
    for _name, _r in _sel.items():
        if _r.get('verdict') == 'insufficient_data':
            print(f'Selection holdout ({_name}): insufficient data')
            continue
        print(f"Selection holdout ({_name}): {_r['selected_method']} of "
              f"{_r['n_candidates']} vs {_r['best_naive']} | "
              f"selection {_r['selection_skill']:+.3f} -> holdout "
              f"{_r['holdout_skill']:+.3f} (shrinkage {_r['shrinkage']:+.3f}, "
              f"DM p={_r['holdout_dm_p']:.4f}, n={_r['n_holdout_predictions']}) | "
              f"{_r['verdict']}")

    # Size-and-power study: what false-positive rate does a mean-free pool
    # actually produce on data with nothing in it, and does the fix keep power?
    from ph_economic_ai.benchmark import baseline_size
    _bs = baseline_size.run()
    artifact('baseline_size.json').write_text(json.dumps(_bs, indent=2), encoding='utf-8')
    print(f"Baseline size: worst-case false-positive rate "
          f"{_bs['worst_case_false_positive_rate']:.1%} without the mean "
          f"(nominal {_bs['alpha']:.0%}), {_bs['max_size_with_mean']:.1%} with it; "
          f"worsens with more data = {_bs['worsens_with_more_data']}")

    # How much of the standard macro target space sits in the vulnerable regime?
    # Reads a frozen FRED-MD snapshot; refresh with tools.refresh_fredmd.
    try:
        from ph_economic_ai.benchmark import vulnerability_survey
        _vs = vulnerability_survey.run()
        artifact('vulnerability_survey.json').write_text(
            json.dumps(_vs, indent=2), encoding='utf-8')
        print(f"Vulnerability census (FRED-MD): {_vs['n_vulnerable']}/{_vs['n_series']} "
              f"= {_vs['share_vulnerable']:.1%} of series have rho < 0.5 "
              f"(median rho {_vs['median_rho']:+.3f})")
    except FileNotFoundError:
        print('Vulnerability census: skipped (run tools.refresh_fredmd first)')

    # Figures for the baseline-specification results. Rendered from the artifacts
    # just written, so a figure can never assert a number the report contradicts.
    try:
        from ph_economic_ai.benchmark import baseline_figures
        baseline_figures.main()
    except Exception as _bf_err:
        print(f'baseline figures: skipped ({type(_bf_err).__name__}: {_bf_err})')

    # Sentiment keystone: does social search interest nowcast inflation beyond
    # naive? Ties the app's exploratory social layer to the validated spine, so it
    # belongs in the one-command run rather than being invoked by hand.
    from ph_economic_ai.benchmark import sentiment_nowcast
    _sn = sentiment_nowcast.run()
    sentiment_nowcast._write(_sn)
    if _sn.get('status') == 'computed':
        print(f"Sentiment nowcast: edge_anywhere={_sn['sentiment_edge_anywhere']} — "
              f"{_sn['finding']}")
    else:
        print(f"Sentiment nowcast: {_sn['status']} (run tools.refresh_social --trends)")


if __name__ == '__main__':
    main()
