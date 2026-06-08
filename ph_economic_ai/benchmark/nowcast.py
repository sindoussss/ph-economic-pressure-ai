"""CPI inflation nowcasting: estimate month-t inflation BEFORE PSA's release using
intra-month-observable drivers.

Integrity rule: only features whose month-t value is published before the CPI_t
release may enter the frame — month-t oil, FX, and fuel (all complete by end of t;
CPI_t releases ~7 days into t+1), plus the already-published prev_inflation. No
same-month CPI-derived feature is used. The walk-forward trains on past complete
(drivers, inflation) pairs only, then applies the mapping to month-t drivers to
estimate the not-yet-released inflation_t — a causal nowcast, not a forecast of
the unknowable future.
"""
import pandas as pd

from ph_economic_ai.benchmark.targets import _features, load_inflation


def build_nowcast_frame(target_loader=None, prev_col: str = 'prev_inflation') -> pd.DataFrame:
    """Contemporaneous nowcast frame: oil/fx/fuel (month t) + <prev_col> (t-1) +
    target. target_loader defaults to load_inflation (resolved at call time so tests
    can monkeypatch the module-level loader). For MoM, pass
    target_loader=load_inflation_mom and prev_col='prev_mom'."""
    loader = target_loader if target_loader is not None else load_inflation
    tgt = loader()
    feats = _features()
    base = pd.DataFrame({
        'oil': feats['oil_price'],
        'fx': feats['usd_php'],
        'fuel': feats['gas_price'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base[prev_col] = base['target'].shift(1)
    return base[['oil', 'fx', 'fuel', prev_col, 'target']].dropna()


from ph_economic_ai.benchmark.backtest import walk_forward
from ph_economic_ai.benchmark.conformal import build_calibration_table
from ph_economic_ai.benchmark.efficiency import run_panel
from ph_economic_ai.benchmark.forecasters import make_forecaster

PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']
FEATURE_COLS = ['oil', 'fx', 'fuel', 'prev_inflation']
CONFORMAL_LEVELS = (0.5, 0.8, 0.9, 0.95)


def run_nowcast(min_train: int = 24, frame=None) -> dict:
    """Nowcast inflation via the panel; naive baseline = last published inflation.
    Verdict 'beats_naive' if any method significantly beats naive (dm_p<0.05,
    skill>0), else 'no_better_than_naive'."""
    if frame is None:
        frame = build_nowcast_frame()
    if len(frame) < min_train + 5:
        return {'verdict': 'insufficient_data', 'n': int(len(frame))}

    panel = run_panel(frame, PANEL_METHODS, min_train, FEATURE_COLS, target_col='target')
    beats = [r for r in panel
             if r['dm_p'] is not None and r['dm_p'] < 0.05 and r['skill_vs_rw'] > 0]
    if beats:
        best = max(beats, key=lambda r: r['skill_vs_rw'])
        verdict = 'beats_naive'
    else:
        best = next(r for r in panel if r['method'] == 'random_walk')
        verdict = 'no_better_than_naive'

    y = frame['target'].to_numpy(dtype=float)
    X = frame[FEATURE_COLS].to_numpy(dtype=float)
    bt = walk_forward(y, X, make_forecaster(best['method']), min_train)
    res = bt['y_true'] - bt['y_pred']
    half = max(1, len(res) // 2)
    calib = build_calibration_table(res[:half], bt['y_true'][half:], bt['y_pred'][half:],
                                    CONFORMAL_LEVELS) if len(res) > 3 else []
    return {
        'verdict': verdict,
        'best_method': best['method'],
        'best_skill': best['skill_vs_rw'],
        'best_dm_p': best.get('dm_p'),
        'n': int(panel[0]['n']),
        'calibration': calib,
        'panel': panel,
    }


from ph_economic_ai.benchmark.metrics import rmse as _rmse
from ph_economic_ai.benchmark.significance import diebold_mariano
from ph_economic_ai.benchmark.targets import load_inflation_mom

BASELINE_POOL = ('random_walk', 'seasonal_naive', 'drift')


def mom_verdict(rmse_by_method: dict, loss_by_method: dict,
                baseline_pool=BASELINE_POOL) -> dict:
    """Verdict for the MoM nowcast: a model 'beats_best_naive' only if it has lower
    RMSE than the BEST simple baseline AND a significant Diebold-Mariano edge over
    it (p<0.05, lower loss). Otherwise 'no_better_than_naive'. Pure function."""
    pool = [m for m in baseline_pool if m in rmse_by_method] or ['random_walk']
    best_naive = min(pool, key=lambda m: rmse_by_method[m])
    base_rmse = rmse_by_method[best_naive]
    base_loss = loss_by_method[best_naive]

    winners = []
    for m in rmse_by_method:
        if m in baseline_pool or rmse_by_method[m] >= base_rmse:
            continue
        dm = diebold_mariano(loss_by_method[m], base_loss)
        if dm['p_value'] < 0.05 and dm['dm_stat'] < 0:
            winners.append(m)

    if winners:
        best_method = min(winners, key=lambda m: rmse_by_method[m])
        dm_p = diebold_mariano(loss_by_method[best_method], base_loss)['p_value']
        return {'verdict': 'beats_best_naive', 'best_method': best_method,
                'best_naive': best_naive,
                'best_skill_vs_naive': round(1 - rmse_by_method[best_method] / base_rmse, 4),
                'dm_p': round(dm_p, 4)}
    return {'verdict': 'no_better_than_naive', 'best_method': best_naive,
            'best_naive': best_naive, 'best_skill_vs_naive': 0.0, 'dm_p': None}


def run_mom_nowcast(min_train: int = 24, baseline_pool=BASELINE_POOL, frame=None,
                    methods=None) -> dict:
    """Nowcast MoM inflation; verdict via DM against the best simple baseline."""
    if frame is None:
        frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    if len(frame) < min_train + 5:
        return {'verdict': 'insufficient_data', 'n': int(len(frame))}
    methods = list(PANEL_METHODS) if methods is None else list(methods)

    feature_cols = [c for c in frame.columns if c != 'target']
    y = frame['target'].to_numpy(dtype=float)
    X = frame[feature_cols].to_numpy(dtype=float)

    rmse_by, loss_by, n_pred = {}, {}, 0
    for m in methods:
        bt = walk_forward(y, X, make_forecaster(m), min_train)
        loss_by[m] = (bt['y_true'] - bt['y_pred']) ** 2
        rmse_by[m] = _rmse(bt['y_true'], bt['y_pred'])
        n_pred = len(bt['y_true'])

    v = mom_verdict(rmse_by, loss_by, baseline_pool)

    bt = walk_forward(y, X, make_forecaster(v['best_method']), min_train)
    res = bt['y_true'] - bt['y_pred']
    half = max(1, len(res) // 2)
    calib = build_calibration_table(res[:half], bt['y_true'][half:], bt['y_pred'][half:],
                                    CONFORMAL_LEVELS) if len(res) > 3 else []
    return {**v, 'n': int(n_pred), 'calibration': calib,
            'rmse_by_method': {k: round(val, 4) for k, val in rmse_by.items()}}


def run_driver_only_ablation(min_train: int = 24, frame=None) -> dict:
    """Isolate the pure within-month driver edge: drop the own-lag (prev_mom) and
    let only driver regressors {ridge, hgb} compete against the simple baselines.
    Adds boolean 'driver_edge' (True iff a driver regressor beats the best simple
    baseline, DM-significant)."""
    if frame is None:
        frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom')
    driver_frame = frame.drop(columns=['prev_mom'], errors='ignore')
    res = run_mom_nowcast(min_train, frame=driver_frame,
                          methods=['random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb'])
    res['driver_edge'] = (res.get('verdict') == 'beats_best_naive')
    return res
