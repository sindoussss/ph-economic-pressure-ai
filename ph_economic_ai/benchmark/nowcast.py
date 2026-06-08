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


def build_nowcast_frame() -> pd.DataFrame:
    """Columns: oil, fx, fuel (contemporaneous, month t), prev_inflation (t-1),
    target (= inflation_t). Inner-joined on the monthly index, dropna'd."""
    infl = load_inflation()
    feats = _features()
    base = pd.DataFrame({
        'oil': feats['oil_price'],
        'fx': feats['usd_php'],
        'fuel': feats['gas_price'],
    })
    base = base.join(infl.rename('target'), how='inner').sort_index()
    base['prev_inflation'] = base['target'].shift(1)
    return base[['oil', 'fx', 'fuel', 'prev_inflation', 'target']].dropna()


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
