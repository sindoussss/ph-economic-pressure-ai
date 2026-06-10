"""MoM Food-CPI nowcast — global food-commodity pass-through, tested honestly.

Builds a food nowcast frame from free agri-futures + oil + FX predictors and the
PSA Food-CPI target, then reuses the existing MoM nowcast runners (unchanged) with
the same preliminary-data robustness re-test as the Transport nowcast.
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation, run_mom_nowcast
from ph_economic_ai.benchmark.psa_cpi import load_food_mom

FOOD_FEATURES_CSV = Path(__file__).parent / 'data' / 'food_features_monthly.csv'


def load_food_features(csv_path: Path = FOOD_FEATURES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={'date': str}).set_index('date').sort_index()


def _build_food_frame(features: pd.DataFrame) -> pd.DataFrame:
    tgt = load_food_mom()
    base = pd.DataFrame({
        'rice': features['rice'], 'wheat': features['wheat'], 'corn': features['corn'],
        'soybean': features['soybean'], 'oil': features['oil_price'],
        'fx': features['usd_php'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base['prev_mom'] = base['target'].shift(1)
    cols = ['rice', 'wheat', 'corn', 'soybean', 'oil', 'fx', 'prev_mom', 'target']
    return base[cols].dropna()


def run_food_nowcast(min_train: int = 24, features=None, prelim_months: int = 6) -> dict:
    """Food-MoM nowcast + driver-only ablation + trailing-preliminary robustness
    re-test. `driver_edge_robust` is the canonical verdict."""
    feats = load_food_features() if features is None else features
    frame = _build_food_frame(feats)
    drop = ('panel', 'calibration')

    def _slim(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in drop}

    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    robust_frame = (frame.iloc[:-prelim_months]
                    if prelim_months and len(frame) > prelim_months + min_train
                    else frame)
    r_abl = run_driver_only_ablation(min_train, frame=robust_frame)

    return {
        'n': int(mom.get('n', len(frame))),
        'mom': _slim(mom),
        'driver_ablation': _slim(abl),
        'driver_edge': bool(abl.get('driver_edge', False)),
        'robust': {
            'prelim_months_dropped': int(prelim_months),
            'n': int(r_abl.get('n', len(robust_frame))),
            'driver_ablation': _slim(r_abl),
            'driver_edge': bool(r_abl.get('driver_edge', False)),
        },
        'driver_edge_robust': bool(r_abl.get('driver_edge', False)),
    }
