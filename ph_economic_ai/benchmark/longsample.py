"""Longer-sample confirmation of the MoM CPI nowcast.

Feeds a longer feature history (features_monthly_long.csv) into the existing
MoM nowcast + driver-only ablation to test whether the result holds on ~2-3x the
sample. Isolated from the main pipeline.
"""
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.nowcast import (
    build_nowcast_frame, run_driver_only_ablation, run_mom_nowcast,
)
from ph_economic_ai.benchmark.targets import load_inflation_mom

LONG_FEATURES_CSV = Path(__file__).parent / 'data' / 'features_monthly_long.csv'


def load_long_features(csv_path: Path = LONG_FEATURES_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, dtype={'date': str}).set_index('date').sort_index()


def run_mom_longsample(min_train: int = 24, features=None) -> dict:
    """Run the MoM nowcast + driver-only ablation on the long feature history.
    Returns {n_long, mom, driver_ablation} with heavy internals dropped."""
    feats = load_long_features() if features is None else features
    frame = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom',
                                features=feats)
    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    drop = ('panel', 'calibration')
    return {
        'n_long': int(mom.get('n', len(frame))),
        'mom': {k: v for k, v in mom.items() if k not in drop},
        'driver_ablation': {k: v for k, v in abl.items() if k not in drop},
    }
