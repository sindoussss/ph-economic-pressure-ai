"""MoM Transport-CPI nowcast — the fuel->inflation pass-through, tested honestly.

Reuses the existing MoM nowcast runners with the PSA Transport-CPI target and the
long feature panel. The driver-only ablation is the key test: does observable
within-month fuel nowcast transport inflation beyond persistence?
"""
from ph_economic_ai.benchmark.nowcast import (
    build_nowcast_frame, run_driver_only_ablation, run_mom_nowcast,
)
from ph_economic_ai.benchmark.psa_cpi import load_transport_mom
from ph_economic_ai.benchmark.longsample import load_long_features


def run_transport_nowcast(min_train: int = 24, features=None) -> dict:
    """Run the Transport-MoM nowcast + driver-only ablation. Returns
    {n, mom, driver_ablation, driver_edge} with heavy internals dropped."""
    feats = load_long_features() if features is None else features
    frame = build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom',
                                features=feats)
    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)
    drop = ('panel', 'calibration')
    return {
        'n': int(mom.get('n', len(frame))),
        'mom': {k: v for k, v in mom.items() if k not in drop},
        'driver_ablation': {k: v for k, v in abl.items() if k not in drop},
        'driver_edge': bool(abl.get('driver_edge', False)),
    }
