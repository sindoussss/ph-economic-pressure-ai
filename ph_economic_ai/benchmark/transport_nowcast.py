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


def run_transport_nowcast(min_train: int = 24, features=None,
                          prelim_months: int = 6) -> dict:
    """Run the Transport-MoM nowcast + driver-only ablation, with a robustness
    re-test that drops the trailing preliminary window (PSA revises the most
    recent CPI prints).

    Returns {n, mom, driver_ablation, driver_edge (full sample),
    robust:{...}, driver_edge_robust}. `driver_edge_robust` is the canonical,
    defensible verdict; `driver_edge` (full sample) may be inflated by
    not-yet-revised recent observations and is kept only for transparency.
    """
    feats = load_long_features() if features is None else features
    frame = build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom',
                                features=feats)
    drop = ('panel', 'calibration')

    def _slim(d: dict) -> dict:
        return {k: v for k, v in d.items() if k not in drop}

    mom = run_mom_nowcast(min_train, frame=frame)
    abl = run_driver_only_ablation(min_train, frame=frame)

    # Robustness: drop the trailing preliminary months and re-test the edge.
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
