"""MoM nowcast for PSA's food CPI sub-categories, tested honestly.

One parameterized module for all six categories rather than six near-copies
of food_nowcast.py — the only thing that varies per category is which PSA
series is the target; the predictor frame (global agri-futures + oil + FX)
and the whole statistical pipeline (mean-corrected baseline pool, DM test,
preliminary-data robustness re-test) are exactly food_nowcast.py's, reused
via nowcast.run_mom_nowcast / run_driver_only_ablation as-is.

Deliberately reuses the existing food predictor set rather than researching
bespoke per-category predictors — real future work, out of scope here (see
docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md §2).

Not yet wired into any production entry point (`run.py`, `accuracy_report.json`,
the app's category cards) -- that's deliberately deferred, gated by the
selection-holdout pre-registration at
docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md. Until
that runs, this module is exercised only by its tests; that is intentional,
not dead code awaiting deletion.
"""
import pandas as pd

from ph_economic_ai.benchmark import psa_cpi
from ph_economic_ai.benchmark.calendar_index import (
    calendar_lag, require_complete_calendar)
from ph_economic_ai.benchmark.food_nowcast import load_food_features
from ph_economic_ai.benchmark.nowcast import run_driver_only_ablation, run_mom_nowcast

CATEGORIES = ['rice', 'meat', 'fish', 'dairy_eggs', 'vegetables', 'sugar']

# Maps a category name to the psa_cpi loader function name that returns its
# MoM series. Read via getattr(psa_cpi, ...) rather than a direct import list
# so tests can monkeypatch psa_cpi.load_rice_mom etc. in place, the same
# pattern food_nowcast's own tests already use for load_food_mom.
_LOADER_BY_CATEGORY = {
    'rice': 'load_rice_mom', 'meat': 'load_meat_mom', 'fish': 'load_fish_mom',
    'dairy_eggs': 'load_dairy_eggs_mom', 'vegetables': 'load_vegetables_mom',
    'sugar': 'load_sugar_mom',
}


def _build_subcategory_frame(category: str, features: pd.DataFrame) -> pd.DataFrame:
    """Same shape as food_nowcast._build_food_frame: the category's own CPI
    MoM as target, the existing food predictor frame, calendar_lag for
    prev_mom (not a row lag — the feature panel skips months, per every
    other nowcast frame builder's own comment on this exact point)."""
    if category not in CATEGORIES:
        raise ValueError(f'unknown category {category!r}, expected one of {CATEGORIES}')
    loader = getattr(psa_cpi, _LOADER_BY_CATEGORY[category])
    tgt = loader()
    base = pd.DataFrame({
        'rice': features['rice'], 'wheat': features['wheat'], 'corn': features['corn'],
        'soybean': features['soybean'], 'oil': features['oil_price'],
        'fx': features['usd_php'],
    })
    base = base.join(tgt.rename('target'), how='inner').sort_index()
    base['prev_mom'] = calendar_lag(base['target'], 1)
    cols = ['rice', 'wheat', 'corn', 'soybean', 'oil', 'fx', 'prev_mom', 'target']
    return require_complete_calendar(base[cols].dropna(), f'{category} nowcast frame')


def run_subcategory_nowcast(category: str, min_train: int = 24, features=None,
                            prelim_months: int = 6) -> dict:
    """One category's MoM nowcast + driver-only ablation + trailing-preliminary
    robustness re-test. Same return shape as food_nowcast.run_food_nowcast."""
    if category not in CATEGORIES:
        raise ValueError(f'unknown category {category!r}, expected one of {CATEGORIES}')
    feats = load_food_features() if features is None else features
    frame = _build_subcategory_frame(category, feats)
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
