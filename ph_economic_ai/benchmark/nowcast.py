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
