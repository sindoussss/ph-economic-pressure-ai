"""1-month-ahead candidate predictors + variant registry for Phase-2 ablation.

All features are lagged (known at month i-1) so every variant is a true forecast.
A single shared frame guarantees identical date support across variants. The
structural hybrid predicts the residual over the lagged RBOB proxy, reconstructed
as proxy_lag1 + residual_pred (leakage-free: the model learns the time-varying gap
including its mean bias).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Variant:
    name: str
    dates: list
    X: np.ndarray          # design matrix; row i predicts y_actual[i]
    y_actual: np.ndarray   # ron95 (for fair baseline comparison), per row
    y_model: np.ndarray    # what the regressor fits (y_actual - structural)
    structural: np.ndarray # per-row structural component (0.0 for plain variants)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """df indexed 'YYYY-MM' with oil_price, usd_php, gas_price, demand_index, ron95.

    Returns a frame of lagged candidate features + 'ron95' target + 'proxy_lag1',
    dropna'd to common support (longest lag = 3 months)."""
    f = pd.DataFrame(index=df.index)
    f['prev_ron95']  = df['ron95'].shift(1)
    f['oil_lag1']    = df['oil_price'].shift(1)
    f['usd_lag1']    = df['usd_php'].shift(1)
    f['gas_lag1']    = df['gas_price'].shift(1)
    f['demand_lag1'] = df['demand_index'].shift(1)
    f['gas_lag2']    = df['gas_price'].shift(2)
    f['gas_lag3']    = df['gas_price'].shift(3)
    f['gas_ma3']     = df['gas_price'].shift(1).rolling(3).mean()
    f['fx_ma3']      = df['usd_php'].shift(1).rolling(3).mean()
    f['gas_delta1']  = df['gas_price'].shift(1) - df['gas_price'].shift(2)
    f['proxy_lag1']  = df['gas_price'].shift(1)
    f['ron95']       = df['ron95']
    return f.dropna()


VARIANTS: dict = {
    'baseline':          {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'demand_lag1'],
                          'structural': None},
    'drop_demand':       {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1'],
                          'structural': None},
    'passthrough_lags':  {'cols': ['prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'gas_lag2',
                                   'gas_lag3', 'gas_ma3', 'fx_ma3', 'gas_delta1', 'demand_lag1'],
                          'structural': None},
    'finished_gas':      {'cols': ['prev_ron95', 'gas_lag1', 'gas_lag2', 'gas_lag3', 'gas_ma3', 'usd_lag1'],
                          'structural': None},
    'structural_hybrid': {'cols': ['gas_delta1', 'fx_ma3', 'demand_lag1', 'gas_lag2'],
                          'structural': 'proxy_lag1'},
}


def make_variant(name: str, frame: pd.DataFrame) -> Variant:
    spec = VARIANTS[name]
    X = frame[spec['cols']].to_numpy(dtype=float)
    y_actual = frame['ron95'].to_numpy(dtype=float)
    if spec['structural'] is not None:
        structural = frame[spec['structural']].to_numpy(dtype=float)
        y_model = y_actual - structural
    else:
        structural = np.zeros(len(frame))
        y_model = y_actual.copy()
    return Variant(name=name, dates=frame.index.tolist(), X=X,
                   y_actual=y_actual, y_model=y_model, structural=structural)


def build_target_frame(target_series, driver_df, target_name: str, drivers: list):
    """Generic 1-month-ahead frame for any target. Produces prev_<target_name>,
    plus lag-1 and 3-month-MA of each driver, and a 'target' column. All features
    are lagged (known at t-1). Inner-joins target and drivers on the date index,
    dropna'd to common support."""
    base = pd.DataFrame({'__t__': target_series})
    joined = base.join(driver_df[list(drivers)], how='inner').sort_index()
    f = pd.DataFrame(index=joined.index)
    f[f'prev_{target_name}'] = joined['__t__'].shift(1)
    for d in drivers:
        f[f'{d}_lag1'] = joined[d].shift(1)
        f[f'{d}_ma3'] = joined[d].shift(1).rolling(3).mean()
    f['target'] = joined['__t__']
    return f.dropna()
