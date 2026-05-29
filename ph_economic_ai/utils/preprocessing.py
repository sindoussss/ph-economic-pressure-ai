import numpy as np
import pandas as pd


def build_gas_features(df: pd.DataFrame):
    """Feature builder for the gas price model."""
    df = df.copy()
    df['prev_gas_price'] = df['gas_price'].shift(1)
    df = df.dropna(subset=['prev_gas_price']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("DataFrame is empty after removing NaN rows. Input df too short (minimum 2 rows required).")
    base_cols  = ['oil_price', 'usd_php', 'demand_index']
    extra_cols = ['psei', 'cpi', 'bsp_rate', 'remittances']
    available  = [c for c in extra_cols if c in df.columns]
    feature_cols = base_cols + available + ['prev_gas_price']
    X = df[feature_cols].values.astype(float)
    y = df['gas_price'].values.astype(float)
    return X, y, feature_cols, df


def build_features(df: pd.DataFrame):
    """Backward-compatible alias for build_gas_features."""
    return build_gas_features(df)


def build_food_features(df: pd.DataFrame, gas_pred):
    """Feature builder for the food price index model.

    Parameters
    ----------
    gas_pred : float or array-like
        Predicted gas price for the current period. Pass a scalar at inference time
        (broadcasts to all rows) or a 1-D array at training time. If an array, it must
        have at least as many elements as the post-dropna df; the tail is sliced to match
        the post-dropna row count (safe because dropna only removes the first row from lag).
        The returned df contains a 'gas_pred' column — do not cache it (data separation invariant).
    """
    df = df.copy()
    df['food_price_idx_lag1'] = df['food_price_idx'].shift(1)
    df = df.dropna(subset=['food_price_idx', 'food_price_idx_lag1']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("DataFrame is empty after removing NaN rows. Input df too short (minimum 2 rows required).")

    if np.isscalar(gas_pred):
        df['gas_pred'] = float(gas_pred)
    else:
        arr = np.asarray(gas_pred, dtype=float)
        if len(arr) < len(df):
            raise ValueError(
                f"gas_pred array (len {len(arr)}) is shorter than df after dropna (len {len(df)}). "
                "Pass the full gas predictions array before lag/dropna filtering."
            )
        df['gas_pred'] = arr[-len(df):]  # align tail to post-dropna length

    feature_cols = [
        'oil_price', 'usd_php', 'cpi', 'rainfall_mm', 'temp_c',
        'food_price_idx_lag1', 'gas_pred',
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(float)
    y = df['food_price_idx'].values.astype(float)
    return X, y, available, df


def build_electricity_features(df: pd.DataFrame, gas_pred):
    """Feature builder for the electricity rate model.

    Parameters
    ----------
    gas_pred : float or array-like
        Predicted gas price for the current period. Pass a scalar at inference time
        (broadcasts to all rows) or a 1-D array at training time. If an array, it must
        have at least as many elements as the post-dropna df; the tail is sliced to match
        the post-dropna row count (safe because dropna only removes the first row from lag).
        The returned df contains a 'gas_pred' column — do not cache it (data separation invariant).
    """
    df = df.copy()
    df['electricity_rate_lag1'] = df['electricity_rate'].shift(1)
    df = df.dropna(subset=['electricity_rate', 'electricity_rate_lag1']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("DataFrame is empty after removing NaN rows. Input df too short (minimum 2 rows required).")

    if np.isscalar(gas_pred):
        df['gas_pred'] = float(gas_pred)
    else:
        arr = np.asarray(gas_pred, dtype=float)
        if len(arr) < len(df):
            raise ValueError(
                f"gas_pred array (len {len(arr)}) is shorter than df after dropna (len {len(df)}). "
                "Pass the full gas predictions array before lag/dropna filtering."
            )
        df['gas_pred'] = arr[-len(df):]

    feature_cols = [
        'oil_price', 'usd_php', 'bsp_rate', 'electricity_rate_lag1', 'gas_pred',
    ]
    available = [c for c in feature_cols if c in df.columns]
    X = df[available].values.astype(float)
    y = df['electricity_rate'].values.astype(float)
    return X, y, available, df


def build_all_features(df: pd.DataFrame, gas_pred) -> dict:
    """Orchestrator — returns {sector: (X, y, cols, df)} for all three sectors."""
    return {
        'gas':         build_gas_features(df),
        'food':        build_food_features(df, gas_pred),
        'electricity': build_electricity_features(df, gas_pred),
    }


def compute_index(current_oil: float, current_usd: float, current_demand: float,
                  df: pd.DataFrame) -> tuple:
    """Return (pressure_index 0-100, oil_delta, usd_delta, demand_norm)."""
    oil_mean, oil_std = df['oil_price'].mean(), df['oil_price'].std()
    usd_mean, usd_std = df['usd_php'].mean(), df['usd_php'].std()

    oil_deltas   = (df['oil_price'] - oil_mean) / oil_std
    usd_deltas   = (df['usd_php'] - usd_mean) / usd_std
    demand_norms = df['demand_index'] / 100.0
    raw_series   = oil_deltas * 0.50 + usd_deltas * 0.30 + demand_norms * 0.20
    raw_min, raw_max = float(raw_series.min()), float(raw_series.max())

    oil_delta    = float((current_oil - oil_mean) / oil_std)
    usd_delta    = float((current_usd - usd_mean) / usd_std)
    demand_norm  = float(current_demand / 100.0)
    raw = oil_delta * 0.50 + usd_delta * 0.30 + demand_norm * 0.20

    if raw_max > raw_min:
        normalized = (raw - raw_min) / (raw_max - raw_min) * 100.0
    else:
        normalized = 50.0

    return float(np.clip(normalized, 0.0, 100.0)), oil_delta, usd_delta, demand_norm


def pressure_band(index: float) -> str:
    if index <= 30:
        return 'Stable'
    elif index <= 60:
        return 'Rising'
    elif index <= 80:
        return 'High'
    return 'Critical'
