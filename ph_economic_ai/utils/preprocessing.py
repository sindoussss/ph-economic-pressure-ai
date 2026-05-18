import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame):
    """Add lagged gas price, drop first row (NaN), return X, y, col names, df."""
    df = df.copy()
    df['prev_gas_price'] = df['gas_price'].shift(1)
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['oil_price', 'usd_php', 'demand_index', 'prev_gas_price']
    X = df[feature_cols].values.astype(float)
    y = df['gas_price'].values.astype(float)
    return X, y, feature_cols, df


def compute_index(current_oil: float, current_usd: float, current_demand: float,
                  df: pd.DataFrame) -> tuple:
    """Return (pressure_index 0-100, oil_delta, usd_delta, demand_norm)."""
    oil_mean, oil_std = df['oil_price'].mean(), df['oil_price'].std()
    usd_mean, usd_std = df['usd_php'].mean(), df['usd_php'].std()

    oil_deltas = (df['oil_price'] - oil_mean) / oil_std
    usd_deltas = (df['usd_php'] - usd_mean) / usd_std
    demand_norms = df['demand_index'] / 100.0
    raw_series = oil_deltas * 0.50 + usd_deltas * 0.30 + demand_norms * 0.20
    raw_min, raw_max = float(raw_series.min()), float(raw_series.max())

    oil_delta = float((current_oil - oil_mean) / oil_std)
    usd_delta = float((current_usd - usd_mean) / usd_std)
    demand_norm = float(current_demand / 100.0)
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
