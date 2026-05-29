import numpy as np
import pandas as pd
from ph_economic_ai.fetcher import fetch_dataset, _RAINFALL_NORMS_MM, _TEMP_NORMS_C  # noqa: F401


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120

    dates = pd.date_range('2024-01', periods=n, freq='MS').strftime('%Y-%m').tolist()

    oil = np.empty(n)
    oil[0] = 85.0
    for i in range(1, n):
        oil[i] = oil[i - 1] + rng.normal(0.2, 1.8)
    oil = np.clip(oil, 75.0, 105.0)

    usd = np.empty(n)
    usd[0] = 56.5
    for i in range(1, n):
        drift = (oil[i] - oil[i - 1]) * 0.04
        usd[i] = usd[i - 1] + drift + rng.normal(0.0, 0.25)
    usd = np.clip(usd, 54.0, 62.0)

    t = np.arange(n)
    demand = 72.0 + 9.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0.0, 2.5, n)
    demand = np.clip(demand, 55.0, 90.0)

    gas = (
        62.0
        + (oil - 75.0) * 0.38
        + (usd - 54.0) * 0.75
        + (demand - 55.0) * 0.10
        + rng.normal(0.0, 0.4, n)
    )
    gas = np.clip(gas, 62.0, 82.0)

    # Synthetic weather: seasonal pattern for PH agricultural zones
    months = [int(d[5:7]) for d in dates]
    rainfall = np.array([_RAINFALL_NORMS_MM[m - 1] for m in months]) + rng.normal(0.0, 10.0, n)
    rainfall = np.clip(rainfall, 0.0, 300.0).round(1)

    temp = np.array([_TEMP_NORMS_C[m - 1] for m in months]) + rng.normal(0.0, 0.5, n)
    temp = np.clip(temp, 22.0, 36.0).round(2)

    # Synthetic food price index: base + gas pass-through + rainfall impact
    gas_delta = np.diff(gas, prepend=gas[0])
    rain_deficit = np.clip(
        (np.array([_RAINFALL_NORMS_MM[m - 1] for m in months]) - rainfall)
        / np.array([_RAINFALL_NORMS_MM[m - 1] for m in months]),
        0.0, 1.0
    )
    food_idx = np.empty(n)
    food_idx[0] = 100.0
    for i in range(1, n):
        food_idx[i] = food_idx[i - 1] + gas_delta[i] * 0.22 + rain_deficit[i] * 0.15
    food_idx = np.clip(food_idx, 80.0, 180.0).round(2)

    # Synthetic electricity rate: base + gas pass-through
    elec = np.empty(n)
    elec[0] = 11.20
    for i in range(1, n):
        elec[i] = elec[i - 1] + gas_delta[i] * 0.18
    elec = np.clip(elec, 8.0, 18.0).round(2)

    return pd.DataFrame({
        'date':             dates,
        'oil_price':        oil.round(2),
        'usd_php':          usd.round(2),
        'demand_index':     demand.round(1),
        'gas_price':        gas.round(2),
        'rainfall_mm':      rainfall,
        'temp_c':           temp,
        'food_price_idx':   food_idx,
        'electricity_rate': elec,
    })
