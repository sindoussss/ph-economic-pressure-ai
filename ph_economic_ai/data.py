import numpy as np
import pandas as pd


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
        + (demand - 55.0) * 0.04
        + rng.normal(0.0, 0.4, n)
    )
    gas = np.clip(gas, 62.0, 82.0)

    return pd.DataFrame({
        'date': dates,
        'oil_price': oil.round(2),
        'usd_php': usd.round(2),
        'demand_index': demand.round(1),
        'gas_price': gas.round(2),
    })
