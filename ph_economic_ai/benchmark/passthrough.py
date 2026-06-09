"""Measure the DOE pass-through: how a change in landed cost (RBOB->PHP proxy)
flows into the retail pump price, and confirm the driver is itself ~random walk.
"""
import numpy as np
import pandas as pd


def estimate_passthrough(df: pd.DataFrame, cost_col: str = 'gas_price',
                         pump_col: str = 'ron95') -> dict:
    """OLS Δpump_t = α + β0·Δcost_t + β1·Δcost_{t-1} + ε with HAC (Newey-West) SE.

    Returns alpha, beta0, beta1, beta_total, r2, driver_acf1, n. Coeffs are None
    if fewer than 10 usable rows.
    """
    import statsmodels.api as sm

    d = df[[cost_col, pump_col]].dropna().sort_index()
    dpump = d[pump_col].diff()
    dcost = d[cost_col].diff()
    reg = pd.DataFrame({
        'dpump': dpump,
        'dcost': dcost,
        'dcost1': dcost.shift(1),
    }).dropna()

    if len(reg) < 10:
        return {'n': int(len(reg)), 'alpha': None, 'beta0': None, 'beta1': None,
                'beta_total': None, 'r2': None, 'driver_acf1': None}

    X = sm.add_constant(reg[['dcost', 'dcost1']])
    model = sm.OLS(reg['dpump'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
    b0 = float(model.params['dcost'])
    b1 = float(model.params['dcost1'])
    driver_acf1 = float(pd.Series(dcost.dropna().to_numpy()).autocorr(lag=1))
    return {
        'n': int(len(reg)),
        'alpha': round(float(model.params['const']), 4),
        'beta0': round(b0, 4),
        'beta1': round(b1, 4),
        'beta_total': round(b0 + b1, 4),
        'r2': round(float(model.rsquared), 4),
        'driver_acf1': round(driver_acf1, 4),
    }
