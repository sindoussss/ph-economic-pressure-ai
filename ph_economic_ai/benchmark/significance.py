"""Diebold-Mariano test of equal predictive accuracy, with the Harvey-Leybourne-
Newbold (1997) small-sample correction. Pure numpy/scipy.
"""
import numpy as np
from scipy import stats


def diebold_mariano(loss_a, loss_b, h: int = 1) -> dict:
    """Compare two per-step loss series. d = loss_a - loss_b.

    Returns {'dm_stat', 'p_value'}. Positive stat => series A has higher loss
    (worse) than series B. p_value is two-sided (Student-t, df = n-1).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    d_bar = float(np.mean(d))
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var_d = gamma0
    for k in range(1, h):
        cov_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_d += 2.0 * cov_k
    if n < 2 or var_d <= 0:
        return {'dm_stat': 0.0, 'p_value': 1.0}
    dm = d_bar / np.sqrt(var_d / n)
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = float(dm * corr)
    p = float(2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=n - 1)))
    return {'dm_stat': dm_hln, 'p_value': p}
