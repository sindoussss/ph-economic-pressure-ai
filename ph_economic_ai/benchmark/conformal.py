"""Split-conformal prediction intervals from out-of-sample residuals.

q̂ for a target level is the finite-sample-corrected quantile of the absolute
calibration residuals. Empirical coverage is then measured on a separate
validation set so the displayed interval can be verified, not asserted.
"""
import numpy as np


def conformal_quantile(cal_residuals: np.ndarray, level: float) -> float:
    """Finite-sample conformal quantile of |residuals| at the given level.

    Uses the ceil((n+1)*level)/n rank to guarantee >= level coverage.
    """
    abs_res = np.abs(np.asarray(cal_residuals, dtype=float))
    n = len(abs_res)
    if n == 0:
        raise ValueError('cal_residuals is empty')
    rank = int(np.ceil((n + 1) * level))
    rank = min(rank, n)                       # clamp; level near 1 with small n
    return float(np.sort(abs_res)[rank - 1])


def coverage(y_true: np.ndarray, y_pred: np.ndarray, qhat: float) -> float:
    """Fraction of points whose true value lies within y_pred +/- qhat."""
    inside = np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= qhat
    return float(np.mean(inside))


def build_calibration_table(cal_residuals, y_true, y_pred, levels=(0.5, 0.8, 0.9, 0.95)):
    """Per-level rows: {'nominal', 'qhat', 'measured'} for the report + UI."""
    table = []
    for level in levels:
        qhat = conformal_quantile(cal_residuals, level)
        table.append({
            'nominal': level,
            'qhat': round(qhat, 4),
            'measured': round(coverage(y_true, y_pred, qhat), 4),
        })
    return table
