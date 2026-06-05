"""Strictly-causal, expanding-window walk-forward backtest.

At each step i (i >= min_train) the forecaster is given only data with index
< i and must predict y[i]. No future information can leak in.
"""
from typing import Callable, Optional

import numpy as np


def walk_forward(
    y: np.ndarray,
    X: Optional[np.ndarray],
    predict_fn: Callable[[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]], float],
    min_train: int,
) -> dict:
    """Return dict with 'y_true', 'y_pred', 'residuals', 'index' (all 1-D arrays).

    predict_fn(X_train, y_train, x_next) -> float
      X_train : feature rows for indices [0, i)  (or None if X is None)
      y_train : targets for indices [0, i)
      x_next  : feature row i                     (or None if X is None)
    """
    n = len(y)
    if min_train < 1 or min_train >= n:
        raise ValueError(f'min_train={min_train} invalid for series of length {n}')

    y_true, y_pred, idx = [], [], []
    for i in range(min_train, n):
        y_train = y[:i]
        X_train = X[:i] if X is not None else None
        x_next = X[i] if X is not None else None
        pred = float(predict_fn(X_train, y_train, x_next))
        y_pred.append(pred)
        y_true.append(float(y[i]))
        idx.append(i)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'residuals': y_true - y_pred,
        'index': np.array(idx),
    }
