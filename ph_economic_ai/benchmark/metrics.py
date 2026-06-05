"""Forecast error metrics. All take 1-D numpy arrays of equal length."""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Ignores zero-truth rows."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Mean absolute scaled error: model MAE / in-sample naive (lag-1) MAE."""
    naive_mae = float(np.mean(np.abs(np.diff(y_train))))
    if naive_mae == 0:
        return float('inf')
    return mae(y_true, y_pred) / naive_mae


def skill_score(rmse_model: float, rmse_baseline: float) -> float:
    """1 - RMSE_model / RMSE_baseline. Positive => beats the baseline."""
    if rmse_baseline == 0:
        return float('-inf')
    return 1.0 - (rmse_model / rmse_baseline)
