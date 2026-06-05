import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.metrics import mae, rmse, mape, mase, skill_score


def test_mae_simple():
    assert mae(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx(2.0 / 3.0)


def test_rmse_simple():
    # errors 0,0,2 -> mean sq = 4/3 -> sqrt
    assert rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 5.0])) == pytest.approx((4.0 / 3.0) ** 0.5)


def test_mape_is_percent():
    # |2-1|/2 = 0.5 -> 50%
    assert mape(np.array([2.0, 2.0]), np.array([1.0, 3.0])) == pytest.approx(50.0)


def test_mase_scaled_by_naive():
    y_train = np.array([10.0, 11.0, 12.0, 13.0])      # naive MAE = 1.0
    y_true = np.array([14.0, 15.0])
    y_pred = np.array([14.0, 13.0])                    # abs errors 0,2 -> MAE 1.0
    assert mase(y_true, y_pred, y_train) == pytest.approx(1.0)


def test_skill_score_positive_when_better():
    assert skill_score(rmse_model=1.0, rmse_baseline=2.0) == pytest.approx(0.5)


def test_skill_score_zero_when_equal():
    assert skill_score(rmse_model=2.0, rmse_baseline=2.0) == pytest.approx(0.0)
