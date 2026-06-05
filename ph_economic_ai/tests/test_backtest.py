import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.backtest import walk_forward


def test_walk_forward_predicts_each_step_after_min_train():
    y = np.arange(10, dtype=float)
    # random-walk style predict_fn using only y_train
    def predict_fn(X_train, y_train, x_next):
        return float(y_train[-1])
    res = walk_forward(y=y, X=None, predict_fn=predict_fn, min_train=3)
    # steps predicted: indices 3..9 -> 7 predictions
    assert len(res['y_pred']) == 7
    assert len(res['y_true']) == 7
    # random walk on 0..9 predicts the previous value each step
    assert res['y_pred'][0] == pytest.approx(2.0)   # predict index 3 from y[:3]=[0,1,2]
    assert res['y_true'][0] == pytest.approx(3.0)


def test_walk_forward_is_causal_no_leakage():
    """predict_fn must never receive a training array containing the target."""
    y = np.arange(10, dtype=float)
    seen_lengths = []
    def predict_fn(X_train, y_train, x_next):
        seen_lengths.append(len(y_train))
        # The value being predicted must NOT be inside y_train
        assert y_train[-1] != y[len(y_train)], 'leakage: target in training set'
        return float(y_train[-1])
    walk_forward(y=y, X=None, predict_fn=predict_fn, min_train=3)
    # training set grows by exactly one each step (expanding window)
    assert seen_lengths == [3, 4, 5, 6, 7, 8, 9]


def test_walk_forward_passes_feature_rows_when_X_given():
    y = np.arange(6, dtype=float)
    X = np.arange(12, dtype=float).reshape(6, 2)
    captured = {}
    def predict_fn(X_train, y_train, x_next):
        captured['x_next'] = x_next
        captured['X_train_rows'] = X_train.shape[0]
        return 0.0
    walk_forward(y=y, X=X, predict_fn=predict_fn, min_train=5)
    # only one prediction (index 5); x_next is row 5 of X
    assert captured['X_train_rows'] == 5
    assert list(captured['x_next']) == [10.0, 11.0]
