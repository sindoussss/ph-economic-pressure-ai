import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.baselines import random_walk_next, seasonal_naive_next


def test_random_walk_returns_last_value():
    assert random_walk_next(np.array([60.0, 61.0, 62.5])) == pytest.approx(62.5)


def test_random_walk_single_point():
    assert random_walk_next(np.array([55.0])) == pytest.approx(55.0)


def test_seasonal_naive_returns_value_one_season_back():
    # season=12, history has 13 points; next ~ value 12 steps before the end
    hist = np.arange(13, dtype=float)  # 0..12
    assert seasonal_naive_next(hist, season=12) == pytest.approx(1.0)


def test_seasonal_naive_falls_back_to_random_walk_when_short():
    hist = np.array([5.0, 6.0, 7.0])
    assert seasonal_naive_next(hist, season=12) == pytest.approx(7.0)
