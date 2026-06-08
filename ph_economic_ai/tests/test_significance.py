import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.significance import diebold_mariano


def test_identical_losses_give_zero_stat_p_one():
    loss = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    r = diebold_mariano(loss, loss.copy())
    assert r['dm_stat'] == pytest.approx(0.0)
    assert r['p_value'] == pytest.approx(1.0)


def test_clearly_worse_a_gives_positive_stat_small_p():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.0, 200)
    loss_a = base + 1.0
    loss_b = base
    r = diebold_mariano(loss_a, loss_b)
    assert r['dm_stat'] > 0
    assert r['p_value'] < 0.05


def test_antisymmetry_of_stat():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 100)
    b = rng.uniform(0, 1, 100)
    r1 = diebold_mariano(a, b)
    r2 = diebold_mariano(b, a)
    assert r1['dm_stat'] == pytest.approx(-r2['dm_stat'], rel=1e-6)
