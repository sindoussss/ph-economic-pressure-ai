import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark import baseline_size as bs
from ph_economic_ai.benchmark import baseline_theory as bt


def test_pure_noise_is_falsely_significant_without_the_mean():
    """The headline size result. On a white-noise target with pure-noise features
    there is nothing to find, so every rejection is a false positive. Without the
    mean in the pool the protocol rejects nearly always; with it, never."""
    c = bs.cell(rho=0.0, n=151, reps=25, seed=11)
    assert c['rate_pool_without_mean'] > 0.80     # catastrophic, not merely inflated
    assert c['rate_pool_with_mean'] <= 0.08       # nominal-ish at alpha = 0.05


def test_more_data_makes_the_distortion_worse():
    """The sharpest practical consequence: 're-run it on a longer sample' — the
    standard robustness move — AMPLIFIES this error instead of exposing it. It is
    why the long-sample re-run tightened the artifact's p-value from 0.032 to
    0.001 rather than dissolving it."""
    small = bs.cell(rho=0.0, n=61, reps=25, seed=12)
    large = bs.cell(rho=0.0, n=151, reps=25, seed=13)
    assert large['rate_pool_without_mean'] > small['rate_pool_without_mean']


def test_distortion_vanishes_above_the_theoretical_crossover():
    """Independent confirmation of baseline_theory: S(rho) is positive only for
    rho < 1/2, so a persistent target should show no distortion at all. Two
    derivations — one analytic, one by simulation — agreeing on the same
    threshold."""
    assert bt.spurious_skill(0.7) < 0             # theory: no spurious edge
    c = bs.cell(rho=0.7, n=151, reps=25, seed=14)
    assert c['rate_pool_without_mean'] <= 0.08    # simulation agrees
    assert c['rate_pool_with_mean'] <= 0.08


def test_corrected_pool_still_detects_a_real_driver():
    """The fix must not simply blind the audit: given a genuine, recoverable
    driver at a realistic sample size, the mean-inclusive pool still finds it."""
    c = bs.cell(rho=0.0, n=151, reps=25, beta=0.6, seed=15)
    assert c['rate_pool_with_mean'] > 0.5


def test_old_pool_never_lets_the_mean_win():
    """Guards the experiment's own validity. Under the OLD pool the mean was never
    evaluated, so it must be excluded as a candidate too — otherwise we would be
    simulating a protocol nobody ran, and the mean could 'win' as a model."""
    rng = np.random.default_rng(3)
    y, X = bs._sim_frame(0.0, 120, rng)
    rm, loss = bs._losses(y, X, bs._METHODS, 24)
    assert 'mean' in rm                                   # it was measured...
    from ph_economic_ai.benchmark.nowcast import mom_verdict
    keep = [m for m in rm if m in bs.POOL_OLD or m not in bs.POOL_NEW]
    assert 'mean' not in keep                             # ...but not offered to OLD
    v = mom_verdict({m: rm[m] for m in keep}, {m: loss[m] for m in keep},
                    baseline_pool=bs.POOL_OLD)
    assert v['best_method'] != 'mean'


def test_pools_are_scored_on_identical_losses():
    """The comparison is paired by construction: same data, same fitted models,
    only the pool differs. If this ever stops holding the result is confounded."""
    rng = np.random.default_rng(4)
    y, X = bs._sim_frame(0.2, 100, rng)
    rm, loss = bs._losses(y, X, bs._METHODS, 24)
    a = bs._verdict_under(bs.POOL_OLD, rm, loss)
    b = bs._verdict_under(bs.POOL_NEW, rm, loss)
    assert a in ('beats_best_naive', 'no_better_than_naive')
    assert b in ('beats_best_naive', 'no_better_than_naive')
    # scoring the same pool twice must be deterministic
    assert bs._verdict_under(bs.POOL_NEW, rm, loss) == b
