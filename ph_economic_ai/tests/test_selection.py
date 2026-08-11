"""Selection-honest evaluation.

`RSK-004`: the audit picks the best of K methods and reports that winner's
p-value, which is optimistic because the winner was chosen for looking good. These
tests pin the two properties that make the holdout protocol worth trusting: it
must not confirm noise, and it must not quietly re-select on the evaluation data.
"""
import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark import selection

POOL = ('random_walk', 'drift', 'seasonal_naive', 'mean')
METHODS = ['random_walk', 'drift', 'seasonal_naive', 'mean', 'ridge', 'hgb']


def _frame(target, **features):
    data = dict(features)
    data['target'] = target
    return pd.DataFrame(data, index=pd.period_range('2005-01', periods=len(target),
                                                    freq='M').strftime('%Y-%m'))


def test_selection_corrected_p_is_a_bonferroni_step():
    assert selection.selection_corrected_p(0.01, 4) == pytest.approx(0.04)
    assert selection.selection_corrected_p(0.4, 5) == 1.0        # clipped
    assert selection.selection_corrected_p(0.02, 1) == pytest.approx(0.02)


def test_selection_corrected_p_rejects_a_nonsense_candidate_count():
    with pytest.raises(ValueError):
        selection.selection_corrected_p(0.01, 0)


def test_split_point_leaves_both_sides_usable():
    cut = selection.split_point(200, min_train=24, holdout_frac=0.3)
    assert 24 < cut < 200 - selection.MIN_HOLDOUT_PREDICTIONS + 1
    assert 200 - cut >= selection.MIN_HOLDOUT_PREDICTIONS


def test_split_point_never_starves_the_holdout():
    """Even a tiny holdout_frac must leave enough predictions to test on."""
    cut = selection.split_point(120, min_train=24, holdout_frac=0.01)
    assert 120 - cut >= selection.MIN_HOLDOUT_PREDICTIONS


def test_noise_is_not_confirmed():
    """The property that matters most. On a target with no signal, a panel will
    still produce a winner during selection; the holdout must refuse to confirm it."""
    rng = np.random.default_rng(11)
    n = 160
    target = rng.normal(0.3, 0.5, n)
    frame = _frame(target,
                   prev=np.roll(target, 1),
                   junk1=rng.normal(0, 1, n),
                   junk2=rng.normal(0, 1, n))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['verdict'] == 'not_confirmed_on_holdout', result['interpretation']


def test_a_real_signal_is_confirmed():
    """And it must not be so conservative that it rejects a genuine driver."""
    rng = np.random.default_rng(5)
    n = 200
    driver = rng.normal(0, 1, n)
    target = 0.3 + 1.5 * driver + rng.normal(0, 0.25, n)
    frame = _frame(target, driver=driver, prev=np.roll(target, 1))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['verdict'] == 'confirmed_on_holdout', result['interpretation']
    assert result['holdout_skill'] > 0
    assert result['holdout_dm_p'] < 0.05


def test_the_baseline_is_fixed_during_selection_not_rechosen_on_the_holdout():
    """Re-picking the best naive on the evaluation segment would be a second
    selection performed on the data reserved to test the first one."""
    rng = np.random.default_rng(3)
    n = 150
    target = rng.normal(0.2, 0.4, n)
    frame = _frame(target, prev=np.roll(target, 1), junk=rng.normal(0, 1, n))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    # On a mean-reverting rate the mean is the strong naive and must be the one
    # carried into the holdout.
    assert result['best_naive'] == 'mean'


def test_holdout_predictions_come_only_from_after_the_cut():
    rng = np.random.default_rng(9)
    n = 150
    target = rng.normal(0.2, 0.4, n)
    frame = _frame(target, prev=np.roll(target, 1), junk=rng.normal(0, 1, n))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['n_holdout_predictions'] == n - result['cut']


def test_shrinkage_is_reported():
    """The gap between selection and holdout skill is the visible signature of
    selection bias, so it must be surfaced rather than left to be computed."""
    rng = np.random.default_rng(7)
    n = 150
    target = rng.normal(0.2, 0.4, n)
    frame = _frame(target, prev=np.roll(target, 1), junk=rng.normal(0, 1, n))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['shrinkage'] == pytest.approx(
        result['selection_skill'] - result['holdout_skill'], abs=1e-9)


def test_too_short_a_frame_reports_insufficient_data_rather_than_guessing():
    rng = np.random.default_rng(1)
    target = rng.normal(0, 1, 30)
    frame = _frame(target, prev=np.roll(target, 1))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['verdict'] == 'insufficient_data'


def test_holdout_below_the_documented_minimum_is_refused():
    """split_point()'s own clamps can force a cut that leaves fewer than
    MIN_HOLDOUT_PREDICTIONS on the holdout side: at n=42 with the default
    min_train=24, the selection-segment floor and the holdout-size floor
    conflict and only 6 predictions are left after the cut. The module's own
    docstring says 12 is the minimum "for a DM test to mean anything", so a DM
    test must never actually run on fewer than that -- regardless of what
    split_point computed -- and this must be refused as insufficient_data
    rather than silently producing a confirmed/not_confirmed verdict."""
    rng = np.random.default_rng(2)
    n = 42
    target = rng.normal(0.2, 0.4, n)
    frame = _frame(target, prev=np.roll(target, 1))
    result = selection.run_selection_holdout(frame, METHODS, POOL, min_train=24)
    assert result['verdict'] == 'insufficient_data'
    assert result['n_holdout_predictions'] < selection.MIN_HOLDOUT_PREDICTIONS
