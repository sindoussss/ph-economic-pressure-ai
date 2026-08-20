import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark.targets import Target
from ph_economic_ai.benchmark.audit import verdict_from_panel, run_audit


def test_verdict_predictable_when_a_method_beats_rw():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'ridge', 'skill_vs_rw': 0.2, 'dm_p': 0.01},
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'predictable' and best['method'] == 'ridge'


def test_verdict_efficient_when_none_significantly_better():
    panel = [
        {'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None},
        {'method': 'hgb', 'skill_vs_rw': -0.05, 'dm_p': 0.9},
        {'method': 'arima', 'skill_vs_rw': -0.2, 'dm_p': 0.01},
    ]
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'efficient' and best['method'] == 'random_walk'


def _predictable_target():
    idx = pd.date_range('2016-01', periods=80, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(7)
    drv = np.cumsum(rng.normal(0, 1, 80))
    y = np.r_[0, 0.9 * np.diff(drv)] + 50
    frame = pd.DataFrame({'prev_t': np.r_[y[0], y[:-1]],
                          'drv_lag1': np.r_[0, np.diff(drv)], 'target': y}, index=idx)
    return Target('synthetic', lambda: pd.Series(y, index=idx), lambda: frame)


def test_run_audit_reports_per_target_verdict():
    reg = {'synthetic': _predictable_target()}
    rows = run_audit(['synthetic'], min_train=24, registry=reg)
    assert rows[0]['target'] == 'synthetic'
    assert rows[0]['verdict'] in ('predictable', 'efficient')
    assert 'panel' in rows[0] and rows[0]['n'] > 0


def test_run_audit_insufficient_data():
    short = Target('short', lambda: pd.Series(dtype=float),
                   lambda: pd.DataFrame({'a': [1.0, 2.0], 'target': [1.0, 2.0]},
                                        index=['2020-01', '2020-02']))
    rows = run_audit(['short'], min_train=24, registry={'short': short})
    assert rows[0]['verdict'] == 'insufficient_data'


# ── The verdict must pay for the selection it performs ───────────────────────
#
# Found 2026-08-20 while resolving Gate 7. The committed artifact read
#
#     fuel  predictable  best_method ridge  best_skill 0.1211  best_dm_p 0.0337
#
# on a bare `dm_p < 0.05`. But the verdict does not test one hypothesis: it takes
# the BEST of the non-baseline candidates and reports whether that one is
# significant. Selecting a maximum over k tests and then judging it at the
# uncorrected alpha inflates the false-positive rate by construction.
#
# The same result, scored in the declared 24-test family, reads bonferroni_p
# 0.7104 and bh_q 0.2368: it survives neither. So the artifact was asserting
# "predictable" for a target that the project's own multiple-testing artifact
# lists under `survive_neither`, and four documents inherited that claim.
#
# The correction here is deliberately the SELF-CONTAINED one, over the k
# candidates the panel actually compares. It cannot depend on the family-wide
# artifact, which is computed downstream of this verdict and would be circular.
# The family correction remains the stricter authority; this only stops the
# verdict claiming more than its own comparison supports.

import pytest

from ph_economic_ai.benchmark import audit as _audit


def _panel(*candidates):
    rows = [{'method': 'random_walk', 'skill_vs_rw': 0.0, 'dm_p': None, 'n': 72}]
    rows += [{'method': m, 'skill_vs_rw': s, 'dm_p': p, 'n': 72}
             for m, s, p in candidates]
    return rows


def test_the_threshold_divides_by_the_candidates_actually_compared():
    four = _panel(('arima', 0.01, 0.9), ('ets', 0.02, 0.8),
                  ('ridge', 0.12, 0.0337), ('hgb', 0.03, 0.7))
    assert _audit.selection_threshold(four) == pytest.approx(0.05 / 4)

    one = _panel(('ridge', 0.12, 0.0337))
    assert _audit.selection_threshold(one) == pytest.approx(0.05)


def test_a_candidate_whose_dm_p_is_missing_is_not_counted():
    """k is the number of tests performed, not the number of methods named. A
    method that produced no p-value was not a comparison and must not make the
    threshold stricter for the ones that were."""
    panel = _panel(('arima', 0.01, None), ('ridge', 0.12, 0.0337))
    assert _audit.selection_threshold(panel) == pytest.approx(0.05 / 1)


def test_the_real_fuel_panel_is_no_longer_called_predictable():
    """The regression, with the artifact's own numbers.

    ridge at dm_p 0.0337 clears a bare 0.05 and does not clear 0.05/4 = 0.0125.
    """
    panel = _panel(('arima', -0.04, 0.72), ('ets', -0.01, 0.55),
                   ('ridge', 0.1211, 0.0337), ('hgb', 0.02, 0.41))
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'efficient'
    assert best['method'] == 'random_walk'


def test_a_genuinely_strong_result_still_reads_predictable():
    """The correction must not make the verdict unreachable, or it would stop
    being a test and become a constant."""
    panel = _panel(('arima', 0.01, 0.9), ('ets', 0.02, 0.8),
                   ('ridge', 0.30, 0.0001), ('hgb', 0.03, 0.7))
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'predictable' and best['method'] == 'ridge'


def test_the_nominal_result_is_recorded_not_erased():
    """Correcting the verdict must not delete the evidence behind it.

    A reader has to be able to see that ridge was nominally significant and that
    the correction is why the verdict says otherwise. Hiding it would trade one
    misleading artifact for another.
    """
    panel = _panel(('ridge', 0.1211, 0.0337), ('hgb', 0.02, 0.41),
                   ('arima', -0.04, 0.72), ('ets', -0.01, 0.55))
    info = _audit.selection_detail(panel)
    assert info['n_candidates'] == 4
    assert info['threshold'] == pytest.approx(0.0125)
    assert info['nominally_significant'] is True
    assert info['nominal_best_method'] == 'ridge'
    assert info['nominal_best_dm_p'] == pytest.approx(0.0337)


def test_baselines_are_still_never_the_winning_method():
    """The pre-existing property must survive the change: a mean-reverting series
    makes the historical mean beat the random walk by construction."""
    panel = _panel(('arima', -0.5, 0.9))
    panel.append({'method': 'mean', 'skill_vs_rw': 0.4, 'dm_p': 0.0001, 'n': 72})
    verdict, best = verdict_from_panel(panel)
    assert verdict == 'efficient'
    assert best['method'] == 'random_walk'
