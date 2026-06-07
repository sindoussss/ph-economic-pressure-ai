import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from ph_economic_ai.benchmark.conformal import (
    conformal_quantile, coverage, build_calibration_table,
)


def test_conformal_quantile_matches_gaussian():
    rng = np.random.default_rng(0)
    cal = rng.normal(0.0, 1.0, 20000)          # residuals ~ N(0,1)
    qhat = conformal_quantile(cal, level=0.90)
    # |N(0,1)| 90th percentile ~ 1.645
    assert qhat == pytest.approx(1.645, abs=0.05)


def test_coverage_near_nominal_on_fresh_sample():
    rng = np.random.default_rng(1)
    cal = rng.normal(0.0, 2.0, 20000)
    qhat = conformal_quantile(cal, level=0.90)
    y_true = rng.normal(50.0, 2.0, 20000)
    y_pred = np.full_like(y_true, 50.0)         # residuals ~ N(0,2)
    cov = coverage(y_true, y_pred, qhat)
    assert cov == pytest.approx(0.90, abs=0.02)


def test_calibration_table_has_row_per_level():
    rng = np.random.default_rng(2)
    cal = np.abs(rng.normal(0.0, 1.0, 5000))
    y_true = rng.normal(10.0, 1.0, 5000)
    y_pred = np.full_like(y_true, 10.0)
    table = build_calibration_table(cal, y_true, y_pred, levels=(0.5, 0.8, 0.9, 0.95))
    assert [r['nominal'] for r in table] == [0.5, 0.8, 0.9, 0.95]
    assert all('measured' in r and 'qhat' in r for r in table)


from ph_economic_ai.benchmark.conformal import (
    normalized_conformal_quantile, normalized_coverage,
)


def test_normalized_coverage_near_nominal_heteroscedastic():
    rng = np.random.default_rng(7)
    n = 20000
    sigma = rng.uniform(0.5, 3.0, n)
    cal_res = rng.normal(0, 1, n) * sigma
    qn = normalized_conformal_quantile(cal_res, sigma, level=0.90)
    sigma_v = rng.uniform(0.5, 3.0, n)
    val_res = rng.normal(0, 1, n) * sigma_v
    cov = normalized_coverage(val_res, sigma_v, qn)
    assert cov == pytest.approx(0.90, abs=0.02)


def test_normalized_bands_are_wider_where_sigma_larger():
    rng = np.random.default_rng(8)
    sigma = np.array([1.0] * 1000 + [3.0] * 1000)
    cal_res = rng.normal(0, 1, 2000) * sigma
    qn = normalized_conformal_quantile(cal_res, sigma, level=0.90)
    assert qn * 3.0 > qn * 1.0
