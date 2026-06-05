import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.proxy_validation import proxy_vs_gold


def test_perfect_proxy_reports_r_one_zero_bias():
    idx = ['2020-01', '2020-02', '2020-03', '2020-04']
    gold = pd.Series([50.0, 52.0, 51.0, 53.0], index=idx)
    proxy = gold.copy()
    res = proxy_vs_gold(proxy, gold)
    assert res['pearson_r'] == pytest.approx(1.0)
    assert res['bias_mean'] == pytest.approx(0.0)
    assert res['mae'] == pytest.approx(0.0)
    assert res['n'] == 4


def test_constant_offset_shows_in_bias_not_correlation():
    idx = ['2020-01', '2020-02', '2020-03', '2020-04']
    gold = pd.Series([50.0, 52.0, 51.0, 53.0], index=idx)
    proxy = gold + 2.0
    res = proxy_vs_gold(proxy, gold)
    assert res['pearson_r'] == pytest.approx(1.0)
    assert res['bias_mean'] == pytest.approx(2.0)
    assert res['mae'] == pytest.approx(2.0)


def test_aligns_on_shared_dates_only():
    gold = pd.Series([50.0, 52.0], index=['2020-01', '2020-02'])
    proxy = pd.Series([50.0, 99.0], index=['2020-01', '2020-09'])
    res = proxy_vs_gold(proxy, gold)
    assert res['n'] == 1
