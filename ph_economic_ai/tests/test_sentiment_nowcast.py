import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd

from ph_economic_ai.benchmark import sentiment_nowcast as sn
from ph_economic_ai.benchmark.nowcast import build_nowcast_frame
from ph_economic_ai.benchmark.targets import load_inflation_mom


def test_no_trends_data_is_graceful():
    # An empty frame stands in for "refresh_social not run yet".
    res = sn.run(trends=pd.DataFrame())
    assert res['status'] == 'no_trends_data'


def _trends_like(index, cols, values):
    return pd.DataFrame({c: values for c in cols}, index=index)


def test_uninformative_sentiment_is_a_null():
    """A constant (zero-information) trend feature cannot beat the mean baseline —
    Ridge on a constant column collapses to the intercept — so sentiment_edge is
    False. This is the deterministic version of the expected real-world null."""
    idx = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom').index
    trends = _trends_like(idx, ['presyo ng gas', 'meralco bill'], values=50.0)
    res = sn.run(trends=trends)
    assert res['status'] == 'computed'
    assert 'headline' in res['targets']
    head = res['targets']['headline']
    # structure
    for k in ('sentiment_only', 'drivers_only', 'drivers_plus_sentiment'):
        assert 'verdict' in head[k]
    assert head['trend_cols'] == ['trend_presyo_ng_gas', 'trend_meralco_bill']
    # uninformative sentiment must not manufacture an edge
    assert head['sentiment_only']['verdict'] == 'no_better_than_naive'
    assert res['sentiment_edge_anywhere'] is False


def test_column_names_are_sanitised():
    idx = build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom').index
    trends = _trends_like(idx, ['Presyo ng GAS!!', 'bigas   presyo'], values=50.0)
    res = sn.run(trends=trends)
    cols = res['targets']['headline']['trend_cols']
    assert cols == ['trend_presyo_ng_gas', 'trend_bigas_presyo']
