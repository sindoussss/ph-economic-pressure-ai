import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.nowcast import build_nowcast_frame


def _fake_sources(monkeypatch, n=40):
    idx = pd.date_range('2017-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(0)
    infl = pd.Series(3.0 + rng.normal(0, 0.5, n), index=idx)
    feats = pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': 60 + np.cumsum(rng.normal(0, 0.5, n)),
        'demand_index': 70 + rng.normal(0, 2, n),
    }, index=idx)
    import ph_economic_ai.benchmark.nowcast as nc
    monkeypatch.setattr(nc, 'load_inflation', lambda: infl)
    monkeypatch.setattr(nc, '_features', lambda: feats)
    return idx, infl, feats


def test_frame_has_contemporaneous_drivers_and_lagged_target(monkeypatch):
    idx, infl, feats = _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    assert list(f.columns) == ['oil', 'fx', 'fuel', 'prev_inflation', 'target']
    assert not f.isna().any().any()
    t = f.index[5]
    pos = list(idx).index(t)
    assert f.loc[t, 'oil'] == pytest.approx(feats['oil_price'].iloc[pos])
    assert f.loc[t, 'fuel'] == pytest.approx(feats['gas_price'].iloc[pos])
    assert f.loc[t, 'prev_inflation'] == pytest.approx(infl.iloc[pos - 1])
    assert f.loc[t, 'target'] == pytest.approx(infl.iloc[pos])


def test_no_same_month_cpi_feature_leak(monkeypatch):
    _fake_sources(monkeypatch)
    f = build_nowcast_frame()
    cpi_like = [c for c in f.columns if 'infl' in c.lower() or 'cpi' in c.lower()]
    assert cpi_like == ['prev_inflation']
