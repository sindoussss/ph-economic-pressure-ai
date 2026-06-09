import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.features import (
    build_feature_frame, make_variant, VARIANTS,
)


def _df(n=40):
    idx = pd.date_range('2018-01', periods=n, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(0)
    gas = 50 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        'oil_price': 70 + np.cumsum(rng.normal(0, 1, n)),
        'usd_php': 55 + np.cumsum(rng.normal(0, 0.1, n)),
        'gas_price': gas,
        'demand_index': 70 + rng.normal(0, 2, n),
        'ron95': gas + 6 + rng.normal(0, 0.3, n),
    }, index=idx)


def test_frame_columns_present_and_lagged():
    f = build_feature_frame(_df())
    for col in ('prev_ron95', 'oil_lag1', 'usd_lag1', 'gas_lag1', 'demand_lag1',
                'gas_lag2', 'gas_lag3', 'gas_ma3', 'fx_ma3', 'gas_delta1',
                'proxy_lag1', 'ron95'):
        assert col in f.columns
    assert not f.isna().any().any()
    raw = _df()
    t = f.index[5]
    pos = list(raw.index).index(t)
    assert f.loc[t, 'gas_lag1'] == pytest.approx(raw['gas_price'].iloc[pos - 1])


def test_make_variant_plain_has_identity_structural():
    f = build_feature_frame(_df())
    v = make_variant('baseline', f)
    assert v.X.shape[0] == len(f)
    assert np.allclose(v.structural, 0.0)
    assert np.allclose(v.y_model, v.y_actual)
    assert v.X.shape[1] == len(VARIANTS['baseline']['cols'])


def test_make_variant_structural_hybrid_decomposes():
    f = build_feature_frame(_df())
    v = make_variant('structural_hybrid', f)
    assert np.allclose(v.y_model + v.structural, v.y_actual)
    assert not np.allclose(v.structural, 0.0)


def test_registry_columns_exist_in_frame():
    f = build_feature_frame(_df())
    for name, spec in VARIANTS.items():
        for c in spec['cols']:
            assert c in f.columns, f'{name} references missing column {c}'
        if spec['structural']:
            assert spec['structural'] in f.columns


from ph_economic_ai.benchmark.features import build_target_frame


def test_build_target_frame_lags_and_target():
    idx = pd.date_range('2018-01', periods=30, freq='MS').strftime('%Y-%m')
    rng = np.random.default_rng(3)
    tgt = pd.Series(50 + np.cumsum(rng.normal(0, 1, 30)), index=idx)
    drivers = pd.DataFrame({'oil': 70 + np.cumsum(rng.normal(0, 1, 30)),
                            'fx': 55 + np.cumsum(rng.normal(0, 0.1, 30))}, index=idx)
    f = build_target_frame(tgt, drivers, 'fx', ['oil', 'fx'])
    assert 'target' in f.columns
    assert 'prev_fx' in f.columns
    for c in ('oil_lag1', 'oil_ma3', 'fx_lag1', 'fx_ma3'):
        assert c in f.columns
    assert not f.isna().any().any()
    t = f.index[4]
    pos = list(tgt.index).index(t)
    assert f.loc[t, 'prev_fx'] == pytest.approx(tgt.iloc[pos - 1])
    assert f.loc[t, 'oil_lag1'] == pytest.approx(drivers['oil'].iloc[pos - 1])


def test_build_target_frame_inner_join_on_dates():
    tgt = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0],
                    index=['2020-01', '2020-02', '2020-03', '2020-04', '2020-05'])
    drivers = pd.DataFrame({'oil': [10.0, 11.0, 12.0]},
                           index=['2020-01', '2020-02', '2020-03'])
    f = build_target_frame(tgt, drivers, 'x', ['oil'])
    assert len(f) <= 1
