import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.targets import (
    Target, TARGETS, load_fx, load_inflation, cpi_to_yoy,
)


def test_cpi_to_yoy_computes_year_on_year_percent():
    idx = pd.date_range('2019-01', periods=14, freq='MS').strftime('%Y-%m')
    cpi = pd.Series(100.0 * (1.03) ** (np.arange(14) / 12.0), index=idx)
    infl = cpi_to_yoy(cpi)
    assert infl.index[0] == '2020-01'
    assert infl.iloc[0] == pytest.approx(3.0, abs=0.2)


def test_load_fx_reads_csv(tmp_path):
    p = tmp_path / 'fx.csv'
    p.write_text('date,usd_php\n2020-01,50.0\n2020-02,51.0\n', encoding='utf-8')
    s = load_fx(p)
    assert list(s.index) == ['2020-01', '2020-02']
    assert s.iloc[1] == pytest.approx(51.0)


def test_load_inflation_reads_index_csv(tmp_path):
    idx = pd.date_range('2019-01', periods=14, freq='MS').strftime('%Y-%m')
    vals = 100.0 * (1.04) ** (np.arange(14) / 12.0)
    p = tmp_path / 'cpi.csv'
    p.write_text('date,cpi_index\n' + '\n'.join(f'{d},{v:.4f}' for d, v in zip(idx, vals)) + '\n',
                 encoding='utf-8')
    infl = load_inflation(p)
    assert infl.iloc[0] == pytest.approx(4.0, abs=0.2)


def test_registry_has_three_targets():
    assert set(TARGETS) == {'fuel', 'fx', 'inflation'}
    for name, t in TARGETS.items():
        assert isinstance(t, Target) and t.name == name
        assert callable(t.load_gold) and callable(t.build_frame)
