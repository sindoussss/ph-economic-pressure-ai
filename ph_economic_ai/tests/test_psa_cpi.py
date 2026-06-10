import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.benchmark.psa_cpi import _label_to_ym, load_transport_cpi, load_transport_mom


def test_label_to_ym_handles_formats():
    assert _label_to_ym('1994M01') == '1994-01'
    assert _label_to_ym('1994 M01') == '1994-01'
    assert _label_to_ym('January 1994') == '1994-01'
    assert _label_to_ym('1994 January') == '1994-01'
    assert _label_to_ym('2018-03') == '2018-03'
    assert _label_to_ym('not a date') is None


def test_load_transport_cpi_and_mom(tmp_path):
    p = tmp_path / 't.csv'
    p.write_text('date,transport_cpi\n2018-01,100.0\n2018-02,101.0\n2018-03,101.0\n',
                 encoding='utf-8')
    idx = load_transport_cpi(p)
    assert list(idx.index) == ['2018-01', '2018-02', '2018-03']
    assert idx['2018-02'] == pytest.approx(101.0)
    mom = load_transport_mom(p)
    assert mom['2018-02'] == pytest.approx(1.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index


from ph_economic_ai.benchmark.psa_cpi import _resolve_commodity_id, load_food_mom


def _commodity_var():
    return {
        'code': 'Commodity Description',
        'values': ['0', '1', '2', '203'],
        'valueTexts': ['0 - ALL ITEMS', '01 - FOOD AND NON-ALCOHOLIC BEVERAGES',
                       '01.1 - FOOD', '07 - TRANSPORT'],
    }


def test_resolve_commodity_id_by_coicop_prefix():
    v = _commodity_var()
    assert _resolve_commodity_id(v, '01') == '1'      # division, not '01.1 - FOOD'
    assert _resolve_commodity_id(v, '07') == '203'


def test_resolve_commodity_id_missing_raises():
    import pytest
    with pytest.raises(ValueError):
        _resolve_commodity_id(_commodity_var(), '99')


def test_load_food_mom(tmp_path):
    import pytest
    p = tmp_path / 'food.csv'
    p.write_text('date,food_cpi\n2018-01,100.0\n2018-02,102.0\n2018-03,102.0\n',
                 encoding='utf-8')
    mom = load_food_mom(p)
    assert mom['2018-02'] == pytest.approx(2.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index


from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom


def test_load_electricity_mom(tmp_path):
    import pytest
    p = tmp_path / 'elec.csv'
    p.write_text('date,electricity_cpi\n2018-01,100.0\n2018-02,103.0\n2018-03,103.0\n',
                 encoding='utf-8')
    mom = load_electricity_mom(p)
    assert mom['2018-02'] == pytest.approx(3.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
