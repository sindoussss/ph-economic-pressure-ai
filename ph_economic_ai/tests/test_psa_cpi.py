import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.benchmark.psa_cpi import _label_to_ym, load_transport_cpi, load_transport_mom, fetch_cpi_subcategory


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


def test_fetch_cpi_subcategory_raises_on_too_few_rows(tmp_path, monkeypatch):
    def fake_fetch_px_table(url, first_year, coicop_prefix):
        return {'2020-01': 100.0, '2020-02': 101.0}  # only 2 rows

    import ph_economic_ai.benchmark.psa_cpi as psa_cpi
    monkeypatch.setattr(psa_cpi, '_fetch_px_table', fake_fetch_px_table)

    out = tmp_path / 'tiny.csv'
    with pytest.raises(ValueError, match='too short'):
        fetch_cpi_subcategory('99.9', out, 'tiny_cpi', 'test source', min_rows=50)


from ph_economic_ai.benchmark.psa_cpi import (
    load_rice_mom, load_meat_mom, load_fish_mom, load_dairy_eggs_mom,
    load_vegetables_mom, load_sugar_mom,
)

_SUBCATEGORY_LOADERS = {
    'rice_cpi': load_rice_mom, 'meat_cpi': load_meat_mom,
    'fish_cpi': load_fish_mom, 'dairy_eggs_cpi': load_dairy_eggs_mom,
    'vegetables_cpi': load_vegetables_mom, 'sugar_cpi': load_sugar_mom,
}


@pytest.mark.parametrize('column,loader', _SUBCATEGORY_LOADERS.items())
def test_load_subcategory_mom(tmp_path, column, loader):
    p = tmp_path / f'{column}.csv'
    p.write_text(f'date,{column}\n2018-01,100.0\n2018-02,104.0\n2018-03,104.0\n',
                 encoding='utf-8')
    mom = loader(p)
    assert mom['2018-02'] == pytest.approx(4.0)
    assert mom['2018-03'] == pytest.approx(0.0)
    assert '2018-01' not in mom.index
