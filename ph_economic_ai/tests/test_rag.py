import pytest
from ph_economic_ai.engine.rag import RagEngine


def test_chunk_splits_long_text():
    engine = RagEngine()
    text = 'oil price fuel gasoline Philippines economy ' * 100
    chunks = engine._chunk(text, source='test')
    assert len(chunks) > 1
    assert all(len(c.text) <= 2048 for c in chunks)
    assert chunks[0].text[-200:] in chunks[1].text


def test_add_text_and_query():
    engine = RagEngine()
    engine.add_text('DOE', 'The pump price will increase next week due to Brent crude oil rising.')
    results = engine.query('gasoline price increase oil', top_k=1)
    assert len(results) == 1
    assert results[0]['source'] == 'DOE'
    assert results[0]['score'] > 0


def test_query_empty_engine():
    engine = RagEngine()
    assert engine.query('anything') == []


def test_toggle_source_excludes_chunks():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price fuel gasoline oil Philippines cost')
    engine.add_text('BSP', 'pump price fuel gasoline oil Philippines cost')
    engine.toggle_source('DOE', False)
    results = engine.query('pump price fuel gasoline', top_k=10)
    assert all(r['source'] != 'DOE' for r in results)


def test_toggle_source_reenabled():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price fuel gasoline oil Philippines')
    engine.toggle_source('DOE', False)
    engine.toggle_source('DOE', True)
    results = engine.query('pump price fuel', top_k=5)
    assert any(r['source'] == 'DOE' for r in results)


def test_query_filter_by_sources():
    engine = RagEngine()
    engine.add_text('DOE', 'pump price adjustment next week gasoline diesel oil')
    engine.add_text('BSP', 'monetary policy rate decision inflation peso dollar')
    results = engine.query('gasoline price oil', top_k=5, sources=['DOE'])
    assert all(r['source'] == 'DOE' for r in results)


def test_chunk_count_property():
    engine = RagEngine()
    engine.add_text('X', 'word ' * 2000)
    assert engine.chunk_count > 0


def test_add_pdf_missing_file():
    engine = RagEngine()
    with pytest.raises(Exception):
        engine.add_pdf('/nonexistent/path/file.pdf')


def test_fetch_all_handles_http_error():
    from unittest.mock import patch
    import requests
    engine = RagEngine()
    with patch('requests.get', side_effect=requests.ConnectionError("timeout")):
        result = engine.fetch_all()
    # All sources fail gracefully — counts are 0 or absent, no exception raised
    assert isinstance(result, dict)
    for count in result.values():
        assert count == 0


def test_toggle_source_reenabled_with_two_sources():
    engine = RagEngine()
    engine.add_text('DOE', 'Philippine oil price bulletin from DOE.')
    engine.add_text('BSP', 'BSP monetary policy rate decision.')
    engine.toggle_source('DOE', False)
    engine.toggle_source('DOE', True)
    results = engine.query('oil price bulletin DOE', top_k=5)
    assert any(r['source'] == 'DOE' for r in results)


def test_toggle_and_filter_interaction():
    engine = RagEngine()
    engine.add_text('DOE', 'Philippine gasoline price DOE bulletin.')
    engine.add_text('BSP', 'BSP monetary policy rate.')
    engine.toggle_source('DOE', False)
    results = engine.query('oil price', top_k=5, sources=['DOE'])
    assert results == []


# ── JSON parser tests (pure functions, no network) ───────────────────────────

def test_parse_open_meteo_minimal():
    from ph_economic_ai.engine.rag import _parse_open_meteo
    txt = _parse_open_meteo({
        'current': {'temperature_2m': 30.1, 'wind_speed_10m': 8},
        'daily':   {'time': ['2026-05-28'], 'temperature_2m_max': [33.0],
                    'temperature_2m_min': [26.0], 'precipitation_sum': [4.2],
                    'weather_code': [61]},
        'hourly':  {'relative_humidity_2m': [70, 80, 75]},
    })
    assert 'Manila' in txt
    assert '30.1' in txt
    assert '4.2mm' in txt
    assert '2026-05-28' in txt


def test_parse_wb_phil_food_minimal():
    from ph_economic_ai.engine.rag import _parse_wb_phil_food
    # World Bank API shape: [meta, [rows...]]
    txt = _parse_wb_phil_food([
        {'page': 1, 'pages': 1, 'per_page': 30, 'total': 2},
        [
            {'date': '2024', 'value': 105.4, 'indicator': {'value': 'Food production index'}},
            {'date': '2018', 'value': 101.0, 'indicator': {'value': 'Food production index'}},
        ],
    ])
    assert 'food' in txt.lower()
    assert '101.00' in txt
    assert '105.40' in txt
    # Trend line should be present
    assert 'change' in txt.lower()


def test_parse_wb_phil_food_empty():
    from ph_economic_ai.engine.rag import _parse_wb_phil_food
    assert _parse_wb_phil_food([{}, []]) == ''
    assert _parse_wb_phil_food([]) == ''


def test_parse_eia_minimal():
    from ph_economic_ai.engine.rag import _parse_eia
    txt = _parse_eia({'response': {'data': [
        {'period': '2020', 'activityName': 'Generation',  'value': '100.0', 'unit': 'BKWH'},
        {'period': '2020', 'activityName': 'Consumption', 'value': '90.0',  'unit': 'BKWH'},
        {'period': '2024', 'activityName': 'Generation',  'value': '120.0', 'unit': 'BKWH'},
        {'period': '2024', 'activityName': 'Consumption', 'value': '110.0', 'unit': 'BKWH'},
        # Should be filtered out (wrong unit)
        {'period': '2024', 'activityName': 'Generation',  'value': '0.12', 'unit': 'QBTU'},
    ]}})
    assert 'Philippine electricity' in txt
    assert '2020' in txt and '2024' in txt
    assert 'Generation' in txt and '120.0' in txt
    # Growth line: 2020→2024 Generation +20%, Consumption +22.2%
    assert 'Growth' in txt
    assert '+20.0%' in txt


def test_eia_disabled_without_key(monkeypatch, tmp_path):
    """When EIA_API_KEY is unset and no config file exists, _fetch_json returns []."""
    monkeypatch.delenv('EIA_API_KEY', raising=False)
    # Point Path.home() at an empty tmp dir so the config.json lookup misses
    monkeypatch.setattr('ph_economic_ai.engine.rag.Path.home', lambda: tmp_path)
    engine = RagEngine()
    chunks = engine._fetch_json(
        'EIAElectricity',
        'https://api.eia.gov/v2/...?api_key={EIA_API_KEY}',
    )
    assert chunks == []
