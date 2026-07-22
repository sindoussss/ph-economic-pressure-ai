import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import ph_economic_ai.engine.entity_extractor as ee


def test_parse_extraction_reads_entities_and_relations():
    text = (
        'ENTITY: diesel | commodity\n'
        'ENTITY: Meralco | organization\n'
        'garbage line that should be ignored\n'
        'REL: diesel -> pump price | drives\n'
    )
    res = ee.parse_extraction(text)
    assert {'name': 'diesel', 'type': 'commodity'} in res['entities']
    assert len(res['entities']) == 2
    assert res['relations'] == [{'a': 'diesel', 'b': 'pump price', 'kind': 'drives'}]


def test_parse_extraction_empty_on_garbage():
    assert ee.parse_extraction('no structured lines here') == {'entities': [], 'relations': []}


def test_extract_degrades_to_empty(monkeypatch):
    """This layer is optional enrichment — it must never break the graph."""
    monkeypatch.setattr(ee.llm, 'is_configured', lambda: True)

    # provider raising -> empty, never propagates
    def boom(*a, **k):
        raise RuntimeError('provider down')
    monkeypatch.setattr(ee.llm, 'complete', boom)
    assert ee.extract('some real chunk text', 'DOE') == {'entities': [], 'relations': []}

    # no provider configured at all -> empty, and no call attempted
    monkeypatch.setattr(ee.llm, 'is_configured', lambda: False)
    monkeypatch.setattr(ee.llm, 'complete', _never_called)
    assert ee.extract('text', 'DOE') == {'entities': [], 'relations': []}

    # empty input -> empty (no call)
    assert ee.extract('   ', 'DOE') == {'entities': [], 'relations': []}


def test_extract_parses_a_successful_response(monkeypatch):
    monkeypatch.setattr(ee.llm, 'is_configured', lambda: True)
    monkeypatch.setattr(ee.llm, 'complete',
                        lambda *a, **k: 'ENTITY: diesel | commodity\n')
    assert ee.extract('chunk', 'DOE')['entities'] == [
        {'name': 'diesel', 'type': 'commodity'}
    ]


def _never_called(*a, **k):              # pragma: no cover - must not run
    raise AssertionError('extract called the provider when none was configured')
