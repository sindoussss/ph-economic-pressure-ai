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
    # model raising -> empty, never propagates
    def boom(*a, **k):
        raise RuntimeError('ollama down')
    monkeypatch.setattr(ee, 'ollama', type('M', (), {'chat': staticmethod(boom)}))
    assert ee.extract('some real chunk text', 'DOE') == {'entities': [], 'relations': []}
    # no ollama at all -> empty
    monkeypatch.setattr(ee, 'ollama', None)
    assert ee.extract('text', 'DOE') == {'entities': [], 'relations': []}
    # empty input -> empty (no call)
    assert ee.extract('   ', 'DOE') == {'entities': [], 'relations': []}
