import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui.kg_extract_worker import enrich_with_entities


def test_enrich_applies_extraction_to_evidence(monkeypatch):
    b = KnowledgeGraphBuilder()
    ev = b.add_evidence('DOE', 0, 'diesel down on Brent slide')
    import ph_economic_ai.ui.kg_extract_worker as w
    monkeypatch.setattr(w, 'extract', lambda text, source='': {
        'entities': [{'name': 'diesel', 'type': 'commodity'}], 'relations': []})
    n = enrich_with_entities(b, extract_fn=w.extract)
    assert n == 1                                   # one chunk processed
    assert b.node('ent:diesel') is not None
    assert any(e.kind == 'mentions' for e in b.edges_of('ent:diesel'))
