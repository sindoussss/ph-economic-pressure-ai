import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder, KGNode, KGEdge


def test_evidence_links_to_source_and_is_idempotent():
    b = KnowledgeGraphBuilder()
    b.add_source('DOE', 'http://doe')
    e1 = b.add_evidence('DOE', 3, 'diesel up 1.2', 'http://doe')
    e2 = b.add_evidence('DOE', 3, 'diesel up 1.2', 'http://doe')   # same chunk again
    assert e1 == e2
    nodes, edges = b.snapshot()
    assert len([n for n in nodes if n.kind == 'evidence']) == 1     # merged, not duplicated
    assert KGEdge(e1, 'src:DOE', 'from_source') in edges
    assert b.node(e1).payload['text'] == 'diesel up 1.2'


def test_agent_claim_and_aggregation_edges():
    b = KnowledgeGraphBuilder()
    b.add_master(-1.8)
    b.add_judge('NCR', -1.7)
    b.add_edge('master', 'judge:NCR', 'aggregates')
    a = b.add_agent('FCST-NCR', 'Forecaster', 'NCR', -1.9)
    b.add_edge('judge:NCR', a, 'aggregates')
    c = b.add_claim(a, -1.9, 'pump easing')
    assert b.node(c).payload['estimate'] == -1.9
    assert any(e.kind == 'claims' and e.src == a for e in b.edges_of(a))
    assert any(e.kind == 'aggregates' and e.dst == a for e in b.edges_of(a))


def test_entity_provenance_accumulates_across_chunks():
    b = KnowledgeGraphBuilder()
    ev1 = b.add_evidence('DOE', 1, 'diesel ...')
    ev2 = b.add_evidence('CRUDE', 2, 'diesel ...')
    n1 = b.add_entity('Diesel', 'commodity', ev1, 'DOE')
    n2 = b.add_entity('diesel', 'commodity', ev2, 'CRUDE')   # case-insensitive same entity
    assert n1 == n2
    node = b.node(n1)
    assert len(node.payload['provenance']) == 2
    assert {e.dst for e in b.edges_of(n1) if e.kind == 'mentions'} == {ev1, ev2}
