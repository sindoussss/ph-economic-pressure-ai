import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.engine.kg_assemble import assemble_structured, apply_extraction


def _inputs():
    return dict(
        sources=['DOE', 'CRUDE'],
        data_inputs={'oil_price': 82.0, 'usd_php': 57.0},
        regionals=[{'region': 'NCR', 'estimate': -1.7}],
        agents=[{'name': 'FCST-NCR', 'role': 'Forecaster', 'region': 'NCR',
                 'estimate': -1.9, 'statement': 'easing'}],
        retrievals={'FCST-NCR': [{'source': 'DOE', 'idx': 1, 'text': 'diesel down'}]},
        master_estimate=-1.8,
    )


def test_assemble_structured_builds_real_graph():
    b = KnowledgeGraphBuilder()
    assemble_structured(b, **_inputs())
    nodes, edges = b.snapshot()
    kinds = {n.id: n.kind for n in nodes}
    assert kinds['master'] == 'master'
    assert kinds['judge:NCR'] == 'judge'
    assert kinds['agent:FCST-NCR'] == 'agent'
    assert kinds['ev:DOE#1'] == 'evidence'
    assert kinds['data:oil_price'] == 'data_input'
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('master', 'judge:NCR', 'aggregates') in ek
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek
    assert ('agent:FCST-NCR', 'ev:DOE#1', 'retrieved') in ek
    assert ('ev:DOE#1', 'src:DOE', 'from_source') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek


def test_apply_extraction_grounds_entities():
    b = KnowledgeGraphBuilder()
    ev = b.add_evidence('DOE', 1, 'diesel down')
    apply_extraction(b, ev, 'DOE', {
        'entities': [{'name': 'diesel', 'type': 'commodity'}],
        'relations': [{'a': 'diesel', 'b': 'pump', 'kind': 'drives'}],
    })
    assert b.node('ent:diesel').payload['provenance'][0] == {'chunk_id': ev, 'source': 'DOE'}
    ek = {(e.src, e.dst, e.kind) for e in b.edges_of('ent:diesel')}
    assert ('ent:diesel', ev, 'mentions') in ek
    assert ('ent:diesel', 'ent:pump', 'relates') in ek
