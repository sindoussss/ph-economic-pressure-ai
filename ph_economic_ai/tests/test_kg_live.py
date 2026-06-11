import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from types import SimpleNamespace as NS
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui import kg_live


class _Rag:
    all_source_names = ['DOE', 'CRUDE']
    def query(self, text, top_k=3, sources=None):
        return [{'source': (sources or ['DOE'])[0], 'text': 'diesel down', 'url': ''}]


def test_live_build_sequence():
    b = KnowledgeGraphBuilder()
    scenario = {'current_price': 60.0, 'oil_price': 82.0}
    kg_live.seed(b, ['DOE', 'CRUDE'], scenario)
    meta = {'FCST-NCR': NS(role='Forecaster', region_name='NCR', rag_sources=['DOE'])}
    resp = [NS(agent_name='FCST-NCR', statement='easing', price_estimate=-1.9)]
    kg_live.add_round(b, resp, meta, _Rag(), scenario)
    kg_live.add_round(b, resp, meta, _Rag(), scenario)            # idempotent re-emit
    kg_live.add_regional(b, ('NCR', 'CENTRAL LUZON'), -1.7, meta)
    kg_live.add_master(b, -1.8)
    nodes, edges = b.snapshot()
    ids = {n.id: n for n in nodes}
    assert 'src:DOE' in ids and 'data:oil_price' in ids
    assert ids['agent:FCST-NCR'].payload['estimate'] == -1.9
    assert ids['master'].payload['final_estimate'] == -1.8
    assert 'ev:DOE#0' in ids and ids['ev:DOE#0'].kind == 'evidence'
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('agent:FCST-NCR', 'ev:DOE#0', 'retrieved') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek
    assert ('master', 'judge:NCR', 'aggregates') in ek
    assert len([n for n in nodes if n.kind == 'evidence']) == 1


def test_sector_agent():
    b = KnowledgeGraphBuilder()
    kg_live.add_sector_agent(b, 'AGRI', 'food', -2.6, 'rice eases')
    n = b.node('agent:AGRI')
    assert n.cluster == 'food' and n.payload['estimate'] == -2.6
    assert any(e.kind == 'claims' for e in b.edges_of('agent:AGRI'))
