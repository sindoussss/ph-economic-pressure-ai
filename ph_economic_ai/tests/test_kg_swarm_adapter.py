import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from types import SimpleNamespace as NS
from ph_economic_ai.engine.kg_swarm_adapter import build_graph


class _Rag:
    all_source_names = ['DOE', 'CRUDE']
    def query(self, text, top_k=3, sources=None):
        return [{'source': (sources or ['DOE'])[0], 'text': 'diesel down', 'url': 'u'}]


def _master():
    return NS(
        final_estimate=-1.8,
        regional_verdicts=[NS(judge_id=0, region_pair=('NCR', 'CENTRAL LUZON'), estimate=-1.7)],
        all_responses=[NS(agent_name='FCST-NCR', statement='easing', price_estimate=-1.9)],
    )


def test_build_graph_from_real_swarm_shapes():
    agents = [NS(name='FCST-NCR', role='Forecaster', region_name='NCR',
                 rag_sources=['DOE'])]
    b = build_graph(_master(), agents, {'current_price': 60.0, 'oil_price': 82.0}, _Rag())
    nodes, edges = b.snapshot()
    ids = {n.id: n for n in nodes}
    assert ids['agent:FCST-NCR'].payload['estimate'] == -1.9          # from all_responses
    assert 'ev:DOE#0' in ids and ids['ev:DOE#0'].kind == 'evidence'   # re-query retrieval
    assert 'data:oil_price' in ids
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek         # agent joined to its judge
    assert ('agent:FCST-NCR', 'ev:DOE#0', 'retrieved') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek
