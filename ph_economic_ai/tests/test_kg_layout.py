import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui.kg_layout import render_model, node_color


def _graph():
    b = KnowledgeGraphBuilder()
    b.add_master(-1.8)
    j = b.add_judge('NCR', -1.7)
    b.add_edge('master', j, 'aggregates')
    a = b.add_agent('FCST', 'Forecaster', 'NCR', -1.9)
    b.add_edge(j, a, 'aggregates')
    ev = b.add_evidence('DOE', 0, 'x')
    b.add_edge(a, ev, 'retrieved')
    return b


def test_render_model_positions_and_colours():
    nodes, edges = _graph().snapshot()
    rm = render_model(nodes, edges, width=800, height=600, seed=3)
    assert len(rm['nodes']) == len(nodes)
    for rn in rm['nodes']:
        assert 0 <= rn['x'] <= 800 and 0 <= rn['y'] <= 600
        assert rn['color'].startswith('#')
    # deterministic for a fixed seed
    rm2 = render_model(nodes, edges, width=800, height=600, seed=3)
    assert [n['x'] for n in rm['nodes']] == [n['x'] for n in rm2['nodes']]
    # hub (master, degree>=1) at least as large as a leaf evidence node
    by_id = {n['id']: n for n in rm['nodes']}
    assert by_id['master']['r'] >= by_id['ev:DOE#0']['r']


def test_node_color_by_kind_and_sector():
    from ph_economic_ai.engine.knowledge_graph import KGNode
    assert node_color(KGNode('agent:x', 'agent', 'x')) == '#E5484D'
    assert node_color(KGNode('a', 'agent', 'x', cluster='food')) == '#15A150'
