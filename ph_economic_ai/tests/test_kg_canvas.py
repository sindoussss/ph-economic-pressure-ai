import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _builder():
    b = KnowledgeGraphBuilder()
    a = b.add_agent('FCST', 'Forecaster', 'NCR', -1.9)
    ev = b.add_evidence('DOE', 0, 'diesel down')
    b.add_edge(a, ev, 'retrieved')
    b.add_entity('diesel', 'commodity', ev, 'DOE')
    return b


def test_canvas_renders_snapshot_and_focus(app):
    from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
    c = KnowledgeGraphCanvas()
    nodes, edges = _builder().snapshot()
    c.set_snapshot(nodes, edges)
    assert c.node_item_count() == len(nodes)
    # focusing the agent highlights its incident edges (red fan)
    c.focus('agent:FCST')
    assert c.focused_edge_count() >= 1
    # node details payload available for the honesty surface
    info = c.node_info('ev:DOE#0')
    assert info['kind'] == 'evidence' and 'diesel' in info['payload']['text']


def test_canvas_empty_snapshot_no_crash(app):
    from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
    c = KnowledgeGraphCanvas()
    c.set_snapshot([], [])
    assert c.node_item_count() == 0
