import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


class _Rag:
    def query(self, text, top_k=3, sources=None):
        src = (sources or ['DOE'])[0]
        return [{'source': src, 'text': f'{src} chunk {i}'} for i in range(top_k)]


def test_add_evidence_layer_populates(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import _SwarmCanvas, _EvidenceNode
    c = _SwarmCanvas()
    n_agents = len(c._agents)
    assert n_agents > 0
    c.add_evidence_layer(_Rag(), {'current_price': 60.0}, top_k=3)
    ev = [it for it in c._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev) == n_agents * 3                      # each agent gained 3 real chunks
    assert ev[0]._source and ev[0]._text               # carries provenance
    # idempotent: calling again clears + rebuilds (not doubles)
    c.add_evidence_layer(_Rag(), {'current_price': 60.0}, top_k=3)
    ev2 = [it for it in c._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev2) == n_agents * 3


def test_evidence_hover_animates_scale(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import _EvidenceNode
    n = _EvidenceNode('DOE', 'diesel eases')
    n.hoverEnterEvent(None)                              # grows on hover
    assert n._scale_anim.endValue() > 1.0
    n.hoverLeaveEvent(None)                              # shrinks back
    assert n._scale_anim.endValue() == 1.0


def test_evidence_click_emits_provenance(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import _SwarmCanvas
    c = _SwarmCanvas()
    got = []
    c.node_clicked.connect(lambda d: got.append(d))
    c._emit_evidence_click('DOE', 'diesel eases 0.20')
    assert got and got[0]['kind'] == 'evidence'
    assert got[0]['payload'] == {'source': 'DOE', 'text': 'diesel eases 0.20'}
