import sys
import pytest
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)

def test_stage3_swarm_panel_builds(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    assert hasattr(panel, '_canvas')
    assert hasattr(panel, '_log')

def test_reset_clears_state(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    panel._groups_done = 5
    panel.reset()
    assert panel._groups_done == 0

def test_swarm_complete_emits_signal(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.swarm import MasterVerdict
    received = []
    panel = Stage3SwarmPanel()
    panel.swarm_complete.connect(lambda mv: received.append(mv))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80,
                       dissenting_regions=[], reasoning='test', regional_verdicts=[])
    panel._on_swarm_complete(mv)
    assert len(received) == 1
    assert received[0].confidence_pct == 80


def test_live_graph_grows_and_view_report(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    p = Stage3SwarmPanel()
    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None): return [{'source': 'DOE', 'text': 'x'}]
    p._begin_live_graph(_Rag(), {'current_price': 60.0}, {})
    assert not p._canvas.isHidden()                        # old structured arena is the live view
    p._on_group_round_done(0, 1, [NS(agent_name='FCST', statement='s', price_estimate=-1.8)])
    p._flush_kg()
    assert p._kg_canvas.node_item_count() > 0              # graph grew
    fired = []
    p.view_report_requested.connect(lambda: fired.append(True))
    p._on_swarm_complete(NS(final_estimate=-1.8, confidence_pct=80, regional_verdicts=[],
                            dissenting_regions=[], all_responses=[]))
    assert not p._toast.isHidden()                         # completion toast slid in
    p._toast_btn.click()                                   # its "View report →" button
    assert fired == [True]


def test_connect_thread_populates_evidence(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel, _EvidenceNode
    p = Stage3SwarmPanel()

    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None):
            return [{'source': 'DOE', 'text': f'c{i}'} for i in range(top_k)]

    thread = NS(_rag=_Rag(), _scenario={'current_price': 60.0},
                agent_typing=NS(connect=lambda *a: None),
                agent_done_typing=NS(connect=lambda *a: None),
                group_round_done=NS(connect=lambda *a: None),
                group_eliminated=NS(connect=lambda *a: None),
                group_survivor=NS(connect=lambda *a: None),
                regional_done=NS(connect=lambda *a: None),
                swarm_complete=NS(connect=lambda *a: None))
    p.connect_thread(thread)
    ev = [it for it in p._canvas._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev) > 0                                  # canvas populated from run start
