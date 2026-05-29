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
