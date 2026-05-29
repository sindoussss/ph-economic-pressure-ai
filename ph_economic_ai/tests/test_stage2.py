import sys
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    a = QApplication.instance() or QApplication(sys.argv)
    yield a


def test_run_requested_emits_four_args(app):
    from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel
    from ph_economic_ai.engine.debate import DEFAULT_AGENTS
    received = []
    panel = Stage2SetupPanel(DEFAULT_AGENTS)
    panel.run_requested.connect(lambda s, a, sw, pn: received.append((s, a, sw, pn)))
    panel._on_run()
    assert len(received) == 1
    _, _, sw, pn = received[0]
    assert sw is True
    assert pn == 4


def test_swarm_toggle_disables_parallel_slider(app):
    from ph_economic_ai.ui.stage2_setup import Stage2SetupPanel
    from ph_economic_ai.engine.debate import DEFAULT_AGENTS
    panel = Stage2SetupPanel(DEFAULT_AGENTS)
    assert panel._parallel_slider.isEnabled()
    panel._swarm_btn.setChecked(False)
    assert not panel._parallel_slider.isEnabled()
    panel._swarm_btn.setChecked(True)
    assert panel._parallel_slider.isEnabled()
