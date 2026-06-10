import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QWidget


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_right_pane_has_outputs_and_interact(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    interact = QWidget()
    panel = Stage4ReportPanel(interact_panel=interact)
    assert panel._right_stack.count() == 2
    assert panel._right_stack.widget(1) is interact
    panel._set_right_pane(1)
    assert panel._right_stack.currentIndex() == 1
    panel._set_right_pane(0)
    assert panel._right_stack.currentIndex() == 0


def test_no_interact_panel_outputs_only(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    assert panel._right_stack.count() == 1
    assert panel._right_stack.currentIndex() == 0
