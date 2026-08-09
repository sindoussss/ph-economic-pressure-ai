import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_sector_forecasts_show_agreement_for_all_three(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10,
                               gas_agreement=70, food_agreement=62, elec_agreement=81)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert any('70%' in t for t in labels)
    assert any('62%' in t for t in labels)
    assert any('81%' in t for t in labels)


def test_sector_forecasts_omits_agreement_when_zero(app):
    """Old callers/tests still calling positionally-only must not crash, and a
    0 agreement (the pre-Task-4 default, or a genuinely unmeasured run) must
    not print a nonsense '0% agreement' caption."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert not any('% agreement' in t for t in labels)
