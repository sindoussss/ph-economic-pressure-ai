import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_trajectories_built_with_markers(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p._df = pd.DataFrame({'gas_price': [58., 59., 60., 61., 60.5, 60.]})
    p.set_sector_forecasts(-1.8, -2.6, 0.18)
    assert p._trajectory_holder.isVisible()
    canvases = p._trajectory_holder.findChildren(FigureCanvasQTAgg)
    assert len(canvases) >= 1                       # >=1 (3 when gold present)
    texts = ' || '.join(l.text() for l in p._trajectory_holder.findChildren(QLabel))
    assert 'kWh' in texts                            # electricity note (gold present)


def test_trajectories_graceful_without_df(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p._df = None
    p.set_sector_forecasts(-1.8, -2.6, 0.18)        # must not raise
    assert p._sector_holder.isVisible()              # bar card still renders
