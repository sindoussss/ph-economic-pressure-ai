import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_interact_ask_tab_has_exploratory_caption(app):
    from unittest.mock import MagicMock
    import numpy as np, pandas as pd
    from ph_economic_ai.engine.rag import RagEngine
    from ph_economic_ai.engine.debate import DEFAULT_AGENTS
    from ph_economic_ai.ui.stage5_interact import Stage5InteractPanel
    df = pd.DataFrame({'date': pd.date_range('2024-01', periods=3, freq='M'),
                       'gas_price': [58., 59., 60.]})
    reg = MagicMock(); reg.predict.return_value = np.array([60.])
    panel = Stage5InteractPanel(RagEngine(), list(DEFAULT_AGENTS), reg, df, 0.0)
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'exploratory' in texts.lower()
