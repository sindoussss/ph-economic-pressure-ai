import sys
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_populate_swarm_exists(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    assert hasattr(panel, 'populate_swarm')


def test_populate_swarm_updates_detail(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict
    from unittest.mock import MagicMock
    import numpy as np

    panel = Stage4ReportPanel()

    rv = RegionalVerdict(judge_id=0, region_pair=('NCR', 'CAR'),
                         estimate=1.5, confidence=0.8,
                         reasoning='test', survivor_names=('A', 'B'))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80,
                       dissenting_regions=[], reasoning='test',
                       regional_verdicts=[rv])

    # Mock regressor and df so _build_right doesn't crash
    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([60.0])
    mock_reg.feature_importances_ = np.array([0.5, 0.3, 0.2])

    import pandas as pd
    df = pd.DataFrame({
        'date': pd.date_range('2024-01', periods=3, freq='M'),
        'gas_price': [58.0, 59.0, 60.0],
        'oil_price': [80.0, 81.0, 82.0],
        'usd_php': [56.0, 56.5, 57.0],
        'demand_index': [70.0, 72.0, 74.0],
        'cpi': [120.0, 121.0, 122.0],
        'remittances': [2.5, 2.6, 2.7],
    })

    panel.populate_swarm(mv, mock_reg, df, 0.5, {'oil_pct': 5.0, 'usd_pct': 2.0})
    assert '1 regional verdicts' in panel._detail_lbl.text()


def test_consensus_marked_exploratory(app):
    from PyQt6.QtWidgets import QLabel
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict
    from unittest.mock import MagicMock
    import numpy as np, pandas as pd
    panel = Stage4ReportPanel()
    rv = RegionalVerdict(judge_id=0, region_pair=('NCR', 'CAR'), estimate=1.5,
                         confidence=0.8, reasoning='', survivor_names=('a', 'b'))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80, dissenting_regions=[],
                       reasoning='', regional_verdicts=[rv])
    df = pd.DataFrame({'date': pd.date_range('2024-01', periods=3, freq='M'),
                       'gas_price': [58., 59., 60.], 'oil_price': [80., 81., 82.],
                       'usd_php': [56., 56.5, 57.], 'cpi': [120., 121., 122.],
                       'remittances': [2.5, 2.6, 2.7], 'demand_index': [70., 71., 72.]})
    reg = MagicMock(); reg.predict.return_value = np.array([60.])
    reg.feature_importances_ = np.array([.5, .3, .2])
    panel.populate_swarm(mv, reg, df, 0.5, {'oil_pct': 5.0, 'usd_pct': 2.0})
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'exploratory' in texts and 'varies per run' in texts


def test_restyle_keeps_consensus_content(app):
    from PyQt6.QtWidgets import QLabel
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict
    from unittest.mock import MagicMock
    import numpy as np, pandas as pd
    panel = Stage4ReportPanel()
    rv = RegionalVerdict(judge_id=0, region_pair=('NCR', 'CAR'), estimate=1.5,
                         confidence=0.8, reasoning='', survivor_names=('a', 'b'))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80, dissenting_regions=[],
                       reasoning='', regional_verdicts=[rv])
    df = pd.DataFrame({'date': pd.date_range('2024-01', periods=3, freq='M'),
                       'gas_price': [58., 59., 60.], 'oil_price': [80., 81., 82.],
                       'usd_php': [56., 56.5, 57.], 'cpi': [120., 121., 122.],
                       'remittances': [2.5, 2.6, 2.7], 'demand_index': [70., 71., 72.]})
    reg = MagicMock(); reg.predict.return_value = np.array([60.])
    reg.feature_importances_ = np.array([.5, .3, .2])
    panel.populate_swarm(mv, reg, df, 0.5, {'oil_pct': 5.0, 'usd_pct': 2.0})
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'varies per run' in texts          # SP2a honesty note survived the restyle
    assert 'SWARM CONSENSUS' in texts          # card title (now an eyebrow, uppercased)
