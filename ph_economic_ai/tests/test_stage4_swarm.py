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
