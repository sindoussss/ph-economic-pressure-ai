import sys
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _responses(estimates):
    from ph_economic_ai.engine.debate import AgentResponse
    return [
        AgentResponse(agent_name=f'Agent{i}', round_num=1, thinking='',
                      statement=f'reasoning {i}', price_estimate=e)
        for i, e in enumerate(estimates)
    ]


def _consensus_for(estimates):
    from ph_economic_ai.engine.debate import DebateEngine
    engine = DebateEngine(agents=[], rag=None, scenario={})
    engine._history = _responses(estimates)
    return engine.consensus()


def _panel(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from unittest.mock import MagicMock
    import numpy as np
    import pandas as pd
    panel = Stage4ReportPanel()
    mock_reg = MagicMock()
    mock_reg.predict.return_value = np.array([60.0])
    mock_reg.feature_importances_ = np.array([0.5, 0.3, 0.2])
    df = pd.DataFrame({
        'date': pd.period_range('2024-01', periods=3, freq='M').to_timestamp(how='end').normalize(),
        'gas_price': [58.0, 59.0, 60.0],
        'oil_price': [80.0, 81.0, 82.0],
        'usd_php': [56.0, 56.5, 57.0],
        'demand_index': [70.0, 72.0, 74.0],
        'cpi': [120.0, 121.0, 122.0],
        'remittances': [2.5, 2.6, 2.7],
    })
    return panel, mock_reg, df


def test_classic_mode_card_flags_a_collapsed_room(app):
    """RSK-038 follow-up: classic mode's own report card (populate ->
    _build_left) showed a bare 'N% agent agreement' with none of the
    collapse-detection swarm mode's card already had two rows down in the
    same file. Three near-identical estimates must trip the same caveat."""
    panel, reg, df = _panel(app)
    consensus = _consensus_for([1.50, 1.50, 1.51])
    panel.populate(_responses([1.50, 1.50, 1.51]), consensus, reg, df, 0.5, {})
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'distinct value' in texts
    assert 'room has narrowed' in texts


def test_classic_mode_card_silent_on_a_spread_room(app):
    panel, reg, df = _panel(app)
    consensus = _consensus_for([1.0, 1.8, 0.5, 2.1])
    panel.populate(_responses([1.0, 1.8, 0.5, 2.1]), consensus, reg, df, 0.5, {})
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'room has narrowed' not in texts


def test_classic_left_caveats_survive_the_restyle(app):
    """Pin test for the Task 8 token-restyle of _build_left: sub_lbl,
    basis_lbl, caveat_lbl, and range-row values change stylesheet only.
    Text and visibility must be identical before and after."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'verdicts': [{'estimate': -0.08}, {'estimate': -0.08}, {'estimate': -0.09}],
    }
    panel._build_left(consensus, responses=[])
    texts = [c.text() for c in panel.findChildren(QLabel)]
    assert any('100% agent agreement' in t for t in texts)
