import sys
import pytest
from PyQt6.QtWidgets import QApplication
import numpy as np
import pandas as pd


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def window(app):
    from ph_economic_ai.ui.main_window import SimMainWindow
    df = pd.DataFrame({
        'date': pd.date_range('2024-01', periods=3, freq='M'),
        'gas_price': [58.0, 59.0, 60.0],
        'oil_price': [80.0, 81.0, 82.0],
        'usd_php': [56.0, 56.5, 57.0],
        'cpi': [120.0, 121.0, 122.0],
        'remittances': [2.5, 2.6, 2.7],
        'demand_index': [70.0, 71.0, 72.0],
    })
    from unittest.mock import MagicMock
    reg = MagicMock()
    reg.predict.return_value = np.array([60.0])
    reg.feature_importances_ = np.array([0.5, 0.3, 0.2])
    return SimMainWindow(df, reg)


def test_main_window_has_swarm_panel(window):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    assert hasattr(window, '_stage3_swarm')
    assert isinstance(window._stage3_swarm, Stage3SwarmPanel)


def test_on_run_requested_accepts_4_args(window):
    from ph_economic_ai.ui.stage2_setup import Scenario
    scenario = Scenario()
    # Should not raise with swarm_mode=False
    from ph_economic_ai.engine.debate import DEFAULT_AGENTS
    window._on_run_requested(scenario, list(DEFAULT_AGENTS), False, 4)
    # Running a simulation navigates to the Simulation page. Per the current
    # nav order (Home=0, Overview=1, Simulation=2, ...), that is stack index 2.
    assert window._stack.currentIndex() == 2


def test_stage5_has_set_swarm_context(window):
    assert hasattr(window._stage5, 'set_swarm_context')


def test_completion_stays_on_simulation_then_button_navigates(window):
    # the Report stack index is 3 (the workbench)
    window._stack.setCurrentIndex(2)               # on Simulation
    window._stage3_swarm.view_report_requested.emit()
    assert window._stack.currentIndex() == 3        # navigated to Report on demand


def test_home_run_chains_monitor_then_swarm(window, monkeypatch):
    # Don't actually start the Monitor thread or the swarm — capture the wiring.
    started = {}
    monkeypatch.setattr(window._monitor, 'start', lambda: started.__setitem__('m', True))
    calls = {}
    monkeypatch.setattr(window, '_on_run_requested',
                        lambda sc, agents, swarm_mode=False, parallel_n=4:
                        calls.update(swarm_mode=swarm_mode))
    window._on_landing_run()
    assert window._stack.currentIndex() == 7        # Monitor shown first
    assert started.get('m') is True                 # Monitor started
    assert calls == {}                              # swarm NOT started yet
    window._monitor.run_finished.emit()             # Monitor finishes ->
    assert calls.get('swarm_mode') is True          # ... now the tournament runs


def test_learning_tab_present_and_refreshes(window):
    from ph_economic_ai.ui.main_window import _TopNavBar
    from ph_economic_ai.ui.learning_view import LearningView
    labels = [lbl for _idx, lbl, _lk in _TopNavBar._ITEMS]
    assert 'Learning' in labels                                  # nav tab exists
    # its stack page is a LearningView and is reachable
    learn_idx = next(i for i, lbl, _ in _TopNavBar._ITEMS if lbl == 'Learning')
    window._stack.setCurrentIndex(learn_idx)
    assert isinstance(window._stack.widget(learn_idx), LearningView)
    window._learning.refresh(None)                              # no crash, empty-states
    assert '0 runs logged' in window._learning._track_lbl.text() or 'runs logged' in window._learning._track_lbl.text()
