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
    win = SimMainWindow(df, reg)
    yield win
    # __init__ starts a real DOECheckerThread (main_window.py); Qt's own
    # contract is that destroying a QThread while it's still running is
    # undefined behavior (see closeEvent's own comment). .close() triggers
    # closeEvent(), which stops and joins it -- the same cleanup
    # test_monitor.py's inline try/finally already does for its one window.
    # Without this, every test using this fixture leaks a running thread for
    # the rest of the process's life (RSK-056).
    win.close()


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


class _FakeDebateEngine:
    """Stands in for DebateEngine.consensus() without running any model calls."""
    def __init__(self, consensus_dict):
        self._c = consensus_dict

    def consensus(self):
        return self._c


def test_format_gas_verdict_names_distinct_values_when_estimates_given():
    """Same honesty treatment as food/electricity: given the raw agent
    estimates, a bare 'agent agreement N%' becomes a checkable distinct-value
    and span breakdown, with the collapse caveat when the room has narrowed.
    This text feeds an LLM prompt directly and, via stage5_interact, the
    user's own screen, so it gets the same treatment on all three sectors."""
    from ph_economic_ai.ui.main_window import _format_gas_verdict
    v = _format_gas_verdict(estimate=-0.5, agreement_pct=100,
                            estimates=[-0.5, -0.5, -0.49])
    assert '2 distinct values' in v
    assert 'agent agreement 100%' not in v
    assert 'room has narrowed' in v


def test_format_gas_verdict_falls_back_to_bare_percentage_without_estimates():
    """Backward compatible: a caller with only the percentage (no raw
    estimates available) still gets a verdict string, not a crash."""
    from ph_economic_ai.ui.main_window import _format_gas_verdict
    v = _format_gas_verdict(estimate=-0.5, agreement_pct=84, estimates=None)
    assert 'agent agreement 84%' in v


def test_food_verdict_names_distinct_values_not_bare_percentage(window):
    """Food's confidence has none of swarm.py's collapse/echo detection
    (ADR-009), so the verdict string must lead with the distinct-value count
    and spread (`honesty.spread_line`), the same honesty treatment fuel's
    card already gets, rather than a bare 'confidence N%' a reader cannot
    check against anything."""
    window._last_scenario = {'oil_pct': 1.0, 'usd_pct': 0.0}
    window._food_engine = _FakeDebateEngine({
        'weighted_avg': 0.3, 'confidence_pct': 100,
        'verdicts': [
            {'agent': 'A', 'estimate': 0.30, 'statement': 'x'},
            {'agent': 'B', 'estimate': 0.30, 'statement': 'x'},
            {'agent': 'C', 'estimate': 0.31, 'statement': 'x'},
        ],
    })
    window._on_food_complete([1, 2, 3])
    assert '2 distinct values' in window._food_verdict
    assert 'confidence 100%' not in window._food_verdict
    # Collapsed room (2 distinct among 3): the same caveat fuel's card shows.
    assert 'room has narrowed' in window._food_verdict


def test_food_verdict_on_failed_debate_falls_back_honestly(window):
    """No responses at all must still produce a grounded, honest verdict
    (the persistence anchor), not a crash and not a bare 'confidence 0%'."""
    window._last_scenario = {}
    window._food_engine = None
    window._on_food_complete([])
    assert 'no agent produced a usable estimate' in window._food_verdict


def test_electricity_verdict_names_distinct_values(window):
    window._last_scenario = {'oil_pct': 1.0, 'usd_pct': 0.0}
    window._elec_engine = _FakeDebateEngine({
        'weighted_avg': 0.05, 'confidence_pct': 62,
        'verdicts': [
            {'agent': 'A', 'estimate': 0.03, 'statement': 'x'},
            {'agent': 'B', 'estimate': 0.07, 'statement': 'x'},
        ],
    })
    window._on_elec_complete([1, 2])
    assert '2 distinct values' in window._elec_verdict
    assert 'confidence 62%' not in window._elec_verdict
    # `agreement_caveat`'s own distinct-count rounding (2dp) must match what
    # spread_line just printed for the same estimates -- the two must never
    # tell a reader a different distinct count for one set of numbers.
    import re
    m = re.search(r'(\d+) distinct values', window._elec_verdict)
    assert m and f'only {m.group(1)} distinct value' in window._elec_verdict


def test_electricity_anchor_not_claimed_as_validated_predictor(window):
    """Regression guard for the withdrawn-finding overclaim this fix removed:
    the electricity anchor must never again be described as a 'validated
    signal' or 'genuinely predictive' -- both manuscript S5.7 and
    anchoring.electricity_passthrough_anchor's own docstring say the
    opposite (a magnitude guard only, corr ~0.03-0.13)."""
    import inspect
    from ph_economic_ai.ui import main_window as mw
    src = inspect.getsource(mw.SimMainWindow._on_elec_complete)
    assert 'genuinely predictive' not in src
    assert 'validated signal' not in src


def test_close_waits_for_a_still_running_kg_worker(window):
    """Qt's contract: destroying a QThread while it's still running is
    undefined behavior (some builds hard-abort the process). closeEvent
    already joins _doe_checker this way; _kg_worker (entity enrichment,
    spawned by stage3_swarm_canvas.py after a swarm run) had no such guard,
    so closing the app moments after a run finished could hit that abort
    mid-enrichment."""
    from unittest.mock import MagicMock
    from PyQt6.QtGui import QCloseEvent
    fake_worker = MagicMock()
    fake_worker.isRunning.return_value = True
    window._stage3_swarm._kg_worker = fake_worker
    window.closeEvent(QCloseEvent())
    fake_worker.wait.assert_called_once()


def test_close_is_a_noop_when_no_kg_worker_has_run_yet(window):
    from PyQt6.QtGui import QCloseEvent
    assert not hasattr(window._stage3_swarm, '_kg_worker')
    window.closeEvent(QCloseEvent())   # must not raise


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


def test_push_sector_forecasts_passes_agreement_values(window):
    window._gas_estimate, window._food_estimate, window._elec_estimate = -0.08, -0.21, -0.10
    window._gas_agreement, window._food_agreement, window._elec_agreement = 70, 62, 81
    captured = {}
    window._stage4.set_sector_forecasts = lambda *args, **kw: captured.update(kw)
    window._push_sector_forecasts()
    assert captured['gas_agreement'] == 70
    assert captured['food_agreement'] == 62
    assert captured['elec_agreement'] == 81


def test_the_window_fixture_leaves_no_running_doe_checker_behind(window):
    """Every earlier test in this file that used `window` got its own
    SimMainWindow, which starts a real DOECheckerThread (a native QThread) at
    construction. Qt's own contract: destroying a QThread while it's still
    running is undefined behavior (RSK-040's closeEvent comment). If the
    fixture doesn't stop each window's thread in teardown, they pile up for
    the rest of the process's life -- confirmed via CI's own crash dump for
    RSK-056: 11 DOECheckerThreads found alive simultaneously, all idle in
    ground_truth.py's `_stop_event.wait()`, in a worker process that had run
    this file's tests. Placed last so the whole file's fixture-teardown
    history is checked, not just one instance."""
    import gc
    from ph_economic_ai.engine.ground_truth import DOECheckerThread
    gc.collect()
    # `window` itself is still running here -- its own teardown hasn't fired
    # yet (that happens after this test function returns) -- so it's excluded.
    leaked = [t for t in gc.get_objects()
             if isinstance(t, DOECheckerThread) and t is not window._doe_checker
             and t.isRunning()]
    assert leaked == [], (
        f'{len(leaked)} DOECheckerThread(s) from earlier tests in this file '
        'are still running -- the window fixture must stop each one in '
        'teardown, the same way test_monitor.py\'s inline try/finally: '
        'win.close() already does'
    )
