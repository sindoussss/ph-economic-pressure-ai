import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication

from ph_economic_ai.engine import llm as llm_mod
from ph_economic_ai.engine.monitor import run_pressure_monitor
from ph_economic_ai.engine.pressure_brief import PressureBrief, SectorReading
from ph_economic_ai.engine.outlook import Outlook, SectorOutlook


class FakeRag:
    def add_text(self, source, text, url=''):
        return 1

    def query(self, text, top_k=5, sources=None):
        return []


def _fake_complete(messages, tier=None, max_tokens=None, **kw):
    text = ' '.join(m.get('content', '') for m in messages)
    if '/kWh' in text:
        est = 'ESTIMATE: +₱0.30/kWh'
    elif '/L' in text:
        est = 'ESTIMATE: +₱1.00/L'
    elif '%' in text:
        est = 'ESTIMATE: +0.5%'
    else:
        est = ''
    return 'Rising now. CAUSAL CHAIN: a -> b -> c. ' + est


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_run_pressure_monitor_composes(monkeypatch, tmp_path):
    """Headless Stage 1 -> Stage 2 without Qt or a live LLM."""
    monkeypatch.setattr(llm_mod, 'complete', _fake_complete)
    brief, outlook = run_pressure_monitor(
        FakeRag(), corpus_dir=tmp_path / 'empty', as_of=date(2026, 7, 24),
        rounds=1, run_tournament=False)
    assert len(brief.readings) == 3
    assert {s.sector for s in outlook.sectors} == {'gas', 'food', 'electricity'}
    assert outlook.horizon == 'next month'


def test_panel_renders_without_thread(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    brief = PressureBrief(as_of='2026-07-24', window='this_week', readings=[
        SectorReading('gas', 'rising', 1.0, '₱/L', 100, ['drives it'], ['RedditPH']),
        SectorReading('food', 'flat', None, '%', 0, [], []),
    ], narrative='Pressure rising.')
    panel._on_monitor_ready(brief)                     # must not raise
    outlook = Outlook(as_of='2026-07-24', sectors=[
        SectorOutlook('gas', 'efficient', 1.0, [-2.0, 4.0], '₱/L', 100, 'no exploitable edge'),
    ])
    panel._on_outlook_ready(outlook)                   # must not raise
    assert panel._cards.count() == 2
    assert panel._outlook.count() == 1


def test_main_window_has_monitor_tab(app):
    from ph_economic_ai.ui.main_window import SimMainWindow
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    df = pd.DataFrame({
        'date': pd.date_range('2024-01', periods=3, freq='M'),
        'gas_price': [58.0, 59.0, 60.0], 'oil_price': [80.0, 81.0, 82.0],
        'usd_php': [56.0, 56.5, 57.0], 'cpi': [120.0, 121.0, 122.0],
        'remittances': [2.5, 2.6, 2.7], 'demand_index': [70.0, 71.0, 72.0],
    })
    reg = MagicMock()
    reg.predict.return_value = np.array([60.0])
    reg.feature_importances_ = np.array([0.5, 0.3, 0.2])
    win = SimMainWindow(df, reg)
    try:
        assert isinstance(win._monitor, PressureMonitorPanel)
        assert win._stack.widget(7) is win._monitor      # stack index 7
        win._on_stage_changed(7)
        assert win._stack.currentIndex() == 7            # nav routes to it
    finally:
        win.close()
