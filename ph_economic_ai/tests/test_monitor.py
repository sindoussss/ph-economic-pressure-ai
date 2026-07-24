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


def test_panel_shows_live_forum_cards(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()                                  # drop the initial hint
    panel._on_forum_event('agent_start', {
        'name': 'Andrea Lim', 'occupation': 'Commuter Sentiment Analyst', 'sector': 'gas'})
    assert 'Andrea Lim' in panel._typing.text()
    panel._on_forum_event('agent_message', {
        'name': 'Andrea Lim', 'occupation': 'Commuter Sentiment Analyst', 'sector': 'gas',
        'message': 'Pump prices are climbing. ESTIMATE: +1.00/L', 'estimate': 1.0, 'unit': '₱/L'})
    panel._on_forum_event('moderator', {'sector': 'gas', 'text': 'Stay on the present read.'})
    assert panel._typing.text() == ''                    # cleared once the message lands
    assert panel._feed_count() == 2                      # one chat card + one moderator card


def test_forum_personas_have_names():
    from ph_economic_ai.engine.forum import _capability_agents
    agents = _capability_agents('gas')
    names = {a.name for a in agents}
    assert 'Andrea Lim' in names and 'Diego Ocampo' in names
    assert all(a.role for a in agents)                   # every agent has an occupation


def test_forum_graph_grows_with_turns():
    from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
    from ph_economic_ai.engine.kg_forum_adapter import add_forum_turn
    b = KnowledgeGraphBuilder()
    add_forum_turn(b, 'Andrea Lim', 'Commuter Sentiment Analyst', 'gas', 1.0, 'rising')
    add_forum_turn(b, 'Diego Ocampo', 'Crude & FX Trader', 'gas', 0.9, 'up')
    nodes, edges = b.snapshot()
    ids = {n.id for n in nodes}
    kinds = {n.kind for n in nodes}
    assert {'master', 'agent', 'claim'} <= kinds         # sector hub + agents + claims
    assert 'sector:gas' in ids and 'agent:Andrea Lim' in ids
    assert sum(1 for n in nodes if n.kind == 'master') == 1   # two agents, one shared hub
    assert sum(1 for n in nodes if n.kind == 'agent') == 2


def test_forum_graph_includes_rag_sources():
    from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
    from ph_economic_ai.engine.kg_forum_adapter import add_forum_turn
    b = KnowledgeGraphBuilder()
    add_forum_turn(b, 'Andrea Lim', 'Analyst', 'gas', 1.0, 'x',
                   sources=['RedditPH', 'GoogleTrends'])
    nodes, edges = b.snapshot()
    ids = {n.id for n in nodes}
    assert 'src:RedditPH' in ids and 'src:GoogleTrends' in ids
    assert any(n.id == 'src:RedditPH' and n.label == 'Reddit' for n in nodes)  # short label
    assert any(e.src == 'agent:Andrea Lim' and e.dst == 'src:RedditPH'
               and e.kind == 'retrieved' for e in edges)


def test_panel_updates_debate_graph(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
    panel = PressureMonitorPanel(FakeRag())
    panel._kg_builder = KnowledgeGraphBuilder()
    panel._on_forum_event('agent_message', {
        'name': 'Andrea Lim', 'occupation': 'Commuter Sentiment Analyst', 'sector': 'gas',
        'message': 'rising', 'estimate': 1.0, 'unit': '₱/L'})
    nodes, _ = panel._kg_builder.snapshot()
    assert any(n.id == 'agent:Andrea Lim' for n in nodes)
    assert not panel._kg.isHidden()      # un-hidden once the first turn lands


def test_forum_graph_canvas_draws(app):
    from ph_economic_ai.ui.forum_graph import ForumGraphCanvas
    from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
    from ph_economic_ai.engine.kg_forum_adapter import add_forum_turn
    b = KnowledgeGraphBuilder()
    add_forum_turn(b, 'Andrea Lim', 'Commuter Sentiment Analyst', 'gas', 1.0, 'rising')
    add_forum_turn(b, 'Diego Ocampo', 'Crude & FX Trader', 'gas', 0.9, 'up')
    canvas = ForumGraphCanvas()
    canvas.set_snapshot(*b.snapshot())
    assert canvas.node_item_count() > 0        # ellipses, edges, labels all drawn


def test_seed_sectors_creates_hubs():
    from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
    from ph_economic_ai.engine.kg_forum_adapter import seed_sectors
    b = KnowledgeGraphBuilder()
    seed_sectors(b, ('gas', 'food', 'electricity'))
    nodes, _ = b.snapshot()
    ids = {n.id for n in nodes}
    assert {'sector:gas', 'sector:food', 'sector:electricity'} <= ids
    assert all(n.kind == 'master' for n in nodes)


def test_placeholder_card_builds(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    assert panel._placeholder_card('gas') is not None      # dashed 'analysing…' card


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
