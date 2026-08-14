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
        rounds=1, run_tournament=False, live=False)
    assert len(brief.readings) == 3
    assert {s.sector for s in outlook.sectors} == {'gas', 'food', 'electricity'}
    assert outlook.horizon == 'next month'


def test_panel_renders_without_thread(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    brief = PressureBrief(as_of='2026-07-24', window='this_week', readings=[
        SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1]),
        SectorReading('food', 'flat', None, '%', 0, drivers=[], sources=[]),
    ], narrative='Pressure rising.')
    panel._on_monitor_ready(brief)                     # must not raise
    outlook = Outlook(as_of='2026-07-24', sectors=[
        SectorOutlook('gas', 'efficient', 1.0, [-2.0, 4.0], '₱/L', 100, 'no exploitable edge'),
    ])
    panel._on_outlook_ready(outlook)                   # must not raise
    # Two sector cards plus the track-record strip above them. The strip is one
    # statement about the app, not three copies of it on the sector cards, where
    # repetition turned the honest sentence into wallpaper.
    assert panel._cards.count() == 3
    assert panel._outlook.count() == 1


def test_food_card_shows_subcategory_breakdown(app):
    """Rice/meat/fish/dairy&eggs/vegetables/sugar each get their own signed
    caption; a missing category reads as unavailable, never 0.0%; no value
    ever shows a literal '+' concatenated with a negative number (the exact
    RSK-053 shape, checked explicitly here because a new signed-percentage
    line is exactly where it would recur)."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from PyQt6.QtWidgets import QLabel
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('food', 'rising', 0.4, '%', 64,
                      estimates=[0.3, 0.4, 0.5],
                      subcategories={'rice': 0.1, 'meat': -0.3, 'fish': 0.8,
                                     'vegetables': 0.5})   # dairy_eggs, sugar absent
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'Rice +0.1%' in texts
    assert 'Meat -0.3%' in texts
    assert 'Fish +0.8%' in texts
    assert 'Vegetables +0.5%' in texts
    assert 'Dairy' in texts and '—' in texts   # missing category shows unavailable
    assert '+-' not in texts
    # A reader must not mistake the six category reads for components that
    # sum/average to the headline estimate above them -- the caption says so.
    assert 'not components of the figure above' in texts


def test_food_card_never_shows_peso_prices(app, monkeypatch):
    """The peso-anchor strip (₱ prices + projection) moved to the Report
    page's Food card -- Monitor's own Food card must never show one again,
    even when real subcategory percentages and a real peso_anchor price are
    both available, and even if something re-imports peso_anchor into this
    module by accident. Monitor keeps only the plain percentage breakdown."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.engine import peso_anchor
    from PyQt6.QtWidgets import QLabel

    monkeypatch.setattr(
        peso_anchor, 'get_anchor',
        lambda category, *a, **kw: {'price': 52.36, 'as_of': '2026-07',
                                    'fetched_on': '2026-08-13'})

    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('food', 'rising', 0.4, '%', 64,
                      estimates=[0.3, 0.4, 0.5],
                      subcategories={'rice': 0.3, 'meat': -0.3, 'fish': 0.8,
                                     'vegetables': 0.5})
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))

    assert '₱' not in texts
    assert 'exploratory projection' not in texts
    # The percentage breakdown line itself must still be there -- only the
    # peso strip moved, not the whole per-category feature.
    assert 'Rice +0.3%' in texts


def test_gas_card_is_unaffected_by_the_peso_strip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from PyQt6.QtWidgets import QLabel

    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.20, '₱/L', 70, estimates=[1.2])
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'exploratory projection' not in texts


def test_gas_card_has_no_subcategory_breakdown(app):
    """Gas/electricity cards are untouched by the food-only breakdown row --
    it must not appear at all, not even empty, for non-food sectors."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from PyQt6.QtWidgets import QLabel
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1])
    card = panel._sector_card(r)
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'Rice' not in texts and 'Meat' not in texts and 'Dairy' not in texts


def _find_label(card, substr):
    from PyQt6.QtWidgets import QLabel
    for w in card.findChildren(QLabel):
        if substr in w.text():
            return w
    return None


def test_range_provenance_label_carries_a_plain_language_tooltip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1])
    card = panel._sector_card(r)
    label = _find_label(card, 'range NOT calibrated')
    assert label is not None
    assert label.toolTip() == _honesty.band_provenance_tooltip(
        _find_band_for(panel, r))


def _find_band_for(panel, r):
    from ph_economic_ai.engine import interval as _interval
    return _interval.band(r.estimate, [], sector=r.sector)


def test_spread_label_carries_a_plain_language_tooltip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1])
    card = panel._sector_card(r)
    label = _find_label(card, 'agent estimates')
    assert label is not None
    assert label.toolTip() == _honesty.SPREAD_LINE_TOOLTIP


def test_direction_agreement_label_carries_a_plain_language_tooltip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1], direction_agreement=82)
    card = panel._sector_card(r)
    label = _find_label(card, 'not a probability')
    assert label is not None
    assert label.toolTip() == _honesty.DIRECTION_AGREEMENT_TOOLTIP


def test_anchor_record_label_carries_a_plain_language_tooltip_gas_only(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[0.9, 1.0, 1.1])
    card = panel._sector_card(r)
    anchor_text = _honesty.anchor_record_line()
    if not anchor_text:
        return  # artifact not present in this environment -- nothing to assert
    label = _find_label(card, anchor_text[:30])
    assert label is not None
    assert label.toolTip() == _honesty.ANCHOR_RECORD_TOOLTIP


def test_narrowed_room_caveat_carries_a_plain_language_tooltip(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = PressureMonitorPanel(FakeRag())
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                      drivers=['drives it'], sources=['RedditPH'],
                      estimates=[1.0, 1.0, 1.0, 1.0])   # 1 distinct value -- collapsed
    card = panel._sector_card(r)
    label = _find_label(card, 'room has narrowed')
    assert label is not None
    assert label.toolTip() == _honesty.AGREEMENT_CAVEAT_TOOLTIP


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
    panel._on_forum_event('judge', {'sector': 'gas', 'text': 'On balance, rising.',
                                    'estimate': 1.0, 'unit': '₱/L'})
    assert panel._typing.text() == ''                    # cleared once the message lands
    assert panel._feed_count() == 3                      # chat + moderator + judge cards


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


def test_live_social_refresh_is_graceful(tmp_path, monkeypatch):
    """Hybrid refresh must never raise and must degrade to the frozen snapshot when
    the live deps aren't present. Deps are stubbed absent so the test stays hermetic
    — a unit test must never depend on Google being reachable."""
    from ph_economic_ai.engine import live_social
    monkeypatch.setattr(live_social, '_have', lambda mod: False)
    status = live_social.refresh_social_snapshot(tmp_path)   # must not raise
    assert set(status) >= {'reddit', 'trends', 'live', 'notes'}
    assert status['live'] is False and status['trends'] == 0
    assert not list(tmp_path.glob('*.jsonl'))                # nothing written


def test_live_trends_is_cached_per_day(tmp_path, monkeypatch):
    """A Trends pull already made today is reused rather than re-fetched — Google
    hard-rate-limits anonymous calls, so re-fetching on every Run earns a 429."""
    import datetime as _dt
    from ph_economic_ai.engine import live_social
    (tmp_path / f'trends_{_dt.date.today().isoformat()}.jsonl').write_text(
        '{"date":"x","source":"GoogleTrends","title":"t"}\n', encoding='utf-8')

    def _boom(*a, **k):                    # any network attempt is a failure here
        raise AssertionError('re-fetched despite a same-day cache')
    monkeypatch.setattr(live_social, '_have', lambda mod: mod == 'pytrends')
    stub = type(sys)('pytrends.request')
    stub.TrendReq = _boom
    monkeypatch.setitem(sys.modules, 'pytrends.request', stub)
    status = live_social.refresh_social_snapshot(tmp_path)
    assert status['trends'] == 1 and status['live'] is True   # served from cache


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


# ── streamed cards (50-agent roster) ──────────────────────────────────────────

def test_agent_start_opens_a_live_card_before_any_text(app):
    """The card must exist before the first token, or a minutes-long run shows a
    frozen panel between turns."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    panel._on_forum_event('agent_start', {
        'name': 'Andrea Lim', 'occupation': 'Commuter Sentiment Analyst', 'sector': 'gas'})
    assert panel._feed_count() == 1
    assert panel._live_card is not None
    assert panel._live_card._body.text() == ''          # opened empty, not "(no reading)"


def test_tokens_accumulate_into_the_live_card(app):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    panel._on_forum_event('agent_start', {
        'name': 'Diego Ocampo', 'occupation': 'Crude & FX Trader', 'sector': 'gas'})
    for chunk in ('Pump prices ', 'are climbing ', 'on crude.'):
        panel._on_forum_event('agent_token', {'name': 'Diego Ocampo', 'sector': 'gas',
                                              'text': chunk})
    assert panel._live_card._body.text() == 'Pump prices are climbing on crude.'


def test_long_stream_is_trimmed_to_the_tail(app):
    """Tokens arrive faster than a human reads; the tail is what is being written."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    panel._on_forum_event('agent_start', {'name': 'A', 'occupation': 'x', 'sector': 'gas'})
    panel._on_forum_event('agent_token', {'name': 'A', 'sector': 'gas', 'text': 'y' * 900})
    shown = panel._live_card._body.text()
    assert len(shown) <= 321 and shown.startswith('…')


def test_agent_message_replaces_the_live_card_with_the_finished_one(app):
    """One card per turn — the streamed card is swapped, not left behind, so the feed
    does not end up with 100 cards for 50 agents."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    d = {'name': 'Andrea Lim', 'occupation': 'Commuter Sentiment Analyst', 'sector': 'gas'}
    panel._on_forum_event('agent_start', d)
    panel._on_forum_event('agent_token', {'name': 'Andrea Lim', 'sector': 'gas',
                                          'text': 'rising'})
    panel._on_forum_event('agent_message', dict(d, message='Pump prices are rising.',
                                                estimate=1.0, unit='PHP/L'))
    assert panel._feed_count() == 1                     # replaced, not appended
    assert panel._live_card is None


def test_a_dropped_agent_leaves_no_orphan_card(app):
    """If an agent produces nothing, the moderator/judge turn must still clear the
    open card rather than leaving a permanently blank one in the feed."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    panel._on_forum_event('agent_start', {'name': 'Ghost', 'occupation': 'x',
                                          'sector': 'gas'})
    panel._on_forum_event('moderator', {'sector': 'gas', 'text': 'Stay on the present read.'})
    assert panel._live_card is None
    assert panel._feed_count() == 1                     # only the moderator card


def test_tokens_before_any_card_are_ignored(app):
    """A stray token must not crash the panel — events can arrive out of order across
    the thread boundary."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel(FakeRag())
    panel._clear_feed()
    panel._on_forum_event('agent_token', {'name': 'nobody', 'sector': 'gas', 'text': 'hi'})
    assert panel._feed_count() == 0
