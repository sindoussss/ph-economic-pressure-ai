import sys
import pytest
from PyQt6.QtWidgets import QApplication

@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)

def test_stage3_swarm_panel_builds(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    assert hasattr(panel, '_canvas')
    assert hasattr(panel, '_log')

def test_reset_clears_state(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    panel._groups_done = 5
    panel.reset()
    assert panel._groups_done == 0

def test_swarm_complete_emits_signal(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.swarm import MasterVerdict
    received = []
    panel = Stage3SwarmPanel()
    panel.swarm_complete.connect(lambda mv: received.append(mv))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80,
                       dissenting_regions=[], reasoning='test', regional_verdicts=[])
    panel._on_swarm_complete(mv)
    assert len(received) == 1
    assert received[0].confidence_pct == 80


def test_live_graph_grows_and_view_report(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    p = Stage3SwarmPanel()
    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None): return [{'source': 'DOE', 'text': 'x'}]
    p._begin_live_graph(_Rag(), {'current_price': 60.0}, {})
    assert not p._canvas.isHidden()                        # old structured arena is the live view
    p._on_group_round_done(0, 1, [NS(agent_name='FCST', statement='s', price_estimate=-1.8)])
    p._flush_kg()
    assert p._kg_canvas.node_item_count() > 0              # graph grew
    fired = []
    p.view_report_requested.connect(lambda: fired.append(True))
    p._on_swarm_complete(NS(final_estimate=-1.8, confidence_pct=80, regional_verdicts=[],
                            dissenting_regions=[], all_responses=[]))
    assert not p._toast.isHidden()                         # completion toast slid in
    p._toast_btn.click()                                   # its "View report →" button
    assert fired == [True]


def test_swarm_complete_flags_a_collapsed_room(app):
    """RSK-038 follow-up: this canvas label stays on screen after the run
    completes and showed a bare 'N% confidence' with none of the collapse
    signal the Stage 4 report card already gives the same number."""
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.swarm import MasterVerdict
    panel = Stage3SwarmPanel()
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=100,
                       dissenting_regions=[], reasoning='test', regional_verdicts=[],
                       agreement_n=3, agreement_distinct=2)
    panel._on_swarm_complete(mv)
    assert 'narrow room' in panel._gas_sub.text()
    assert 'narrow room' in panel._toast_sub.text()


def test_swarm_complete_silent_on_a_spread_room(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.swarm import MasterVerdict
    panel = Stage3SwarmPanel()
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=60,
                       dissenting_regions=[], reasoning='test', regional_verdicts=[],
                       agreement_n=4, agreement_distinct=4)
    panel._on_swarm_complete(mv)
    assert 'narrow room' not in panel._gas_sub.text()


def test_food_consensus_flags_a_collapsed_room(app):
    """Same RSK-038 gap as the gas label: 'consensus reached' claimed
    agreement with no distinct-value check behind it."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    responses = [NS(price_estimate=0.30), NS(price_estimate=0.30), NS(price_estimate=0.31)]
    panel._on_food_canvas_complete(responses)
    assert 'narrow room' in panel._food_sub.text()


def test_food_consensus_silent_on_a_spread_room(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    responses = [NS(price_estimate=0.10), NS(price_estimate=0.30), NS(price_estimate=0.50)]
    panel._on_food_canvas_complete(responses)
    assert panel._food_sub.text() == '100% agreement'
    assert 'narrow room' not in panel._food_sub.text()


def test_food_consensus_shows_the_calibrated_agreement_percentage(app):
    """The live tile used to show a bare 'consensus reached' -- the calibrated
    number `DebateEngine.consensus()` already computes (RSK-042's band) sat
    one call away, unused. This pins the tile to that same computation."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.debate import DebateEngine
    panel = Stage3SwarmPanel()
    estimates = [0.10, 0.30, 0.50, 0.35]
    responses = [NS(price_estimate=e) for e in estimates]
    panel._on_food_canvas_complete(responses)

    engine = DebateEngine.__new__(DebateEngine)
    engine._sector = 'food'
    engine._history = [NS(round_num=1, price_estimate=e, agent_name='a', statement='')
                        for e in estimates]
    expected_pct = engine.consensus()['confidence_pct']
    assert panel._food_sub.text().startswith(f'{expected_pct}% agreement')


def test_elec_consensus_flags_a_collapsed_room(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    responses = [NS(price_estimate=0.03), NS(price_estimate=0.03), NS(price_estimate=0.04)]
    panel._on_elec_canvas_complete(responses)
    assert 'narrow room' in panel._elec_sub.text()


def test_elec_consensus_silent_on_a_spread_room(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    responses = [NS(price_estimate=0.01), NS(price_estimate=0.03), NS(price_estimate=0.07)]
    panel._on_elec_canvas_complete(responses)
    assert panel._elec_sub.text() == '100% agreement'


def test_elec_consensus_shows_the_calibrated_agreement_percentage(app):
    """Same pin as the food test above, for electricity's own (tighter, 0.10)
    band."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    from ph_economic_ai.engine.debate import DebateEngine
    panel = Stage3SwarmPanel()
    estimates = [0.01, 0.03, 0.07, 0.09]
    responses = [NS(price_estimate=e) for e in estimates]
    panel._on_elec_canvas_complete(responses)

    engine = DebateEngine.__new__(DebateEngine)
    engine._sector = 'electricity'
    engine._history = [NS(round_num=1, price_estimate=e, agent_name='a', statement='')
                        for e in estimates]
    expected_pct = engine.consensus()['confidence_pct']
    assert panel._elec_sub.text().startswith(f'{expected_pct}% agreement')


def test_graph_metrics_reflect_the_real_scene_not_a_hardcoded_literal(app):
    """NODES/EDGES/density/avg_deg used to be literal strings ('37', '110+',
    '0.087', '4.21') that never changed no matter what was actually built."""
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    canvas = panel._canvas
    gm = canvas._graph_metrics()
    expected_nodes = (len(canvas._agents) + len(canvas._regionals) + 1
                      + len(canvas._rag_nodes) + len(canvas._sector_agents) + 2)
    assert gm['nodes'] == expected_nodes
    assert gm['nodes'] != 37                    # the old hardcoded literal
    assert gm['edges'] > 0
    assert gm['edges'] != 110                    # the old hardcoded literal ('110+')
    assert 0 < gm['density'] <= 1
    assert gm['avg_deg'] == pytest.approx(2 * gm['edges'] / gm['nodes'])
    assert gm['clusters'] == 6


def test_console_header_matches_the_real_session_id(app):
    """The console header showed a fixed 'swarm_session_001' forever, right
    next to the SWARM SESSION card's real per-run generated id."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    assert panel._console_id.text() == 'swarm_session_001'
    thread = NS(_rag=None, _scenario={'current_price': 60.0},
                agent_typing=NS(connect=lambda *a: None),
                agent_done_typing=NS(connect=lambda *a: None),
                group_round_done=NS(connect=lambda *a: None),
                group_eliminated=NS(connect=lambda *a: None),
                group_survivor=NS(connect=lambda *a: None),
                regional_done=NS(connect=lambda *a: None),
                swarm_complete=NS(connect=lambda *a: None))
    panel.connect_thread(thread)
    assert panel._console_id.text() != 'swarm_session_001'
    assert panel._console_id.text() == panel._canvas._session['session_id']


def test_connect_thread_populates_evidence(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel, _EvidenceNode
    p = Stage3SwarmPanel()

    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None):
            return [{'source': 'DOE', 'text': f'c{i}'} for i in range(top_k)]

    thread = NS(_rag=_Rag(), _scenario={'current_price': 60.0},
                agent_typing=NS(connect=lambda *a: None),
                agent_done_typing=NS(connect=lambda *a: None),
                group_round_done=NS(connect=lambda *a: None),
                group_eliminated=NS(connect=lambda *a: None),
                group_survivor=NS(connect=lambda *a: None),
                regional_done=NS(connect=lambda *a: None),
                swarm_complete=NS(connect=lambda *a: None))
    p.connect_thread(thread)
    ev = [it for it in p._canvas._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev) > 0                                  # canvas populated from run start


# ── a sign that lies, RSK-031 lineage never reached this file ────────────────

def test_swarm_complete_signs_a_falling_gas_estimate_correctly(app):
    """A literal '+' concatenated before the value produced '+₱-3.53/L' for
    every falling swarm-mode forecast on the sidebar's main GAS figure --
    the same bug RSK-031 already fixed in swarm.py's prompt and the Agent
    Performance table, never reaching this file at all."""
    from ph_economic_ai.engine.swarm import MasterVerdict
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    mv = MasterVerdict(final_estimate=-3.53, confidence_pct=70,
                       dissenting_regions=[], reasoning='test', regional_verdicts=[])
    panel._on_swarm_complete(mv)
    assert panel._gas_val.text() == '₱-3.53/L'
    assert '+₱-' not in panel._gas_val.text()


def test_regional_done_log_signs_a_falling_estimate_correctly(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    verdict = NS(estimate=-1.20, region_pair=('NCR', 'Central Luzon'))
    panel._on_regional_done(0, verdict)
    log_text = panel._console.toPlainText()
    assert '-1.20/L' in log_text
    assert '+-' not in log_text


def test_elec_agent_done_log_signs_a_falling_estimate_correctly(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    resp = NS(agent_name='Grid Analyst', price_estimate=-0.0250, statement='easing')
    panel._on_elec_agent_done(resp)
    log_text = panel._console.toPlainText()
    assert '-0.0250/kWh' in log_text
    assert '+₱-' not in log_text


def test_elec_canvas_complete_signs_a_falling_average_correctly(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    panel = Stage3SwarmPanel()
    responses = [NS(price_estimate=-0.02), NS(price_estimate=-0.04)]
    panel._on_elec_canvas_complete(responses)
    assert panel._elec_val.text() == '₱-0.0300/kWh'
    assert '+₱-' not in panel._elec_val.text()
