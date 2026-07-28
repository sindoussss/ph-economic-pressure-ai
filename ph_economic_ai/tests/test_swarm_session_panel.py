"""The SWARM SESSION panel must report the run, not a literal.

Observed live: the header read `ALIVE 7 / ELIM 13` while the panel beside it read
`AGENTS 20/20 alive` and `PHASE group_arena`. All four panel rows and the session
id were hardcoded strings in the painter, so they never changed for any run.
"""
import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication  # noqa: E402

from ph_economic_ai.ui import stage3_swarm_canvas as sc  # noqa: E402


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(app):
    return sc.Stage3SwarmPanel()


def test_total_is_derived_from_the_roster_not_hardcoded():
    from ph_economic_ai.engine.swarm import REGIONS, _ROLE_ORDER
    assert sc.TOTAL_AGENTS == len(REGIONS) * len(_ROLE_ORDER)


def test_panel_starts_consistent_with_the_header(panel):
    assert panel._canvas._session['alive'] == sc.TOTAL_AGENTS
    assert panel._canvas._session['total'] == sc.TOTAL_AGENTS
    assert int(panel._alive_val.text()) == panel._canvas._session['alive']


def test_eliminations_move_both_the_header_and_the_panel(panel):
    """The exact contradiction from the screenshot: 13 eliminations."""
    for i in range(13):
        panel._on_group_eliminated(0, f'NCR Agent{i}', 0.5, 1)

    assert panel._alive_val.text() == '7'
    assert panel._elim_val.text() == '13'
    assert panel._canvas._session['alive'] == 7, (
        'the session panel still disagrees with the header')


def test_the_two_counters_never_diverge_across_a_whole_run(panel):
    for i in range(sc.TOTAL_AGENTS - 1):
        panel._on_group_eliminated(0, f'A{i}', 0.4, 1)
        assert int(panel._alive_val.text()) == panel._canvas._session['alive']


def test_phase_reaches_the_panel(panel):
    panel._set_phase(1, 'Group arena')
    assert panel._canvas._session['phase'] == 'group_arena'
    panel._set_phase(2, 'Regional judges')
    assert panel._canvas._session['phase'] == 'regional_judges'
    panel._set_phase(3, 'Master verdict')
    assert panel._canvas._session['phase'] == 'master_verdict'


def test_alive_never_goes_negative(panel):
    for i in range(sc.TOTAL_AGENTS + 5):
        panel._on_group_eliminated(0, f'A{i}', 0.4, 1)
    assert panel._canvas._session['alive'] >= 0
    assert int(panel._alive_val.text()) >= 0


def test_reset_restores_a_full_consistent_roster(panel):
    for i in range(5):
        panel._on_group_eliminated(0, f'A{i}', 0.4, 1)
    panel.reset()
    assert panel._canvas._session['alive'] == sc.TOTAL_AGENTS
    assert int(panel._alive_val.text()) == sc.TOTAL_AGENTS
    assert panel._canvas._session['session_id'] == 'session_pending'


def test_set_session_rejects_an_unknown_field(panel):
    """A typo must fail loudly rather than write a key the painter never reads."""
    with pytest.raises(KeyError):
        panel._canvas.set_session(aliv=3)


def test_set_session_updates_only_what_it_is_given(panel):
    before = dict(panel._canvas._session)
    panel._canvas.set_session(phase='master_verdict')
    after = panel._canvas._session
    assert after['phase'] == 'master_verdict'
    assert after['alive'] == before['alive']
    assert after['session_id'] == before['session_id']
