"""The Monitor's analysts must not invent the numbers they quote.

A live run produced this, from an electricity coal-import analyst:

    "The peso's current exchange rate is around 1 USD = 50.5 PHP, which is a
     0.5% decline from last week's rate."

The true rate at that moment was 61.61, held in the corpus with thirty days of
history. The analyst was out by 18 percent and invented a week-on-week change,
stated more precisely than the honest answer would have been.

The cause was not the model. `SECTOR_SOURCES` gave the electricity market lane
`['YahooFinanceCrude', 'EIAElectricity']` and no FX feed, so that analyst could
not retrieve the peso rate however hard he tried — while
`anchoring.electricity_passthrough_anchor(oil_pct, usd_pct)` says the exchange
rate is half of what drives his sector. The source map contradicted the physics,
and the model filled the gap from memory.
"""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from ph_economic_ai.engine.auto_assemble import SECTOR_SOURCES, sector_corpus


# ── Every market lane can see the exchange rate ───────────────────────────────

@pytest.mark.parametrize('sector', ['gas', 'food', 'electricity'])
def test_every_market_lane_can_read_fx(sector):
    """Each sector's anchor is a function of the exchange rate, so each sector's
    market analyst has to be able to look it up."""
    assert 'YahooFinanceForex' in SECTOR_SOURCES[sector]['market'], (
        f'{sector} market analysts cannot source the peso rate they will discuss')


def test_electricity_keeps_its_own_feeds_too():
    """Adding FX must not displace the lane's existing evidence."""
    market = SECTOR_SOURCES['electricity']['market']
    assert 'EIAElectricity' in market
    assert 'YahooFinanceCrude' in market


def test_the_channels_remain_distinct():
    """The Forum's design is evidence TYPED, not shared: a social agent handed
    market data cites BusinessWorld for a mood reading. FX joins the market lane
    only."""
    for sector in ('gas', 'food', 'electricity'):
        assert 'YahooFinanceForex' not in SECTOR_SOURCES[sector]['social']
        assert 'YahooFinanceForex' not in SECTOR_SOURCES[sector]['news']


@pytest.mark.parametrize('sector', ['gas', 'food', 'electricity'])
def test_fx_reaches_the_sector_corpus(sector):
    assert 'YahooFinanceForex' in sector_corpus(sector)


# ── And the prompt forbids filling gaps from memory ───────────────────────────

def _agent_prompt_text():
    """The user-side prompt one Forum agent receives."""
    from unittest.mock import MagicMock

    from ph_economic_ai.engine import forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    f = forum.Forum.__new__(forum.Forum)
    f._as_of = '2026-07-28'
    f._window = 'this_week'
    f._rag = MagicMock()
    f._rag.query.return_value = []
    f._rag_text = lambda agent, query: ('no context', [])

    ctx = SectorContext(sector='electricity', unit='PHP/kWh',
                        verdict_note='note', social_counts={'today': 1})
    agent = MagicMock()
    agent.system_prompt = 'you are an analyst'
    agent.rag_sources = ['YahooFinanceCrude']
    msgs, _ = f._agent_prompt(agent, ctx, history=[], steer='')
    return msgs[1]['content']


def test_the_prompt_forbids_recalling_a_figure_from_memory():
    text = _agent_prompt_text()
    assert 'NUMBERS RULE' in text
    assert 'ONLY if it appears in your retrieved context' in text


def test_the_prompt_offers_an_honest_alternative():
    """A rule that only forbids invites evasion. Saying "no reading in my
    channel" has to be an explicitly acceptable answer."""
    text = _agent_prompt_text()
    assert 'no current reading in my channel' in text
    assert 'valid and useful' in text


# ── The card shows the whole statement ────────────────────────────────────────

@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _card(message):
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel
    panel = PressureMonitorPanel.__new__(PressureMonitorPanel)
    return PressureMonitorPanel._chat_card(panel, {
        'name': 'Tonio Sarmiento', 'occupation': 'Coal Import Analyst',
        'sector': 'electricity', 'message': message,
        'estimate': 0.25, 'unit': 'PHP/kWh',
    })


LONG = ('The peso depreciation against the US dollar is putting upward pressure '
        'on imported coal costs, which feed the generation charge. ') * 6


def test_a_long_statement_is_previewed_not_destroyed(qapp):
    """The regression: the tail used to be cut off before reaching the widget, so
    it could not be read at all, however the user tried."""
    from ph_economic_ai.ui.pressure_monitor import _PREVIEW_CHARS
    card = _card(LONG)
    assert len(card._body.text()) <= _PREVIEW_CHARS + 1     # +1 for the ellipsis
    assert card._full == ' '.join(LONG.split())


def test_clicking_reveals_every_word(qapp):
    card = _card(LONG)
    card.toggle()
    assert card._body.text() == ' '.join(LONG.split())


def test_clicking_again_collapses(qapp):
    from ph_economic_ai.ui.pressure_monitor import _PREVIEW_CHARS
    card = _card(LONG)
    card.toggle()
    card.toggle()
    assert len(card._body.text()) <= _PREVIEW_CHARS + 1


def test_the_affordance_says_how_much_is_hidden(qapp):
    card = _card(LONG)
    assert 'Show more' in card._hint.text()
    assert 'more characters' in card._hint.text()
    assert not card._hint.isHidden()


def test_a_short_statement_offers_no_affordance(qapp):
    """Nothing is hidden, so nothing should suggest otherwise."""
    card = _card('Short reading, nothing more to say.')
    assert card._hint.isHidden()
    assert card._body.text() == 'Short reading, nothing more to say.'


def test_a_short_statement_cannot_be_toggled(qapp):
    card = _card('Short reading.')
    card.toggle()
    assert card._body.text() == 'Short reading.'


def test_an_empty_statement_still_renders(qapp):
    card = _card('')
    assert card._body.text() == '(no reading)'
    assert card._hint.isHidden()
