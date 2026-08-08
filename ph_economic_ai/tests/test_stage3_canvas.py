import sys
from types import SimpleNamespace as NS

import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _estimate_label_text(row) -> str:
    """The one QLabel in a _LogRow that carries the peso estimate."""
    for lbl in row.findChildren(QLabel):
        if lbl.text().startswith('₱') or lbl.text() == '—':
            return lbl.text()
    raise AssertionError('no estimate label found in _LogRow')


def test_log_row_falling_estimate_does_not_double_sign(app):
    """RSK-031 lineage: a literal '+' concatenated before a signed number reads
    '+₱-3.53' for any falling forecast. `{:+.2f}` is the fix used everywhere
    else in the app; this call site never got it."""
    from ph_economic_ai.ui.stage3_canvas import _LogRow
    resp = NS(agent_name='Market Analyst', round_num=1, price_estimate=-3.53,
              statement='easing')
    row = _LogRow(resp, '#6366F1')
    text = _estimate_label_text(row)
    assert text == '₱-3.53'
    assert '+₱-' not in text


def test_log_row_rising_estimate_keeps_the_plus(app):
    from ph_economic_ai.ui.stage3_canvas import _LogRow
    resp = NS(agent_name='Market Analyst', round_num=1, price_estimate=2.10,
              statement='rising')
    row = _LogRow(resp, '#6366F1')
    text = _estimate_label_text(row)
    assert text == '₱+2.10'


def test_log_row_no_estimate_shows_dash(app):
    from ph_economic_ai.ui.stage3_canvas import _LogRow
    resp = NS(agent_name='Market Analyst', round_num=1, price_estimate=None,
              statement='no read')
    row = _LogRow(resp, '#6366F1')
    assert _estimate_label_text(row) == '—'
