import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QFrame


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_tokens():
    from ph_economic_ai.ui import theme
    for c in (theme.SURFACE, theme.CARD, theme.INK, theme.MUTED, theme.FAINT,
              theme.HAIRLINE, theme.UP, theme.DOWN, theme.NEUTRAL):
        assert isinstance(c, str) and c.startswith('#')
    assert theme.direction_color('up') == theme.UP
    assert theme.direction_color('down') == theme.DOWN
    assert theme.direction_color('na') == theme.FAINT


def test_helpers(app):
    from ph_economic_ai.ui import theme
    assert isinstance(theme.eyebrow('hi'), QLabel) and theme.eyebrow('hi').text() == 'HI'
    assert theme.serif_number('1.8').text() == '1.8'
    assert isinstance(theme.muted('x'), QLabel)
    assert isinstance(theme.hairline(), QFrame)
    frame, layout = theme.card('Title')
    assert isinstance(frame, QFrame)
    assert 'TITLE' in [c.text() for c in frame.findChildren(QLabel)]
    assert theme.tag('validated').text() == 'validated'
