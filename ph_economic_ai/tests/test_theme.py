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


def test_confidence_bar(app):
    from ph_economic_ai.ui import theme
    bar = theme.confidence_bar(0.3, 0.2)
    assert isinstance(bar, QFrame)
    fills = [c for c in bar.findChildren(QFrame)]
    assert len(fills) == 1
    assert bar.height() == 5


def test_confidence_bar_positioning(app):
    """Verify that confidence_bar's _position() correctly computes fill geometry."""
    from ph_economic_ai.ui import theme
    bar = theme.confidence_bar(0.3, 0.2)
    # Set fixed width and show the widget to trigger resizeEvent -> _position()
    bar.setFixedWidth(200)
    bar.show()
    # Verify the fill widget's geometry: x = int(200*0.3)=60, width = int(200*0.2)=40
    fill = [c for c in bar.findChildren(QFrame)][0]
    assert fill.geometry().x() == 60
    assert fill.geometry().y() == 0
    assert fill.geometry().width() == 40
    assert fill.geometry().height() == 5


def test_warning_token():
    from ph_economic_ai.ui import theme
    assert theme.WARNING == '#B45309'


def test_stat_card_full(app):
    from ph_economic_ai.ui import theme
    frame, layout = theme.stat_card(
        'GAS / FUEL', '-P0.08/L', color=theme.DOWN,
        meta='70% agent agreement', tag_kind='exploratory',
        confidence_frac=(0.38, 0.24))
    labels = [c.text() for c in frame.findChildren(QLabel)]
    assert 'GAS / FUEL' in labels
    assert '-P0.08/L' in labels
    assert '70% agent agreement' in labels
    assert 'exploratory' in labels
    assert len(frame.findChildren(QFrame)) >= 1  # the confidence bar's track


def test_stat_card_without_confidence_or_tag(app):
    from ph_economic_ai.ui import theme
    frame, layout = theme.stat_card('FOOD', '-0.21%', tag_kind=None)
    labels = [c.text() for c in frame.findChildren(QLabel)]
    assert 'FOOD' in labels
    assert '-0.21%' in labels
    assert 'exploratory' not in labels and 'validated' not in labels
