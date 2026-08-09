import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_set_sector_forecasts_renders_card(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-2.40, 0.50, 0.05)
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    # Sector labels now render through theme.eyebrow(), which uppercases
    # (card grid instead of the old mixed-case row list) -- compare
    # case-insensitively; the check itself (label + unit present) is unchanged.
    upper = texts.upper()
    assert 'SECTOR FORECAST' in upper
    assert 'GAS / FUEL' in upper and '/L' in texts
    assert 'FOOD' in upper and '%' in texts
    assert 'ELECTRICITY' in upper and 'kWh' in texts
    assert 'exploratory' in texts.lower()


def test_sector_card_renders_bars(app):
    from PyQt6.QtWidgets import QFrame, QLabel
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p.set_sector_forecasts(-1.8, -2.6, 0.18)
    texts = ' || '.join(l.text() for l in p.findChildren(QLabel))
    assert '1.80' in texts and '2.60' in texts and '0.1800' in texts
    assert len(p._sector_holder.findChildren(QFrame)) >= 3   # a bar track per sector
