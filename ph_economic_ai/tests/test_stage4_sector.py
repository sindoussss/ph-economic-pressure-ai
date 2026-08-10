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


def test_sector_forecasts_show_agreement_for_all_three(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10,
                               gas_agreement=70, food_agreement=62, elec_agreement=81)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert any('70%' in t for t in labels)
    assert any('62%' in t for t in labels)
    assert any('81%' in t for t in labels)


def test_sector_forecasts_omits_agreement_when_zero(app):
    """Old callers/tests still calling positionally-only must not crash, and a
    0 agreement (the pre-Task-4 default, or a genuinely unmeasured run) must
    not print a nonsense '0% agreement' caption."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert not any('% agreement' in t for t in labels)


def test_calling_set_sector_forecasts_twice_does_not_leak_the_old_cards(app):
    """Re-rendering (main_window.py calls this from 5 sites) must actually
    remove the previous cards, not just lose track of them.

    deleteLater() only *schedules* destruction; the QObject stays a live
    child (and shows up in findChildren) until the event loop actually
    processes the deferred-delete event. Qt only posts that event once the
    widget's parent-of-parents chain has been asked, so we flush it
    explicitly rather than relying on an implicit event-loop spin.
    """
    from PyQt6.QtCore import QCoreApplication, QEvent
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10, gas_agreement=70)
    first_children = set(panel._sector_holder.findChildren(QLabel))
    panel.set_sector_forecasts(0.15, 0.30, 0.05, gas_agreement=40)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    second_children = set(panel._sector_holder.findChildren(QLabel))
    assert not (first_children & second_children)
    assert any('40%' in c.text() for c in second_children)
    assert not any('70%' in c.text() for c in second_children)
