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


def test_sector_holder_items_are_all_widgets_so_the_clear_loop_can_remove_them(app):
    """Pins the actual mechanism the widget-leak fix depends on, not just its
    downstream symptom.

    set_sector_forecasts()'s own clear loop only calls .deleteLater() on
    it.widget() results:
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
    A QLayoutItem for a bare addLayout()'d sub-layout always returns None
    from .widget() -- so if the sector-card row were ever added via
    addLayout() again instead of wrapped in a QWidget + addWidget(), this
    clear loop would silently stop removing it, orphaning every previous
    render's cards. Checking that every item in the layout is a widget item
    pins the fix directly.

    An earlier version of this test instead forced Qt's deferred-delete
    queue to run early (QCoreApplication.sendPostedEvents(None,
    QEvent.Type.DeferredDelete) + processEvents()) to observe the actual
    destruction. That proved unstable on this project's CI -- a segfault in
    PyQt6/Qt internals under the Linux/offscreen/xdist-parallel runner,
    non-deterministic (passed once, crashed the next run), not reproducible
    locally. This version tests the same regression without touching Qt's
    event-processing internals at all.
    """
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10, gas_agreement=70)
    layout = panel._sector_holder_layout
    assert layout.count() > 0
    for i in range(layout.count()):
        item = layout.itemAt(i)
        assert item.widget() is not None, (
            f"item {i} in _sector_holder_layout has no widget() -- it was "
            "added via addLayout(), which the clear loop's deleteLater() "
            "cleanup can never reach, silently orphaning it on the next render")


def test_calling_set_sector_forecasts_twice_shows_the_latest_values(app):
    """Re-rendering (main_window.py calls this from 5 sites) must not raise,
    and the new render's values must appear.

    Does not assert the first render's labels are absent: deleteLater()
    only schedules destruction, so the old QLabels are still findable until
    the event loop actually processes it -- checking that would need the
    same forced event-flush this file deliberately avoids (see the test
    above). The structural test above is what actually pins the leak fix;
    this one just confirms a second call renders correctly and doesn't
    raise.
    """
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10, gas_agreement=70)
    panel.set_sector_forecasts(0.15, 0.30, 0.05, gas_agreement=40)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert any('40%' in t for t in labels)


def test_food_card_shows_peso_anchor_strip(app, monkeypatch):
    """Report's Food card gets the peso-anchor strip that used to live on
    Monitor -- driven by food_subcategories (this run's own debate output,
    from DebateEngine.consensus()'s new 'subcategories' key), not Monitor's
    separate ForumEngine run."""
    from ph_economic_ai.ui import stage4_report
    from ph_economic_ai.engine import peso_anchor

    def fake_get_anchor(category, *a, **kw):
        prices = {
            'rice': {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
            'meat': {'price': 185.40, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
            'fish': None,
            'vegetables': {'price': 62.10, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
        }
        return prices.get(category)
    monkeypatch.setattr(peso_anchor, 'get_anchor', fake_get_anchor)

    panel = stage4_report.Stage4ReportPanel()
    panel.set_sector_forecasts(
        -0.08, 0.20, -0.10,
        food_subcategories={'rice': 0.3, 'meat': -0.3, 'fish': 0.8, 'vegetables': 0.5})
    texts = ' || '.join(l.text() for l in panel._sector_holder.findChildren(QLabel))

    assert 'Rice ₱52.36 → ₱52.52' in texts
    assert 'Meat ₱185.40 → ₱184.84' in texts
    assert 'Fish —' in texts
    assert 'Vegetables ₱62.10 → ₱62.41' in texts
    assert 'exploratory projection, not a validated prediction' in texts


def test_food_card_omits_the_peso_strip_when_no_food_subcategories(app):
    """Backward-compatible default: callers that never pass
    food_subcategories (or pass {}) get no peso strip -- no crash, and
    nothing implying data that was never provided."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, 0.20, -0.10)
    texts = ' || '.join(l.text() for l in panel._sector_holder.findChildren(QLabel))
    assert 'exploratory projection' not in texts


def test_gas_and_electricity_cards_never_show_a_peso_strip(app, monkeypatch):
    """food_subcategories only ever anchors the Food card -- gas/electricity
    must not pick up a stray peso strip even though they're built in the
    same loop."""
    from ph_economic_ai.ui import stage4_report
    from ph_economic_ai.engine import peso_anchor
    monkeypatch.setattr(
        peso_anchor, 'get_anchor',
        lambda category, *a, **kw: {'price': 52.36, 'as_of': '2026-07',
                                    'fetched_on': '2026-08-13'})
    panel = stage4_report.Stage4ReportPanel()
    panel.set_sector_forecasts(
        -0.08, 0.20, -0.10, food_subcategories={'rice': 0.3})
    # Exactly one peso strip -- the food card's -- not one per sector card.
    # (Gas/electricity's own value labels legitimately contain '₱' too, as
    # part of their ₱/L, ₱/kWh units, so that substring alone can't be the
    # check here -- the strip's own caption text is unique to it.)
    all_texts = [l.text() for l in panel._sector_holder.findChildren(QLabel)]
    assert sum('exploratory projection' in t for t in all_texts) == 1
