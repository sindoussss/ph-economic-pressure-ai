import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


class _FakeStore:
    def __init__(self, runs):
        self._runs = runs

    def get_recent_runs(self, limit=20):
        return self._runs[:limit]


def test_landing_latest_shows_three_sectors(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [{'run_id': 4, 'timestamp': '2026-06-10T00:00:00+00:00',
             'final_estimate': -2.40, 'confidence_pct': 54,
             'food_estimate': 0.50, 'electricity_estimate': 0.05,
             'actual_price_change': None}]
    panel = LandingPanel(store=_FakeStore(runs))
    panel._refresh_recent_runs()
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in texts
    assert 'Gas / fuel' in texts and 'Food' in texts and 'Electricity' in texts
    assert '/L' in texts and '%' in texts and 'kWh' in texts


def test_landing_empty_store_no_crash(app):
    from ph_economic_ai.ui.landing import LandingPanel
    panel = LandingPanel(store=_FakeStore([]))
    panel._refresh_recent_runs()  # must not raise
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'No simulations on record yet.' in texts
