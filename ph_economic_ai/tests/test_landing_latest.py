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


def test_refresh_recent_picks_up_late_sector_data(app):
    # Reproduces the bug: gas saves first; food/electricity arrive later.
    # A public refresh must re-read the store and show the now-complete sectors.
    from ph_economic_ai.ui.landing import LandingPanel
    run = {'run_id': 6, 'timestamp': '2026-06-10T00:00:00+00:00',
           'final_estimate': -1.8, 'confidence_pct': 72,
           'food_estimate': None, 'electricity_estimate': None,
           'actual_price_change': None}
    panel = LandingPanel(store=_FakeStore([run]))
    panel.refresh_recent()                       # initial: sectors None -> em dash
    # sector debates finish later and write the row:
    run['food_estimate'] = -2.6
    run['electricity_estimate'] = 0.18
    panel.refresh_recent()                       # must re-read and show them
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert '2.60' in texts and '%' in texts       # food now shown
    assert '0.1800' in texts and 'kWh' in texts   # electricity now shown


def _run(rid, ts, gas, conf, food=None, elec=None):
    return {'run_id': rid, 'timestamp': ts, 'final_estimate': gas,
            'confidence_pct': conf, 'food_estimate': food,
            'electricity_estimate': elec, 'actual_price_change': None}


def test_latest_heading_shows_date_and_agreement(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18),
            _run(5, '2026-06-10T00:00:00+00:00', -1.5, 50),
            _run(4, '2026-06-09T00:00:00+00:00', -2.4, 54)]
    panel = LandingPanel(store=_FakeStore(runs))
    panel.refresh_recent()
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in texts
    assert '72% agreement' in texts
    assert 'Jun 10' in texts


def test_track_record_excludes_latest_run(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18),
            _run(5, '2026-06-10T00:00:00+00:00', -1.5, 50),
            _run(4, '2026-06-09T00:00:00+00:00', -2.4, 54)]
    panel = LandingPanel(store=_FakeStore(runs))
    panel.refresh_recent()
    labels = [l.text() for l in panel.findChildren(QLabel)]
    text = ' || '.join(labels)
    assert 'FUEL TRACK RECORD' in text
    assert any(t.startswith('#5') for t in labels)
    assert not any(t.startswith('#6') for t in labels)
    assert 'agreement' in text and 'confidence' not in text


def test_single_run_track_record_placeholder(app):
    from ph_economic_ai.ui.landing import LandingPanel
    panel = LandingPanel(store=_FakeStore([_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18)]))
    panel.refresh_recent()
    text = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in text
    assert 'No simulations on record yet.' in text
