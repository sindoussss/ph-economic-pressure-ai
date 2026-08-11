import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_set_alert_shows_plain_header_no_colored_box(app):
    from ph_economic_ai.ui.causal_chain_widget import BSPAlertBanner
    banner = BSPAlertBanner()
    banner.set_alert({
        'severity': 'CRITICAL', 'projected_cpi': 6.03, 'current_cpi': 6.2,
        'cpi_as_of': 'PSA, Jul 2026', 'sector_cpi_impact': -0.17,
        'breakdown': {'fuel': -0.01, 'food': -0.08, 'electricity': -0.07},
    })
    assert not banner.isHidden()
    labels = [c.text() for c in banner.findChildren(QLabel)]
    assert any('6.03%' in t for t in labels)
    assert any('CRITICALLY EXCEEDED' in t.upper() for t in labels)
    # No QFrame in the banner (besides the banner itself) should carry a
    # colored background/border fill -- the boxed-alert pattern is gone.
    style = banner.styleSheet()
    assert 'background' not in style or '#FEF2F2' not in style


def test_set_alert_sector_breakdown_has_one_sign_not_two(app):
    """RSK-031's shape a ninth time: 'Fuel: +{value:.2f}ppt' hardcoded a literal
    '+' in front of a value that is negative when a sector is falling, producing
    'Fuel: +-0.01ppt' -- the same double-sign bug already fixed eight times
    elsewhere in this codebase, in a location none of those fixes reached."""
    from ph_economic_ai.ui.causal_chain_widget import BSPAlertBanner
    banner = BSPAlertBanner()
    banner.set_alert({
        'severity': 'CRITICAL', 'projected_cpi': 6.03, 'current_cpi': 6.2,
        'cpi_as_of': 'PSA, Jul 2026', 'sector_cpi_impact': -0.17,
        'breakdown': {'fuel': -0.01, 'food': -0.08, 'electricity': 0.07},
    })
    labels = [c.text() for c in banner.findChildren(QLabel)]
    combined = ' || '.join(labels)
    assert '+-' not in combined
    assert 'Fuel: -0.01ppt' in combined
    assert 'Food: -0.08ppt' in combined
    assert 'Elec: +0.07ppt' in combined


def test_set_alert_stable_severity_still_shows(app):
    from ph_economic_ai.ui.causal_chain_widget import BSPAlertBanner
    banner = BSPAlertBanner()
    banner.set_alert({
        'severity': 'STABLE', 'projected_cpi': 3.1, 'current_cpi': 3.0,
        'cpi_as_of': 'PSA, Jul 2026', 'sector_cpi_impact': 0.1, 'breakdown': {},
    })
    assert not banner.isHidden()
    labels = [c.text() for c in banner.findChildren(QLabel)]
    assert any('3.10%' in t for t in labels)
