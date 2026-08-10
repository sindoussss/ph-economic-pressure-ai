import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _mv(**overrides):
    from types import SimpleNamespace as NS
    base = dict(final_estimate=-0.08, confidence_pct=70, dissenting_regions=[],
               regional_verdicts=[], physical_anchor=None)
    base.update(overrides)
    return NS(**base)


def test_every_honesty_caveat_survives_the_restyle(app):
    """Pins every caveat string _build_swarm_left can render, so a future
    styling change can't silently drop one -- the exact regression shape
    this project's own audits keep finding (RSK-038 and its recurrences)."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'agreement_n': 3, 'agreement_distinct': 2, 'agreement_regions': (2, 2),
        'agreement_echo_n': 1, 'agreement_diversity': 0.3,
        'agreement_models': {}, 'unscored_regions': 1, 'outside_regional': True,
    }
    panel._build_swarm_left(_mv(confidence_pct=100), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    joined = ' '.join(texts)
    assert '100% agent agreement' in joined
    assert 'outside the regional range above' in joined
    assert any('unscored' in t.lower() or 'tie-break' in t.lower() for t in texts)


def test_narrow_room_caveat_visible_when_collapsed(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'agreement_n': 3, 'agreement_distinct': 1, 'agreement_regions': (2, 2),
        'agreement_echo_n': 0, 'agreement_diversity': 0.0,
        'agreement_models': {}, 'unscored_regions': 0,
    }
    panel._build_swarm_left(_mv(confidence_pct=100), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    expected_caveat = _honesty.agreement_caveat(3, 1, 0.0)
    assert expected_caveat in texts


def test_regional_table_keeps_every_value_and_the_missing_estimate_note(app):
    """Pins the Regional Verdicts restructure specifically: name, estimate
    (or 'discarded'), agreement%, and the missing-estimate honesty note all
    have to survive moving from one-box-per-region to a compact table."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel, _missing_estimate_note
    panel = Stage4ReportPanel()
    regions = [
        NS(region_pair=('NCR', 'Central Luzon'), estimate=-0.07, confidence=0.63,
           rejected_estimate=None),
        NS(region_pair=('Western Visayas', 'Davao'), estimate=None, confidence=0.0,
           rejected_estimate=99.0),
    ]
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 70, 'low': -0.10, 'high': -0.07,
        'agreement_n': 2, 'agreement_distinct': 2, 'agreement_regions': (2, 2),
        'agreement_echo_n': 0, 'agreement_diversity': 0.5,
        'agreement_models': {}, 'unscored_regions': 0,
    }
    panel._build_swarm_left(_mv(confidence_pct=70, regional_verdicts=regions), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    assert any('NCR & Central Luzon' in t for t in texts)
    assert any('-0.07' in t for t in texts)
    assert any('63%' in t for t in texts)
    assert any('discarded' in t for t in texts)
    expected_note = _missing_estimate_note(None, 99.0)
    assert any(expected_note in t for t in texts)
