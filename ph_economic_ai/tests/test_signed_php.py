"""The peso formatter must not render a fall as '+₱-0.25'."""
import pytest

from ph_economic_ai.ui.stage4_report import _signed_php


def test_positive_gets_a_plus():
    assert _signed_php(0.35) == '+₱0.35'


def test_negative_gets_a_minus_not_plus_minus():
    assert _signed_php(-0.25) == '-₱0.25'
    assert '+₱-' not in _signed_php(-0.25)


def test_zero_reads_as_plus_zero():
    assert _signed_php(0.0) == '+₱0.00'


def test_suffix_is_appended():
    assert _signed_php(-1.2, '/L') == '-₱1.20/L'


def test_pdf_export_uses_the_fixed_formatter_not_a_bare_literal():
    """The PDF export path built its own 'Consensus: +{avg:.2f}/L' line,
    bypassing _signed_php entirely -- a falling forecast saved to a file the
    user downloads and keeps would have read 'Consensus: +-3.53/L'."""
    import inspect
    from ph_economic_ai.ui import stage4_report
    src = inspect.getsource(stage4_report.Stage4ReportPanel._on_export)
    assert "f'Consensus: +{avg:.2f}/L'" not in src
    assert '_signed_php(avg)' in src
