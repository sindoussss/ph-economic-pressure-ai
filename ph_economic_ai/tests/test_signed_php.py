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
