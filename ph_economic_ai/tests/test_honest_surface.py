import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.ui.honest_surface import (
    conformal_halfwidth, validated_summary_lines,
)

_FULL = {
    'headline_skill_vs_random_walk': -0.18,
    'conformal_widths': {'0.5': 2.4, '0.9': 10.42, '0.95': 16.0},
    'audit': [
        {'target': 'fuel', 'verdict': 'efficient'},
        {'target': 'fx', 'verdict': 'efficient'},
        {'target': 'inflation', 'verdict': 'efficient'},
    ],
    'mom_longsample': {'n_long': 143, 'mom': {'verdict': 'beats_best_naive',
                                              'best_method': 'arima'}},
}


def test_conformal_halfwidth_reads_level():
    assert conformal_halfwidth(_FULL, '0.9') == pytest.approx(10.42)
    assert conformal_halfwidth(_FULL, '0.5') == pytest.approx(2.4)


def test_conformal_halfwidth_missing_returns_none():
    assert conformal_halfwidth({'conformal_widths': {}}, '0.9') is None
    assert conformal_halfwidth(None) is None
    assert conformal_halfwidth({}, '0.9') is None


def test_summary_lines_full_report():
    lines = validated_summary_lines(_FULL)
    text = ' || '.join(lines)
    assert 'efficient' in text and 'random walk' in text
    assert '10.42' in text
    assert 'predictable' in text
    assert any('Methodology' in l for l in lines)


def test_summary_lines_none_report():
    lines = validated_summary_lines(None)
    assert len(lines) == 1
    assert 'benchmark.run' in lines[0]


def test_summary_lines_missing_keys_no_crash():
    lines = validated_summary_lines({'something': 1})
    assert isinstance(lines, list)
    assert any('Methodology' in l for l in lines)


from ph_economic_ai.ui.honest_surface import calibrated_interval_line


def test_calibrated_interval_line_full():
    line = calibrated_interval_line({'conformal_widths': {'0.9': 10.42}})
    assert line is not None
    assert '10.42' in line and 'calibrated' in line and '90%' in line


def test_calibrated_interval_line_custom_level():
    line = calibrated_interval_line({'conformal_widths': {'0.8': 5.0}}, level='0.8')
    assert '80%' in line and '5.00' in line


def test_calibrated_interval_line_missing_returns_none():
    assert calibrated_interval_line(None) is None
    assert calibrated_interval_line({'conformal_widths': {}}) is None
    assert calibrated_interval_line({}) is None
