"""Extractor tests for the sector estimates.

These parsers decide what number reaches the projected-CPI headline, and they
fail silently: a misparse produces a confident wrong figure, never an error.
The food extractor in particular feeds the 0.388 basket weight directly into
the BSP-target banner.
"""
import pytest

from ph_economic_ai.engine.debate import (
    _extract_percent, _extract_price, _extract_electricity_change,
    _MAX_REALISTIC_FOOD_PCT,
)


# ── The bug seen in a real run ────────────────────────────────────────────────

def test_yoy_citation_in_prose_does_not_beat_the_estimate_line():
    """The observed failure: an agent cites a year-on-year figure while
    reasoning, and a first-match scan returned the citation as the forecast.
    7.60% x 0.388 became +2.95ppt of projected CPI."""
    text = (
        'Food inflation ran 7.6% year-on-year per PSA, but base effects are '
        'fading, so I expect a much smaller monthly move.\n'
        'ESTIMATE: +0.4%'
    )
    assert _extract_percent(text) == pytest.approx(0.4)


def test_revised_estimate_wins_over_an_earlier_one():
    """Agents restate and revise; the final ESTIMATE line is the answer."""
    text = 'Initially ESTIMATE: +3.0%\nOn reflection that is too high.\nESTIMATE: +0.8%'
    assert _extract_percent(text) == pytest.approx(0.8)


def test_implausible_monthly_food_change_is_rejected():
    """A double-digit 'monthly' figure can only be a misparse — better to
    surface no estimate than a confident wrong one."""
    assert _extract_percent(f'ESTIMATE: +{_MAX_REALISTIC_FOOD_PCT + 5:.1f}%') is None


def test_plain_prose_still_parses_when_no_estimate_line():
    """Fallback must survive: not every agent emits the required line."""
    assert _extract_percent('I project +1.2% for the month.') == pytest.approx(1.2)


def test_negative_percent():
    assert _extract_percent('ESTIMATE: -0.6%') == pytest.approx(-0.6)


# ── Price / electricity ───────────────────────────────────────────────────────

def test_price_prefers_the_estimate_line_over_a_baseline_mention():
    text = 'Pump prices sit near a base of +₱60.00/L today.\nESTIMATE: +₱1.25/L'
    assert _extract_price(text) == pytest.approx(1.25)


def test_price_accepts_peso_written_as_php():
    assert _extract_price('ESTIMATE: +PHP 2.50/L') == pytest.approx(2.50)


def test_unsigned_baseline_is_not_mistaken_for_a_change():
    assert _extract_price('the rate is ₱14.33/kWh') is None


def test_electricity_rejects_an_absolute_rate_parsed_as_a_change():
    """₱14.33 is the Meralco base rate, not a monthly change — accepting it
    would put a ~100x error into the electricity CPI contribution."""
    assert _extract_electricity_change('ESTIMATE: +₱14.33/kWh') is None


def test_electricity_accepts_a_realistic_change():
    assert _extract_electricity_change('ESTIMATE: +₱0.45/kWh') == pytest.approx(0.45)


def test_no_estimate_anywhere_returns_none():
    for fn in (_extract_percent, _extract_price, _extract_electricity_change):
        assert fn('The outlook is broadly stable.') is None
