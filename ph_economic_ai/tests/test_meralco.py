"""Meralco's generation charge, read from data instead of hardcoded.

`anchoring._GEN_CHARGE_PHP_KWH` was frozen at 5.50 while the actual charge had
reached 9.2504 by 2026-07 -- 41% low, multiplied straight through every
electricity estimate. A constant that must be *remembered* is a constant that
goes stale, so the level is now read from a committed series and the fallback
exists only for when that series cannot be loaded at all.

Meralco publishes one "Generation" PDF per month and blocks automated fetches
(HTTP 403), so this cannot be a network fetcher. It is the next best thing: a
parser that turns the published PDF into a row, so adding a month is a command
rather than an edit.
"""
import pytest

from ph_economic_ai.benchmark import meralco


def _write(tmp_path, body):
    p = tmp_path / 'gc.csv'
    p.write_text(body, encoding='utf-8')
    return p


# ── the series ───────────────────────────────────────────────────────────────

def test_loads_the_charge_by_month(tmp_path):
    p = _write(tmp_path, 'date,generation_charge_php_kwh\n'
                         '2026-05,8.7942\n2026-06,9.0704\n2026-07,9.2504\n')
    s = meralco.load_generation_charge(p)
    assert list(s.index) == ['2026-05', '2026-06', '2026-07']
    assert s['2026-07'] == pytest.approx(9.2504)


def test_month_over_month_change_is_in_php_per_kwh(tmp_path):
    """The quantity the app estimates: a PHP/kWh change, not a percentage."""
    p = _write(tmp_path, 'date,generation_charge_php_kwh\n'
                         '2026-05,8.7942\n2026-06,9.0704\n2026-07,9.2504\n')
    mom = meralco.load_generation_charge_mom(p)
    assert mom['2026-06'] == pytest.approx(0.2762)
    assert mom['2026-07'] == pytest.approx(0.1800)
    assert '2026-05' not in mom.index, 'the first month has no prior to difference'


def test_latest_is_the_newest_published_level(tmp_path):
    p = _write(tmp_path, 'date,generation_charge_php_kwh\n'
                         '2026-05,8.7942\n2026-07,9.2504\n2026-06,9.0704\n')
    assert meralco.latest_generation_charge(p) == pytest.approx(9.2504)


def test_a_missing_series_falls_back_rather_than_crashing(tmp_path):
    """The anchor may not take the app down because a data file is absent."""
    got = meralco.latest_generation_charge(tmp_path / 'nope.csv', default=7.77)
    assert got == pytest.approx(7.77)


def test_the_shipped_series_is_present_and_plausible():
    s = meralco.load_generation_charge()
    assert len(s) >= 3
    assert all(4.0 < v < 20.0 for v in s), 'PHP/kWh, not centavos or a percentage'
    assert meralco.latest_generation_charge() > 8.0, 'the 5.50 era is long past'


# ── the parser: adding a month is a command, not an edit ─────────────────────

_PDF_TEXT = """BREAKDOWN OF GENERATION CHARGE
1. First Gas Power Corporation (FGPC) 13.2% 474,982,370 5,377,773,226 11.1827
TOTAL 100.0% 3,613,373,791 32,277,086,927 9.2656
Other Generation Adjustments (OGA)
1. Pilferage Recovery (0.0197)
2. ILP Recovery 0.0000
3. High Load Factor Rider 0.0001
4. TOU Differential 0.0045
9.2504
*Includes adjustment to align the total energy with metered energy inputs
JULY 2026 GENERATION CHARGE
TOTAL
"""


def test_parser_takes_the_adjusted_headline_not_the_subtotal():
    """9.2656 is the pre-adjustment subtotal; 9.2504 is the charge actually
    billed, after Other Generation Adjustments. Taking the wrong one is a
    plausible, silent error."""
    assert meralco.parse_generation_charge_text(_PDF_TEXT) == pytest.approx(9.2504)


def test_parser_handles_a_longer_adjustment_block():
    """August 2026 carried an extra OGA line (GOUR Recovery), so the headline
    cannot be found by counting lines from the total."""
    text = _PDF_TEXT.replace('4. TOU Differential 0.0045',
                             '4. TOU Differential 0.0065\n5. GOUR Recovery 0.0701')
    text = text.replace('\n9.2504\n', '\n9.2800\n')
    assert meralco.parse_generation_charge_text(text) == pytest.approx(9.2800)


def test_parser_refuses_text_it_does_not_understand():
    """Silently returning a wrong number here would poison the anchor."""
    with pytest.raises(ValueError):
        meralco.parse_generation_charge_text('not a rate table at all')


def test_parser_reads_the_month_from_the_heading():
    assert meralco.parse_generation_charge_month(_PDF_TEXT) == '2026-07'
