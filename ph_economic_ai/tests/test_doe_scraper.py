import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.benchmark.doe_scraper import parse_doe_prices, to_monthly

# Representative DOE-style markup (synthetic test input — NOT committed data).
_SAMPLE_HTML = """
<table>
  <tr><th>Date</th><th>Product</th><th>Price</th></tr>
  <tr><td>2026-05-06</td><td>Gasoline (RON 95)</td><td>62.50</td></tr>
  <tr><td>2026-05-13</td><td>Gasoline (RON 95)</td><td>63.10</td></tr>
  <tr><td>2026-06-03</td><td>Gasoline (RON 95)</td><td>64.05</td></tr>
</table>
"""


def test_parser_extracts_dated_prices():
    prices = parse_doe_prices(_SAMPLE_HTML)
    assert len(prices) >= 2
    for date_str, val in prices.items():
        assert len(date_str) == 10 and date_str[4] == '-'   # YYYY-MM-DD
        assert 20.0 < val < 120.0                            # sane PHP/liter
    assert prices['2026-05-06'] == pytest.approx(62.50)


def test_parser_returns_empty_on_no_rows():
    # Live legacy DOE page serves prices via linked bulletins, not inline HTML.
    assert parse_doe_prices('<html><body>no price table here</body></html>') == {}


def test_to_monthly_averages_within_month():
    daily = {'2026-05-06': 62.5, '2026-05-13': 63.1, '2026-06-03': 64.0}
    monthly = to_monthly(daily)
    assert monthly['2026-05'] == pytest.approx((62.5 + 63.1) / 2)
    assert monthly['2026-06'] == pytest.approx(64.0)


def test_parse_then_monthly_end_to_end():
    monthly = to_monthly(parse_doe_prices(_SAMPLE_HTML))
    assert monthly['2026-05'] == pytest.approx((62.50 + 63.10) / 2)
    assert monthly['2026-06'] == pytest.approx(64.05)
