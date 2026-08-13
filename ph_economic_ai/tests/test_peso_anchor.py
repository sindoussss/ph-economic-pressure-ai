from datetime import date, timedelta

import pytest

from ph_economic_ai.engine import peso_anchor


def test_project_raises_the_price_for_a_positive_percent():
    assert peso_anchor.project(100.0, 5.0) == pytest.approx(105.0)


def test_project_lowers_the_price_for_a_negative_percent():
    """The sign has to actually flip the direction of the arithmetic, not
    just format with a minus -- this is real multiplication."""
    assert peso_anchor.project(100.0, -5.0) == pytest.approx(95.0)


def test_project_is_a_noop_at_zero_percent():
    assert peso_anchor.project(52.36, 0.0) == pytest.approx(52.36)


def test_load_cache_returns_empty_dict_when_file_is_missing(tmp_path):
    assert peso_anchor._load_cache(tmp_path / 'nope.json') == {}


def test_load_cache_returns_empty_dict_when_file_is_corrupt(tmp_path):
    p = tmp_path / 'bad.json'
    p.write_text('{not valid json', encoding='utf-8')
    assert peso_anchor._load_cache(p) == {}


def test_save_then_load_cache_round_trips(tmp_path):
    p = tmp_path / 'cache.json'
    peso_anchor._save_cache({'rice': {'price': 52.36}}, p)
    assert peso_anchor._load_cache(p) == {'rice': {'price': 52.36}}


def test_save_cache_creates_the_parent_directory_if_missing(tmp_path):
    p = tmp_path / 'nested' / 'cache.json'
    peso_anchor._save_cache({'rice': {'price': 52.36}}, p)
    assert peso_anchor._load_cache(p) == {'rice': {'price': 52.36}}


def test_is_fresh_true_for_an_entry_fetched_today():
    today = date(2026, 8, 13)
    entry = {'fetched_on': '2026-08-13'}
    assert peso_anchor._is_fresh(entry, today) is True


def test_is_fresh_false_for_an_entry_fetched_yesterday():
    today = date(2026, 8, 13)
    entry = {'fetched_on': '2026-08-12'}
    assert peso_anchor._is_fresh(entry, today) is False


def test_is_fresh_false_for_a_missing_fetched_on():
    assert peso_anchor._is_fresh({}, date(2026, 8, 13)) is False


def test_is_usable_if_stale_true_at_the_59_day_boundary():
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=59)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is True


def test_is_usable_if_stale_true_at_exactly_60_days():
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=60)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is True


def test_is_usable_if_stale_false_at_61_days():
    """The boundary itself, not just 'a long time ago' -- 61 days is where
    RSK-041's lesson actually bites: PSA's own ~30-day lag is well inside
    this, so the cutoff must not fire on normal reporting cadence."""
    today = date(2026, 8, 13)
    entry = {'fetched_on': (today - timedelta(days=61)).isoformat()}
    assert peso_anchor._is_usable_if_stale(entry, today) is False


def test_is_usable_if_stale_false_for_a_missing_fetched_on():
    assert peso_anchor._is_usable_if_stale({}, date(2026, 8, 13)) is False


def test_category_items_covers_exactly_the_four_confirmed_categories():
    assert set(peso_anchor.CATEGORY_ITEMS) == {'rice', 'meat', 'fish', 'vegetables'}
