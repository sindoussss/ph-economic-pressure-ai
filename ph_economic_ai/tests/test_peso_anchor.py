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


import json as jsonlib
from unittest.mock import MagicMock


def test_fetch_live_returns_none_for_a_category_not_in_scope():
    assert peso_anchor._fetch_live('sugar') is None


def _fake_meta_response(commodity_label):
    r = MagicMock()
    r.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['0'], 'valueTexts': ['Philippines']},
            {'code': 'Commodity', 'values': ['5'], 'valueTexts': [commodity_label]},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': ['2018', '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    return r


def _fake_post_response(rows):
    r = MagicMock()
    r.content = jsonlib.dumps({'data': rows}).encode('utf-8')
    r.raise_for_status = lambda: None
    return r


def test_fetch_live_parses_a_successful_response(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    # key = [Geolocation, Commodity, Year, Period] value-ids;
    # year id '1' -> 2019 (index 1), period id '1' -> February.
    post = _fake_post_response([{'key': ['0', '5', '1', '1'], 'values': ['55.41']}])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


def test_fetch_live_sends_the_hardcoded_national_geolocation_id(monkeypatch):
    """Geolocation must be the hardcoded national id '0' -- matching the
    established psa_cpi.py::_fetch_px_table precedent for this exact field
    -- not resolved positionally from the metadata's values list. This
    table family carries ~119 regional/provincial Geolocation entries
    alongside the national one, so the metadata here deliberately lists a
    non-national entry FIRST: if the query body were still built from
    geo_var['values'][0], this test would catch the silent substitution by
    asserting on the actual POST body sent to PSA."""
    meta = MagicMock()
    meta.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['13700000000', '0'],
             'valueTexts': ['National Capital Region', 'Philippines']},
            {'code': 'Commodity', 'values': ['5'],
             'valueTexts': ['RICE, REGULAR-MILLED, 1 KG']},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': ['2018', '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    post = _fake_post_response([{'key': ['0', '5', '1', '1'], 'values': ['55.41']}])
    captured = {}

    def fake_post(url, json=None, **kw):
        captured['json'] = json
        return post

    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', fake_post)

    peso_anchor._fetch_live('rice')

    geo_query = next(q for q in captured['json']['query'] if q['code'] == 'Geolocation')
    assert geo_query == {'code': 'Geolocation', 'selection': {'filter': 'item', 'values': ['0']}}


def test_fetch_live_picks_the_most_recent_row_when_several_exist(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '0', '0'], 'values': ['50.00']},   # 2018-01
        {'key': ['0', '5', '1', '1'], 'values': ['55.41']},   # 2019-02, latest
        {'key': ['0', '5', '0', '11'], 'values': ['51.00']},  # not present here, illustrative
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


def test_fetch_live_skips_annual_and_blank_rows(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '1', '12'], 'values': ['999.00']},  # Annual, must be skipped
        {'key': ['0', '5', '0', '0'], 'values': ['..']},       # blank, must be skipped
        {'key': ['0', '5', '0', '1'], 'values': ['48.20']},    # 2018-02, the only real row
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 48.20, 'as_of': '2018-02'}


def test_fetch_live_returns_none_when_the_network_call_raises(monkeypatch):
    def raise_error(*a, **kw):
        raise peso_anchor.requests.RequestException('boom')
    monkeypatch.setattr(peso_anchor.requests, 'get', raise_error)
    assert peso_anchor._fetch_live('rice') is None


def test_fetch_live_returns_none_when_the_commodity_label_is_not_found(monkeypatch):
    meta = _fake_meta_response('SOME OTHER ITEM, 1 KG')
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    assert peso_anchor._fetch_live('rice') is None


def test_fetch_live_returns_none_when_the_response_is_malformed(monkeypatch):
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    # A data row with a key shorter than the expected 4 entries: indexing
    # row['key'][2] / row['key'][3] would raise IndexError if that
    # exception weren't caught alongside the others.
    post = _fake_post_response([{'key': ['0', '5'], 'values': ['55.41']}])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') is None


def test_fetch_live_returns_none_when_a_commodity_label_is_none(monkeypatch):
    """A real PX-Web response shape: an untranslated/missing label comes
    back as None in valueTexts, not an empty string. The None entry is
    positioned before the real match so the loop's `txt.strip()` actually
    reaches it -- if AttributeError weren't caught alongside the others,
    this would crash instead of returning None."""
    r = MagicMock()
    r.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['0'], 'valueTexts': ['Philippines']},
            {'code': 'Commodity', 'values': ['4', '5'],
             'valueTexts': [None, 'RICE, REGULAR-MILLED, 1 KG']},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': ['2018', '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: r)

    assert peso_anchor._fetch_live('rice') is None


def test_get_anchor_returns_none_for_a_category_not_in_scope():
    assert peso_anchor.get_anchor('sugar') is None


def test_get_anchor_uses_a_fresh_same_day_cache_without_fetching(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    peso_anchor._save_cache(
        {'rice': {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'}},
        cache_path)

    def fail_if_called(category):
        raise AssertionError('should not fetch when cache is fresh')
    monkeypatch.setattr(peso_anchor, '_fetch_live', fail_if_called)

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'}


def test_get_anchor_fetches_and_caches_when_todays_entry_is_missing(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    monkeypatch.setattr(peso_anchor, '_fetch_live',
                        lambda category: {'price': 52.90, 'as_of': '2026-08'})

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 52.90, 'as_of': '2026-08', 'fetched_on': '2026-08-13'}
    assert peso_anchor._load_cache(cache_path)['rice']['price'] == 52.90


def test_get_anchor_falls_back_to_a_usable_stale_cache_when_the_fetch_fails(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    peso_anchor._save_cache(
        {'rice': {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}},
        cache_path)
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)

    result = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert result == {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}


def test_get_anchor_returns_none_when_fetch_fails_and_cache_is_too_stale(tmp_path, monkeypatch):
    from datetime import timedelta
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    too_old = (today - timedelta(days=61)).isoformat()
    peso_anchor._save_cache(
        {'rice': {'price': 45.00, 'as_of': '2026-06', 'fetched_on': too_old}}, cache_path)
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)

    assert peso_anchor.get_anchor('rice', cache_path=cache_path, today=today) is None


def test_get_anchor_returns_none_when_fetch_fails_and_no_cache_exists(tmp_path, monkeypatch):
    cache_path = tmp_path / 'nope.json'
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)
    assert peso_anchor.get_anchor('rice', cache_path=cache_path, today=date(2026, 8, 13)) is None
