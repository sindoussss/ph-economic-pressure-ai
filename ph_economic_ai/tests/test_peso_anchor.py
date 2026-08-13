from datetime import date, datetime, timedelta

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
    """Geolocation must be the hardcoded national id '000000000' -- the
    real PSGC-style id PSA's own 2M/2018NEW/ retail-commodity tables use
    for "Philippines" (confirmed against the live API: this table family
    carries ~119 regional/provincial Geolocation entries, each keyed by a
    9-digit PSGC code, not a small sequential index) -- not resolved
    positionally from the metadata's values list. The metadata here
    deliberately lists a non-national entry FIRST: if the query body were
    still built from geo_var['values'][0], this test would catch the
    silent substitution by asserting on the actual POST body sent to PSA."""
    meta = MagicMock()
    meta.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['130000000', '000000000'],
             'valueTexts': ['National Capital Region', 'Philippines']},
            {'code': 'Commodity', 'values': ['5'],
             'valueTexts': ['RICE, REGULAR-MILLED, 1 KG']},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': ['2018', '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    post = _fake_post_response([{'key': ['000000000', '5', '1', '1'], 'values': ['55.41']}])
    captured = {}

    def fake_post(url, json=None, **kw):
        captured['json'] = json
        return post

    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', fake_post)

    peso_anchor._fetch_live('rice')

    geo_query = next(q for q in captured['json']['query'] if q['code'] == 'Geolocation')
    assert geo_query == {'code': 'Geolocation',
                          'selection': {'filter': 'item', 'values': ['000000000']}}


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


def test_load_cache_returns_empty_dict_when_file_holds_a_non_dict_json_value(tmp_path):
    """A real JSON file can just as easily hold a list, string, or number as
    an object -- the missing/corrupt-file handling above must extend to a
    wrong-shaped-but-valid JSON file too, never returning it as-is for
    callers that assume a dict."""
    p = tmp_path / 'list.json'
    p.write_text('[1, 2, 3]', encoding='utf-8')
    assert peso_anchor._load_cache(p) == {}


# ── Fix 2: a single unparseable price cell must not discard an
# already-found good row ────────────────────────────────────────────────────

def test_fetch_live_skips_a_comma_formatted_price_row_but_keeps_a_good_one(monkeypatch):
    """A comma-formatted price ('1,250.00') makes float() raise ValueError.
    Before the fix that propagated out of the row loop entirely and was
    caught by the function's outer except, discarding the good row found
    earlier in the same response. Row parsing must be isolated per-row, the
    same shape psa_cpi.py::_fetch_px_table already uses."""
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '0', '1'], 'values': ['48.20']},      # 2018-02, good, found first
        {'key': ['0', '5', '1', '0'], 'values': ['1,250.00']},   # unparseable: comma
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 48.20, 'as_of': '2018-02'}


def test_fetch_live_skips_a_dash_price_row_but_keeps_a_good_one(monkeypatch):
    """A bare dash ('-') is another real PSA 'no data' marker distinct from
    '..'/''/None -- float('-') also raises ValueError and must be isolated
    the same way."""
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '0', '1'], 'values': ['48.20']},   # 2018-02, good, found first
        {'key': ['0', '5', '1', '0'], 'values': ['-']},       # unparseable: dash
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 48.20, 'as_of': '2018-02'}


# ── Fix 3: a `None` entry in the Year variable's labels must never
# silently produce/cache a "None-02"-shaped as_of ──────────────────────────

def test_fetch_live_skips_a_row_with_a_none_year_label(monkeypatch):
    """Same PX-Web 'untranslated/missing label' shape already handled for
    Commodity, but here in Year. The row with the None-labelled year is
    deliberately the one that WOULD be picked as "latest" if 'None-02' were
    allowed through: the string 'None-02' sorts greater than '2019-01'
    lexicographically. Before the fix this silently produced and cached
    that garbage as_of; after the fix the row is skipped and the function
    falls through to the next valid row."""
    r = MagicMock()
    r.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['0'], 'valueTexts': ['Philippines']},
            {'code': 'Commodity', 'values': ['5'],
             'valueTexts': ['RICE, REGULAR-MILLED, 1 KG']},
            {'code': 'Year', 'values': ['0', '1'], 'valueTexts': [None, '2019']},
            {'code': 'Period', 'values': ['0', '1', '12'],
             'valueTexts': ['January', 'February', 'Annual']},
        ]
    }
    post = _fake_post_response([
        {'key': ['0', '5', '0', '1'], 'values': ['99.99']},   # year id '0' -> None label
        {'key': ['0', '5', '1', '0'], 'values': ['48.20']},   # year id '1' -> 2019, valid
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: r)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    result = peso_anchor._fetch_live('rice')
    assert result is not None
    assert 'None' not in result['as_of']
    assert result == {'price': 48.20, 'as_of': '2019-01'}


def test_fetch_live_returns_none_when_every_row_has_a_none_year_label(monkeypatch):
    """If no row survives the None-year guard, the function returns None
    rather than falling through to a garbage as_of."""
    r = MagicMock()
    r.json.return_value = {
        'variables': [
            {'code': 'Geolocation', 'values': ['0'], 'valueTexts': ['Philippines']},
            {'code': 'Commodity', 'values': ['5'],
             'valueTexts': ['RICE, REGULAR-MILLED, 1 KG']},
            {'code': 'Year', 'values': ['0'], 'valueTexts': [None]},
            {'code': 'Period', 'values': ['0', '12'], 'valueTexts': ['January', 'Annual']},
        ]
    }
    post = _fake_post_response([{'key': ['0', '5', '0', '0'], 'values': ['99.99']}])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: r)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') is None


# ── Fix 4: the branch's #1 named constraint -- 2M/2018NEW/, never
# 2M/RP or 2M/NRP -- had zero regression coverage ──────────────────────────

def test_fetch_live_calls_the_2018new_folder_never_rp_or_nrp(monkeypatch):
    """Every other fetch test mocks requests.get/post with a lambda that
    ignores the URL entirely, so nothing else in this suite would catch a
    regression to PSA's 2M/RP or 2M/NRP folders -- both confirmed frozen at
    2021 (see the module docstring); using either would silently anchor the
    app to five-year-old prices with no error."""
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([{'key': ['0', '5', '1', '1'], 'values': ['55.41']}])
    captured_urls = []

    def fake_get(url, *a, **kw):
        captured_urls.append(url)
        return meta

    def fake_post(url, *a, **kw):
        captured_urls.append(url)
        return post

    monkeypatch.setattr(peso_anchor.requests, 'get', fake_get)
    monkeypatch.setattr(peso_anchor.requests, 'post', fake_post)

    peso_anchor._fetch_live('rice')

    assert captured_urls, 'the fetch should have made at least one HTTP call'
    for url in captured_urls:
        assert '2M/2018NEW/' in url
        assert '2M/RP' not in url
        assert '2M/NRP' not in url


# ── Fix 5: negative-cache a failed fetch so a persistent PSA outage
# doesn't repeatedly re-attempt the same slow network call ────────────────

def test_get_anchor_does_not_refetch_within_the_failure_cooldown(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)

    first = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert first is None

    def fail_if_called(category):
        raise AssertionError('should not re-fetch within the failure cooldown')
    monkeypatch.setattr(peso_anchor, '_fetch_live', fail_if_called)

    second = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today)
    assert second is None


def test_recently_failed_is_true_immediately_after_marking(tmp_path):
    cache = {}
    now = datetime(2026, 8, 13, 10, 0, 0)
    peso_anchor._mark_failed(cache, 'rice', now)
    assert peso_anchor._recently_failed(cache, 'rice', now) is True


def test_recently_failed_is_false_once_the_cooldown_window_elapses():
    """The cooldown is a floor against retrying the SAME failure on every
    render within a short window -- it must not become a permanent outage
    flag. One minute past the 1-hour cooldown, a fresh attempt is allowed
    again."""
    cache = {}
    failed_at = datetime(2026, 8, 13, 10, 0, 0)
    peso_anchor._mark_failed(cache, 'rice', failed_at)
    later = failed_at + timedelta(hours=1, minutes=1)
    assert peso_anchor._recently_failed(cache, 'rice', later) is False


def test_clear_failed_removes_the_marker_for_only_that_category():
    cache = {}
    now = datetime(2026, 8, 13, 10, 0, 0)
    peso_anchor._mark_failed(cache, 'rice', now)
    peso_anchor._mark_failed(cache, 'meat', now)
    peso_anchor._clear_failed(cache, 'rice')
    assert peso_anchor._recently_failed(cache, 'rice', now) is False
    assert peso_anchor._recently_failed(cache, 'meat', now) is True


# ── Second residual-review round ────────────────────────────────────────────

# Fix 1: raw = row['values'][0] must be inside the per-row try/except too,
# not just float(raw) and the year/period resolution -- an empty or missing
# `values` list must not discard an already-found good row.

def test_fetch_live_skips_a_row_with_an_empty_values_list_but_keeps_a_good_one(monkeypatch):
    """Before the fix, `raw = row['values'][0]` sat outside the per-row
    try/except. An empty `values` list raises IndexError from that line,
    which propagated past the per-row guard straight to the function's
    outer except -- discarding the good 2019-02 row already found, and
    returning None for the whole fetch instead of just skipping this one
    malformed row."""
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '1', '1'], 'values': ['55.41']},  # 2019-02, good, found first
        {'key': ['0', '5', '0', '0'], 'values': []},         # malformed: empty values list
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


def test_fetch_live_skips_a_row_missing_the_values_key_but_keeps_a_good_one(monkeypatch):
    """Same hazard, but the row is missing the 'values' key entirely --
    row['values'][0] raises KeyError rather than IndexError. Both must be
    isolated to just that row by the per-row try/except."""
    meta = _fake_meta_response('RICE, REGULAR-MILLED, 1 KG')
    post = _fake_post_response([
        {'key': ['0', '5', '1', '1'], 'values': ['55.41']},  # 2019-02, good, found first
        {'key': ['0', '5', '0', '0']},                       # malformed: no 'values' key
    ])
    monkeypatch.setattr(peso_anchor.requests, 'get', lambda *a, **kw: meta)
    monkeypatch.setattr(peso_anchor.requests, 'post', lambda *a, **kw: post)

    assert peso_anchor._fetch_live('rice') == {'price': 55.41, 'as_of': '2019-02'}


# Fix 2: the negative-cache failure marker must be written whenever a fetch
# fails, whether or not a usable stale cache entry is found to fall back to
# -- otherwise the dominant real-world case (a working app with cached data,
# hitting a temporary PSA outage) pays the full slow-fetch cost on every
# single call despite the cache fallback succeeding every time.

def test_get_anchor_marks_failure_even_when_a_stale_cache_fallback_succeeds(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    now = datetime(2026, 8, 13, 9, 0, 0)
    peso_anchor._save_cache(
        {'rice': {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}},
        cache_path)

    calls = []

    def fake_fetch_live(category):
        calls.append(category)
        return None
    monkeypatch.setattr(peso_anchor, '_fetch_live', fake_fetch_live)

    first = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today, now=now)
    assert first == {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}
    assert calls == ['rice']

    def fail_if_called(category):
        raise AssertionError('should not re-fetch within the failure cooldown')
    monkeypatch.setattr(peso_anchor, '_fetch_live', fail_if_called)

    second = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today, now=now)
    assert second == {'price': 51.00, 'as_of': '2026-06', 'fetched_on': '2026-07-01'}


# Fix 3: `now` is injectable in get_anchor, matching the existing `today`
# parameter, so the full expiry-then-success negative-cache cycle can be
# exercised through get_anchor's own public API.

def test_get_anchor_retries_after_the_cooldown_expires_through_the_public_api(tmp_path, monkeypatch):
    cache_path = tmp_path / 'cache.json'
    today = date(2026, 8, 13)
    t0 = datetime(2026, 8, 13, 9, 0, 0)

    monkeypatch.setattr(peso_anchor, '_fetch_live', lambda category: None)
    first = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today, now=t0)
    assert first is None

    t1 = t0 + peso_anchor._FAILURE_COOLDOWN + timedelta(minutes=1)
    monkeypatch.setattr(
        peso_anchor, '_fetch_live',
        lambda category: {'price': 53.10, 'as_of': '2026-08'})

    second = peso_anchor.get_anchor('rice', cache_path=cache_path, today=today, now=t1)
    assert second == {'price': 53.10, 'as_of': '2026-08', 'fetched_on': '2026-08-13'}

    cache = peso_anchor._load_cache(cache_path)
    assert cache.get(peso_anchor._FAILURES_KEY, {}).get('rice') is None


# Fix 4: a future-timestamped failure marker (clock skew, or a cache file
# copied between machines) must not keep the cooldown active indefinitely.

def test_recently_failed_is_false_for_a_future_timestamped_marker():
    cache = {}
    now = datetime(2026, 8, 13, 10, 0, 0)
    future_failed_at = now + timedelta(minutes=5)
    peso_anchor._mark_failed(cache, 'rice', future_failed_at)
    assert peso_anchor._recently_failed(cache, 'rice', now) is False


# --- anchor_strip(): shared pure-formatting helper for Monitor and Report ---

def test_anchor_strip_returns_none_when_no_category_has_both_pieces(monkeypatch):
    monkeypatch.setattr(peso_anchor, 'get_anchor', lambda category, *a, **kw: None)
    assert peso_anchor.anchor_strip({}) is None
    assert peso_anchor.anchor_strip({'rice': 0.3}) is None


def test_anchor_strip_builds_text_and_caption_from_price_and_percent(monkeypatch):
    prices = {
        'rice': {'price': 52.36, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
        'meat': {'price': 185.40, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
        'fish': None,
        'vegetables': {'price': 62.10, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
    }
    monkeypatch.setattr(peso_anchor, 'get_anchor', lambda category, *a, **kw: prices.get(category))

    result = peso_anchor.anchor_strip(
        {'rice': 0.3, 'meat': -0.3, 'fish': 0.8, 'vegetables': 0.5})

    assert result['text'] == (
        'Rice ₱52.36 → ₱52.52  ·  Meat ₱185.40 → ₱184.84  ·  Fish —  ·  '
        'Vegetables ₱62.10 → ₱62.41')
    assert 'as of 2026-07 or later' in result['caption']
    assert 'exploratory projection, not a validated prediction' in result['caption']


def test_anchor_strip_reports_the_oldest_as_of_month(monkeypatch):
    prices = {
        'rice': {'price': 52.36, 'as_of': '2026-05', 'fetched_on': '2026-08-13'},
        'meat': {'price': 185.40, 'as_of': '2026-07', 'fetched_on': '2026-08-13'},
        'fish': None,
        'vegetables': None,
    }
    monkeypatch.setattr(peso_anchor, 'get_anchor', lambda category, *a, **kw: prices.get(category))

    result = peso_anchor.anchor_strip({'rice': 0.3, 'meat': -0.3})

    assert 'as of 2026-05 or later' in result['caption']


def test_anchor_strip_only_calls_get_anchor_for_categories_with_a_percent(monkeypatch):
    """A category never fetched at all when this cycle's debate produced no
    percentage for it -- matching the existing Monitor behaviour, not a new
    cost paid on every render for categories with nothing to anchor."""
    calls = []

    def fake_get_anchor(category, *a, **kw):
        calls.append(category)
        return {'price': 10.0, 'as_of': '2026-07', 'fetched_on': '2026-08-13'}

    monkeypatch.setattr(peso_anchor, 'get_anchor', fake_get_anchor)
    peso_anchor.anchor_strip({'rice': 0.3})
    assert calls == ['rice']
