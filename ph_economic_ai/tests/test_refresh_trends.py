"""The Google Trends refresher, and the drift that had no guard.

`refresh_trends` sent all four terms in ONE `build_payload` while the provenance
record it writes says `payload: one term per request`, and explains why: a shared
payload rescales every term against the most popular one, so a low-volume term is
reported as a fraction of `diesel price` rather than against its own history.

The committed data matched the RECORD, not the code -- every column reaches 100,
which only happens when each term is scaled alone -- so the code had drifted after
the file was built. Running it would have degraded the series AND written a record
asserting a method it did not use.

**It had no test at all.** That is the actual defect these cover: not the payload
shape, which is one line, but that nothing tied the code to the claim its own
output makes about it.

Network-free. `pytrends` is replaced by a stub that records how it was called.
"""
import sys
import types

import pandas as pd
import pytest

from ph_economic_ai.tools import refresh_social


class _FakeTrendReq:
    """Records every payload built, so the test can see the request SHAPE."""

    calls: list = []
    fail_terms: set = set()

    def __init__(self, **kw):
        self._terms: list = []

    def build_payload(self, kw_list, **kw):
        type(self).calls.append(list(kw_list))
        self._terms = list(kw_list)
        if type(self).fail_terms & set(kw_list):
            raise RuntimeError('429 Too Many Requests')

    def interest_over_time(self):
        idx = pd.date_range('2016-01-01', periods=8, freq='W')
        frame = pd.DataFrame({t: range(len(idx)) for t in self._terms}, index=idx)
        frame['isPartial'] = False
        return frame


@pytest.fixture
def fake_pytrends(monkeypatch):
    _FakeTrendReq.calls = []
    _FakeTrendReq.fail_terms = set()
    module = types.ModuleType('pytrends.request')
    module.TrendReq = _FakeTrendReq
    pkg = types.ModuleType('pytrends')
    pkg.request = module
    monkeypatch.setitem(sys.modules, 'pytrends', pkg)
    monkeypatch.setitem(sys.modules, 'pytrends.request', module)
    # The real thing spaces requests by 12 seconds; the shape is what is tested.
    monkeypatch.setattr(refresh_social, '_TRENDS_SPACING', 0)
    monkeypatch.setattr(refresh_social, '_TRENDS_BACKOFF', 0)
    return _FakeTrendReq


# ── The request shape the provenance record claims ───────────────────────────

def test_each_term_is_requested_on_its_own(fake_pytrends, tmp_path):
    """The defect. A shared payload scales every term against the most popular
    one, so `presyo ng gas` would be reported as a fraction of `diesel price`
    instead of against its own history."""
    out = tmp_path / 'trends.csv'
    terms = ['a', 'b', 'c', 'd']
    refresh_social.refresh_trends(terms=terms, out=out)

    assert len(fake_pytrends.calls) == len(terms), 'one request per term'
    for call in fake_pytrends.calls:
        assert len(call) == 1, f'{call} bundles terms into one payload'
    assert [c[0] for c in fake_pytrends.calls] == terms


def test_the_written_record_describes_what_the_code_actually_did(fake_pytrends, tmp_path):
    """The guard that would have caught the drift. The record claims one term per
    request; this asserts the claim against the recorded call shape rather than
    against the wording alone, so the two cannot separate again."""
    import json

    out = tmp_path / 'trends.csv'
    refresh_social.refresh_trends(terms=['a', 'b'], out=out)
    record = json.loads((tmp_path / 'trends.csv.provenance.json').read_text('utf-8'))

    assert record['request_params']['payload'] == 'one term per request'
    assert all(len(c) == 1 for c in fake_pytrends.calls), (
        'the record claims one term per request and the code must match it')


def test_every_term_becomes_its_own_column(fake_pytrends, tmp_path):
    out = tmp_path / 'trends.csv'
    refresh_social.refresh_trends(terms=['a', 'b', 'c'], out=out)
    written = pd.read_csv(out, index_col=0)
    assert list(written.columns) == ['a', 'b', 'c']
    assert 'isPartial' not in written.columns


# ── All or nothing ───────────────────────────────────────────────────────────

def test_one_failed_term_writes_nothing(fake_pytrends, tmp_path):
    """`DEC-059`. Partial coverage is a DIFFERENT series, not a shorter one: the
    nowcast would keep running with a driver silently missing."""
    fake_pytrends.fail_terms = {'c'}
    out = tmp_path / 'trends.csv'
    assert refresh_social.refresh_trends(terms=['a', 'b', 'c'], out=out) == 0
    assert not out.exists()
    assert not (tmp_path / 'trends.csv.provenance.json').exists()


def test_a_failing_term_is_retried_before_being_given_up_on(fake_pytrends, tmp_path):
    """A 429 is transient, and two earlier refreshes died on one. Retried with
    backoff rather than abandoned on first sight.

    Asserted as `> 1`, NOT against `_TRENDS_ATTEMPTS`. Reading the same constant
    the code reads makes the assertion `n == n`, which holds however small n gets:
    setting the constant to 1 disabled retries entirely and this test still
    passed. A guard that quotes the value it is guarding cannot fail."""
    fake_pytrends.fail_terms = {'b'}
    refresh_social.refresh_trends(terms=['a', 'b'], out=tmp_path / 'trends.csv')
    attempts_at_b = [c for c in fake_pytrends.calls if c == ['b']]
    assert len(attempts_at_b) > 1, 'a transient failure must be retried at least once'
    assert refresh_social._TRENDS_ATTEMPTS > 1


def test_an_existing_file_survives_a_failed_refresh(fake_pytrends, tmp_path):
    """The refusal must not damage what is already committed."""
    out = tmp_path / 'trends.csv'
    out.write_text('date,a\n2016-01,1.0\n', encoding='utf-8')
    fake_pytrends.fail_terms = {'a'}
    refresh_social.refresh_trends(terms=['a'], out=out)
    assert out.read_text(encoding='utf-8') == 'date,a\n2016-01,1.0\n'


def test_a_missing_pytrends_is_skipped_rather_than_crashing(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'pytrends', None)
    monkeypatch.setitem(sys.modules, 'pytrends.request', None)
    assert refresh_social.refresh_trends(terms=['a'], out=tmp_path / 'x.csv') == 0


# ── The committed file still shows the per-term signature ────────────────────

def test_the_committed_series_is_scaled_per_term():
    """Every column reaching 100 is the fingerprint of per-term scaling, and it
    is how the drift was caught: the data matched the record while the code did
    not. If a shared payload ever ships again, this fails on the next refresh."""
    from ph_economic_ai.benchmark.paths import data
    frame = pd.read_csv(data('google_trends_monthly.csv'), index_col=0)
    assert len(frame.columns) == 4
    for column in frame.columns:
        assert frame[column].max() == 100.0, (
            f'{column} peaks at {frame[column].max()}, so it was scaled against '
            f'another term rather than its own history')
