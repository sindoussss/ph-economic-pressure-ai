import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from datetime import date

import pytest

from ph_economic_ai.engine import social_snapshot as ss


def _write(tmp_path, rows):
    d = tmp_path / 'social'
    d.mkdir()
    (d / 'reddit_2026-07-24.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return d


def test_absent_snapshot_is_graceful(tmp_path):
    assert ss.load_social_snapshot(tmp_path / 'does_not_exist') == []


def test_load_skips_malformed_lines(tmp_path):
    d = tmp_path / 'social'
    d.mkdir()
    (d / 'reddit_2026-07-24.jsonl').write_text(
        '{"date":"2026-07-24","source":"RedditPH","title":"ok","text":"x"}\n'
        'not json\n'
        '{"source":"RedditPH"}\n',   # missing required 'date' -> skipped
        encoding='utf-8')
    posts = ss.load_social_snapshot(d)
    assert len(posts) == 1 and posts[0].title == 'ok'


def test_window_slice(tmp_path):
    d = _write(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'today', 'text': 'x'},
        {'date': '2026-07-19', 'source': 'RedditPH', 'title': '5d ago', 'text': 'x'},
        {'date': '2026-06-15', 'source': 'RedditPH', 'title': 'old', 'text': 'x'},
    ])
    posts = ss.load_social_snapshot(d)
    as_of = date(2026, 7, 24)
    assert len(ss.window_slice(posts, 'today', as_of)) == 1
    assert len(ss.window_slice(posts, 'this_week', as_of)) == 2
    assert len(ss.window_slice(posts, 'this_month', as_of)) == 2   # 06-15 is >30d out
    with pytest.raises(ValueError):
        ss.window_slice(posts, 'this_year', as_of)


def test_to_rag_text_bundles_one_source(tmp_path):
    d = _write(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'gas up', 'text': 'presyo tumaas'},
        {'date': '2026-07-24', 'source': 'GoogleTrends', 'title': 'interest 80', 'text': ''},
    ])
    posts = ss.load_social_snapshot(d)
    blob = ss.to_rag_text(posts, 'RedditPH')
    assert 'gas up' in blob and '2026-07-24' in blob
    assert 'interest 80' not in blob   # other source excluded


def test_register_feeds_rag_via_add_text(tmp_path):
    d = _write(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'gas up', 'text': 'presyo'},
        {'date': '2026-07-24', 'source': 'GoogleTrends', 'title': 'interest 80', 'text': ''},
    ])

    class FakeRag:
        def __init__(self):
            self.calls = []

        def add_text(self, source, text, url=''):
            self.calls.append((source, text, url))
            return 1

    rag = FakeRag()
    counts = ss.register_social_sources(rag, d)
    assert counts == {'GoogleTrends': 1, 'RedditPH': 1}
    assert {c[0] for c in rag.calls} == {'RedditPH', 'GoogleTrends'}
    # never a live URL — always the frozen-snapshot scheme
    assert all(c[2].startswith('social-snapshot://') for c in rag.calls)
