import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from datetime import date

from ph_economic_ai.engine import llm as llm_mod
from ph_economic_ai.engine import forum
from ph_economic_ai.engine.auto_assemble import auto_assemble


class FakeRag:
    def add_text(self, source, text, url=''):
        return 1

    def query(self, text, top_k=5, sources=None):
        return []


def _fake_complete(messages, tier=None, max_tokens=None, **kw):
    """Return a sector-appropriate ESTIMATE based on the unit hint in the prompt.
    Moderator/synth calls (no unit) get a plain summary."""
    text = ' '.join(m.get('content', '') for m in messages)
    if '/kWh' in text:
        est = 'ESTIMATE: +₱0.30/kWh'
    elif '/L' in text:
        est = 'ESTIMATE: +₱1.00/L'
    elif '%' in text:
        est = 'ESTIMATE: +0.5%'
    else:
        est = ''
    return ('Prices are rising now. '
            'CAUSAL CHAIN: oil up -> pump up -> households pay more. ' + est)


def _snapshot(tmp_path, rows):
    d = tmp_path / 'social'
    d.mkdir()
    (d / 'reddit_2026-07-24.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    return d


def test_auto_assemble_builds_sector_contexts(tmp_path):
    d = _snapshot(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'gas up', 'text': 'presyo'},
        {'date': '2026-07-20', 'source': 'GoogleTrends', 'title': 'interest', 'text': ''},
    ])
    asm = auto_assemble(rag=FakeRag(), corpus_dir=d, as_of=date(2026, 7, 24),
                        window='this_week', report_path=tmp_path / 'no_report.json')
    assert [c.sector for c in asm.contexts] == ['gas', 'food', 'electricity']
    assert asm.as_of == '2026-07-24' and asm.window == 'this_week'
    for c in asm.contexts:
        assert set(c.social_counts) == {'today', 'this_week', 'this_month'}
        assert c.verdict_note   # always a non-empty honesty note


def test_verdict_note_carries_efficiency(tmp_path):
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'audit': [{'target': 'fuel', 'verdict': 'efficient'}]}),
                      encoding='utf-8')
    asm = auto_assemble(rag=FakeRag(), corpus_dir=tmp_path / 'empty',
                        as_of=date(2026, 7, 24), report_path=report)
    gas = next(c for c in asm.contexts if c.sector == 'gas')
    assert 'EFFICIENT' in gas.verdict_note.upper()


def test_forum_produces_pressure_brief(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, 'complete', _fake_complete)
    d = _snapshot(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'x', 'text': 'y'},
    ])
    brief = forum.run_monitor(FakeRag(), corpus_dir=d, as_of=date(2026, 7, 24),
                              window='this_week', rounds=1)
    assert brief.as_of == '2026-07-24' and brief.window == 'this_week'
    by = {r.sector: r for r in brief.readings}
    assert set(by) == {'gas', 'food', 'electricity'}
    assert by['gas'].estimate == 1.0 and by['gas'].direction == 'rising'
    assert by['food'].estimate == 0.5 and by['food'].direction == 'rising'
    assert by['electricity'].estimate == 0.3 and by['electricity'].direction == 'rising'
    assert all(r.confidence == 100 for r in brief.readings)   # agents agree
    assert all(r.drivers for r in brief.readings)             # causal chains captured
    assert brief.narrative                                    # synthesised summary
    # serialisation round-trips
    assert set(brief.to_dict()) == {'as_of', 'window', 'narrative', 'readings'}


def test_forum_handles_unparseable_estimates(monkeypatch, tmp_path):
    monkeypatch.setattr(llm_mod, 'complete',
                        lambda *a, **k: 'No number here.')   # no ESTIMATE line
    brief = forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'empty',
                              as_of=date(2026, 7, 24), rounds=1)
    for r in brief.readings:
        assert r.estimate is None and r.direction == 'unknown' and r.confidence == 0
