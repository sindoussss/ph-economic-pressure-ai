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


def _patch_llm(monkeypatch, complete_fn=None):
    """Patch BOTH seams.

    The forum streams only when an event sink is attached, so a test that attaches
    one and patches `complete` alone would silently make real network calls — with a
    50-agent roster that is a multi-minute hang, which is exactly how this was found.
    Deriving the stream fake from the complete fake also means the two paths can
    never disagree about what an agent said.
    """
    fn = complete_fn or _fake_complete
    monkeypatch.setattr(llm_mod, 'complete', fn)
    monkeypatch.setattr(
        llm_mod, 'stream',
        lambda m, tier=None, max_tokens=None, **kw: iter(
            [fn(m, tier=tier, max_tokens=max_tokens, **kw)]))


def test_food_magnitude_is_anchor_guarded(monkeypatch, tmp_path):
    """A weak-model +5%/month food read (a YoY-leak error) must be clamped back to
    the anchor band, not shown as-is — the §6.6 guard the Monitor was bypassing."""
    def high_food(messages, tier=None, max_tokens=None, **kw):
        return 'Rising. CAUSAL CHAIN: high prices -> spending -> budgets. ESTIMATE: +5.0%'
    monkeypatch.setattr(llm_mod, 'complete', high_food)
    brief = forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'nope',
                              as_of=date(2026, 7, 24), sectors=('food',), rounds=1,
                              live=False)
    from ph_economic_ai.engine import anchoring
    food = brief.readings[0]
    cap = anchoring.food_persistence_anchor([]) + anchoring.FOOD_TOLERANCE_PCT
    assert food.estimate is not None and food.estimate <= cap + 1e-6   # clamped down
    assert food.estimate < 5.0
    assert food.confidence == 100                                      # agents agreed (raw)
    assert food.drivers and 'ESTIMATE' not in food.drivers[0]          # driver trimmed


def test_confidence_scales_with_corroboration():
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = SectorContext(sector='gas', unit='₱/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=1)
    lone = [AgentResponse('A', 1, '', 'CAUSAL CHAIN: x. ESTIMATE: +1.00/L', 1.0)]
    assert f._aggregate(ctx, lone).confidence == 50                    # one voice, not 100%
    pair = lone + [AgentResponse('B', 1, '', 'CAUSAL CHAIN: y. ESTIMATE: +1.00/L', 1.0)]
    assert f._aggregate(ctx, pair).confidence == 100                   # two agree -> full


def test_judge_synthesis_overrides_the_mean():
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = SectorContext(sector='gas', unit='₱/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=1)
    finals = [AgentResponse('A', 1, '', 'CAUSAL CHAIN: x. ESTIMATE: +1.00/L', 1.0),
              AgentResponse('B', 1, '', 'CAUSAL CHAIN: y. ESTIMATE: +1.00/L', 1.0)]
    r = f._aggregate(ctx, finals, judged=0.40)
    assert r.estimate == 0.40        # the judge's synthesis, not the +1.00 agent mean
    assert r.confidence == 100       # agents still agreed (on +1.00)
    r2 = f._aggregate(ctx, finals, judged=None)
    assert r2.estimate == 1.00       # no judge number -> falls back to the agent mean


def test_cites_only_sources_actually_retrieved(monkeypatch, tmp_path):
    """A channel whose feeds return nothing must NOT be cited. Reddit is no longer
    reachable (API behind a researcher review), so citing it — on the card or in the
    debate map — would claim evidence the run never read."""
    class PartialRag:
        def add_text(self, source, text, url=''):
            return 1

        def query(self, text, top_k=5, sources=None):        # social returns nothing
            return [{'source': s, 'text': 'crude up 3pct'} for s in (sources or [])
                    if s not in ('RedditPH', 'GoogleTrends')]

    _patch_llm(monkeypatch)
    seen = []
    brief = forum.run_monitor(PartialRag(), corpus_dir=tmp_path / 'empty',
                              as_of=date(2026, 7, 24), sectors=('gas',), rounds=1,
                              live=False, on_event=lambda k, d: seen.append((k, d)))
    cited = brief.readings[0].sources
    assert 'RedditPH' not in cited and 'GoogleTrends' not in cited   # never read
    assert 'YahooFinanceCrude' in cited and 'DOEBulletin' in cited   # genuinely read
    social = next(d for k, d in seen
                  if k == 'agent_message' and d['name'] == 'Andrea Lim')
    assert social['sources'] == []          # the social lane cites nothing, honestly


def test_no_sources_at_all_cites_nothing(monkeypatch, tmp_path):
    """Total retrieval failure yields an empty citation list, not a wishlist."""
    monkeypatch.setattr(llm_mod, 'complete', _fake_complete)
    brief = forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'empty',
                              as_of=date(2026, 7, 24), sectors=('gas',), rounds=1,
                              live=False)
    assert brief.readings[0].sources == []


def test_social_counts_are_sector_specific(tmp_path):
    """Rice chatter must not be counted as gas evidence."""
    d = _snapshot(tmp_path, [
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'bigas presyo tumaas',
         'text': 'rice is expensive'},
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'diesel price hike',
         'text': 'fuel up again'},
        {'date': '2026-07-24', 'source': 'RedditPH', 'title': 'meralco bill',
         'text': 'kuryente mahal'},
    ])
    asm = auto_assemble(rag=FakeRag(), corpus_dir=d, as_of=date(2026, 7, 24),
                        window='this_week', report_path=tmp_path / 'none.json')
    by = {c.sector: c.social_counts['this_week'] for c in asm.contexts}
    assert by == {'gas': 1, 'food': 1, 'electricity': 1}   # one each, not 3/3/3


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
                              window='this_week', rounds=1, live=False)
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
                              as_of=date(2026, 7, 24), rounds=1, live=False)
    for r in brief.readings:
        assert r.estimate is None and r.direction == 'unknown' and r.confidence == 0


# ── 50-agent roster, adaptive rebuttals, streaming ────────────────────────────

def test_roster_is_fifty_agents_across_three_sectors():
    from ph_economic_ai.engine.forum import _capability_agents, roster_size
    assert roster_size() == 50
    names = [a.name for s in ('gas', 'food', 'electricity')
             for a in _capability_agents(s)]
    assert len(set(names)) == 50, 'names must be unique - the UI keys cards on them'


def test_each_channel_keeps_its_lane_at_scale():
    """Every one of the 50 belongs to exactly one evidence channel and carries its own
    vantage; without that, 17 agents per channel would be clones and the forum would
    just be an expensive single model."""
    from ph_economic_ai.engine.forum import _capability_agents
    for sector in ('gas', 'food', 'electricity'):
        for a in _capability_agents(sector):
            lanes = sum(x in a.system_prompt for x in
                        ('ONLY to public attention', 'ONLY to reported events',
                         'ONLY to the underlying market'))
            assert lanes == 1, f'{a.name} does not sit in exactly one lane'
            assert 'vantage point is' in a.system_prompt


def test_agents_are_interleaved_by_channel():
    """A live viewer should see all three lanes in the first few cards rather than six
    consecutive social reads."""
    from ph_economic_ai.engine.forum import _capability_agents
    lanes = set()
    for a in _capability_agents('gas')[:3]:
        lanes.add('social' if 'public attention' in a.system_prompt
                  else 'news' if 'reported events' in a.system_prompt else 'market')
    assert lanes == {'social', 'news', 'market'}


def test_per_channel_cap_shrinks_the_roster():
    from ph_economic_ai.engine.forum import roster_size
    assert roster_size(per_channel=1) == 9          # the original 3x3 cast
    assert roster_size(per_channel=2) == 18
    assert roster_size(per_channel=1) < roster_size()


def _ctx(sector='gas', unit='PHP/L'):
    from ph_economic_ai.engine.auto_assemble import SectorContext
    return SectorContext(sector=sector, unit=unit, verdict_note='', anchor=0.0)


def test_round_two_only_invites_the_divergent():
    """A second full round would roughly double a multi-minute run, mostly on agents
    restating themselves. Only the outliers answer the moderator.

    `k` is a CAP, not a quota: agents inside the agreement band have nothing to
    revise, so with five agents agreeing and one outlier, exactly one is called
    even though two slots were offered. Padding the invitation to fill k would
    spend a turn asking an agent who already agrees to reconsider.
    """
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = _ctx()
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=2)
    agents = _capability_agents('gas')
    resp = [AgentResponse(a.name, 1, '', 'x', 1.0) for a in agents[:5]]
    resp.append(AgentResponse(agents[5].name, 1, '', 'x', 9.0))      # the outlier
    picked = [a.name for a in f._divergent(ctx, resp, agents, k=2)]
    assert picked == [agents[5].name]


def test_divergent_skips_agents_with_no_estimate():
    """An agent with no parsable number has nothing to defend - usually a dropped
    call - so spending a rebuttal on it is waste."""
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = _ctx()
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=2)
    agents = _capability_agents('gas')
    resp = [AgentResponse(agents[0].name, 1, '', 'x', 1.0),
            AgentResponse(agents[1].name, 1, '', 'x', 2.0),
            AgentResponse(agents[2].name, 1, '', 'x', None)]
    got = [a.name for a in f._divergent(ctx, resp, agents, k=3)]
    assert agents[2].name not in got


def test_consensus_uses_every_agents_latest_word_not_the_last_round():
    """The bug adaptive rebuttals would otherwise introduce: filtering history by
    round number would cut a 50-agent consensus down to the few who rebutted."""
    from ph_economic_ai.engine.forum import Forum, _latest_per_agent
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = _ctx()
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=2)
    hist = [AgentResponse(f'A{i}', 1, '', 'x', 1.0) for i in range(10)]
    hist.append(AgentResponse('A0', 2, '', 'x', 3.0))               # A0 rebuts
    latest = _latest_per_agent(hist)
    assert len(latest) == 10                                        # all ten, not one
    assert next(r for r in latest if r.agent_name == 'A0').price_estimate == 3.0
    assert f._aggregate(ctx, hist).confidence > 0



def test_judge_compresses_a_large_roster_but_keeps_the_extremes(monkeypatch):
    """50 full statements would overflow the deep tier. The extremes are what a judge
    resolving a disagreement needs verbatim; the rest collapse to name/estimate."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.debate import AgentResponse
    seen = {}

    def spy(messages, tier=None, max_tokens=None, **kw):
        seen['body'] = messages[-1]['content']
        return 'On balance rising. ESTIMATE: +1.00/L'

    monkeypatch.setattr(llm_mod, 'complete', spy)
    ctx = _ctx()
    f = Forum(FakeRag(), [ctx], as_of='2026-07-24', window='this_week', rounds=1)
    finals = [AgentResponse(f'A{i}', 1, '', f'view {i}', float(i)) for i in range(30)]
    f._judge_sector(ctx, finals)
    body = seen['body']
    assert 'view 0' in body and 'view 29' in body         # both extremes verbatim
    assert 'Other analysts' in body                        # the middle summarised
    assert len(body) < 6000                                # and it stays bounded


def test_streaming_emits_tokens_only_when_someone_is_listening(monkeypatch, tmp_path):
    """Tokens exist to fill a card live, so a headless run takes the cheaper
    `complete` path - which is also the seam the rest of these tests patch."""
    _patch_llm(monkeypatch)
    seen = []
    forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'x', as_of=date(2026, 7, 24),
                      sectors=('gas',), rounds=1, live=False,
                      on_event=lambda k, d: seen.append((k, d)))
    assert 'agent_token' in [k for k, _ in seen]
    tok = next(d for k, d in seen if k == 'agent_token')
    assert tok['text'] and 'name' in tok and 'sector' in tok


def test_headless_run_never_streams(monkeypatch, tmp_path):
    used = []

    def spy_complete(messages, tier=None, max_tokens=None, **kw):
        used.append(1)
        return _fake_complete(messages, tier=tier, max_tokens=max_tokens, **kw)

    def boom(*a, **k):
        raise AssertionError('a headless run must not stream')

    monkeypatch.setattr(llm_mod, 'complete', spy_complete)
    monkeypatch.setattr(llm_mod, 'stream', boom)
    forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'x', as_of=date(2026, 7, 24),
                      sectors=('gas',), rounds=1, live=False)       # no on_event
    assert used


# ── estimate plausibility guard (found by a live run) ─────────────────────────

def test_absolute_price_is_rejected_not_shown_as_a_change():
    """A live 17-agent gas run produced +150.00/L and +100.00/L — the model quoting
    a pump PRICE where a CHANGE was asked for. The unguarded gas extractor put both
    on agent cards."""
    from ph_economic_ai.engine.forum import _extract_guarded
    accepted, rejected = _extract_guarded('gas', 'ESTIMATE: +150.00/L')
    assert accepted is None and rejected == 150.0


def test_plausible_values_still_pass():
    from ph_economic_ai.engine.forum import _extract_guarded
    for sector, text, want in [('gas', 'ESTIMATE: +1.20/L', 1.2),
                               ('food', 'ESTIMATE: +0.8%', 0.8),
                               ('electricity', 'ESTIMATE: +0.30/kWh', 0.3)]:
        accepted, rejected = _extract_guarded(sector, text)
        assert accepted == want and rejected is None


def test_no_estimate_is_distinct_from_a_rejected_one():
    """The distinction the swarm already draws: 'said nothing' and 'said something
    absurd' are different events and must not collapse."""
    from ph_economic_ai.engine.forum import _extract_guarded
    assert _extract_guarded('gas', 'prices feel high lately') == (None, None)
    assert _extract_guarded('gas', 'ESTIMATE: +99.00/L')[1] == 99.0


def test_forum_and_swarm_share_one_fuel_bound():
    """Two debate systems must not drift to different notions of a plausible move."""
    from ph_economic_ai.engine import swarm
    from ph_economic_ai.engine.debate import _MAX_REALISTIC_FUEL_PHP_L
    from ph_economic_ai.engine.forum import _PLAUSIBLE
    assert _PLAUSIBLE['gas'] == _MAX_REALISTIC_FUEL_PHP_L
    assert swarm._MAX_REALISTIC_FUEL_CHANGE == _MAX_REALISTIC_FUEL_PHP_L


def test_rejected_estimate_reaches_the_event(monkeypatch, tmp_path):
    """The UI can only report a discarded number if the event carries it."""
    def absurd(messages, tier=None, max_tokens=None, **kw):
        return 'Pumps are moving. CAUSAL CHAIN: a -> b -> c. ESTIMATE: +150.00/L'

    _patch_llm(monkeypatch, absurd)
    seen = []
    brief = forum.run_monitor(FakeRag(), corpus_dir=tmp_path / 'x',
                              as_of=date(2026, 7, 26), sectors=('gas',), rounds=1,
                              live=False, on_event=lambda k, d: seen.append((k, d)))
    msg = next(d for k, d in seen if k == 'agent_message')
    assert msg['estimate'] is None
    assert msg['rejected_estimate'] == 150.0
    # and nothing absurd survives into the sector reading
    assert brief.readings[0].estimate is None or abs(brief.readings[0].estimate) <= 8.0


# ── round-1 independence (found by a live run) ────────────────────────────────

def test_round_one_agents_do_not_see_each_other():
    """Agreement between agents that have read each other is herding, not
    corroboration — and the card's confidence number is computed from exactly that
    agreement. A live run also showed a 3b model copying its neighbours verbatim:
    8 distinct openings out of 20, with a news reporter reciting the social lane's
    line. Round 1 is therefore blind."""
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.auto_assemble import SectorContext
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-26', window='this_week', rounds=2)
    history = [AgentResponse('Someone Else', 1, '', 'CRUDE IS SURGING WILDLY', 4.0)]
    msgs, _ = f._agent_prompt(_capability_agents('gas')[0], ctx, history, steer='')
    body = msgs[-1]['content']
    assert 'CRUDE IS SURGING WILDLY' not in body
    # the SECTION must be absent; the phrase itself still appears in the anti-echo
    # instruction that names the headings an agent must not parrot
    assert 'Prior statements:\n' not in body


def test_rebuttal_agents_do_see_the_debate():
    """Responding to what others said is the whole point of a rebuttal, so the
    blindness is lifted once the moderator has spoken."""
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.auto_assemble import SectorContext
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-26', window='this_week', rounds=2)
    history = [AgentResponse('Someone Else', 1, '', 'CRUDE IS SURGING WILDLY', 4.0)]
    msgs, _ = f._agent_prompt(_capability_agents('gas')[0], ctx, history,
                              steer='Stay on the present read.')
    body = msgs[-1]['content']
    assert 'CRUDE IS SURGING WILDLY' in body
    assert 'Stay on the present read.' in body


def test_prompt_forbids_echoing_its_own_scaffolding():
    """Live agents opened with 'BENCHMARK NOTE:' and 'Retrieved context:' — the
    prompt read back instead of answered."""
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.auto_assemble import SectorContext
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-26', window='this_week', rounds=1)
    body = f._agent_prompt(_capability_agents('gas')[0], ctx, [], '')[0][-1]['content']
    assert 'in your OWN words' in body
    assert "another analyst's name" in body


def test_blind_round_one_keeps_prompts_small():
    """The independence fix is also the scale fix: with no prior statements, an
    opening prompt does not grow with the roster."""
    from ph_economic_ai.engine.forum import Forum, _capability_agents
    from ph_economic_ai.engine.auto_assemble import SectorContext
    from ph_economic_ai.engine.debate import AgentResponse
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='', anchor=0.0)
    f = Forum(FakeRag(), [ctx], as_of='2026-07-26', window='this_week', rounds=1)
    agent = _capability_agents('gas')[0]
    empty = len(f._agent_prompt(agent, ctx, [], '')[0][-1]['content'])
    crowded = [AgentResponse(f'A{i}', 1, '', 'x' * 280, 1.0) for i in range(49)]
    late = len(f._agent_prompt(agent, ctx, crowded, '')[0][-1]['content'])
    assert late == empty, 'the 50th agent must get the same prompt as the first'


# ── the estimate instruction (found by a live run) ────────────────────────────

def test_estimate_line_is_an_instruction_not_a_template():
    """A small model copies a template. With the old wording ('ESTIMATE: +P X.XX/L')
    live agents answered with the placeholder itself, which parses to nothing — most
    of the roster's estimates were being lost to that one formatting choice."""
    from ph_economic_ai.engine.forum import _EST_LINE, _extract_guarded
    for sector, line in _EST_LINE.items():
        assert 'X.X' not in line.replace('never write X.XX', '').replace(
            'never write X.X', ''), f'{sector} still shows a copyable placeholder'
        assert 'worked example' in line
        assert 'your own number' in line
        # the worked example itself must be parsable, or we are teaching a format
        # the parser rejects
        import re
        for m in re.findall(r'"(ESTIMATE:[^"]+)"', line):
            assert _extract_guarded(sector, m)[0] is not None, f'{sector}: {m!r}'


def test_food_category_lines_are_instructions_not_templates():
    """Same guard as the ESTIMATE line above, for the six new food
    sub-category lines: each must carry a worked example and must not leave
    a copyable 'X.X' placeholder outside its own worked example."""
    from ph_economic_ai.engine.forum import _FOOD_CATEGORY_LINES
    from ph_economic_ai.engine.debate import _CATEGORY_LABELS, _extract_category_percents
    assert set(_FOOD_CATEGORY_LINES) == set(_CATEGORY_LABELS)
    for category, line in _FOOD_CATEGORY_LINES.items():
        assert 'worked example' in line
        assert 'your own number' in line
        import re
        for m in re.findall(rf'"({_CATEGORY_LABELS[category]}:[^"]+)"', line):
            assert category in _extract_category_percents(m), f'{category}: {m!r}'


def test_judge_sector_returns_subcategories_for_food(monkeypatch):
    """The judge's synthesis, for food, must feed _extract_category_percents
    and thread the result through -- not just the blended estimate."""
    import pytest
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    def fake_complete(msgs, **kw):
        return ('Prices are broadly steady with a modest rice uptick.\n'
                'RICE: +0.3%\nMEAT: +0.0%\nFISH: -0.2%\n'
                'DAIRY_EGGS: +0.0%\nVEGETABLES: +0.1%\nSUGAR: +0.0%\n'
                'ESTIMATE: +0.1%')

    monkeypatch.setattr(llm_mod, 'complete', fake_complete)
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='food', unit='%', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert estimate == pytest.approx(0.1)
    assert subcategories == {'rice': 0.3, 'meat': 0.0, 'fish': -0.2,
                             'dairy_eggs': 0.0, 'vegetables': 0.1, 'sugar': 0.0}


def test_judge_sector_returns_empty_subcategories_for_gas(monkeypatch):
    """Gas and electricity have no PSA sub-categories -- the judge must not
    try to parse category lines that were never asked for."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    monkeypatch.setattr(llm_mod, 'complete',
                        lambda msgs, **kw: 'Steady. ESTIMATE: +0.10/L')
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert subcategories == {}


def test_food_blended_estimate_does_not_silently_become_a_category_value(monkeypatch):
    """CRITICAL regression: `_extract_percent`'s prose fallback grabs the FIRST
    signed percent anywhere in the text whenever the anchored ESTIMATE: line
    fails to parse. The food judge prompt always asks for six category lines
    before its ESTIMATE line, so an unparseable ESTIMATE: ('broadly unchanged')
    must not let that fallback silently adopt RICE (or any other category) as
    the headline blended estimate."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    monkeypatch.setattr(llm_mod, 'complete', lambda msgs, **kw: (
        'Prices look broadly steady this cycle.\n'
        'RICE: +0.3%\nMEAT: +0.0%\nFISH: -0.2%\n'
        'DAIRY_EGGS: +0.0%\nVEGETABLES: +0.1%\nSUGAR: +0.0%\n'
        'ESTIMATE: broadly unchanged'  # unparseable blended line, deliberately
    ))
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='food', unit='%', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert estimate is None, (
        f'blended estimate should be None when ESTIMATE: fails to parse, '
        f'not silently adopt a category value -- got {estimate}'
    )
    assert subcategories == {'rice': 0.3, 'meat': 0.0, 'fish': -0.2,
                             'dairy_eggs': 0.0, 'vegetables': 0.1, 'sugar': 0.0}


def test_food_blended_estimate_does_not_leak_from_category_line_commentary(monkeypatch):
    """Same failure class as the sibling test above, but the leak comes from
    trailing commentary on a category line (e.g. '(down from -1.2% last
    month)') rather than the category value's own token -- the first fix
    round only stripped the label+value, not the rest of the line."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    monkeypatch.setattr(llm_mod, 'complete', lambda msgs, **kw: (
        'ESTIMATE: broadly unchanged\n'
        'RICE: +0.3% (down from -1.2% last month)\n'
        'MEAT: +0.0%\nFISH: +0.0%\nDAIRY_EGGS: +0.0%\n'
        'VEGETABLES: +0.0%\nSUGAR: +0.0%'
    ))
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='food', unit='%', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert estimate is None, (
        f'blended estimate should be None -- got {estimate}, leaked from '
        f'trailing commentary on the RICE line'
    )
    assert subcategories['rice'] == 0.3, (
        'subcategories parsing (on the full unstripped text) must still '
        'correctly find rice=0.3 despite the trailing commentary'
    )


def test_food_judge_prompt_actually_contains_the_category_lines(monkeypatch):
    """`fake_complete`-style tests elsewhere in this file ignore `msgs` entirely
    and return a canned response, so they would stay green even if the prompt-
    building code that adds the six category lines were deleted. This asserts
    directly on the prompt content sent to the judge, following the
    `msgs[-1]['content']` idiom used at test_rebuttal_agents_do_see_the_debate
    and test_round_one_agents_do_not_see_each_other."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    captured = {}
    def fake_complete(msgs, **kw):
        captured['content'] = msgs[-1]['content']
        return ('Steady. RICE: +0.1%\nMEAT: +0.0%\nFISH: +0.0%\n'
                'DAIRY_EGGS: +0.0%\nVEGETABLES: +0.0%\nSUGAR: +0.0%\n'
                'ESTIMATE: +0.1%')

    monkeypatch.setattr(llm_mod, 'complete', fake_complete)
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='food', unit='%', verdict_note='exploratory',
                        anchor=None, social_counts={})
    f._judge_sector(ctx, finals=[])
    for label in ('RICE:', 'MEAT:', 'FISH:', 'DAIRY_EGGS:', 'VEGETABLES:', 'SUGAR:'):
        assert label in captured['content'], f'{label} missing from food judge prompt'


def test_gas_judge_prompt_does_not_contain_category_lines(monkeypatch):
    """Gas has no PSA sub-categories -- its judge prompt must not carry the
    food-only category lines."""
    from ph_economic_ai.engine.forum import Forum
    from ph_economic_ai.engine.auto_assemble import SectorContext

    captured = {}
    def fake_complete(msgs, **kw):
        captured['content'] = msgs[-1]['content']
        return 'Steady. ESTIMATE: +0.10/L'

    monkeypatch.setattr(llm_mod, 'complete', fake_complete)
    f = Forum(FakeRag(), [], as_of='2026-08-12', window='this_week')
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='exploratory',
                        anchor=None, social_counts={})
    f._judge_sector(ctx, finals=[])
    for label in ('RICE:', 'MEAT:', 'FISH:', 'DAIRY_EGGS:', 'VEGETABLES:', 'SUGAR:'):
        assert label not in captured['content'], f'{label} unexpectedly in gas judge prompt'


def test_placeholder_answer_parses_to_nothing():
    """Guards the regression directly: if an agent ever echoes a placeholder again,
    it must come through as 'no estimate' rather than a bogus number."""
    from ph_economic_ai.engine.forum import _extract_guarded
    assert _extract_guarded('gas', 'ESTIMATE: +X.XX/L') == (None, None)
    assert _extract_guarded('food', 'ESTIMATE: +X.X%') == (None, None)
