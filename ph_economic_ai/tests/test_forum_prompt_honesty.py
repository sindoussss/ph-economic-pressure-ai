"""Prompt constraints that keep the forum honest to the benchmark.

These pin two failures observed in a real run.

1. The electricity judge reasoned "a decrease in mentions of electricity prices
   could indicate a temporary relief", inferring a price direction from search
   volume. The sentiment nowcast tested those exact terms and returned
   `no_better_than_naive` with skill 0.0, so the app was treating a measured-null
   channel as directional evidence.

2. The gas judge reported "significant recent increases in gas prices due to
   official announcements" as its read, which describes an adjustment that had
   already taken effect rather than pressure building into the next one.

Prompt text is normally not worth asserting on. These lines are, because each one
encodes an empirical result, and losing them silently would put the app back to
contradicting its own benchmark.
"""
import pytest

from ph_economic_ai.engine import forum


def test_social_channel_forbids_inferring_direction_from_attention():
    social = forum._CHANNEL_TEMPLATES['social']
    assert 'attention is NOT direction' in social
    assert 'naive baseline' in social
    assert 'falling mentions do not mean falling prices' in social


def test_social_channel_still_renders_per_sector():
    for sector in ('gas', 'food', 'electricity'):
        rendered = forum._CHANNEL_TEMPLATES['social'].format(sector=sector)
        assert sector in rendered
        assert '{' not in rendered, 'an unfilled placeholder would reach the model'


def test_judge_may_not_take_a_direction_from_mood():
    judge = forum._JUDGE_SYSTEM
    assert 'WEIGHTING RULE' in judge
    assert 'may NOT on its own decide' in judge
    assert 'keep the estimate near zero' in judge


def test_judge_is_warned_off_restating_a_change_already_in_effect():
    assert 'BACKWARD-LOOKING TRAP' in forum._JUDGE_SYSTEM
    assert 'already in the price' in forum._JUDGE_SYSTEM


def test_judge_still_refuses_to_become_a_forecaster():
    """The new dated anchor must not have turned a present read into a forecast."""
    judge = forum._JUDGE_SYSTEM
    assert 'never a confident' in judge
    assert 'present read' in judge


@pytest.mark.parametrize('sector,expect', [
    ('gas', 'fuel adjustment cadence'),
    ('food', 'cpi cadence'),
    ('electricity', 'cpi cadence'),
])
def test_each_sector_is_anchored_to_a_real_dated_event(sector, expect):
    label = forum._next_change_label(sector)
    assert label and label != 'the next scheduled adjustment'
    # A real date carries a year; the fallback string does not.
    assert '20' in label


def test_gas_and_cpi_sectors_use_different_schedules():
    """Fuel moves weekly, CPI monthly. Collapsing them would put a wrong date on
    two of the three cards."""
    assert forum._next_change_label('gas') != forum._next_change_label('food')
    assert forum._next_change_label('food') == forum._next_change_label('electricity')


def test_label_degrades_to_a_safe_phrase_if_the_calendar_fails(monkeypatch):
    """A broken clock must not put a wrong date in front of a user."""
    from ph_economic_ai.engine import price_calendar

    def boom(*a, **k):
        raise RuntimeError('calendar unavailable')

    monkeypatch.setattr(price_calendar, 'describe_next_fuel_adjustment', boom)
    assert forum._next_change_label('gas') == 'the next scheduled adjustment'


# ── Agreement is centred on the median ───────────────────────────────────────
# The mean is dragged by one extreme agent, moving the centre off the cluster and
# counting agents who genuinely agree as disagreeing. The regional verdicts in a
# live run were -0.20 and -2.15, exactly that shape, with gas agreement at 54%.

def _medoid(ests):
    return min(ests, key=lambda c: (sum(abs(e - c) for e in ests), c))


def _agreement(ests, sector='gas', centre_fn=None, band=None):
    """Reimplements the shipped formula so centres and bands can be compared.

    `band` defaults to the shipped one. Tests that demonstrate a historical
    artefact pass the band it occurred at explicitly, so the evidence stays
    readable after the shipped band changes.
    """
    if band is None:
        band = forum._BAND.get(sector, 0.2)
    n = len(ests)
    centre = (centre_fn or _medoid)(ests)
    within = sum(1 for e in ests if abs(e - centre) <= band)
    return int((within / n) * 100 * min(n, 2) / 2)


def test_a_far_outlier_destroys_the_mean_centre_but_not_the_medoid():
    """Six agents agreeing near -0.6 with one at +6.0. The mean lands at +0.34,
    off every cluster member, and scores the room at 0. The medoid is an estimate
    an agent gave, so the six who agree are counted."""
    ests = [-0.6, -0.5, -0.7, -0.55, -0.65, -0.6, 6.0]
    mean_based = _agreement(ests, centre_fn=lambda x: sum(x) / len(x))
    medoid_based = _agreement(ests)
    assert mean_based == 0, mean_based
    assert medoid_based > 80, medoid_based


def test_a_genuine_half_split_reads_half_not_zero():
    """Three agents near -0.6 and three near -2.15. The abstract median lands at
    -1.33, a value nobody said, in the gap between the clusters, and read 0%
    against a room where half the agents DO agree with each other. The medoid is
    an actual estimate, so a clean half-split reads 50%: low, honest, and not a
    quantisation artefact."""
    ests = [-0.6, -0.5, -0.55, -2.1, -2.15, -2.2]
    assert _agreement(ests) == 50
    assert _agreement(ests, centre_fn=lambda x: sum(x) / len(x)) == 0


def test_the_live_gap_artefact_is_fixed():
    """The mixed live run: six of eight agents between 1.0 and 2.0, scored 0%.

    Two compounding artefacts, both shown at the band they occurred at (0.20).
    The median landed at 1.25, a value NO agent said, in the gap between the two
    sub-clusters. The band was also finer than the 0.5 grid the models emit on, so
    it could only ever match exactly. Fixing the centre alone lifts it off zero;
    the shipped configuration does better still.
    """
    import statistics
    ests = [-1.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 6.0]
    assert _agreement(ests, centre_fn=statistics.median, band=0.20) == 0   # artefact
    assert _agreement(ests, band=0.20) > 0                                 # medoid alone
    assert _agreement(ests) > _agreement(ests, band=0.20)                  # + band
    assert _medoid(ests) in ests, 'the centre must be a value an agent gave'


def test_a_tight_field_still_scores_full_agreement():
    assert _agreement([-0.6, -0.55, -0.65, -0.6, -0.58]) == 100


def test_the_band_is_never_wide_enough_to_merge_genuine_disagreement():
    """The band moved from 0.20 to 0.50 for gas, so the guard that matters is no
    longer "unchanged" but "still separates real disagreement".

    Two controls: clusters 1.5 apart and 0.9 apart. Both must stay at 50 percent,
    i.e. the band may never fuse two genuinely separate groups into a consensus.
    A band of 1.00 fails the second control, which is why it was rejected despite
    producing better live numbers.
    """
    for control in ([-0.6, -0.5, -0.55, -2.1, -2.15, -2.2],
                    [-0.6, -0.5, -0.55, -1.4, -1.45, -1.5]):
        assert _agreement(control) == 50, control
    assert forum._BAND['gas'] <= 0.50, 'gas band widened past the control limit'
    assert forum._BAND['food'] == 0.3
    assert forum._BAND['electricity'] == 0.10


def test_a_band_of_one_peso_would_have_failed_the_control():
    """Pins the rejected alternative, so the reasoning survives the decision."""
    near_split = [-0.6, -0.5, -0.55, -1.4, -1.45, -1.5]
    assert _agreement(near_split) == 50                           # at the shipped band
    band = 1.00
    c = _medoid(near_split)
    n = len(near_split)
    inflated = int(sum(1 for e in near_split if abs(e - c) <= band) / n * 100)
    assert inflated == 100, 'the rejected band no longer inflates; revisit the choice'


def test_the_shipped_code_uses_the_medoid():
    import inspect
    src = inspect.getsource(forum.Forum._aggregate)
    assert 'min(ests, key=' in src                # medoid: an actual estimate
    assert 'centre = sum(ests) / n' not in src    # never the draggable mean


def test_direction_agreement_is_computed_and_coarser():
    """Live rising-oil run: [1.0 x2, 2.0 x4, 2.5 x2]. Magnitude agreement 50%,
    but every agent said up. Both numbers are true; they answer different
    questions, and the card shows them side by side rather than swapping one
    for the other."""
    import inspect
    src = inspect.getsource(forum.Forum._aggregate)
    assert 'direction_agreement' in src
    from ph_economic_ai.engine.pressure_brief import SectorReading
    r = SectorReading(sector='gas', direction='rising', estimate=2.0,
                      unit='PHP/L', confidence=50, direction_agreement=100)
    assert r.direction_agreement == 100


# ── Round 2 is Delphi feedback, not a pre-seeded anchor ──────────────────────

def _resp(name, est):
    from ph_economic_ai.engine.debate import AgentResponse
    return AgentResponse(name, 1, '', f'reasoning by {name}', est)


def _ctx(sector='gas', unit='PHP/L'):
    from types import SimpleNamespace
    return SimpleNamespace(sector=sector, unit=unit)


def test_delphi_line_states_the_median_not_the_mean():
    responses = [_resp('a', -0.6), _resp('b', -0.5), _resp('c', -0.7),
                 _resp('d', -0.6), _resp('e', 2.0)]          # one outlier
    line = forum.Forum._delphi_line(_ctx(), responses)
    assert '-0.60' in line, line                              # median, unmoved
    assert '+0.12' not in line                                # the dragged mean


def test_delphi_line_offers_stay_and_cite_not_only_convergence():
    """The outlier must have a stated path to keep its number. Feedback that only
    says 'move' is pressure, not deliberation."""
    line = forum.Forum._delphi_line(_ctx(), [_resp('a', -0.6), _resp('b', -0.5)])
    assert 'cite' in line
    assert 'revise toward' in line


def test_delphi_line_needs_at_least_two_estimates():
    assert forum.Forum._delphi_line(_ctx(), [_resp('a', -0.6)]) == ''
    assert forum.Forum._delphi_line(_ctx(), []) == ''


def test_divergent_selection_centres_on_the_median():
    """With a mean centre, the outlier at +2.0 drags the centre to +0.12 and the
    cluster members look almost as divergent as the outlier. The median keeps the
    centre on the cluster so the outlier is ranked first."""
    from types import SimpleNamespace
    agents = [SimpleNamespace(name=n) for n in 'abcde']
    responses = [_resp('a', -0.6), _resp('b', -0.5), _resp('c', -0.7),
                 _resp('d', -0.6), _resp('e', 2.0)]
    f = forum.Forum.__new__(forum.Forum)
    picked = f._divergent(_ctx(), responses, agents, k=1)
    assert [a.name for a in picked] == ['e']


# ── Round 2 reaches the agents who actually disagree ─────────────────────────
# It used to invite a fixed top-4 by rank, so on a 50-agent roster 46 never saw
# the group's centre and stayed locked to a round-1 answer given before any
# feedback existed. Agreement was measured over a room that had not been told
# what the room thought.

def _agents(names):
    from types import SimpleNamespace
    return [SimpleNamespace(name=n) for n in names]


def test_only_agents_outside_the_band_are_invited():
    f = forum.Forum.__new__(forum.Forum)
    resps = [_resp('a', -0.6), _resp('b', -0.6), _resp('c', -0.55),
             _resp('d', -0.65), _resp('e', -2.0)]
    picked = [a.name for a in f._divergent(_ctx(), resps, _agents('abcde'), k=16)]
    assert picked == ['e']


def test_a_unanimous_room_spends_no_calls():
    """Nothing to resolve means nothing to ask."""
    f = forum.Forum.__new__(forum.Forum)
    resps = [_resp(n, -0.6) for n in 'abcdef']
    assert f._divergent(_ctx(), resps, _agents('abcdef'), k=16) == []


def test_the_cap_bounds_cost_on_a_badly_split_room():
    f = forum.Forum.__new__(forum.Forum)
    resps = [_resp(f'a{i}', float(i)) for i in range(40)]
    picked = f._divergent(_ctx(), resps, _agents([f'a{i}' for i in range(40)]), k=16)
    assert len(picked) == 16


def test_invitations_are_ranked_by_distance_from_the_centre():
    f = forum.Forum.__new__(forum.Forum)
    resps = [_resp('near', -1.4), _resp('far', 5.0),
             _resp('a', -0.6), _resp('b', -0.6), _resp('c', -0.6)]
    picked = [a.name for a in
              f._divergent(_ctx(), resps, _agents(['near', 'far', 'a', 'b', 'c']), k=16)]
    assert picked[0] == 'far'


def test_divergent_uses_the_same_centre_as_the_agreement_metric():
    """If the two disagreed, round 2 would chase agents the metric considers fine."""
    import inspect
    src = inspect.getsource(forum.Forum._divergent)
    assert 'min(values, key=' in src          # medoid, as in _aggregate
    assert 'statistics.median' not in src
    assert '_BAND' in src                      # the metric's own band


# ── The judge may verify, but not originate ──────────────────────────────────
# The judge had no evidence at all and a blanket "do NOT introduce new facts",
# so it could only average what the analysts said, including claims nothing
# supported. It now reads the same sector corpus they did. The ban narrowed from
# "no facts" to "no NEW drivers": check the room, do not replace it.

def test_judge_may_check_the_analysts_against_evidence():
    j = forum._JUDGE_SYSTEM
    assert 'EVIDENCE RULE' in j
    assert 'discount that analyst' in j
    assert 'Do NOT introduce new facts' not in j     # the blanket ban is gone


def test_judge_still_may_not_originate_a_driver():
    """Reading evidence must not turn the judge into a 51st analyst."""
    j = forum._JUDGE_SYSTEM
    assert 'may NOT introduce a driver no analyst raised' in j
    assert 'you resolve it' in j


def test_judge_reports_disagreement_rather_than_overruling_the_panel():
    assert 'report the disagreement rather than substituting your own read' in forum._JUDGE_SYSTEM


def test_judge_evidence_comes_from_the_sector_corpus_not_a_fresh_query():
    """It checks what the room was working from; it does not go hunting."""
    import inspect
    src = inspect.getsource(forum.Forum._judge_sector)
    assert 'sector_corpus(ctx.sector)' in src
    assert 'judge_evidence' in src


def test_judge_still_works_when_retrieval_fails():
    """A judge that cannot read must still resolve, not crash."""
    import inspect
    src = inspect.getsource(forum.Forum._judge_sector)
    assert 'except Exception' in src
    assert "judge_evidence = ''" in src


def test_the_forecast_ban_survived_the_change():
    assert 'never a confident' in forum._JUDGE_SYSTEM
    assert 'present read' in forum._JUDGE_SYSTEM
