"""What "agent agreement" is measured over.

The report showed a 38% headline sitting directly above two regional cards
reading 74% and 53%. All three said "agent agreement" and all three used the
same function; they disagreed because they were fed different populations. The
cards read every agent estimate in their groups. The headline read the group
SURVIVORS — one agent per group, two per judge — and when one of those two had
no parseable number the pair was dropped entirely, leaving the national figure
resting on a single pair of agents.

38 is also the floor of the metric for any two estimates ₱3.00 or more apart, so
the headline was a constant reported as a measurement.

These tests pin the population, not the percentage.
"""
from unittest.mock import MagicMock, patch

import pytest

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.engine.swarm import (
    GroupSurvivor, MasterJudge, RegionalVerdict, _robust_confidence_pct,
    measure_agreement,
)

SCENARIO = {'oil_pct': -2.0, 'usd_pct': 0.0, 'current_price': 98.82}

# Run 28, 2026-07-28 — the run in the screenshot. Estimates as recorded in
# cache/trust.db, grouped by region. Kept as literals so the incident stays
# reproducible without the database.
RUN_28 = {
    0: [-2.5],                                  # NCR: two agents parsed to None
    1: [-4.0, -2.0, -2.0],                      # Central Luzon
    2: [0.5, -1.77, -5.0],                      # Western Visayas
    3: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],          # Davao: unanimous, wrong sign
}


def _history(estimates, round_num=1):
    return [AgentResponse(f'agent{i}', round_num, '', 'x', e)
            for i, e in enumerate(estimates)]


def _histories(by_group):
    return {gid: _history(est) for gid, est in by_group.items()}


def _survivor(group_id, estimate):
    return GroupSurvivor(
        group_id=group_id, region_name=f'R{group_id}',
        response=AgentResponse('winner', 2, '', 'x', estimate),
        combined_score=0.8, agent_role='Forecaster', agent_model='fast',
    )


def _verdict(estimate, judge_id=0):
    return RegionalVerdict(
        judge_id=judge_id, region_pair=('A', 'B'), estimate=estimate,
        confidence=0.7, reasoning='x', survivor_names=('a', 'b'),
    )


def _master(by_group, survivors=(), statement='ESTIMATE: -₱1.00/L', anchor=None):
    judge = MasterJudge(
        verdicts=[_verdict(-1.0, 0), _verdict(-1.2, 1)],
        rag=MagicMock(), scenario=SCENARIO,
        survivors=list(survivors),
        group_histories=_histories(by_group),
        anchor=anchor,
    )
    with patch('ph_economic_ai.engine.swarm.llm.stream', return_value=[statement]):
        return judge.run()


# ── The population ────────────────────────────────────────────────────────────

def test_agreement_is_measured_over_agents_not_survivors():
    """The defect, stated directly: 20 agents ran, 4 survived, 4 were counted."""
    mv = _master(
        {0: [-1.0, -1.1, -0.9, -1.2, -1.0],
         1: [-1.0, -1.1, -0.9, -1.2, -1.0],
         2: [-1.0, -1.1, -0.9, -1.2, -1.0],
         3: [-1.0, -1.1, -0.9, -1.2, -1.0]},
        survivors=[_survivor(g, -1.0) for g in range(4)],
    )
    assert mv.agreement_n == 20
    assert mv.agreement_regions == (4, 4)


def test_the_headline_is_never_measured_over_fewer_agents_than_a_card():
    """A regional card reads two groups. The headline reads all four, so it can
    never rest on a smaller population than the card printed beneath it."""
    by_group = {0: [-1.0, -1.1], 1: [-1.0, -0.9, -1.3],
                2: [-2.0, -2.1], 3: [-2.0, -2.2, -1.8, -2.1]}
    mv = _master(by_group, survivors=[_survivor(g, -1.0) for g in range(4)])
    biggest_card = max(len(by_group[0]) + len(by_group[1]),
                       len(by_group[2]) + len(by_group[3]))
    assert mv.agreement_n >= biggest_card


def test_a_survivor_without_an_estimate_no_longer_deletes_a_region():
    """Run 28's NCR survivor parsed to None and took the whole NCR & Central
    Luzon pair out of the average with it."""
    mv = _master({0: [-1.0, -1.1, -0.9], 1: [-1.0, -1.2, -1.1]},
                 survivors=[_survivor(0, None), _survivor(1, -1.0)])
    assert mv.agreement_n == 6
    assert mv.agreement_regions == (2, 2)
    assert mv.confidence_pct > 50


def test_the_run_28_headline_stops_being_a_two_agent_measurement():
    mv = _master(RUN_28, survivors=[_survivor(2, -5.0), _survivor(3, 2.0)])
    assert mv.agreement_n == 12          # was 2
    assert mv.agreement_regions == (3, 4)   # NCR never produced a second estimate
    assert mv.confidence_pct != 38


def test_the_run_28_room_is_still_reported_as_divided():
    """The fix must not be an inflation. Davao unanimous at the wrong sign and
    Western Visayas spanning ₱5.50 is a divided room, and the number should say
    so — just over the whole room rather than over two of its members."""
    mv = _master(RUN_28, survivors=[_survivor(2, -5.0), _survivor(3, 2.0)])
    assert 50 <= mv.confidence_pct <= 75


def test_a_group_that_never_produced_two_estimates_is_counted_out_loud():
    mv = _master({0: [-1.0], 1: [-1.0, -1.1, -1.05]})
    assert mv.agreement_regions == (1, 2)


def test_regions_at_different_levels_are_not_a_disagreement():
    """Freight multipliers mean Davao is SUPPOSED to differ from NCR. Two
    internally unanimous groups at different levels is agreement, not conflict —
    the cross-region gap is reported separately as dissent."""
    mv = _master({0: [-0.50, -0.52, -0.48], 1: [-2.90, -2.88, -2.92]})
    assert mv.confidence_pct > 90


def test_nothing_usable_reports_no_population_rather_than_a_percentage():
    mv = _master({0: [], 1: []})
    assert mv.agreement_n == 0
    assert mv.agreement_regions == (0, 2)


def test_the_fallback_pools_agents_not_judges():
    """When no single group reaches quorum the pool is still the agents. The old
    fallback scored the survivors against the regional verdicts — two judge
    outputs, published as "agent agreement"."""
    mv = _master({0: [-1.0], 1: [-1.1], 2: [-0.9], 3: [-1.2]},
                 survivors=[_survivor(g, 5.0) for g in range(4)])
    assert mv.agreement_n == 4             # the four agents, not the survivors
    assert mv.agreement_regions == (0, 4)  # pooled: no per-region split possible
    assert mv.confidence_pct > 90          # and those four agents do agree


# ── The metric itself ─────────────────────────────────────────────────────────

def test_a_lone_estimate_is_not_sixty_five_percent():
    """`return 65 if valid else 0` — a hardcoded confidence dressed as a
    measurement, the same defect as the 0.75 the regional cards used to ship."""
    assert _robust_confidence_pct([-1.03], None) == 0


def test_zero_is_reserved_for_nothing_to_measure():
    """A real room cannot reach 0: the centre always agrees with itself, so the
    floor for n estimates is 1/n. That keeps 0 unambiguous."""
    assert _robust_confidence_pct([], None) == 0
    scattered = _robust_confidence_pct([-8.0, -4.0, 0.0, 4.0, 8.0], None)
    assert scattered > 0


def test_measure_agreement_returns_the_count_with_the_number():
    pct, n = measure_agreement([-1.0, -1.1, None, -0.9, 99.0])
    assert n == 3                     # None and the absolute-price parse dropped
    assert pct == _robust_confidence_pct([-1.0, -1.1, -0.9], None)


def test_two_estimates_still_score_but_the_caller_can_see_it_was_two():
    """The metric is near-binary on a pair, and past the bands it pins to 38
    however far apart the two actually are. The count is what lets the report
    refuse to headline it."""
    assert measure_agreement([-1.0, -4.0]) == (38, 2)
    assert measure_agreement([-1.0, -7.0]) == (38, 2)   # ₱6 apart, same number


@pytest.mark.parametrize('gap,expected', [(0.50, 96), (0.51, 71), (3.00, 38)])
def test_the_two_point_cliff_is_real_and_documented(gap, expected):
    """The cliff sits at the agreement band, so it moved with it.

    It was pinned at ₱1.00 -> 92 and ₱1.01 -> 67 while the band was 1.00. The
    band is now 0.50, matching the Forum and the control study that disqualified
    1.00 for merging two clusters 0.9 apart into a consensus, and the cliff moved
    to match. The property being pinned is unchanged: on a two-point sample this
    metric is a step function, which is why the population count is reported
    beside it.
    """
    assert measure_agreement([-1.0, -1.0 - gap])[0] == expected


def test_the_band_matches_the_forum():
    """Two halves of one app must not disagree about what agreement means. The
    swarm sat at 1.00 while the Forum moved to 0.50 on a control study, and the
    swarm was simply never brought along."""
    from ph_economic_ai.engine import forum, swarm as swarm_mod
    assert swarm_mod._AGREEMENT_BAND == forum._BAND['gas']


def test_the_disqualified_band_would_have_called_a_split_a_consensus():
    """The control the vault ran and the swarm never adopted: two tight clusters
    0.9 apart. Under the old band this scored 92 percent."""
    split = [-2.0, -2.0, -2.0, -2.0, -1.1, -1.1, -1.1, -1.1]
    assert _robust_confidence_pct(split, None) < 70


# ── How much of the agreement is the anchor talking to itself ────────────────

def test_anchor_echoes_are_counted(store=None):
    """Every agent is handed the mechanical pass-through, so an agent restating
    it scores identically to one independently arriving at it. Measured at 25.0
    percent of estimates locally and 19.4 percent hosted, which is about a
    quarter of the number the card reports."""
    mv = _master({0: [-2.42, -2.42, -2.0], 1: [-2.42, -1.0, 0.5]},
                 anchor=-2.42)
    assert mv.agreement_n == 6
    assert mv.agreement_echo_n == 3


def test_a_room_that_ignores_the_anchor_reports_no_echo():
    mv = _master({0: [-1.0, -1.1, -0.9], 1: [-1.2, -1.05, -0.95]}, anchor=-2.42)
    assert mv.agreement_echo_n == 0


def test_the_echo_count_only_covers_the_measured_population():
    """A group below quorum is not scored, so its estimates must not be counted
    as echoes either — the two numbers have to describe the same room."""
    mv = _master({0: [-2.42], 1: [-2.42, -2.42, -1.0]}, anchor=-2.42)
    assert mv.agreement_n == 3
    assert mv.agreement_echo_n == 2


def test_the_card_states_the_echo_share():
    from ph_economic_ai.ui import honesty
    line = honesty.agreement_basis(15, (4, 4), echo_n=5)
    assert '15 agent estimates' in line
    assert '5 of them (33%)' in line
    assert 'restate the physical anchor' in line


def test_the_card_stays_quiet_when_nothing_echoed():
    from ph_economic_ai.ui import honesty
    assert 'restate' not in honesty.agreement_basis(15, (4, 4), echo_n=0)


# ── A percentage cannot tell consensus from copying ──────────────────────────

def _rich(by_group):
    """Histories with distinct statements, so diversity is measurable."""
    return {gid: [AgentResponse(f'a{i}', 1, '', text, est)
                  for i, (est, text) in enumerate(items)]
            for gid, items in by_group.items()}


def _master_rich(by_group, anchor=None):
    judge = MasterJudge(
        verdicts=[_verdict(-1.0, 0)], rag=MagicMock(), scenario=SCENARIO,
        group_histories=_rich(by_group), anchor=anchor)
    with patch('ph_economic_ai.engine.swarm.llm.stream',
               return_value=['ESTIMATE: -₱1.00/L']):
        return judge.run()


def test_a_collapsed_room_is_visible_despite_a_perfect_percentage():
    """The finding a blind-arm experiment produced: 32 agents, 2 distinct
    estimates, 100 percent agreement. Blinding them tripled the distinct values
    and widened the spread nine-fold while the percentage moved 8 points, which
    is the metric failing to respond to the thing that matters."""
    mv = _master_rich({0: [(-2.10, 'Prices ease.')] * 6,
                       1: [(-2.20, 'Prices ease.')] * 6})
    assert mv.confidence_pct >= 95
    assert mv.agreement_distinct == 2
    assert mv.agreement_diversity < 0.5


def test_a_varied_room_reports_its_variety():
    mv = _master_rich({0: [(-2.1, 'Crude fell.'), (-1.4, 'Freight lags.'),
                           (-2.6, 'Peso steady.')],
                       1: [(-1.9, 'Demand soft.'), (-2.4, 'Parity drops.'),
                           (-1.6, 'Stocks high.')]})
    assert mv.agreement_distinct == 6
    assert mv.agreement_diversity == 1.0


def test_the_counts_describe_the_scored_population_only():
    """A group below quorum is not scored, so it must not contribute distinct
    values either — otherwise the two numbers describe different rooms."""
    mv = _master_rich({0: [(-2.10, 'Only one here.')],
                       1: [(-2.10, 'A.'), (-2.20, 'B.'), (-3.30, 'C.')]})
    assert mv.agreement_n == 3
    assert mv.agreement_distinct == 3


def test_the_caveat_fires_on_a_narrowed_room():
    from ph_economic_ai.ui import honesty
    text = honesty.agreement_caveat(16, distinct=2, diversity=0.06)
    assert 'only 2 distinct values' in text
    assert 'one view held widely' in text


def test_the_caveat_stays_silent_on_a_healthy_room():
    from ph_economic_ai.ui import honesty
    assert honesty.agreement_caveat(16, distinct=8, diversity=1.0) == ''


def test_the_caveat_needs_a_measurable_population():
    from ph_economic_ai.ui import honesty
    assert honesty.agreement_caveat(1, distinct=1, diversity=0.0) == ''


def test_the_basis_line_states_the_distinct_count():
    from ph_economic_ai.ui import honesty
    line = honesty.agreement_basis(32, (4, 4), echo_n=0, distinct=2)
    assert 'taking 2 distinct values' in line


# ── Blinding is the shipped default ──────────────────────────────────────────

def test_the_estimating_roles_are_blind_by_default():
    """Shipped 2026-07-29. An agent forming its opening read no longer sees what
    its neighbours just said, which is what the Forum has always done and the
    swarm never did.

    Measured over one paired run: 2 distinct estimates became 6 and the spread
    went from 0.26 to 2.50 PHP/L, at a cost of 8 points of reported agreement.
    The lower number over six real opinions is the better one; the previous
    configuration was scoring the room's collapse as its consensus.
    """
    from ph_economic_ai.engine.swarm import GroupArena, build_swarm_agents

    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    arena = GroupArena(group_id=0, agents=agents, rag=rag, scenario=SCENARIO)
    peers = [AgentResponse('NCR Forecaster', 1, '', 'Prices fall sharply.', -2.0)]

    for role in ('Forecaster', 'DataExtractor', 'Synthesizer'):
        agent = next(a for a in agents if a.role == role)
        prompt = ' '.join(m['content'] for m in arena._build_prompt(agent, 1, peers))
        assert 'This round so far' not in prompt, f'{role} can still see its peers'


def test_the_scoring_roles_still_read_their_peers():
    """They score other agents by name. A blind Critic has nothing to critique,
    and the elimination bracket then measures nothing."""
    from ph_economic_ai.engine.swarm import GroupArena, build_swarm_agents

    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    arena = GroupArena(group_id=0, agents=agents, rag=rag, scenario=SCENARIO)
    peers = [AgentResponse('NCR Forecaster', 1, '', 'Prices fall sharply.', -2.0)]

    for role in ('Critic', 'ConfidenceScorer'):
        agent = next(a for a in agents if a.role == role)
        prompt = ' '.join(m['content'] for m in arena._build_prompt(agent, 1, peers))
        assert 'This round so far' in prompt, f'{role} was blinded and cannot score'


def test_earlier_rounds_are_never_hidden():
    """Round 2 exists so an agent can respond to what the room said. Blinding is
    about same-round contamination, not about removing the debate."""
    from ph_economic_ai.engine.swarm import GroupArena, build_swarm_agents

    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    arena = GroupArena(group_id=0, agents=agents, rag=rag, scenario=SCENARIO)
    arena._history = [AgentResponse('a', 1, '', 'Round one view.', -2.0),
                      AgentResponse('b', 1, '', 'Another view.', -1.0)]
    agent = next(a for a in agents if a.role == 'Forecaster')
    prompt = ' '.join(m['content'] for m in arena._build_prompt(agent, 2, []))
    assert 'Previous rounds' in prompt


def test_the_orchestrator_ships_the_same_default():
    """A default that only holds when GroupArena is constructed directly would
    never reach a real run."""
    from ph_economic_ai.engine.swarm import SwarmOrchestrator

    orch = SwarmOrchestrator(rag=MagicMock(), scenario=SCENARIO)
    assert orch._blind_round_one is True
    assert orch._reconcile is True, (
        'the reconciliation rule was left on: the blind-arm experiment could not '
        'separate its effect from noise')


def test_peer_visibility_can_still_be_turned_back_on():
    """The experiment has to be able to re-run the old configuration."""
    from ph_economic_ai.engine.swarm import GroupArena, build_swarm_agents

    rag = MagicMock()
    rag.query.return_value = []
    agents = [a for a in build_swarm_agents() if a.group_id == 0]
    arena = GroupArena(group_id=0, agents=agents, rag=rag, scenario=SCENARIO,
                       blind_round_one=False)
    peers = [AgentResponse('NCR Forecaster', 1, '', 'Prices fall sharply.', -2.0)]
    agent = next(a for a in agents if a.role == 'Forecaster')
    prompt = ' '.join(m['content'] for m in arena._build_prompt(agent, 1, peers))
    assert 'This round so far' in prompt


# ── The diversity metric must be able to reach its own maximum ───────────────

def _resp(name, rnd, text):
    return AgentResponse(name, rnd, '', text, -1.0)


def test_diversity_can_reach_one():
    """The first version divided distinct openings by RESPONSES, and an agent
    speaks in both rounds. 32 responses from 20 agents capped it at 0.625, so the
    0.5 caveat threshold sat at four fifths of an unreachable maximum and would
    have fired on almost every healthy run. One statement per agent."""
    from ph_economic_ai.engine.swarm import opening_diversity

    rs = [_resp(f'a{i}', 1, f'Opening view {i}.') for i in range(20)]
    rs += [_resp(f'a{i}', 2, f'Opening view {i}.') for i in range(12)]
    assert opening_diversity(rs) == 1.0


def test_diversity_still_catches_herding():
    from ph_economic_ai.engine.swarm import opening_diversity

    rs = [_resp(f'a{i}', 1, 'Prices will ease this week.') for i in range(15)]
    rs += [_resp(f'a{15 + i}', 1, f'Distinct take {i}.') for i in range(5)]
    assert opening_diversity(rs) == pytest.approx(0.3)


def test_a_reworded_second_round_does_not_inflate_diversity():
    """Round 2 is SUPPOSED to respond to the room, so it says nothing about
    whether the opening read was independent."""
    from ph_economic_ai.engine.swarm import opening_diversity

    rs = [_resp('a', 1, 'Same opening.'), _resp('a', 2, 'A different follow up.'),
          _resp('b', 1, 'Same opening.'), _resp('b', 2, 'Another follow up.')]
    assert opening_diversity(rs) == pytest.approx(0.5)


def test_diversity_reads_the_earliest_round_per_agent():
    """The blind round is where contamination would show."""
    from ph_economic_ai.engine.swarm import opening_diversity

    rs = [_resp('a', 2, 'Late and distinct.'), _resp('a', 1, 'Shared opening.'),
          _resp('b', 1, 'Shared opening.')]
    assert opening_diversity(rs) == pytest.approx(0.5)


def test_diversity_of_an_empty_room_is_zero():
    from ph_economic_ai.engine.swarm import opening_diversity
    assert opening_diversity([]) == 0.0
