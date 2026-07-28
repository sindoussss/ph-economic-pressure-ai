"""The elimination bracket must select on quality, not on role position.

Observed in a live run: the console showed agent after agent eliminated with
`score=0.50`, including "Davao R2 Forecaster". 0.50 is exactly what
`compute_combined_score` returns when the critic score and the confidence both
fall back to their 0.5 defaults, so the whole field was tied.

`eliminate_bottom_n` then sorted on score alone. Python's sort is stable, `alive`
is ordered by `_ROLE_ORDER`, and `Forecaster` is first in that list -- so on a tie
the forecaster was eliminated first on every single run, and the
ConfidenceScorer, whose job is not forecasting, survived by construction.
"""
import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.engine.debate import AgentResponse


def _agent(name, role, score):
    a = swarm.SwarmAgent.__new__(swarm.SwarmAgent)
    a.name, a.role, a.combined_score, a.is_alive = name, role, score, True
    return a


def _field(score=0.50):
    """One agent per role, ordered the way the arena orders them."""
    agents = [_agent(role, role, score) for role in swarm._ROLE_ORDER]
    return sorted(agents, key=lambda a: swarm._ROLE_ORDER.index(a.role))


# ── the defect ───────────────────────────────────────────────────────────────

def test_defaults_produce_exactly_the_observed_score():
    """Pins where 0.50 came from, so the diagnosis stays reproducible."""
    resp = AgentResponse('A', 1, '', 'text', 1.0)
    assert swarm.compute_combined_score(
        resp, critic_score=0.5, confidence=0.5, group_estimates=[1.0]) == 0.50


def test_a_tied_field_is_detected_as_degenerate():
    assert swarm.scores_are_degenerate(_field()) is True


def test_a_differentiated_field_is_not_degenerate():
    agents = _field()
    agents[0].combined_score = 0.9
    assert swarm.scores_are_degenerate(agents) is False


def test_the_forecaster_is_not_eliminated_first_on_a_tie():
    """The regression. Before the fix this eliminated Forecaster on every run."""
    alive = _field()
    _, eliminated = swarm.eliminate_bottom_n(alive, n=1)
    assert eliminated[0].role != 'Forecaster'


def test_a_tie_does_not_follow_role_order():
    alive = _field()
    order = []
    for _rnd, n in swarm._BRACKET:
        alive, elim = swarm.eliminate_bottom_n(alive, n=n)
        order += [e.role for e in elim]
    assert order != swarm._ROLE_ORDER[:len(order)], (
        'elimination is still following the hardcoded role order')


def test_tie_break_is_reproducible():
    """Determinism still holds (ADR-002): same names, same outcome."""
    first = [e.role for _, e in
             [(None, x) for x in swarm.eliminate_bottom_n(_field(), n=2)[1]]]
    second = [e.role for _, e in
              [(None, x) for x in swarm.eliminate_bottom_n(_field(), n=2)[1]]]
    assert first == second


# ── real scores must still win ───────────────────────────────────────────────

def test_the_lowest_real_score_is_eliminated_regardless_of_role():
    alive = _field()
    for a in alive:
        a.combined_score = 0.9
    target = next(a for a in alive if a.role == 'ConfidenceScorer')
    target.combined_score = 0.1
    _, eliminated = swarm.eliminate_bottom_n(alive, n=1)
    assert eliminated[0].role == 'ConfidenceScorer'


def test_a_forecaster_that_genuinely_scores_worst_is_still_eliminated():
    """The fix must not make the forecaster immune, only unbiased."""
    alive = _field()
    for a in alive:
        a.combined_score = 0.9
    next(a for a in alive if a.role == 'Forecaster').combined_score = 0.05
    _, eliminated = swarm.eliminate_bottom_n(alive, n=1)
    assert eliminated[0].role == 'Forecaster'


# ── why the field tied: strict parsing ───────────────────────────────────────

@pytest.mark.parametrize('text', [
    'SCORE: Forecaster: 8',
    'SCORE: Forecaster: 8/10',
    '**SCORE: Forecaster: 8**',
    'SCORE - Forecaster - 8',
    'score: forecaster: 8',
])
def test_score_parsing_accepts_the_formats_models_actually_emit(text):
    assert swarm._parse_scores(text, ['Forecaster'])['Forecaster'] == pytest.approx(0.8)


@pytest.mark.parametrize('text,expected', [
    ('CONFIDENCE: Forecaster: 0.85', 0.85),
    ('CONFIDENCE: Forecaster: 85%', 0.85),
    ('CONFIDENCE: Forecaster: 85', 0.85),
    ('CONFIDENCE - Forecaster - 0.7', 0.7),
])
def test_confidence_parsing_handles_percent_and_bare_integers(text, expected):
    got = swarm._parse_confidence(text, ['Forecaster'])['Forecaster']
    assert got == pytest.approx(expected)


def test_an_unscored_agent_still_defaults_rather_than_crashing():
    assert swarm._parse_scores('no scores here', ['Forecaster'])['Forecaster'] == 0.5
    assert swarm._parse_confidence('nothing', ['Forecaster'])['Forecaster'] == 0.5


def test_confidence_is_clamped_to_a_probability():
    assert 0.0 <= swarm._parse_confidence(
        'CONFIDENCE: Forecaster: 250%', ['Forecaster'])['Forecaster'] <= 1.0


def test_parsed_scores_break_the_tie_that_caused_the_bias():
    """End to end: once the critic's scores parse, the field is no longer tied."""
    text = ' '.join(f'SCORE: {r}: {i + 3}' for i, r in enumerate(swarm._ROLE_ORDER))
    scores = swarm._parse_scores(text, swarm._ROLE_ORDER)
    assert len(set(scores.values())) == len(swarm._ROLE_ORDER)


# ── Estimate parsing feeds the whole bracket ─────────────────────────────────
# A missed estimate is not a small loss. compute_combined_score returns 0.0 when
# price_estimate is None, so one unparsed format ties an entire group at 0.00 and
# the elimination stops measuring anything. A live run showed exactly that:
# "all combined scores equal (0.00)" for every round of a group.

@pytest.mark.parametrize('text,expected', [
    ('ESTIMATE: -0.50/L', -0.5),
    ('ESTIMATE: +0.50/L', 0.5),
    ('ESTIMATE: 0.50/L', 0.5),
    ('ESTIMATE: -P0.50/L', -0.5),
    ('ESTIMATE: P-0.50/L', -0.5),          # currency BEFORE the sign
    ('ESTIMATE: PHP-0.50/L', -0.5),        # the format that was missing entirely
    ('ESTIMATE: -0.50 PHP/L', -0.5),
    ('**ESTIMATE: -0.50/L**', -0.5),
    ('ESTIMATE: - 0.50 / L', -0.5),
    ('ESTIMATE: minus 0.50/L', -0.5),
    ('ESTIMATE: plus 1.20/L', 1.2),
])
def test_every_format_a_model_writes_parses(text, expected):
    accepted, _ = swarm.parse_fuel_estimate(text)
    assert accepted == pytest.approx(expected)


def test_the_plausibility_guard_still_rejects_absurd_values():
    """Widening the parser must not widen what counts as a believable move."""
    accepted, rejected = swarm.parse_fuel_estimate('ESTIMATE: -150.00/L')
    assert accepted is None
    assert rejected == pytest.approx(-150.0)


def test_a_parsed_estimate_produces_a_nonzero_score():
    """The link between parsing and the bracket, stated directly."""
    from ph_economic_ai.engine.debate import AgentResponse
    parsed = AgentResponse('A', 1, '', 'ESTIMATE: -0.50/L', -0.5)
    unparsed = AgentResponse('B', 1, '', 'no number here', None)
    assert swarm.compute_combined_score(parsed, 0.5, 0.5, [-0.5]) > 0
    assert swarm.compute_combined_score(unparsed, 0.5, 0.5, [-0.5]) == 0.0


# ── The calibration rule must not point at a nonexistent anchor ──────────────
# It previously emitted the anchor bullets unconditionally, substituting the
# literal phrase "the ML anchor if supplied by the prompt". Agents were told to
# treat that sentence as their centre of gravity and stay within P1.00/L of it.

def _arena(ml_baseline):
    a = swarm.GroupArena.__new__(swarm.GroupArena)
    a._ml_baseline = ml_baseline
    return a


def test_no_anchor_means_no_anchor_bullets():
    rule = _arena('')._calibration_rule()
    assert 'center of gravity' not in rule
    assert 'ML anchor' not in rule
    assert 'if supplied by the prompt' not in rule


def test_no_anchor_still_keeps_the_output_constraints():
    """Those two bullets are what stop absolute pump prices and absurd values,
    so they must survive the branch."""
    rule = _arena('')._calibration_rule()
    assert 'Output only the next price CHANGE' in rule
    assert 'is invalid' in rule


def test_a_real_anchor_is_named_verbatim():
    rule = _arena('-1.12 PHP/L')._calibration_rule()
    assert '-1.12 PHP/L' in rule
    assert 'center of gravity' in rule
    assert 'that anchor' in rule


def test_the_anchored_rule_also_keeps_the_output_constraints():
    rule = _arena('-1.12 PHP/L')._calibration_rule()
    assert 'Output only the next price CHANGE' in rule


def test_the_two_branches_are_labelled_differently():
    """A reader of the prompt should be able to tell which regime is in force."""
    assert 'OUTPUT RULE' in _arena('')._calibration_rule()
    assert 'CALIBRATION RULE' in _arena('-1.12 PHP/L')._calibration_rule()


# ── Swarm agreement is about the AGENTS, not the judge ───────────────────────
# The card labels this "agent agreement", but it used to centre on the judge's
# verdict, which is anchored: when the agents give nothing usable or the guard
# clamps them, it lands on the mechanical pass-through. The number then measured
# distance from the anchor. A live run showed agents clustered near -2.0 with the
# judge on the anchor at -1.03, scoring 50% on screen.

def test_agreement_is_centred_on_the_agents_not_the_judge():
    agents = [-2.0, -2.0, -2.1, -1.9, -2.87, -2.8, -2.9, -2.0]
    on_anchor = swarm._robust_confidence_pct(agents, -1.03)
    on_agents = swarm._robust_confidence_pct(agents, -2.0)
    assert on_anchor == on_agents, (
        'the judge estimate still moves the number; it must not')
    assert on_anchor > 85, on_anchor


def test_a_unanimous_room_reads_one_hundred():
    """The 95 cap understated a room that genuinely agrees, and the Forum reports
    100. The two halves of the app must mean the same thing by 100 percent."""
    assert swarm._robust_confidence_pct([-1.0] * 8, -1.0) == 100


def test_a_genuine_split_stays_low():
    """The control. Two tight clusters far apart must not read as consensus just
    because each cluster is internally compact."""
    assert swarm._robust_confidence_pct([-3.0, -2.9, -3.1, 2.9, 3.0, 3.1], 0.0) < 45


def test_spread_is_measured_over_the_whole_room():
    """Measuring it on the agreeing subset only is near-tautological."""
    import inspect
    src = inspect.getsource(swarm._robust_confidence_pct)
    assert 'statistics.pstdev(valid)' in src
    assert 'pstdev(usable)' not in src


def test_a_single_outlier_does_not_dominate():
    seven_agree = [-1.0] * 7 + [6.0]
    assert swarm._robust_confidence_pct(seven_agree, -1.0) > 60
