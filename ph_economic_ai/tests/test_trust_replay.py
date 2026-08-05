"""Trust must be reproducible from the evidence that still stands.

The residue `RSK-023` left behind. Withdrawing a grade cleared the grade and
left the trust it had moved, because trust was a running EMA: it kept a
conclusion and destroyed the evidence for it. The roster carried movement from
grades that no longer existed, and the run showed as ungraded while the agents
still wore the score it had given them.

Inverting the EMA is not the fix. `old = (new - a*raw)/(1-a)` holds only when
nothing clamped and nothing has happened since, and by the time a grade is
withdrawn both are usually false. Trust is now DEFINED as the replay of its
event log, so removing evidence removes its effect exactly.
"""
import sqlite3

import pytest

from ph_economic_ai.engine.store import (
    _EMA_ALPHA, _TRUST_INIT, AgentTrustStore, trust_tier)


def _wrong_week(store, run_id):
    """A `graded_against` stamp from a week the run did not forecast.

    Derived from the run rather than written in. A hardcoded date lands in the
    same cycle as the target on some days of the year and a different one on
    others, and the test would then pass or fail by calendar.
    """
    from datetime import timedelta
    week = store.target_cycle(dict(store.get_run(run_id)))
    return f'{(week - timedelta(days=7)).isoformat()} (gap 0.71d)'


def _responses(names, estimate=1.0, internal=0.8):
    return [{'agent_name': n, 'round_num': 1, 'estimate': estimate,
             'statement': 'x', 'citation_count': 1, 'has_causal_chain': 1,
             'internal_score': internal, 'model_used': 'm'} for n in names]


# ── the property the whole change exists for ─────────────────────────────────

def test_withdrawing_a_grade_returns_trust_to_where_it_was(due_run):
    """Grade, withdraw, and every agent is exactly where it started.

    "Exactly" is the point. The old behaviour left the movement in place and
    described it as fading, which is true of a decay and not of a correction.
    """
    store, run_id = due_run(baseline=84.38, estimate=-1.28, price=89.51)
    store.save_agent_responses(run_id, _responses(['A', 'B']))
    store.update_trust('A', internal_score=0.8)
    store.update_trust('B', internal_score=0.4)
    before = store.get_all_trust()

    store.apply_ground_truth_grade(run_id, actual_change=5.13,
                                   graded_against=_wrong_week(store, run_id))
    graded = store.get_all_trust()
    assert graded != before, 'the grade must actually move trust, or nothing is proven'

    assert [w['run_id'] for w in store.withdraw_cross_cycle_grades()] == [run_id]
    assert store.get_all_trust() == pytest.approx(before)


def test_the_withdrawal_reports_what_it_moved(due_run):
    """A silent correction to a trust score is the same class of problem as the
    residue it fixes."""
    store, run_id = due_run(baseline=84.38, estimate=-1.28, price=89.51)
    store.save_agent_responses(run_id, _responses(['A']))
    store.update_trust('A', internal_score=0.8)
    store.apply_ground_truth_grade(run_id, actual_change=5.13,
                                   graded_against=_wrong_week(store, run_id))
    entry = store.withdraw_cross_cycle_grades()[0]
    assert 'A' in entry['trust_moved']
    moved = entry['trust_moved']['A']
    assert moved['before'] != moved['after']
    assert moved['events'] >= 1


def test_a_correct_grade_survives_a_withdrawal_pass(due_run):
    """The guard has to discriminate. A cycle-aligned grade keeps its trust."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _responses(['A'], estimate=-0.6))
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    assert find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 1
    graded = store.get_all_trust()

    assert store.withdraw_cross_cycle_grades() == []
    assert store.get_all_trust() == pytest.approx(graded)


# ── replay is the definition, not a repair tool ──────────────────────────────

def test_replay_reproduces_a_score_nothing_has_disturbed(due_run):
    """If replay disagreed with the live path on untouched history, one of the
    two would be wrong and there would be no way to tell which."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _responses(['A', 'B']))
    store.update_trust('A', internal_score=0.9)
    store.update_trust('B', internal_score=0.2)
    store.update_trust('A', internal_score=0.3)
    live = store.get_all_trust()

    assert store.replay_trust() == {}, 'nothing should move'
    assert store.get_all_trust() == pytest.approx(live)


def test_replay_follows_the_ema_by_hand(due_run):
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    store.update_trust('A', internal_score=1.0)
    store.update_trust('A', internal_score=0.0)
    expected = _EMA_ALPHA * 1.0 + (1 - _EMA_ALPHA) * _TRUST_INIT
    expected = _EMA_ALPHA * 0.0 + (1 - _EMA_ALPHA) * expected
    assert store.get_trust('A') == pytest.approx(expected)
    store.replay_trust()
    assert store.get_trust('A') == pytest.approx(expected)


def test_events_are_replayed_in_the_order_they_happened(due_run):
    """A grade lands days after the run it grades, so later runs have already
    moved the score. Replaying by run order rather than by clock would apply the
    updates in an order that never occurred."""
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    con = sqlite3.connect(store._path)
    con.executemany(
        'INSERT INTO trust_events (occurred_at, agent_name, kind, raw, run_id, '
        'internal_score, accuracy_score) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [('2026-08-03T00:00:00+00:00', 'A', 'response', 1.0, 9, 1.0, None),
         ('2026-08-01T00:00:00+00:00', 'A', 'response', 0.0, 7, 0.0, None)])
    con.commit()
    con.close()

    store.replay_trust()
    # Clock order is 0.0 then 1.0; row order would give 1.0 then 0.0.
    expected = _EMA_ALPHA * 0.0 + (1 - _EMA_ALPHA) * _TRUST_INIT
    expected = _EMA_ALPHA * 1.0 + (1 - _EMA_ALPHA) * expected
    assert store.get_trust('A') == pytest.approx(expected)


def test_a_bench_recovery_is_logged_and_replayed(due_run):
    """A decay missing from the log would return an agent to a score it had
    already recovered from."""
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    for _ in range(6):
        store.update_trust('A', internal_score=0.0)
    low = store.get_trust('A')
    recovered = store.recover_benched('A')
    assert recovered > low

    assert store.replay_trust() == {}
    assert store.get_trust('A') == pytest.approx(recovered)


def test_replay_keeps_participation_and_quality_untouched(due_run):
    """These describe how much an agent ran and how well it wrote. No grade and
    no withdrawal can change either, and ADR-008's reset kept the same line."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, _responses(['A']))
    store.update_trust('A', internal_score=0.8)
    before = [r for r in store.get_all_trust_rows() if r['agent_name'] == 'A'][0]

    store.replay_trust()
    after = [r for r in store.get_all_trust_rows() if r['agent_name'] == 'A'][0]
    assert after['runs_participated'] == before['runs_participated']
    assert after['avg_internal_score'] == before['avg_internal_score']


def test_the_tier_moves_with_the_replayed_score(due_run):
    """A corrected score that leaves a stale tier behind would keep an agent
    benched on evidence that has been withdrawn."""
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    for _ in range(8):
        store.update_trust('A', internal_score=0.0)
    store.replay_trust()
    row = [r for r in store.get_all_trust_rows() if r['agent_name'] == 'A'][0]
    assert row['current_model_tier'] == trust_tier(row['trust_score'])
    assert row['current_model_tier'] == 'demoted'


# ── history recorded before the log existed ──────────────────────────────────

def test_reconstruction_refuses_to_run_over_an_existing_log(due_run):
    """Rebuilding on top of real events would double every movement."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, _responses(['A']))
    store.update_trust('A', internal_score=0.8)
    report = store.reconstruct_trust_events()
    assert report['reconstructed'] == 0
    assert 'already exists' in report['skipped']


def test_reconstruction_reports_that_it_is_not_exact(due_run):
    """`recover_benched` wrote no history before the log, so a pre-log decay is
    gone. Saying so is the difference between a limit and a silent error."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, _responses(['A', 'B']))
    con = sqlite3.connect(store._path)
    con.execute('DELETE FROM trust_events')       # a store written before the log
    con.commit()
    con.close()

    report = store.reconstruct_trust_events()
    assert report['exact'] is False
    assert 'bench recoveries' in report['cannot_recover']
    assert report['reconstructed'] == 2 and report['agents'] == 2


def test_reconstruction_gives_one_response_event_per_agent_per_run(due_run):
    """`update_trust` is called once per agent from the scores dict, not once
    per response row. An agent that spoke in three rounds moved trust once."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, [
        {'agent_name': 'A', 'round_num': r, 'estimate': 1.0, 'statement': 's',
         'citation_count': 0, 'has_causal_chain': 0, 'internal_score': 0.7,
         'model_used': 'm'} for r in (1, 2, 3)])
    con = sqlite3.connect(store._path)
    con.execute('DELETE FROM trust_events')
    con.commit()
    con.close()

    assert store.reconstruct_trust_events()['reconstructed'] == 1


def test_reconstruction_grades_each_agent_once_on_its_final_word(due_run):
    """The reconstruction must match the path it stands in for.

    It used to build one grade event per RESPONSE row, matching
    `apply_ground_truth_grade` at the time. Both were wrong the same way: an
    agent that spoke three times had one outcome move its trust three times, on
    estimates it had already revised. This test asserted that behaviour and had
    to change with it, which is the signal that matters.
    """
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, [
        {'agent_name': 'A', 'round_num': r, 'estimate': -0.6, 'statement': 's',
         'citation_count': 0, 'has_causal_chain': 0, 'internal_score': 0.7,
         'model_used': 'm'} for r in (1, 2, 3)])
    store.update_trust('A', internal_score=0.7)   # as the swarm does, at run time
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    assert find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 1
    live = store.get_trust('A')

    con = sqlite3.connect(store._path)
    con.execute('DELETE FROM trust_events')
    con.commit()
    con.close()

    report = store.reconstruct_trust_events()
    assert report['reconstructed'] == 2, '1 response + 1 grade, not 1 + 3'
    store.replay_trust()
    assert store.get_trust('A') == pytest.approx(live)


# ── the one-time reset for history the log cannot reach ──────────────────────

def test_the_reset_clears_scores_and_the_log_together(due_run):
    """A reset that left events behind would replay straight back to the score
    it just cleared."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, _responses(['A', 'B']))
    store.update_trust('A', internal_score=0.9)
    store.update_trust('B', internal_score=0.1)
    assert store.get_trust('A') != _TRUST_INIT

    report = store.reset_trust_to_prior()
    assert set(report['reset']) == {'A', 'B'} and report['to'] == _TRUST_INIT
    assert store.get_all_trust() == {'A': _TRUST_INIT, 'B': _TRUST_INIT}

    assert store.replay_trust() == {}
    assert store.get_all_trust() == {'A': _TRUST_INIT, 'B': _TRUST_INIT}


def test_the_reset_keeps_participation_and_quality(due_run):
    """These describe how much an agent ran and how well it wrote. No grading
    defect touched either, and ADR-008 drew the same line."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    store.save_agent_responses(run_id, _responses(['A']))
    store.update_trust('A', internal_score=0.8)
    before = [r for r in store.get_all_trust_rows() if r['agent_name'] == 'A'][0]

    store.reset_trust_to_prior()
    after = [r for r in store.get_all_trust_rows() if r['agent_name'] == 'A'][0]
    assert after['runs_participated'] == before['runs_participated']
    assert after['avg_internal_score'] == before['avg_internal_score']
    assert after['current_model_tier'] == trust_tier(_TRUST_INIT)


def test_scoring_after_a_reset_is_fully_derivable(due_run):
    """The point of the reset. Every score from here traces to its runs."""
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.update_trust('A', internal_score=0.9)
    store.reset_trust_to_prior()

    store.save_agent_responses(run_id, _responses(['A'], estimate=-0.6))
    store.update_trust('A', internal_score=0.7)
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    assert find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 1
    live = store.get_trust('A')

    # Replaying from the prior over only the post-reset events reproduces it.
    assert store.replay_trust() == {}
    assert store.get_trust('A') == pytest.approx(live)

    events = sqlite3.connect(store._path).execute(
        'SELECT kind, run_id FROM trust_events ORDER BY event_id').fetchall()
    assert [e[0] for e in events] == ['response', 'grade']
    assert events[1][1] == run_id, 'a grade must name the run it rests on'


def test_the_fixture_target_week_is_closed(due_run):
    """A guard on the fixture itself, not on the code.

    At `weeks_ago=1` the target week is the CURRENT week, so the price the
    grader records for today lands inside the week under test. Whenever the
    fixture price and the live price differ, that week becomes ambiguous and
    grading silently returns zero -- a test that passes only while two unrelated
    numbers happen to be equal.
    """
    from datetime import datetime, timezone
    from ph_economic_ai.engine import vintage

    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    target = store.target_cycle(dict(store.get_run(run_id)))
    assert target < vintage.fuel_cycle_start(datetime.now(timezone.utc)), (
        'the forecast week must be closed, or today\'s observation falls in it')


# ── the leaderboard says what its numbers rest on ────────────────────────────

def test_an_empty_log_reads_as_a_state_not_a_bug(due_run):
    """Twenty agents all showing 0.50 with no explanation reads as broken. It is
    the honest state, and the line has to say which."""
    from ph_economic_ai.ui import honesty
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    line = honesty.trust_basis(store.trust_provenance())
    assert 'neutral prior' in line
    assert 'could not be reproduced from surviving evidence' in line


def test_the_basis_counts_the_events_behind_the_scores(due_run):
    from ph_economic_ai.ui import honesty
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _responses(['A'], estimate=-0.6))
    store.update_trust('A', internal_score=0.7)
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    find_and_grade_runs(store, current_price=84.38, min_age_days=0)

    prov = store.trust_provenance()
    assert prov['response'] == 1 and prov['grade'] == 1
    line = honesty.trust_basis(prov)
    assert '2 recorded events' in line
    assert '1 from graded outcomes' in line


def test_the_basis_does_not_imply_a_grade_that_has_not_happened(due_run):
    from ph_economic_ai.ui import honesty
    store, _ = due_run(baseline=85.00, estimate=-0.5)
    store.update_trust('A', internal_score=0.7)
    line = honesty.trust_basis(store.trust_provenance())
    assert 'none from a graded outcome yet' in line


def test_the_basis_omits_a_zero_half_rather_than_printing_it():
    """"0 from response quality" reads as a bug rather than as the state after a
    reset, which is exactly when it occurs."""
    from ph_economic_ai.ui import honesty
    line = honesty.trust_basis({'response': 0, 'grade': 32,
                                'since': '2026-08-05T00:00:00+00:00'})
    assert '0 from response quality' not in line
    assert '32 from graded outcomes' in line


# ── one forecast, one movement per agent ─────────────────────────────────────

def _rounds(agent, pairs):
    return [{'agent_name': agent, 'round_num': rn, 'estimate': est,
             'statement': 's', 'citation_count': 0, 'has_causal_chain': 0,
             'internal_score': 0.7, 'model_used': 'm'} for rn, est in pairs]


def test_an_agent_that_spoke_twice_is_graded_once(due_run):
    """One outcome must move an agent's trust once.

    Live on the app's first graded run: 20 agents, 32 responses, 32 grade
    events. Twelve agents took two EMA updates from a single forecast and moved
    further than the eight who spoke once, which weights trust by how much an
    agent talks rather than by how right it was.
    """
    import sqlite3
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _rounds('A', [(1, -0.2), (2, -0.6)])
                               + _rounds('B', [(1, -0.6)]))
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    assert find_and_grade_runs(store, current_price=84.38, min_age_days=0) == 1

    events = sqlite3.connect(store._path).execute(
        "SELECT agent_name, COUNT(*) FROM trust_events WHERE kind='grade' "
        "GROUP BY agent_name").fetchall()
    assert dict(events) == {'A': 1, 'B': 1}


def test_a_revised_estimate_is_not_graded(due_run):
    """An agent's answer is the one it ends on.

    `forum._latest_per_agent` already encodes that for consensus, confidence and
    the judge. Grading was the one place that scored a position the agent had
    retracted: Central Luzon DataExtractor said 1.20, revised to 1.35, and both
    were scored.
    """
    import sqlite3
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    # Round 1 is exactly right, round 2 is exactly wrong. Grading the round the
    # agent withdrew would reward it.
    store.save_agent_responses(run_id, _rounds('A', [(1, -0.62), (2, 2.50)]))
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    find_and_grade_runs(store, current_price=84.38, min_age_days=0)

    acc = sqlite3.connect(store._path).execute(
        "SELECT accuracy_score FROM trust_events WHERE kind='grade'").fetchall()
    assert len(acc) == 1
    assert acc[0][0] == pytest.approx(0.0), 'the final word was wrong by 3.12'


def test_the_highest_round_wins_regardless_of_insert_order(due_run):
    """Rows arrive in whatever order `save_agent_responses` was handed."""
    import sqlite3
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _rounds('A', [(3, -0.62), (1, 2.50)]))
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    find_and_grade_runs(store, current_price=84.38, min_age_days=0)
    acc = sqlite3.connect(store._path).execute(
        "SELECT accuracy_score FROM trust_events WHERE kind='grade'").fetchone()
    assert acc[0] == pytest.approx(1.0), 'round 3 was exactly right'


def test_rebuilding_grade_events_applies_the_current_rule(due_run):
    """The grading rule has changed twice under a stored grade. A trust score is
    only auditable if it reflects the rule in force, not the one that happened
    to be running the day it was written."""
    import sqlite3
    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    store.save_agent_responses(run_id, _rounds('A', [(1, -0.2), (2, -0.6)]))
    from ph_economic_ai.engine.ground_truth import find_and_grade_runs
    find_and_grade_runs(store, current_price=84.38, min_age_days=0)

    con = sqlite3.connect(store._path)
    # Simulate a log written under the old per-response rule.
    con.execute("INSERT INTO trust_events (occurred_at, agent_name, kind, raw, "
                "run_id, internal_score, accuracy_score) "
                "VALUES ('2026-01-01T00:00:00+00:00', 'A', 'grade', 0.5, ?, 0.7, 0.5)",
                (run_id,))
    con.commit()
    assert con.execute("SELECT COUNT(*) FROM trust_events WHERE kind='grade'"
                       ).fetchone()[0] == 2

    report = store.rebuild_grade_events(run_id)
    assert report['was'] == 2 and report['rebuilt'] == 1
    assert con.execute("SELECT COUNT(*) FROM trust_events WHERE kind='grade'"
                       ).fetchone()[0] == 1
    con.close()


def test_rebuilding_refuses_an_ungraded_run(due_run):
    store, run_id = due_run(baseline=85.00, estimate=-0.5)
    assert store.rebuild_grade_events(run_id)['skipped'] == 'run is not graded'
