import pytest
from unittest.mock import patch
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.ground_truth import (
    compute_accuracy_score,
    find_and_grade_runs,
)


@pytest.fixture
def store_with_run(tmp_path):
    """A run whose forecast period has already elapsed.

    `horizon_days=-1` puts the target date one day in the PAST, which is what makes
    the run due for grading. Since RSK-018 a run is graded when its period is over,
    not merely when the row is old, so a fixture that only aged the row would never
    be gradable.
    """
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    run_id = s.save_run(
        scenario={'current_price': 98.82},
        final_estimate=1.42,
        confidence_pct=78,
        horizon_days=-1.0,
    )
    s.save_agent_responses(run_id, [
        {'agent_name': 'Market Analyst', 'round_num': 1, 'estimate': 1.42,
         'statement': 'Rising.', 'citation_count': 1, 'has_causal_chain': 1,
         'internal_score': 0.7, 'model_used': 'deepseek-r1:8b'},
    ])
    return s, run_id


def test_accuracy_score_perfect():
    assert compute_accuracy_score(estimate=1.42, actual=1.42) == 1.0


def test_accuracy_score_half_php_error():
    score = compute_accuracy_score(estimate=1.92, actual=1.42)
    assert abs(score - (1 - 0.5 / 3.0)) < 0.001


def test_accuracy_score_three_php_error():
    score = compute_accuracy_score(estimate=4.42, actual=1.42)
    assert score == 0.0


def test_find_and_grade_runs_skips_a_run_whose_period_has_not_elapsed(tmp_path):
    """A forecast for next week cannot be graded this week."""
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'current_price': 98.82}, final_estimate=1.42,
               confidence_pct=78, horizon_days=7.0)
    assert find_and_grade_runs(s, current_price=100.22) == 0


def test_find_and_grade_runs_grades_a_run_whose_period_has_elapsed(store_with_run):
    store, run_id = store_with_run
    assert find_and_grade_runs(store, current_price=100.22) == 1
    assert store.get_ungraded_runs(min_age_days=0.0) == []


def test_the_grade_records_which_observation_it_used(store_with_run):
    """A grade that cannot be traced to an observation cannot be audited."""
    store, run_id = store_with_run
    find_and_grade_runs(store, current_price=100.22)
    row = store.get_run(run_id) if hasattr(store, 'get_run') else None
    if row is None:
        import sqlite3
        conn = sqlite3.connect(store._path)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone())
    assert row['graded_against'], 'no observation recorded for this grade'
    assert 'gap' in row['graded_against']


def test_a_run_with_no_price_near_its_period_stays_ungraded(tmp_path):
    """The heart of RSK-018. A run whose week has no observation must NOT be
    scored against a price from a different week."""
    from datetime import datetime, timedelta, timezone
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    # Target date sits 60 days in the past; the only observation is today's.
    s.save_run(scenario={'current_price': 98.82}, final_estimate=1.42,
               confidence_pct=78, horizon_days=-60.0)
    assert find_and_grade_runs(s, current_price=100.22) == 0
    assert len(s.get_ungraded_runs(min_age_days=0.0)) == 1

    # Supply an observation for that run's actual period and it becomes gradable.
    s.record_price_observation(
        99.10, observed_at=datetime.now(timezone.utc) - timedelta(days=60))
    assert find_and_grade_runs(s, current_price=100.22) == 1


def test_trust_improves_after_accurate_grade(store_with_run):
    store, run_id = store_with_run
    trust_before = store.get_trust('Market Analyst')
    # actual_change = 100.22 - 98.82 = 1.40, estimate was 1.42, error ≈ ₱0.02.
    # Grade the run directly — this isolates "accurate grade -> trust rises" and
    # removes the wall-clock age filter (the source of the past intermittent
    # failure; see test_find_and_grade_runs_grades_old_run). find_and_grade_runs'
    # age behaviour is covered by the skips_recent / grades_old_run tests.
    store.apply_ground_truth_grade(run_id, actual_change=100.22 - 98.82)
    trust_after = store.get_trust('Market Analyst')
    assert trust_after > trust_before


def test_find_and_grade_runs_skips_missing_current_price(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'fuel_type': 'diesel'}, final_estimate=1.0, confidence_pct=60)
    graded = find_and_grade_runs(s, current_price=100.0)
    assert graded == 0


def test_accuracy_score_symmetric():
    assert compute_accuracy_score(1.92, 1.42) == compute_accuracy_score(1.42, 1.92)


# ── The price table holds observations, not poll counts ──────────────────────

def test_repeating_the_same_price_records_one_observation(tmp_path):
    """`INSERT OR REPLACE` keyed on `observed_at`, which carries microseconds, so
    every call wrote a row. The grading poll runs six-hourly and every run
    records too: the table reached 568 rows holding EIGHT observations, one price
    repeated 157 times in a single day.

    Same failure shape as an agreement percentage -- a number that looks like
    evidence of density and is not. "568 price observations" is a sentence
    someone would say to a panel, and it would be false."""
    import datetime as _dt

    from ph_economic_ai.engine.store import AgentTrustStore

    import sqlite3

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    day = _dt.datetime(2026, 7, 27, 6, 0, tzinfo=_dt.timezone.utc)
    for hour in range(0, 24, 2):
        s.record_price_observation(84.38, day.replace(hour=hour))

    con = sqlite3.connect(path)
    assert con.execute('select count(*) from price_observations').fetchone()[0] == 1
    # And the one kept is still findable, so the poll frequency changed and the
    # grading behaviour did not.
    assert s.price_near('2026-07-27T12:00:00+00:00')['price'] == 84.38
    con.close()


def test_a_price_that_moves_within_a_day_still_records(tmp_path):
    """The key is the PAIR, not the day. A genuine intraday move is information
    and must not be swallowed by the deduplication."""
    import datetime as _dt
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    s = AgentTrustStore(str(tmp_path / 'trust.db'))
    day = _dt.datetime(2026, 7, 27, 6, 0, tzinfo=_dt.timezone.utc)
    s.record_price_observation(84.38, day)
    s.record_price_observation(89.51, day.replace(hour=18))
    con = sqlite3.connect(str(tmp_path / 'trust.db'))
    assert con.execute('select count(*) from price_observations').fetchone()[0] == 2
    con.close()


def test_deduplication_cannot_move_a_graded_price_to_another_day(tmp_path):
    """The claim "information-preserving" was made and was FALSE.

    Collapsing 568 duplicate rows to their 8 real observations moved one run's
    graded outcome from 84.38 to 89.51, because the nearest SURVIVING row fell on
    the next day, which happened to carry a different price. Same information,
    different grade.

    The defect was in `price_near`, not in the deduplication: ranking by raw
    timestamp made the match depend on how many times a day the poll ran. It now
    ranks by calendar-day distance first, so a same-day observation always wins
    and poll frequency cannot change a grade.
    """
    import datetime as _dt
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    con = sqlite3.connect(path)

    # The real shape: one price all of day 1, a different price from day 2.
    day1 = _dt.datetime(2026, 7, 30, 0, 0, tzinfo=_dt.timezone.utc)
    for hour in range(24):
        con.execute('insert or replace into price_observations (observed_at, price)'
                    ' values (?, ?)', (day1.replace(hour=hour).isoformat(), 84.38))
    con.execute('insert or replace into price_observations (observed_at, price)'
                ' values (?, ?)', ('2026-07-31T01:14:00+00:00', 89.51))
    con.commit()

    target = '2026-07-30T17:17:00+00:00'
    before = s.price_near(target)
    s.deduplicate_price_observations()
    after = s.price_near(target)

    assert before['price'] == 84.38
    assert after['price'] == 84.38, (
        'deduplication moved the graded price to the next day')
    assert after['observed_at'][:10] == '2026-07-30', 'same-day must win'
    con.close()


def test_a_same_day_observation_beats_a_closer_one_on_another_day(tmp_path):
    """The property that makes the above hold in general. A target at 23:00 is
    nearer in HOURS to 01:00 the next day than to 06:00 the same day, and the
    same-day price is still the right one: these are weekly step prices sampled
    by a poll, so the day is the resolution and the hour is noise."""
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    con = sqlite3.connect(path)
    con.execute('insert into price_observations (observed_at, price) values (?, ?)',
                ('2026-07-30T06:00:00+00:00', 84.38))
    con.execute('insert into price_observations (observed_at, price) values (?, ?)',
                ('2026-07-31T01:00:00+00:00', 89.51))
    con.commit()

    got = s.price_near('2026-07-30T23:00:00+00:00')
    assert got['price'] == 84.38
    assert got['observed_at'][:10] == '2026-07-30'
    con.close()
