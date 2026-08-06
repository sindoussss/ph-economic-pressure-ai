from datetime import datetime

import pytest
from unittest.mock import patch
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.ground_truth import (
    compute_accuracy_score,
    find_and_grade_runs,
)


@pytest.fixture
def store_with_run(due_run):
    """A run whose forecast week has closed, with that week's price recorded.

    Made a week ago with a normal seven-day horizon, which is how a run becomes
    due in production. It used to be forced with `horizon_days=-1`; a negative
    horizon points at no week at all now that grading is cycle-aligned, and a
    fixture that cannot exist in production cannot pin production behaviour.
    """
    s, run_id = due_run(baseline=98.82, estimate=1.42, price=100.22)
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
    # The provenance is the WEEK, not the nearest timestamp. "nearest
    # observation" was exactly the wording that hid a cross-cycle comparison.
    assert row['graded_against'].startswith('pricing week ')


def test_a_run_with_no_price_near_its_period_stays_ungraded(due_run):
    """The heart of RSK-018. A run whose week has no observation must NOT be
    scored against a price from a different week."""
    from datetime import timedelta

    # Forecast week closed nine weeks ago; the only observation is today's.
    s, run_id = due_run(baseline=98.82, estimate=1.42, weeks_ago=9)
    assert find_and_grade_runs(s, current_price=100.22) == 0
    assert len(s.get_ungraded_runs(min_age_days=0.0)) == 1

    # Supply an observation inside that run's own forecast week and it becomes
    # gradable. Anywhere in the week will do -- the week is the unit.
    week = s.target_cycle(dict(s.get_run(run_id)))
    s.record_price_observation(99.10, observed_at=week + timedelta(days=3))
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


# ── Cycle-aligned grading ────────────────────────────────────────────────────

def test_the_outcome_comes_from_the_pricing_week_not_the_nearest_scrape(tmp_path):
    """PH retail fuel is a step function with a Tuesday 06:00 boundary, so a
    weekly forecast is a claim about the step. Grading against whichever scrape
    landed nearest the target measures the source's sampling as much as the
    forecast."""
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    con = sqlite3.connect(path)
    # One clean price all through the week opened Tuesday 2026-08-04.
    for day in ('04', '05', '06', '07'):
        con.execute('insert into price_observations (observed_at, price) values (?, ?)',
                    (f'2026-08-{day}T09:00:00+08:00', 90.0))
    con.commit()

    got = s.cycle_price('2026-08-06T21:00:00+08:00')
    assert got is not None
    assert got['price'] == 90.0
    assert got['cycle'] == '2026-08-04'
    assert got['n_observations'] == 4
    con.close()


def test_a_week_whose_observations_disagree_has_no_price(tmp_path):
    """The live defect. The source read 84.38 on Thursday 30 July and 89.51 on
    Friday 31 July, both inside the cycle opened 07-28, on days no adjustment
    happens. That +5.13 became the "actual change" for two runs and produced the
    only two zero scores in the record.

    A week with two prices has no price. Refusing is `DEC-045`'s rule: a missing
    grade shrinks the record, a wrong grade corrupts it permanently."""
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    con = sqlite3.connect(path)
    con.execute('insert into price_observations (observed_at, price) values (?, ?)',
                ('2026-07-30T09:00:00+08:00', 84.38))
    con.execute('insert into price_observations (observed_at, price) values (?, ?)',
                ('2026-07-31T09:00:00+08:00', 89.51))
    con.commit()
    assert s.cycle_price('2026-07-30T12:00:00+08:00') is None
    con.close()


def test_a_week_with_no_observation_has_no_price(tmp_path):
    from ph_economic_ai.engine.store import AgentTrustStore
    s = AgentTrustStore(str(tmp_path / 'trust.db'))
    assert s.cycle_price('2026-08-06T12:00:00+08:00') is None


def test_a_grade_records_the_week_it_used(store_with_run):
    """A grade that cannot be traced to a pricing week cannot be audited, and
    "nearest observation" was exactly the provenance that hid the defect."""
    store, run_id = store_with_run
    assert find_and_grade_runs(store, current_price=100.22) == 1

    import sqlite3
    con = sqlite3.connect(store._path)
    con.row_factory = sqlite3.Row
    row = dict(con.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone())
    assert row['graded_against'].startswith('pricing week ')
    assert 'observations)' in row['graded_against']
    # 100.22 observed against the run's own stored baseline of 98.82.
    assert row['actual_price_change'] == pytest.approx(1.40)
    con.close()


def test_a_cross_cycle_grade_is_withdrawn(tmp_path):
    """Every grade in the stored record was one: three runs forecasting the week
    opened 2026-07-28, each scored against the week opened 08-04. A forecast for
    one week compared to another week's price is a grade of a different
    question."""
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(db_path=path)
    run_id = s.save_run(scenario={'current_price': 84.38}, final_estimate=-1.28,
                        confidence_pct=70, horizon_days=-1.0)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("UPDATE runs SET actual_price_change=5.13, accuracy_error=6.41, "
                "graded_at='2026-08-04T15:01:29+00:00', target_date=?, "
                "graded_against='2026-08-04T15:01:29.646293+00:00 (gap 0.71d)' "
                "WHERE run_id=?", ('2026-08-03T21:59:00+00:00', run_id))
    con.commit()
    assert s.get_graded_errors() == [6.41]

    withdrawn = s.withdraw_cross_cycle_grades()
    assert [w['run_id'] for w in withdrawn] == [run_id]
    row = dict(con.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone())
    assert row['graded_at'] is None and row['accuracy_error'] is None
    assert s.get_graded_errors() == [], 'the band must not rest on a wrong week'
    con.close()


def test_a_correctly_aligned_grade_is_not_withdrawn(tmp_path):
    """The guard has to discriminate, or it empties the record."""
    import sqlite3

    from ph_economic_ai.engine.store import AgentTrustStore

    path = str(tmp_path / 'trust.db')
    s = AgentTrustStore(path)
    run_id = s.save_run(scenario={'current_price': 88.0}, final_estimate=2.0,
                        confidence_pct=70, horizon_days=-1.0)
    con = sqlite3.connect(path)
    con.execute("UPDATE runs SET actual_price_change=2.0, accuracy_error=0.0, "
                "graded_at='2026-08-06T09:00:00+00:00', target_date=?, "
                "graded_against='pricing week 2026-08-04 (4 observations)' "
                "WHERE run_id=?", ('2026-08-06T09:00:00+00:00', run_id))
    con.commit()
    assert s.withdraw_cross_cycle_grades() == []
    con.close()


def test_one_batch_of_runs_forecasts_one_week(due_run):
    """Ten runs launched to forecast the same adjustment must agree on the week.

    Runs 23 to 32 in the stored record were one batch, all aimed at the
    adjustment of 2026-08-04. Their stored target instants land between 40
    seconds before 06:00:00 PHT and 36 seconds after it, because each was
    computed as "now plus the time remaining" at a slightly different `now`.
    Bucketing that instant put three of them in the week they forecast and seven
    in the week they started from, on a margin of under a minute.
    """
    import sqlite3
    from datetime import datetime, timedelta

    s, run_id = due_run(baseline=88.0, estimate=-1.0)
    run = dict(s.get_run(run_id))
    boundary = s.target_cycle(run)

    con = sqlite3.connect(s._path)
    con.row_factory = sqlite3.Row
    weeks = set()
    for offset in (-40, -1, 0, +1, +36):
        con.execute('UPDATE runs SET target_date=? WHERE run_id=?',
                    ((boundary + timedelta(seconds=offset)).isoformat(), run_id))
        con.commit()
        weeks.add(s.target_cycle(dict(s.get_run(run_id))))
    con.close()
    assert len(weeks) == 1, 'a sub-minute jitter must not change the week'
    assert weeks.pop() == boundary


def test_a_forecast_is_always_about_a_later_week(due_run):
    """The week a run is graded on cannot be the week it read its baseline in,
    or the "change" is measured against itself."""
    s, run_id = due_run(baseline=88.0, estimate=-1.0)
    run = dict(s.get_run(run_id))
    started = s.cycle_prices(run['timestamp'])[1]
    assert s.target_cycle(run) > started


def test_a_run_whose_own_week_has_two_prices_is_not_graded(due_run):
    """A measured change has two ends and the baseline is the other one.

    Live case: the source read 84.38 on Thursday 30 July and 89.51 on Friday 31
    July, both inside the cycle opened 07-28 and on days no adjustment happens.
    A run that stored 84.38 would score a +5.13 "actual change" that is real
    only if the 89.51 reading was the error, and nothing can say which it was.
    """
    from datetime import timedelta

    # Control: the same run grades while only the target week has a price.
    s, run_id = due_run(baseline=84.38, estimate=-1.28, price=89.51)
    assert find_and_grade_runs(s, current_price=89.51) == 1

    # Now the run's OWN week carries both readings, and the change from it has
    # no single value.
    s2, run2 = due_run(baseline=84.38, estimate=-1.28, price=89.51, db='b.db')
    made = dict(s2.get_run(run2))['timestamp']
    own_week = s2.cycle_prices(made)[1]
    s2.record_price_observation(84.38, observed_at=own_week + timedelta(days=1))
    s2.record_price_observation(89.51, observed_at=own_week + timedelta(days=3))
    assert s2.cycle_prices(made)[0] == {84.38, 89.51}
    assert find_and_grade_runs(s2, current_price=89.51) == 0


def test_a_baseline_week_with_no_observation_still_grades(due_run):
    """Absence is not ambiguity, and conflating them would empty the record.

    Most older runs have no observation inside their own week at all. For those
    the stored baseline is the best record of what the run reasoned from, and
    refusing on that basis would refuse nearly everything.
    """
    s, run_id = due_run(baseline=98.82, estimate=1.42, price=100.22)
    assert s.cycle_prices(dict(s.get_run(run_id))['timestamp'])[0] == set()
    assert find_and_grade_runs(s, current_price=100.22) == 1


# ── the age gate that was declared and never ran ─────────────────────────────

def test_the_grader_takes_no_age_parameter():
    """`min_age_days` sat in this signature and was never read.

    It went dead at `RSK-018`, when `get_ungraded_runs(min_age_days)` was
    replaced by `get_due_runs()`: "the row is old enough" became "the forecast
    period has elapsed", which is the better rule and subsumes it. The parameter
    stayed, so a caller reading the signature believed a five-day age gate was in
    force, and thirty call sites passed `min_age_days=0` to disable something
    that was not running.

    Pinned by signature so it cannot be reintroduced as a parameter that lies.
    """
    import inspect
    params = inspect.signature(find_and_grade_runs).parameters
    assert 'min_age_days' not in params
    with pytest.raises(TypeError):
        find_and_grade_runs(None, current_price=1.0, min_age_days=0)


def test_get_ungraded_runs_still_honours_the_age_filter(due_run):
    """The parameter is real on the method that reads it, and removing it from
    the grader must not be read as removing it everywhere."""
    s, _ = due_run(baseline=98.82, estimate=1.42)
    assert len(s.get_ungraded_runs(min_age_days=0.0)) == 1
    assert s.get_ungraded_runs(min_age_days=365.0) == []


def test_a_short_horizon_run_grades_as_soon_as_its_week_settles(due_run):
    """The case that separates the two rules.

    A run made minutes before an adjustment has a horizon of hours, so its
    forecast week closes almost immediately. Under the declared five-day age gate
    it would have waited five days; under the rule that actually governs it
    grades as soon as that week has a price. Two runs in the stored record were
    made this way, with horizons of 0.021 and 0.006 days.
    """
    from datetime import timedelta

    s, run_id = due_run(baseline=85.00, estimate=-0.5, horizon_days=0.02,
                        weeks_ago=2)
    week = s.target_cycle(dict(s.get_run(run_id)))
    s.record_price_observation(84.38, observed_at=week + timedelta(days=2))
    assert find_and_grade_runs(s, current_price=84.38) == 1
