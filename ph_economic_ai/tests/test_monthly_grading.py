"""Monthly grading for food, and the one rule that makes it worth anything.

Gas reprices weekly. `grade_verdict` grades one run against the pricing WEEK it
forecast, so one run is one observation. Food reprices monthly and PSA publishes
one CPI print per calendar month, so the unit of evidence is a MONTH:

    N runs inside one calendar month are worth ONE graded sample, not N.

The app is run many times a month in normal use and testing. Counting runs would
let `band()` report "12 graded samples" while resting on two or three genuinely
independent months, repeated -- the same class of overclaim `gradable` was added
to prevent on the screen. The rule is enforced twice: runs are collapsed by month
before anything is graded, and `sector_grades` is keyed `PRIMARY KEY (sector,
month)` so the table cannot physically hold two samples for one month.
"""
import pandas as pd
import pytest

from ph_economic_ai.engine.ground_truth_monthly import (
    UNGRADED_REASONS_MONTHLY,
    find_and_grade_months,
    grade_verdict_monthly,
    month_samples,
    run_month,
)
from ph_economic_ai.engine.store import AgentTrustStore

NOW = pd.Timestamp('2026-08-14', tz='UTC')


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def _run(timestamp: str, food=None, elec=None) -> dict:
    return {'run_id': 1, 'timestamp': timestamp,
            'food_estimate': food, 'electricity_estimate': elec}


def _add_run(store, timestamp: str, food=None):
    """Persist a run with a CONTROLLED timestamp (save_run stamps 'now')."""
    rid = store.save_run(scenario={'oil_pct': 1.0}, final_estimate=None,
                         confidence_pct=0)
    store.update_run_sectors(rid, food, None)
    with store._lock:
        store._conn.execute('UPDATE runs SET timestamp=? WHERE run_id=?',
                            (timestamp, rid))
        store._conn.commit()
    return rid


# ── the rule ─────────────────────────────────────────────────────────────────

def test_run_month_is_the_calendar_month_of_the_timestamp():
    assert run_month(_run('2026-07-21T16:13:47+00:00')) == '2026-07'
    assert run_month(_run('2026-06-10T15:54:49.420918+00:00')) == '2026-06'


def test_five_runs_in_one_month_are_one_sample():
    runs = [_run(f'2026-07-0{d}T12:00:00+00:00', food=f)
            for d, f in zip(range(1, 6), (0.5, 1.5, -2.0, 3.0, 0.25))]
    samples = month_samples(runs, 'food', pd.Series({'2026-07': 0.146}), now=NOW)

    assert len(samples) == 1, 'five runs in one month must be ONE sample'
    assert samples[0]['month'] == '2026-07'
    assert samples[0]['n_runs'] == 5


def test_distinct_months_stay_distinct():
    """Dedup must not over-collapse."""
    runs = [_run('2026-06-10T12:00:00+00:00', food=0.5),
            _run('2026-06-11T12:00:00+00:00', food=0.7),
            _run('2026-07-21T12:00:00+00:00', food=1.0)]
    truth = pd.Series({'2026-06': -0.364, '2026-07': 0.146})
    samples = month_samples(runs, 'food', truth, now=NOW)

    assert [s['month'] for s in samples] == ['2026-06', '2026-07']
    assert [s['n_runs'] for s in samples] == [2, 1]


def test_regrading_never_inflates_the_month_count(store):
    _add_run(store, '2026-07-21T12:00:00+00:00', food=0.5)
    _add_run(store, '2026-07-22T12:00:00+00:00', food=1.5)
    truth = pd.Series({'2026-07': 0.146})

    first = find_and_grade_months(store, 'food', truth, now=NOW)
    second = find_and_grade_months(store, 'food', truth, now=NOW)

    assert (first, second) == (1, 0)
    assert store.count_sector_graded_months('food') == 1


def test_twenty_runs_across_two_months_are_two_samples(store):
    for month in ('06', '07'):
        for day in range(1, 11):
            _add_run(store, f'2026-{month}-{day:02d}T12:00:00+00:00', food=0.5)
    truth = pd.Series({'2026-06': -0.364, '2026-07': 0.146})

    find_and_grade_months(store, 'food', truth, now=NOW)

    assert store.count_sector_graded_months('food') == 2
    assert len(store.get_sector_graded_errors('food')) == 2


# ── unsettled months stay ungraded, and say why ──────────────────────────────

def test_month_psa_has_not_published_is_not_graded():
    runs = [_run('2026-08-03T12:00:00+00:00', food=0.5)]
    truth = pd.Series({'2026-06': -0.364, '2026-07': 0.146})
    assert month_samples(runs, 'food', truth, now=NOW) == []


def test_in_progress_month_is_never_graded():
    """Even if the series carries it, the current month is not settled."""
    runs = [_run('2026-08-03T12:00:00+00:00', food=0.5)]
    assert month_samples(runs, 'food', pd.Series({'2026-08': 0.2}), now=NOW) == []


def test_verdict_names_the_obstacle_rather_than_failing_silently(store):
    """Mirrors `grade_verdict`: a reader asking "why is this ungraded" gets the
    real reason, not a generic "pending"."""
    _add_run(store, '2026-08-03T12:00:00+00:00', food=0.5)     # current month
    _add_run(store, '2026-07-02T12:00:00+00:00', food=None)    # no estimate

    v = grade_verdict_monthly(store, 'food', pd.Series({'2026-07': 0.146}), now=NOW)
    obstacles = {r['month']: r['obstacle'] for r in v['blocked']}

    assert obstacles['2026-08'] == 'month_in_progress'
    assert obstacles['2026-07'] == 'no_estimate'
    assert all(o in UNGRADED_REASONS_MONTHLY for o in obstacles.values())
    assert v['gradable'] == []


def test_unpublished_month_reports_no_cpi_yet(store):
    _add_run(store, '2026-07-02T12:00:00+00:00', food=0.5)
    v = grade_verdict_monthly(store, 'food', pd.Series({'2026-05': 0.1}), now=NOW)
    assert v['blocked'][0]['obstacle'] == 'no_cpi_yet'
    assert '2026-07' in v['blocked'][0]['reason']


def test_gradable_months_carry_their_measured_change(store):
    _add_run(store, '2026-07-02T12:00:00+00:00', food=0.5)
    v = grade_verdict_monthly(store, 'food', pd.Series({'2026-07': 0.146}), now=NOW)
    assert v['blocked'] == []
    assert v['gradable'][0]['actual_change'] == pytest.approx(0.146)


# ── the sample's value ───────────────────────────────────────────────────────

def test_month_estimate_is_the_median_of_that_months_runs():
    """Median, not mean: within-month estimates are heavy-tailed and one weak
    run should not define the month."""
    runs = [_run('2026-07-01T12:00:00+00:00', food=0.0),
            _run('2026-07-02T12:00:00+00:00', food=1.0),
            _run('2026-07-03T12:00:00+00:00', food=50.0)]
    s = month_samples(runs, 'food', pd.Series({'2026-07': 0.0}), now=NOW)[0]

    assert s['estimate'] == pytest.approx(1.0)
    assert s['abs_error'] == pytest.approx(1.0)


def test_runs_may_be_any_iterable_not_only_a_list():
    """`runs` is typed Iterable and the grouping used to happen once per month,
    re-consuming it. A generator was exhausted by the first month, so every
    later month silently read as having no estimates and went ungraded."""
    runs = (_run(f'2026-0{m}-02T12:00:00+00:00', food=1.0) for m in (5, 6, 7))
    truth = pd.Series({'2026-05': 0.0, '2026-06': 0.0, '2026-07': 0.0})

    samples = month_samples(runs, 'food', truth, now=NOW)

    assert [s['month'] for s in samples] == ['2026-05', '2026-06', '2026-07']


def test_runs_without_an_estimate_do_not_count_as_runs():
    runs = [_run('2026-07-01T12:00:00+00:00', food=None),
            _run('2026-07-02T12:00:00+00:00', food=1.0)]
    s = month_samples(runs, 'food', pd.Series({'2026-07': 0.146}), now=NOW)
    assert len(s) == 1 and s[0]['n_runs'] == 1


def test_errors_come_back_newest_first_for_the_conformal_window(store):
    for month, est in (('05', 5.0), ('06', 1.0), ('07', 2.0)):
        _add_run(store, f'2026-{month}-02T12:00:00+00:00', food=est)
    truth = pd.Series({'2026-05': 0.0, '2026-06': 0.0, '2026-07': 0.0})
    find_and_grade_months(store, 'food', truth, now=NOW)

    assert store.get_sector_graded_errors('food') == [2.0, 1.0, 5.0]
    assert store.get_sector_graded_errors('food', limit=2) == [2.0, 1.0]
