"""A price with no product attached is not an observation of this series.

The "ambiguous week" that blocked ten runs was not ambiguous. The cycle opened
2026-07-28 held 84.38 on the Thursday and 89.51 on the Friday, on days no
adjustment happens, and the app refused to grade against a week that could not
settle on a price.

They are two different fuels. Until `f050c53` the scraper took the MEDIAN across
every product on the page, and the page recorded that day listed Diesel 81.13,
Diesel Plus 83.94, Unleaded 91 84.38, Premium 95 89.51 and Kerosene 111.43. The
median is 84.38, which is Unleaded 91. This app forecasts RON 95. The changeover
in the stored data falls within fifteen minutes of that commit.

So the week was settled all along and the table could not say so, because it
recorded a number without recording what the number was OF.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from ph_economic_ai.engine.store import (
    FORECAST_GRADE, AgentTrustStore, _GRADE_SELECTION_FIXED_AT)

_WEEK = datetime(2026, 7, 28, 6, 0, tzinfo=timezone(timedelta(hours=8)))


def _store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


# ── the defect itself ────────────────────────────────────────────────────────

def test_two_fuel_grades_in_one_week_is_not_an_unsettled_week(tmp_path):
    """The live case, reproduced. One week, two products, one RON 95 price."""
    s = _store(tmp_path)
    s.record_price_observation(84.38, observed_at=_WEEK + timedelta(days=2),
                               grade='RON 91')
    s.record_price_observation(89.51, observed_at=_WEEK + timedelta(days=3),
                               grade='RON 95')

    prices, start, _ = s.cycle_prices(_WEEK + timedelta(days=3))
    assert prices == {89.51}, 'the other grade is not evidence about this series'
    assert s.cycle_price(_WEEK + timedelta(days=3))['price'] == 89.51
    s.close()


def test_a_genuine_two_price_week_is_still_refused(tmp_path):
    """The guard must not be dissolved by the fix. Two RON 95 readings that
    disagree inside one week is still a week with no settled price."""
    s = _store(tmp_path)
    s.record_price_observation(89.51, observed_at=_WEEK + timedelta(days=2))
    s.record_price_observation(91.20, observed_at=_WEEK + timedelta(days=3))
    assert s.cycle_price(_WEEK + timedelta(days=3)) is None
    s.close()


def test_an_observation_defaults_to_the_forecast_grade(tmp_path):
    """Callers that predate the column must not have their prices vanish."""
    s = _store(tmp_path)
    s.record_price_observation(89.51, observed_at=_WEEK + timedelta(days=2))
    assert s.cycle_prices(_WEEK + timedelta(days=2))[0] == {89.51}
    row = sqlite3.connect(s._path).execute(
        'SELECT grade FROM price_observations').fetchone()
    assert row[0] == FORECAST_GRADE
    s.close()


def test_the_same_price_under_two_grades_is_two_observations(tmp_path):
    """Deduplication keys on (day, price). Two products that happen to cost the
    same on one day are still two facts, and collapsing them would lose which
    series each belongs to."""
    s = _store(tmp_path)
    day = _WEEK + timedelta(days=2)
    s.record_price_observation(89.51, observed_at=day, grade='RON 95')
    s.record_price_observation(89.51, observed_at=day.replace(hour=18),
                               grade='RON 91')
    n = sqlite3.connect(s._path).execute(
        'SELECT COUNT(*) FROM price_observations').fetchone()[0]
    assert n == 2
    assert s.cycle_prices(day)[0] == {89.51}
    s.close()


# ── the migration for rows written before the grade was recorded ─────────────

def test_rows_from_before_the_selection_fix_are_labelled_unknown(tmp_path):
    """`unknown`, not `RON 91`. The evidence that they ARE Unleaded 91 is strong,
    but the rule in force did not select a product at all -- it took the middle
    of a list. What is known is that they are not RON 95, and that is enough."""
    path = str(tmp_path / 'trust.db')
    con = sqlite3.connect(path)
    con.executescript('CREATE TABLE price_observations ('
                      ' observed_at TEXT PRIMARY KEY, price REAL NOT NULL);')
    con.executemany('INSERT INTO price_observations VALUES (?, ?)',
                    [('2026-07-29T22:55:48+00:00', 84.38),
                     ('2026-08-01T02:10:51+00:00', 89.51)])
    con.commit()
    con.close()

    s = AgentTrustStore(db_path=path)
    rows = dict(sqlite3.connect(path).execute(
        'SELECT observed_at, grade FROM price_observations').fetchall())
    assert rows['2026-07-29T22:55:48+00:00'] == 'unknown'
    assert rows['2026-08-01T02:10:51+00:00'] == FORECAST_GRADE
    s.close()


def test_the_migration_boundary_is_the_commit_that_fixed_selection():
    """A date typed into a migration is a claim. This one is checkable: the
    scraper started selecting by name in `f050c53` on 2026-07-31."""
    assert _GRADE_SELECTION_FIXED_AT.startswith('2026-07-31')


def test_the_unknown_rows_do_not_settle_a_week(tmp_path):
    """The whole point. An unidentified product cannot make a week ambiguous,
    and cannot supply its price either."""
    path = str(tmp_path / 'trust.db')
    con = sqlite3.connect(path)
    con.executescript('CREATE TABLE price_observations ('
                      ' observed_at TEXT PRIMARY KEY, price REAL NOT NULL);')
    con.execute('INSERT INTO price_observations VALUES (?, ?)',
                ('2026-07-29T22:55:48+00:00', 84.38))
    con.commit()
    con.close()

    s = AgentTrustStore(db_path=path)
    assert s.cycle_prices('2026-07-29T22:55:48+00:00')[0] == set()
    assert s.cycle_price('2026-07-29T22:55:48+00:00') is None
    s.close()


# ── the grade travels with the price out of the scraper ──────────────────────

def test_the_scraper_reports_which_grade_it_matched(monkeypatch):
    """The preference list falls through to RON 91 when the page omits RON 95.
    A caller storing that as an observation of RON 95 recreates the defect."""
    from ph_economic_ai.engine import swarm

    monkeypatch.setattr(swarm, 'fetch_live_retail_price_checked',
                        lambda: (84.38, True))
    monkeypatch.setattr(swarm, '_LAST_MATCHED_GRADE', 'RON 91')
    price, live, grade = swarm.fetch_live_retail_price_graded()
    assert (price, live, grade) == (84.38, True, 'RON 91')


def test_a_failed_fetch_reports_unknown_not_the_forecast_grade(monkeypatch):
    """The fallback constant is not an observation of anything, so labelling it
    RON 95 would put a made-up number into the series under a real name."""
    from ph_economic_ai.engine import swarm

    monkeypatch.setattr(swarm, 'fetch_live_retail_price_checked',
                        lambda: (swarm._FALLBACK_RETAIL_PRICE_PHP, False))
    monkeypatch.setattr(swarm, '_LAST_MATCHED_GRADE', 'RON 95')
    assert swarm.fetch_live_retail_price_graded()[2] == 'unknown'


# ── a baseline its own week contradicts ──────────────────────────────────────

def test_a_run_whose_baseline_is_not_its_week_price_is_refused(due_run):
    """Runs made before the selection fix stored Unleaded 91 while the app
    forecasts RON 95. Measuring a change from one product to another produces a
    number that looks like a market move and is a relabelling.

    Sharper than the plausibility bound: 84.38 to 89.51 is 5.13, comfortably
    inside the +/-8 that bound allows.
    """
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=84.38, estimate=-1.28, price=89.51)
    run = dict(store.get_run(run_id))
    own_week = store.cycle_prices(run['timestamp'])[1]
    store.record_price_observation(89.51, observed_at=own_week + timedelta(days=2))

    verdict = grade_verdict(store, run)
    assert verdict['obstacle'] == 'baseline_contradicted'
    assert '84.38' in verdict['reason'] and '89.51' in verdict['reason']
    assert abs(89.51 - 84.38) < 8, 'the plausibility bound would have let this pass'


def test_a_baseline_matching_its_week_grades(due_run):
    """The live case that now settles: run 32 stored 89.51, the week it ran in
    settled at 89.51, and the week it forecast also settled at 89.51."""
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=89.51, estimate=0.95, price=89.51)
    run = dict(store.get_run(run_id))
    own_week = store.cycle_prices(run['timestamp'])[1]
    store.record_price_observation(89.51, observed_at=own_week + timedelta(days=2))

    verdict = grade_verdict(store, run)
    assert verdict['obstacle'] is None
    assert verdict['actual_change'] == pytest.approx(0.0)


def test_a_week_with_no_observation_still_trusts_the_stored_baseline(due_run):
    """Absence is not contradiction. Most older runs have no observation in their
    own week, and refusing those would refuse nearly everything."""
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    run = dict(store.get_run(run_id))
    assert store.cycle_prices(run['timestamp'])[0] == set()
    assert grade_verdict(store, run)['obstacle'] is None


# ── a graded tile must show the outcome, not a tick ──────────────────────────

def test_a_graded_tile_reports_the_actual_and_the_error(tmp_path, monkeypatch):
    """"+0.95 ₱/L · graded ✓" reads as vindication. The first graded run in the
    record predicted a rise of 0.95 in a week that did not move at all: small
    error, wrong direction, and the checkmark said neither."""
    import sys
    from PyQt6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(sys.argv)

    from ph_economic_ai.ui.landing import LandingPanel

    class _FakeStore:
        def get_recent_runs(self, limit=4):
            return [{'run_id': 40, 'timestamp': '2026-08-04T00:00:00+00:00',
                     'final_estimate': -1.0, 'confidence_pct': 70},
                    {'run_id': 32, 'timestamp': '2026-08-02T00:00:00+00:00',
                     'final_estimate': 0.95, 'confidence_pct': 98,
                     'actual_price_change': 0.0, 'accuracy_error': 0.95}]

    panel = LandingPanel(store=_FakeStore())
    panel.refresh_recent()
    text = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'actual +0.00' in text
    assert 'off by 0.95' in text
    assert 'graded ✓' not in text, 'a tick is not an outcome'
