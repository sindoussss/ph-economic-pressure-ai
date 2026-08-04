"""Shared fixtures.

`due_run` exists because of one design change. Grading is cycle-aligned: a run
is a claim about a pricing WEEK, and the week it claims is derived from the week
it was MADE in, not from `timestamp + horizon_days`. That derived instant used
to be bucketed to the second, which split ten runs from one batch across two
different weeks on a margin of under a minute.

The old fixtures forced a run to be due with `horizon_days=-1`, a negative
horizon no real run has. Under a cycle rule a negative horizon has no week to
point at -- a forecast is always about a week that has not happened yet. The
honest way to make a run due is the way production makes one due: let it get old.
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from ph_economic_ai.engine.store import AgentTrustStore


def _backdate(store, run_id, weeks):
    """Move a saved run into the past, target date included."""
    delta = timedelta(days=7 * weeks)
    con = sqlite3.connect(store._path)
    con.row_factory = sqlite3.Row
    row = con.execute('SELECT timestamp, target_date FROM runs WHERE run_id=?',
                      (run_id,)).fetchone()
    made = datetime.fromisoformat(row['timestamp']) - delta
    target = (datetime.fromisoformat(row['target_date']) - delta
              if row['target_date'] else None)
    con.execute('UPDATE runs SET timestamp=?, target_date=? WHERE run_id=?',
                (made.isoformat(), target.isoformat() if target else None, run_id))
    con.commit()
    con.close()
    return made


@pytest.fixture
def due_run(tmp_path):
    """Make a run whose forecast week has closed, and the store holding it.

    Returns `make(baseline=..., estimate=..., price=None, weeks_ago=1)`:
    `price`, when given, is recorded as the observed price of the week the run
    forecasts, which is what makes the run gradable.
    """
    stores = []

    def make(baseline=98.82, estimate=1.42, price=None, weeks_ago=2,
             confidence_pct=78, horizon_days=7.0, db=None):
        # Two weeks, not one. At one week the run's target week IS the current
        # week, so `find_and_grade_runs` recording today's price lands inside the
        # very week under test: the fixture's price and the live one share a
        # cycle, and any difference between them makes that week ambiguous and
        # silently grades nothing. It passed while both were the same number and
        # failed once, unreproducibly, which is the worst way for it to fail.
        # At two weeks the target week is closed and today's observation cannot
        # reach it.
        store = AgentTrustStore(db_path=str(tmp_path / (db or 'trust.db')))
        stores.append(store)
        run_id = store.save_run(scenario={'current_price': baseline},
                                final_estimate=estimate,
                                confidence_pct=confidence_pct,
                                horizon_days=horizon_days)
        _backdate(store, run_id, weeks_ago)
        if price is not None:
            run = dict(store.get_run(run_id))
            # Mid-week, so the observation cannot be read as belonging to either
            # neighbouring cycle by a boundary technicality.
            store.record_price_observation(
                price, observed_at=store.target_cycle(run) + timedelta(days=2))
        return store, run_id

    yield make
    for s in stores:
        try:
            s.close()
        except Exception:
            pass
