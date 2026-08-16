"""Sector estimates must survive the race with the gas swarm.

Food and electricity debate *in parallel* with the 39-call gas swarm, and they
are much lighter, so they normally finish first. But the run row does not exist
until the gas swarm completes -- `_current_run_id` is assigned in
`_on_simulation_complete`. A sector callback that fires before that found
`_current_run_id is None`, skipped the write, and nothing ever back-filled it.

The estimate was simply lost. On the stored history that was 6 of 20 runs, and
it cost 2026-05 entirely: one run that month, no estimate recorded, so the month
can never be graded. At one graded sample per calendar month
(`engine/ground_truth_monthly`), a lost month is a month of calibration that
cannot be recovered by any later work.
"""
import pytest

from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.ui.main_window import SimMainWindow


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def _window(store, run_id=None, food=None, elec=None):
    """Just enough SimMainWindow to exercise persistence -- no Qt, no widgets."""
    w = SimMainWindow.__new__(SimMainWindow)
    w._store = store
    w._current_run_id = run_id
    w._food_estimate = food
    w._elec_estimate = elec
    return w


def _run_id(store):
    return store.save_run(scenario={'oil_pct': 1.0}, final_estimate=1.0,
                          confidence_pct=50)


def test_estimates_are_persisted_when_the_run_already_exists(store):
    rid = _run_id(store)
    w = _window(store, run_id=rid, food=0.5, elec=0.05)

    assert w._persist_sector_estimates() is True

    row = store.get_run(rid)
    assert row['food_estimate'] == 0.5
    assert row['electricity_estimate'] == 0.05


def test_a_sector_finishing_before_the_run_exists_is_not_lost(store):
    """The actual defect. The debate completes first, so there is no row yet --
    the estimate must still reach the run once the gas swarm creates it."""
    w = _window(store, run_id=None, food=-1.75, elec=0.18)

    assert w._persist_sector_estimates() is False, 'nothing to write to yet'

    # Gas swarm finishes: the row appears and the id is assigned.
    w._current_run_id = _run_id(store)
    assert w._persist_sector_estimates() is True, 'must back-fill once the row exists'

    row = store.get_run(w._current_run_id)
    assert row['food_estimate'] == -1.75, 'the early estimate was dropped'
    assert row['electricity_estimate'] == 0.18


def test_a_later_refined_estimate_overwrites_the_seeded_anchor(store):
    """Debates seed an anchor immediately and refine it when they finish, so a
    back-fill must not freeze the anchor in place."""
    rid = _run_id(store)
    w = _window(store, run_id=rid, food=0.30, elec=0.02)   # anchors
    w._persist_sector_estimates()

    w._food_estimate = 1.42                                 # debate result
    w._persist_sector_estimates()

    assert store.get_run(rid)['food_estimate'] == 1.42


def test_persistence_is_silent_when_there_is_no_store(store):
    w = _window(None, run_id=1, food=0.5)
    assert w._persist_sector_estimates() is False


def test_a_failing_store_does_not_take_the_run_down(store):
    """Persistence is a side effect of a run, and may not break one."""
    class Broken:
        def update_run_sectors(self, *a, **k):
            raise RuntimeError('db gone')
    w = _window(Broken(), run_id=1, food=0.5)
    assert w._persist_sector_estimates() is False


def test_a_month_with_no_recorded_estimate_cannot_be_graded(store):
    """Why this matters, stated as a test: the loss is permanent. A month whose
    runs recorded no estimate yields no graded sample, ever."""
    import pandas as pd
    from ph_economic_ai.engine.ground_truth_monthly import month_samples

    rid = _run_id(store)
    with store._lock:
        store._conn.execute('UPDATE runs SET timestamp=? WHERE run_id=?',
                            ('2026-05-14T12:00:00+00:00', rid))
        store._conn.commit()

    truth = pd.Series({'2026-05': 0.42})
    now = pd.Timestamp('2026-08-15', tz='UTC')

    assert month_samples(store.get_all_runs(), 'food', truth, now=now) == []

    _window(store, run_id=rid, food=0.9)._persist_sector_estimates()

    samples = month_samples(store.get_all_runs(), 'food', truth, now=now)
    assert [s['month'] for s in samples] == ['2026-05']
