"""Food's band, once it is graded per month — and the guarantees around it.

Three things must hold at once: food can become calibrated, it becomes so only
on twelve independent MONTHS, and gas is completely unaffected. The fourth, and
the one most likely to be broken by a careless edit, is that food's band is
never built from fuel errors.
"""
import pandas as pd
import pytest

from ph_economic_ai.engine import interval as _interval
from ph_economic_ai.engine.ground_truth_monthly import find_and_grade_months
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.ui import honesty as _honesty

NOW = pd.Timestamp('2026-08-14', tz='UTC')


@pytest.fixture
def store(tmp_path):
    return AgentTrustStore(db_path=str(tmp_path / 'trust.db'))


def _grade_months(store, n, sector='food', error=0.4):
    for i in range(n):
        store.upsert_sector_grade(sector, f'{2025 + i // 12:04d}-{i % 12 + 1:02d}',
                                  estimate=error, actual=0.0, abs_error=error,
                                  n_runs=1)


# ── food can now be calibrated, but only on months ───────────────────────────

def test_food_is_a_graded_sector():
    assert 'food' in _interval.GRADED_SECTORS
    assert _interval.SAMPLE_UNIT['food'] == 'months'


@pytest.mark.parametrize('n', [0, 1, 11])
def test_food_below_the_threshold_is_a_stated_prior(store, n):
    _grade_months(store, n)
    b = _interval.band(0.5, store.get_sector_graded_errors('food'), sector='food',
                       errors_from='food')
    assert b['calibrated'] is False
    assert b['gradable'] is True, 'food CAN become calibrated by waiting'
    assert b['half_width'] == _interval.FALLBACK_HALFWIDTH['food'][0.5]


def test_food_calibrates_at_twelve_months(store):
    _grade_months(store, 12, error=0.4)
    b = _interval.band(0.5, store.get_sector_graded_errors('food'), sector='food',
                       errors_from='food')
    assert b['calibrated'] is True
    assert b['n_graded'] == 12
    assert b['half_width'] == pytest.approx(0.4)


def test_many_runs_in_two_months_never_reach_the_threshold(store):
    """The rule, end to end: 24 runs across 2 months is 2 samples, not 24."""
    for month in ('06', '07'):
        for day in range(1, 13):
            rid = store.save_run(scenario={'oil_pct': 1.0}, final_estimate=None,
                                 confidence_pct=0)
            store.update_run_sectors(rid, 0.5, None)
            with store._lock:
                store._conn.execute('UPDATE runs SET timestamp=? WHERE run_id=?',
                                    (f'2026-{month}-{day:02d}T12:00:00+00:00', rid))
                store._conn.commit()
    find_and_grade_months(store, 'food', pd.Series({'2026-06': 0.0, '2026-07': 0.0}),
                          now=NOW)

    assert store.count_sector_graded_months('food') == 2
    b = _interval.band(0.5, store.get_sector_graded_errors('food'), sector='food',
                       errors_from='food')
    assert b['calibrated'] is False, '24 runs in 2 months is 2 samples'
    assert b['n_graded'] == 2


# ── the unit hazard ──────────────────────────────────────────────────────────

def test_food_band_never_uses_fuel_errors(store):
    """Fuel error is PHP/L; food's is percentage points. The monitor used to
    hand one error list to every sector, which was safe only while food was
    ungraded. If that regresses, food gets PHP/L half-widths beside a % sign."""
    from ph_economic_ai.ui.pressure_monitor import PressureMonitorPanel

    for _ in range(20):                       # a deep FUEL history
        rid = store.save_run(scenario={'current_price': 60.0}, final_estimate=1.0,
                             confidence_pct=50)
        store.apply_ground_truth_grade(rid, actual_change=6.0)   # 5.0 PHP/L error

    panel = PressureMonitorPanel.__new__(PressureMonitorPanel)   # no Qt needed
    panel._store = store

    assert panel._graded_errors('gas'), 'gas still reads its own history'

    # This asserted `== []` until food began inheriting the committed shared
    # months (`benchmark/shared_grades.py`). Emptiness was only ever a PROXY for
    # the property under test, and it stopped being a valid one the moment food
    # could legitimately have grades without this store holding them.
    #
    # So the hazard is now named directly instead. Every run above was graded at
    # a 5.0 PHP/L fuel error; if the sector filter regresses, that magnitude
    # appears in food's list and the band puts PHP/L half-widths beside a % sign.
    from ph_economic_ai.benchmark import shared_grades as _shared

    food_errors = panel._graded_errors('food')
    assert 5.0 not in food_errors, 'a PHP/L fuel error reached food'
    # Exact rather than a magnitude bound: this store grades no food month, so
    # food's list must be precisely what the committed shared file carries and
    # nothing else. A magnitude test would false-fail on a real food shock.
    assert food_errors == _shared.merged_errors(_shared.load_shared(), [], 'food')

    b = _interval.band(0.5, panel._graded_errors('food'), sector='food',
                   errors_from='food')
    assert b['calibrated'] is False
    assert b['half_width'] == _interval.FALLBACK_HALFWIDTH['food'][0.5]


def test_sector_errors_are_kept_apart_in_the_store(store):
    rid = store.save_run(scenario={'current_price': 60.0}, final_estimate=1.0,
                         confidence_pct=50)
    store.apply_ground_truth_grade(rid, actual_change=6.0)
    _grade_months(store, 3, error=0.2)

    assert store.get_graded_errors() == [pytest.approx(5.0)]
    assert store.get_sector_graded_errors('food') == [0.2, 0.2, 0.2]


# ── gas is untouched ─────────────────────────────────────────────────────────

def test_gas_wording_and_behaviour_unchanged(store):
    errors = [0.2] * _interval.MIN_GRADED_FOR_CALIBRATION
    b = _interval.band(1.0, errors, sector='gas')
    assert b['calibrated'] is True
    assert b['sample_unit'] == 'runs'
    assert b['source'] == f'calibrated on {len(errors)} graded runs'
    assert _honesty.band_provenance(b) == (
        f'range calibrated on this app’s own {len(errors)} graded runs '
        f'(split conformal)')


def test_gas_uncalibrated_wording_unchanged():
    b = _interval.band(1.0, [0.2], sector='gas')
    line = _honesty.band_provenance(b)
    assert 'graded runs available' in line
    assert 'months' not in line


def test_gas_band_is_not_affected_by_food_grades(store):
    _grade_months(store, 50, sector='food')
    assert store.get_graded_errors() == []
    b = _interval.band(1.0, store.get_graded_errors(), sector='gas')
    assert b['calibrated'] is False


# ── provenance says months for food ──────────────────────────────────────────

def test_food_provenance_counts_months_not_runs(store):
    _grade_months(store, 12)
    line = _honesty.band_provenance(
        _interval.band(0.5, store.get_sector_graded_errors('food'), sector='food',
                       errors_from='food'))
    assert '12 graded months' in line
    assert 'runs' not in line, 'a month is not a run'


def test_food_uncalibrated_promises_a_reachable_threshold(store):
    _grade_months(store, 2)
    line = _honesty.band_provenance(
        _interval.band(0.5, store.get_sector_graded_errors('food'), sector='food',
                       errors_from='food'))
    assert f'2 of {_interval.MIN_GRADED_FOR_CALIBRATION} graded months' in line
    assert 'cannot become calibrated' not in line


def test_electricity_still_says_it_can_never_calibrate():
    """Unchanged: ₱/kWh estimates have no comparable outcome series."""
    line = _honesty.band_provenance(_interval.band(0.05, [], sector='electricity'))
    assert 'cannot become calibrated by waiting' in line
    assert 'electricity' not in _interval.GRADED_SECTORS
