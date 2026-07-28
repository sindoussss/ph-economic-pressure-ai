"""The "when" the app could not previously answer.

Timing here is a calendar fact, not a forecast: the weekly adjustment date is
known, and only its direction and size are uncertain. These tests pin the
boundaries, which is where a date helper actually goes wrong.
"""
import datetime as dt

import pytest

from ph_economic_ai.engine import price_calendar as pc


def _at(y, m, d, hour=12, minute=0):
    return dt.datetime(y, m, d, hour, minute, tzinfo=pc.PH_TZ)


def test_next_adjustment_from_a_monday_is_the_next_day():
    # 2026-07-27 is a Monday.
    assert _at(2026, 7, 27).weekday() == 0
    nxt = pc.next_fuel_adjustment(_at(2026, 7, 27))
    assert nxt.date() == dt.date(2026, 7, 28)
    assert nxt.weekday() == pc.FUEL_ADJUSTMENT_WEEKDAY


def test_a_tuesday_before_the_effective_hour_still_counts_as_today():
    nxt = pc.next_fuel_adjustment(_at(2026, 7, 28, hour=2))
    assert nxt.date() == dt.date(2026, 7, 28)


def test_a_tuesday_after_the_effective_hour_rolls_to_next_week():
    """The boundary that decides whether a user is told 'today' or 'in 7 days'."""
    nxt = pc.next_fuel_adjustment(_at(2026, 7, 28, hour=9))
    assert nxt.date() == dt.date(2026, 8, 4)


def test_exactly_at_the_effective_hour_has_already_happened():
    nxt = pc.next_fuel_adjustment(_at(2026, 7, 28, hour=pc.FUEL_EFFECTIVE_HOUR))
    assert nxt.date() == dt.date(2026, 8, 4)


def test_from_a_wednesday_it_is_six_days_out():
    nxt = pc.next_fuel_adjustment(_at(2026, 7, 29))
    assert nxt.date() == dt.date(2026, 8, 4)


@pytest.mark.parametrize('weekday_date,expected', [
    (dt.date(2026, 7, 27), dt.date(2026, 7, 28)),   # Mon -> Tue
    (dt.date(2026, 7, 29), dt.date(2026, 8, 4)),    # Wed -> next Tue
    (dt.date(2026, 8, 1), dt.date(2026, 8, 4)),     # Sat -> next Tue
    (dt.date(2026, 8, 2), dt.date(2026, 8, 4)),     # Sun -> next Tue
])
def test_every_weekday_lands_on_a_tuesday(weekday_date, expected):
    nxt = pc.next_fuel_adjustment(_at(weekday_date.year, weekday_date.month,
                                      weekday_date.day))
    assert nxt.date() == expected
    assert nxt.weekday() == pc.FUEL_ADJUSTMENT_WEEKDAY


def test_humanize_gives_the_words_the_user_asked_for():
    now = _at(2026, 7, 27)
    assert pc.humanize_until(_at(2026, 7, 27), now) == 'today'
    assert pc.humanize_until(_at(2026, 7, 28), now) == 'tomorrow'
    assert pc.humanize_until(_at(2026, 7, 30), now) == 'in 3 days'


def test_describe_carries_a_real_date_not_just_a_horizon():
    d = pc.describe_next_fuel_adjustment(_at(2026, 7, 27))
    assert d['date'] == '2026-07-28'
    assert d['weekday'] == 'Tuesday'
    assert d['when'] == 'tomorrow'
    assert 'July' in d['label']
    assert d['basis'] == 'scheduled'


def test_horizon_days_can_drive_the_grading_target():
    """The date is only useful if a run can be saved against it, so the horizon has
    to come out in the units the store wants."""
    d = pc.describe_next_fuel_adjustment(_at(2026, 7, 27, hour=6))
    assert 0 < d['horizon_days'] <= 1.0


def test_naive_datetimes_are_treated_as_philippine_time():
    naive = dt.datetime(2026, 7, 27, 12, 0)
    assert pc.next_fuel_adjustment(naive).date() == dt.date(2026, 7, 28)


def test_cpi_release_rolls_into_the_next_month():
    nxt = pc.next_cpi_release(_at(2026, 7, 27))
    assert nxt.date() == dt.date(2026, 8, 5)
    after = pc.next_cpi_release(_at(2026, 8, 6))
    assert after.date() == dt.date(2026, 9, 5)


def test_cpi_release_across_a_year_boundary():
    assert pc.next_cpi_release(_at(2026, 12, 20)).date() == dt.date(2027, 1, 5)


def test_the_module_does_not_predict_a_future_rise_date():
    """Scope guard. Timing is reported as scheduled, never as forecast, because the
    benchmark does not support a dated directional claim."""
    for describe in (pc.describe_next_fuel_adjustment, pc.describe_next_cpi_release):
        assert describe()['basis'] == 'scheduled'


# ── The loop: card date -> stored target -> graded period ────────────────────

def test_the_stored_target_matches_the_adjustment_the_card_announced(tmp_path):
    """The point of connecting the horizon. A run saved with the calendar's
    horizon must come due on the adjustment the user was shown, not on a flat
    seven-day default."""
    from ph_economic_ai.engine.store import AgentTrustStore

    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    event = pc.describe_next_fuel_adjustment()
    rid = s.save_run(scenario={'current_price': 60.0}, final_estimate=1.0,
                     confidence_pct=70, horizon_days=event['horizon_days'])

    row = next(r for r in s.get_ungraded_runs(min_age_days=0.0) if r['run_id'] == rid)
    stored = dt.datetime.fromisoformat(row['target_date']).astimezone(pc.PH_TZ)
    assert stored.date() == dt.date.fromisoformat(event['date']), (
        'the run is graded against a different day than the card announced')


def test_a_flat_default_would_miss_the_announced_adjustment(tmp_path):
    """Why the wiring was needed. On a Monday the next adjustment is one day out,
    so the seven-day default lands on the FOLLOWING week's price."""
    from ph_economic_ai.engine.store import AgentTrustStore, DEFAULT_HORIZON_DAYS

    monday = _at(2026, 7, 27)
    announced = pc.next_fuel_adjustment(monday).date()
    drifted = (monday + dt.timedelta(days=DEFAULT_HORIZON_DAYS)).date()
    assert announced == dt.date(2026, 7, 28)
    assert drifted != announced, 'this test no longer demonstrates the drift'

    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    real = pc.describe_next_fuel_adjustment(monday)['horizon_days']
    assert real < DEFAULT_HORIZON_DAYS


def test_horizon_stays_positive_right_after_an_adjustment():
    """Just after Tuesday 6am the next adjustment is a week away, not in the past.
    A negative horizon would make a run due immediately and grade it against the
    adjustment it was never about."""
    just_after = _at(2026, 7, 28, hour=6, minute=1)
    assert pc.describe_next_fuel_adjustment(just_after)['horizon_days'] > 6.0
