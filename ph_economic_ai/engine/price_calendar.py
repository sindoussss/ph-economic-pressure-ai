"""When the next price change actually happens.

The app could say how much but never when, so a user asking "is it tomorrow, or
the 27th?" got a magnitude floating free of any date. `engine/outlook.py` states
the horizon is always monthly and no weekly number is ever produced, while the
swarm prompt tells agents "typical weekly adjustments are PHP0.20 to PHP3.00/L".
The engine was reasoning weekly and reporting monthly.

The useful insight is that for Philippine fuel, *when* is mostly a calendar fact
rather than a forecast. Oil companies move pump prices on a weekly cycle with a
published effective day, so the date of the next change is known; what is
uncertain is its direction and size. That turns an unanswerable question ("which
future date will prices rise") into an answerable one ("at the next adjustment,
on this date, which way and how much").

Scope limit, deliberate: this module reports the next *scheduled* adjustment only.
It does not predict which future week a rise will occur in. The benchmark does not
support that claim, and a dated prediction is exactly the kind of confident
overreach the audit exists to prevent.
"""
from __future__ import annotations

import datetime as dt

# Philippine Standard Time. No daylight saving.
PH_TZ = dt.timezone(dt.timedelta(hours=8), name='PHT')

# The weekly retail fuel price adjustment takes effect on Tuesday, following the
# Monday announcement by the oil companies. Monday is 0, so Tuesday is 1.
FUEL_ADJUSTMENT_WEEKDAY = 1
# Most companies implement in the early morning. The hour only decides whether a
# Tuesday that has already begun still counts as upcoming.
FUEL_EFFECTIVE_HOUR = 6

# PSA publishes the monthly CPI in the first week of the following month. The day
# moves, so this is the scheduled anchor and not a guarantee.
CPI_RELEASE_DAY = 5


def _now(now=None) -> dt.datetime:
    if now is None:
        return dt.datetime.now(PH_TZ)
    if now.tzinfo is None:
        return now.replace(tzinfo=PH_TZ)
    return now.astimezone(PH_TZ)


def next_fuel_adjustment(now=None) -> dt.datetime:
    """The next weekly fuel price adjustment, in Philippine time.

    A Tuesday before the effective hour still counts as today's adjustment; after
    it, the next one is a week out.
    """
    current = _now(now)
    days_ahead = (FUEL_ADJUSTMENT_WEEKDAY - current.weekday()) % 7
    candidate = (current + dt.timedelta(days=days_ahead)).replace(
        hour=FUEL_EFFECTIVE_HOUR, minute=0, second=0, microsecond=0)
    if candidate <= current:
        candidate += dt.timedelta(days=7)
    return candidate


def next_cpi_release(now=None) -> dt.datetime:
    """The next scheduled PSA CPI release."""
    current = _now(now)
    candidate = current.replace(day=CPI_RELEASE_DAY, hour=9, minute=0,
                                second=0, microsecond=0)
    if candidate <= current:
        first_next = (current.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        candidate = first_next.replace(day=CPI_RELEASE_DAY, hour=9, minute=0,
                                       second=0, microsecond=0)
    return candidate


def humanize_until(target: dt.datetime, now=None) -> str:
    """'today', 'tomorrow', or 'in N days' -- the phrasing a user asked for."""
    current = _now(now)
    delta_days = (target.date() - current.date()).days
    if delta_days <= 0:
        return 'today'
    if delta_days == 1:
        return 'tomorrow'
    return f'in {delta_days} days'


def describe_next_fuel_adjustment(now=None) -> dict:
    """Everything the UI needs to answer "when", with no forecasting involved."""
    current = _now(now)
    target = next_fuel_adjustment(current)
    return {
        'event': 'fuel_price_adjustment',
        'effective_at': target.isoformat(),
        'date': target.date().isoformat(),
        'weekday': target.strftime('%A'),
        'label': target.strftime('%A, %d %B %Y'),
        'when': humanize_until(target, current),
        'days_away': (target.date() - current.date()).days,
        'horizon_days': round((target - current).total_seconds() / 86400.0, 3),
        'basis': 'scheduled',
        'note': ('Weekly cycle: announced Monday, effective Tuesday. A holiday can '
                 'shift it, so treat this as the scheduled date rather than a '
                 'guarantee.'),
    }


def describe_next_cpi_release(now=None) -> dict:
    current = _now(now)
    target = next_cpi_release(current)
    return {
        'event': 'psa_cpi_release',
        'effective_at': target.isoformat(),
        'date': target.date().isoformat(),
        'label': target.strftime('%d %B %Y'),
        'when': humanize_until(target, current),
        'days_away': (target.date() - current.date()).days,
        'horizon_days': round((target - current).total_seconds() / 86400.0, 3),
        'basis': 'scheduled',
        'note': 'PSA publishes in the first week of the following month; the day moves.',
    }
