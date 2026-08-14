"""Calendar-aware helpers for month-indexed frames.

The benchmark indexes frames by the string 'YYYY-MM' and lags with `shift(n)`,
which moves by ROW. That is only a month lag when the index has no gaps. The
feature panel had 25 of 120 months missing, so row lags were reaching two and
three months back at 18 transitions without raising anything.

These helpers make the calendar explicit: a lag is a month lag, and a missing
month yields NaN rather than a silently wrong value.
"""
from __future__ import annotations

import pandas as pd


def to_periods(index) -> pd.PeriodIndex:
    """A monthly PeriodIndex from a 'YYYY-MM' string index.

    `exact=False` is what keeps the documented leniency: a daily string like
    '2020-01-15' collapses to its month rather than raising. pandas used to
    tolerate the trailing '-15' against a '%Y-%m' format and now enforces the
    format exactly, so without this the leniency the tests pin -- and the
    duplicate-month guard in `calendar_lag`, which can only fire once parsing
    succeeds -- both broke on the parse instead.

    The explicit format is kept rather than dropped for free parsing, so a
    genuinely malformed index still raises here instead of being guessed at.
    """
    return pd.PeriodIndex(
        pd.to_datetime(index, format='%Y-%m', exact=False), freq='M')


def missing_months(index) -> list[str]:
    """Calendar months absent between the first and last entry."""
    idx = to_periods(index)
    if len(idx) == 0:
        return []
    expected = pd.period_range(idx.min(), idx.max(), freq='M')
    return [str(p) for p in sorted(set(expected) - set(idx))]


def is_complete(index) -> bool:
    return not missing_months(index)


def calendar_lag(series: pd.Series, months: int = 1) -> pd.Series:
    """Value from `months` calendar months earlier, aligned to `series.index`.

    Returns NaN where that month is absent, so a gap becomes a dropped row rather
    than a wrong lag. On a complete index this is identical to `shift(months)`.
    """
    periods = to_periods(series.index)
    if periods.has_duplicates:
        dupes = sorted({str(p) for p in periods[periods.duplicated()]})
        raise ValueError(
            f'calendar_lag needs one row per month, got duplicate months: '
            f'{", ".join(dupes[:5])}. A daily or intra-month index collapses onto '
            f'the same period and the lag would be ambiguous.')
    by_period = pd.Series(series.to_numpy(), index=periods)
    wanted = periods - months
    values = by_period.reindex(wanted).to_numpy()
    return pd.Series(values, index=series.index, name=series.name)


def require_complete_calendar(frame, label: str):
    """Raise unless `frame` has every calendar month between its first and last.

    The forecasters take plain arrays and lag by POSITION -- `seasonal_naive`
    returns `y_train[-12]`, which is a year earlier only on a gap-free index. That
    assumption is invisible at the point it is relied on, so it is asserted here,
    where a frame is handed over for evaluation.
    """
    missing = missing_months(frame.index)
    if missing:
        shown = ', '.join(missing[:12])
        more = '' if len(missing) <= 12 else f' and {len(missing) - 12} more'
        raise ValueError(
            f'{label}: {len(missing)} calendar months missing ({shown}{more}). '
            f'Positional lags would not be month lags -- seasonal_naive would read '
            f'the wrong year. Rebuild the feature panel before evaluating.')
    return frame


def reindex_complete(frame: pd.DataFrame) -> pd.DataFrame:
    """Insert missing calendar months as all-NaN rows.

    Use when a caller needs the gaps to be visible in the frame itself. This does
    not invent values; it makes absence explicit.
    """
    idx = to_periods(frame.index)
    if len(idx) == 0:
        return frame
    full = pd.period_range(idx.min(), idx.max(), freq='M')
    out = frame.copy()
    out.index = idx
    out = out.reindex(full)
    out.index = full.strftime('%Y-%m')
    return out
