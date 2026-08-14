"""Grading for the sectors that reprice MONTHLY.

`ground_truth.grade_verdict` grades gas. It reprices weekly, each run carries its
own baseline pump price, and one run is one gradable observation against the
pricing WEEK it claimed. That model does not transfer. Food reprices monthly and
PSA publishes one CPI print per calendar month, so the unit of evidence here is
a MONTH, not a run:

    N runs inside one calendar month are worth ONE graded sample, not N.

The app is run many times per month in normal use and testing. Counting runs
would let `interval.band` advertise "12 graded samples" while resting on two or
three genuinely independent months, repeated -- the same family of overclaim the
`gradable` flag was added to stop on screen. The rule is enforced twice: runs are
collapsed by month before anything is graded, and `sector_grades` is keyed
`PRIMARY KEY (sector, month)` so the table *cannot* hold two samples for one
month.

`grade_verdict_monthly` mirrors `grade_verdict`'s stance rather than its
signature: a month that cannot be graded says exactly WHY, from a closed
vocabulary, instead of being silently skipped. The screen and the grader then
cannot give different answers about why a count is still zero.

**Pace, stated plainly.** One sample per month is a hard ceiling, so reaching
`MIN_GRADED_FOR_CALIBRATION` (12) takes twelve independent months of wall-clock.
With the stored backlog contributing only 2026-06 and 2026-07, food is not
expected to show a calibrated band until roughly **mid-2027**. This feature
accrues; it does not finish. Loosening the month rule to make the number move
faster would forfeit the only thing the number is worth.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from statistics import median
from typing import Iterable, Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

#: PSA publishes monthly, so a daily poll is already generous. Gas polls every
#: six hours because DOE reprices weekly; matching that here would be 28 wasted
#: passes per new data point.
_POLL_INTERVAL_MS = 24 * 60 * 60 * 1000

#: Which stored run column carries each sector's monthly estimate.
SECTOR_COLUMN = {
    'food': 'food_estimate',
    'electricity': 'electricity_estimate',
}

#: Why a month was not graded. Ordered from "wait" to "never", matching
#: `ground_truth.UNGRADED_REASONS`.
UNGRADED_REASONS_MONTHLY = ('month_in_progress', 'no_cpi_yet', 'no_estimate')


def run_month(run: dict) -> Optional[str]:
    """The 'YYYY-MM' a run belongs to, or None if the timestamp is unreadable."""
    ts = run.get('timestamp')
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).strftime('%Y-%m')
    except ValueError:
        logging.warning('ground_truth_monthly: unparseable timestamp %r', ts)
        return None


def _current_month(now=None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    return f'{now.year:04d}-{now.month:02d}'


def _as_month_map(outcome_series) -> dict[str, float]:
    """Normalise a pandas Series / dict of monthly outcomes to {'YYYY-MM': float}."""
    out = {}
    for k, v in dict(outcome_series).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _by_month(runs: Iterable[dict], column: str) -> dict[str, list[float]]:
    """Group run estimates by calendar month. THE dedup step: after this a month
    is one bucket, and nothing downstream can turn it back into N samples."""
    grouped: dict[str, list[float]] = {}
    for run in runs:
        month = run_month(run)
        if month is None:
            continue
        grouped.setdefault(month, [])
        est = run.get(column)
        if est is not None:
            grouped[month].append(float(est))
    return grouped


def month_samples(runs: Iterable[dict], sector: str, outcome_series,
                  now=None) -> list[dict]:
    """At most ONE graded sample per settled calendar month, oldest first.

    Rows are {'month', 'estimate', 'actual', 'abs_error', 'n_runs'}. A month is
    emitted only with at least one run carrying an estimate AND a settled
    outcome -- published, and not the month in progress. Everything else is
    omitted rather than approximated; `grade_verdict_monthly` reports why.

    The month's `estimate` is the **median** of that month's runs. Median rather
    than mean because within-month estimates are heavy-tailed: one weak-model run
    can sit an order of magnitude off, and the sample should reflect what the app
    typically claimed that month, not what its worst run claimed.
    """
    return [s for s in _classify(runs, sector, outcome_series, now)['gradable']]


def _classify(runs, sector: str, outcome_series, now=None) -> dict:
    """Split months into gradable samples and blocked ones with a reason."""
    column = SECTOR_COLUMN.get(sector)
    if column is None:
        raise ValueError(
            f'no monthly grading column for sector {sector!r}; known sectors are '
            f'{sorted(SECTOR_COLUMN)}')

    truth = _as_month_map(outcome_series)
    this_month = _current_month(now)
    gradable, blocked = [], []

    # Grouped ONCE. Calling `_by_month` inside the loop re-consumed `runs`, which
    # is typed Iterable: a generator would have been exhausted by the first pass
    # and every month after it would have read as having no estimates.
    grouped = _by_month(runs, column)
    for month in sorted(grouped):
        estimates = grouped[month]
        if month >= this_month:
            blocked.append({
                'month': month, 'obstacle': 'month_in_progress',
                'reason': (f'{month} has not finished, so it has no settled CPI '
                           f'print to grade against'),
                'level': logging.DEBUG, 'actual_change': None})
            continue
        if not estimates:
            blocked.append({
                'month': month, 'obstacle': 'no_estimate',
                'reason': (f'no run in {month} recorded a {sector} estimate, so '
                           f'there is nothing to measure'),
                'level': logging.DEBUG, 'actual_change': None})
            continue
        if month not in truth:
            blocked.append({
                'month': month, 'obstacle': 'no_cpi_yet',
                'reason': (f'PSA has not published the {month} {sector} CPI yet '
                           f'(series ends {max(truth) if truth else "nowhere"})'),
                'level': logging.DEBUG, 'actual_change': None})
            continue
        estimate = float(median(estimates))
        actual = truth[month]
        gradable.append({
            'month': month, 'estimate': estimate, 'actual': actual,
            'actual_change': actual, 'abs_error': abs(estimate - actual),
            'n_runs': len(estimates),
        })
    return {'gradable': gradable, 'blocked': blocked}


def grade_verdict_monthly(store: 'AgentTrustStore', sector: str, outcome_series,
                          now=None) -> dict:
    """Which months can be graded, and for the rest, exactly why not.

    The monthly analogue of `grade_verdict`. Returns
    `{'gradable': [...], 'blocked': [{'month', 'obstacle', 'reason', 'level',
    'actual_change'}]}`, with every obstacle drawn from
    `UNGRADED_REASONS_MONTHLY`.

    Pure with respect to the store -- reads only -- so the dedup decision can be
    inspected and tested without writing anything. Months already recorded in
    `sector_grades` are excluded from both lists: they are settled history, not
    pending work.
    """
    already = store.get_graded_months(sector)
    split = _classify(store.get_all_runs(), sector, outcome_series, now)
    return {
        'gradable': [s for s in split['gradable'] if s['month'] not in already],
        'blocked': [b for b in split['blocked'] if b['month'] not in already],
    }


def find_and_grade_months(store: 'AgentTrustStore', sector: str, outcome_series,
                          now=None) -> int:
    """Grade every newly-settled month. Returns months NEWLY graded -- never
    runs, and never a month counted twice.

    Safe to call repeatedly: already-graded months are filtered here and
    rejected again by the table's primary key.
    """
    graded = 0
    for s in grade_verdict_monthly(store, sector, outcome_series, now=now)['gradable']:
        if store.upsert_sector_grade(sector, s['month'], s['estimate'],
                                     s['actual'], s['abs_error'], s['n_runs']):
            graded += 1
    return graded


def load_series(sector: str):
    """The settled monthly outcome series for a sector.

    Only called for sectors in `interval.GRADED_SECTORS`. Electricity is listed
    because the series exists, NOT because it is usable -- see `interval` for why
    its ₱/kWh estimates cannot be graded against a percentage CPI.
    """
    from ph_economic_ai.benchmark.psa_cpi import load_electricity_mom, load_food_mom
    if sector == 'food':
        return load_food_mom()
    if sector == 'electricity':
        return load_electricity_mom()
    raise ValueError(f'no monthly outcome series for sector {sector!r}')


def series_is_stale(outcome_series, now=None) -> bool:
    """True when the newest published month is more than a month overdue.

    The committed PSA CSVs are frozen gold. Unrefreshed, they stop yielding
    gradable months and grading quietly returns zero forever rather than
    failing -- indistinguishable from "no new evidence yet". This project has
    been bitten by exactly that, so staleness is reported rather than absorbed.
    """
    truth = _as_month_map(outcome_series)
    if not truth:
        return True
    try:
        year, month = (int(p) for p in max(truth).split('-'))
    except ValueError:
        return True
    month += 2                      # overdue once two full months have passed
    year, month = year + (month - 1) // 12, (month - 1) % 12 + 1
    if now is None:
        now = datetime.now(timezone.utc)
    return f'{now.year:04d}-{now.month:02d}' >= f'{year:04d}-{month:02d}'


class PSAMonthlyCheckerThread(QThread):
    """Grades newly-settled months for monthly sectors, in the background.

    The monthly counterpart to `DOECheckerThread`, which is left exactly as it
    is: gas's weekly path is untouched by any of this. Separate threads because
    the cadences differ by 28x, not because the work differs.
    """
    months_graded = pyqtSignal(str, int)      # sector, months newly graded
    feed_stale = pyqtSignal(str)              # sector whose CSV needs refreshing

    def __init__(self, store: 'AgentTrustStore', parent=None):
        super().__init__(parent)
        self._store = store
        self._stop_event = threading.Event()

    def run(self):
        from ph_economic_ai.engine.interval import GRADED_SECTORS
        while not self._stop_event.is_set():
            for sector in sorted(GRADED_SECTORS & set(SECTOR_COLUMN)):
                try:
                    series = load_series(sector)
                    if series_is_stale(series):
                        logging.warning(
                            'PSAMonthlyChecker: the %s outcome series is stale — '
                            'refresh the committed PSA CSV or months will never '
                            'settle and grading will read as "no evidence yet".',
                            sector)
                        self.feed_stale.emit(sector)
                    count = find_and_grade_months(self._store, sector, series)
                    if count:
                        self.months_graded.emit(sector, count)
                except Exception as e:
                    logging.warning('PSAMonthlyChecker(%s): %s', sector, e)
            self._stop_event.wait(timeout=_POLL_INTERVAL_MS / 1000)

    def stop(self):
        self._stop_event.set()
        self.quit()
