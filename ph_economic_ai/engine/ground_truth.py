from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

_POLL_INTERVAL_MS = 6 * 60 * 60 * 1000   # 6 hours in milliseconds

# The same bound the estimate parser uses, applied to the OUTCOME. A weekly DOE
# adjustment beyond this is not a market event, it is a broken baseline. Imported
# so the two cannot drift apart: an outcome the app would refuse to accept as an
# estimate must not be accepted as the truth it is graded against.
from ph_economic_ai.engine.debate import _MAX_REALISTIC_FUEL_PHP_L as _MAX_PLAUSIBLE_CHANGE


def compute_accuracy_score(estimate: float, actual: float) -> float:
    """₱0.00 error → 1.0 | ₱3.00+ error → 0.0 (linear).

    The floor at ₱3.00 is deliberate but blunt, and worth knowing when reading a
    trust score: a run that missed by ₱3.01 and one that missed by ₱20 score the
    same 0.0. On the stored history half the graded runs sat at or beyond that
    floor, so the metric could not rank them against each other. That is
    tolerable for "was this close", and it is why accuracy is only 60 percent of
    the trust update rather than all of it.
    """
    return max(0.0, 1.0 - abs(estimate - actual) / 3.0)


def find_and_grade_runs(
    store: 'AgentTrustStore',
    current_price: float,
    min_age_days: float = 5.0,
) -> int:
    """Grade every run whose forecast period has elapsed, each against a price
    observed near ITS OWN target date. Returns the count graded.

    RSK-018. This used to grade every ungraded run against `current_price`, so a
    run five days old and a run sixty days old were scored against the same
    number. A one-week forecast judged against a price two months later is not a
    wrong grade so much as a grade of a different question, and it fed the trust
    scores and the accuracy view.

    `current_price` is still accepted: it is the observation being reported now, so
    it is recorded into the price history and is the natural match for runs whose
    target date is around today. Runs whose period has no observation close enough
    stay ungraded rather than being scored against a mismatched week.
    """
    store.record_price_observation(current_price)

    graded = 0
    for run in store.get_due_runs():
        try:
            scenario = json.loads(run['scenario_json'])
        except json.JSONDecodeError:
            logging.warning('ground_truth: malformed scenario_json for run_id=%s',
                            run.get('run_id'))
            continue
        baseline = scenario.get('current_price')
        if baseline is None:
            continue
        # No magic-value check on the baseline here, deliberately. Comparing it
        # against the fallback constant was tried and rejected: it cannot tell
        # "this IS the fallback" from "the real price happens to equal it", so a
        # week where the market lands on that number would silently stop grading
        # for good. The liveness is recorded at the source instead — a run whose
        # price could not be fetched stores no baseline at all and is skipped by
        # the `baseline is None` check above.

        # A measured change has TWO ends, and the baseline is the other one. If
        # the week the run started in has no single price, the change from it
        # has no single value either. Live case: the source read 84.38 on
        # Thursday 30 July and 89.51 on Friday 31 July, both inside the cycle
        # opened 07-28. Runs that stored 84.38 as their baseline would score a
        # +5.13 "actual change" that is real only if the 89.51 reading was the
        # error -- and nothing here can know which one was. Refusing costs a
        # grade; grading costs the track record's meaning.
        #
        # Absence is not ambiguity. Most older runs have no observation in their
        # own week at all, and for those the stored baseline is the best record
        # of what the run actually reasoned from. Only disagreement disqualifies.
        started, _, _ = store.cycle_prices(run['timestamp'])
        if len(started) > 1:
            logging.debug('ground_truth: run_id=%s started in a week with no single '
                          'price (%s); the change from it is undefined, leaving '
                          'ungraded', run.get('run_id'), sorted(started))
            continue

        target = store.target_cycle(run)
        # The PRICING WEEK, not the nearest scrape and not a derived instant. A
        # weekly forecast is a claim about the Tuesday step, so grading it
        # against an arbitrary timestamp measures the source's sampling as much
        # as the forecast. Two live defects came from that: runs targeting
        # 2026-08-03 were graded against an 08-04 observation, which is the NEXT
        # cycle, and the source moved 84.38 to 89.51 on a Thursday-to-Friday
        # inside one cycle, turning +5.13 of scrape noise into an "actual
        # change" and producing the only two zero scores on record.
        match = store.cycle_price(target)
        if match is None:
            # Either no observation for that week, or the week's observations
            # disagree and it has no single price. Both leave the run eligible
            # rather than graded against a guess.
            logging.debug('ground_truth: run_id=%s has no unambiguous price for the '
                          'pricing week it forecasts (opened %s); leaving ungraded',
                          run.get('run_id'), target.date().isoformat())
            continue

        actual_change = match['price'] - baseline
        if abs(actual_change) > _MAX_PLAUSIBLE_CHANGE:
            # The app already refuses an ESTIMATE outside this bound as an
            # absolute-price parse. The OUTCOME deserves the same scepticism, and
            # did not get it: a stale baseline in the stored scenario produced an
            # "actual change" of -14.44 PHP/L on every graded run, which is the
            # observed price minus a hardcoded fallback rather than any week's
            # move. Nothing rejected it, so `compute_accuracy_score` floored at
            # zero for every agent and drove seven of twenty below the demotion
            # threshold on a number that never described a real outcome.
            #
            # An implausible outcome means the baseline or the observation is
            # wrong, and grading against either is worse than not grading at all,
            # because a wrong grade is permanent and silently reshapes the roster.
            logging.warning(
                'ground_truth: run_id=%s implies a %+.2f PHP/L change '
                '(observed %.2f vs stored baseline %.2f). That is outside the '
                '+/-%.0f plausibility bound, so the baseline is stale rather than '
                'the market extraordinary. Leaving it ungraded.',
                run.get('run_id'), actual_change, match['price'], baseline,
                _MAX_PLAUSIBLE_CHANGE)
            continue

        store.apply_ground_truth_grade(
            run['run_id'], actual_change,
            graded_against=f"pricing week {match['cycle']} "
                           f"({match['n_observations']} observations)")
        graded += 1
    return graded


class DOECheckerThread(QThread):
    """Background QThread that polls DOE price every 6 hours and grades old runs."""
    grades_applied = pyqtSignal(int)   # count of runs graded

    def __init__(self, store: 'AgentTrustStore', parent=None):
        super().__init__(parent)
        self._store = store
        self._stop_event = threading.Event()

    def run(self):
        from ph_economic_ai.engine.swarm import fetch_live_retail_price_checked
        while not self._stop_event.is_set():
            try:
                current_price, is_live = fetch_live_retail_price_checked()
                if not is_live:
                    # A failed scrape is not an observation. Recording the
                    # fallback as one is how ten runs came to be graded against a
                    # change of exactly +0.00: the constant minus itself, which
                    # reads as a quiet week and scores every estimate as wrong by
                    # its own magnitude.
                    logging.info('DOECheckerThread: price unavailable, skipping '
                                 'this grading pass rather than recording the '
                                 'fallback as an observation')
                    self._stop_event.wait(timeout=_POLL_INTERVAL_MS / 1000)
                    continue
                count = find_and_grade_runs(self._store, current_price)
                if count:
                    self.grades_applied.emit(count)
            except Exception as e:
                logging.warning('DOECheckerThread: %s', e)
            # Sleep in 30-second chunks so stop() wakes us promptly
            self._stop_event.wait(timeout=_POLL_INTERVAL_MS / 1000)

    def stop(self):
        self._stop_event.set()
        self.quit()
