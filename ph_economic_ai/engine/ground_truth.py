from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

_POLL_INTERVAL_MS = 6 * 60 * 60 * 1000   # 6 hours in milliseconds


def compute_accuracy_score(estimate: float, actual: float) -> float:
    """₱0.00 error → 1.0 | ₱3.00+ error → 0.0 (linear)."""
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

        target = store.effective_target_date(run)
        match = store.price_near(target)
        if match is None:
            # No observation for this run's period. Leaving it ungraded is the
            # point of the fix; it stays eligible once a matching price arrives.
            logging.debug('ground_truth: run_id=%s has no price near %s; leaving ungraded',
                          run.get('run_id'), target)
            continue

        actual_change = match['price'] - baseline
        store.apply_ground_truth_grade(
            run['run_id'], actual_change,
            graded_against=f"{match['observed_at']} (gap {match['gap_days']:.2f}d)")
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
        from ph_economic_ai.engine.swarm import fetch_live_retail_price
        while not self._stop_event.is_set():
            try:
                current_price = fetch_live_retail_price()
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
