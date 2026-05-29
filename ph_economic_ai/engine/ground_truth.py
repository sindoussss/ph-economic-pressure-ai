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
    """Find ungraded runs older than min_age_days, grade them against current_price, return count graded."""
    ungraded = store.get_ungraded_runs(min_age_days=min_age_days)
    graded = 0
    for run in ungraded:
        try:
            scenario = json.loads(run['scenario_json'])
        except json.JSONDecodeError:
            logging.warning('ground_truth: malformed scenario_json for run_id=%s', run.get('run_id'))
            continue
        baseline = scenario.get('current_price')
        if baseline is None:
            continue
        actual_change = current_price - baseline
        store.apply_ground_truth_grade(run['run_id'], actual_change)
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
