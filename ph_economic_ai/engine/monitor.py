"""One-click Pressure Monitor pipeline: Gather → Forum debate (Monitor) →
Tournament debate (Outlook).

`run_pressure_monitor` is the headless composition (Stage 1 then Stage 2), so the
whole flow is testable without Qt. `MonitorThread` runs it off the UI thread and
emits the brief the moment Stage 1 finishes — so the app can paint the hero
(present read) immediately and fill in the bounded forecast after.
"""
from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine.forum import run_monitor
from ph_economic_ai.engine.outlook import Outlook, run_outlook
from ph_economic_ai.engine.pressure_brief import PressureBrief


def run_pressure_monitor(rag, corpus_dir=None, as_of=None, window: str = 'this_week',
                         sectors=('gas', 'food', 'electricity'), rounds: int = 2,
                         run_tournament: bool = True,
                         on_event: Optional[Callable[[str, dict], None]] = None
                         ) -> tuple[PressureBrief, Outlook]:
    """Stage 1 (Monitor) then Stage 2 (Outlook). Returns (brief, outlook)."""
    kwargs = {} if corpus_dir is None else {'corpus_dir': corpus_dir}
    brief = run_monitor(rag, as_of=as_of, window=window, sectors=sectors,
                        rounds=rounds, on_event=on_event, **kwargs)
    outlook = run_outlook(brief, rag=rag, run_tournament=run_tournament)
    return brief, outlook


class MonitorThread(QThread):
    """Runs the Monitor pipeline off the UI thread. `monitor_ready` fires first
    (paint the hero), then `outlook_ready` (fill the bounded forecast)."""
    monitor_ready = pyqtSignal(object)   # PressureBrief
    outlook_ready = pyqtSignal(object)   # Outlook
    forum_event = pyqtSignal(str, object)  # kind, data dict (agent_start/agent_message/moderator)
    error_occurred = pyqtSignal(str)

    def __init__(self, rag, window: str = 'this_week', rounds: int = 2,
                 run_tournament: bool = True, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._window = window
        self._rounds = rounds
        self._run_tournament = run_tournament

    def run(self):
        try:
            brief = run_monitor(self._rag, window=self._window, rounds=self._rounds,
                                on_event=lambda kind, data: self.forum_event.emit(kind, data))
            self.monitor_ready.emit(brief)
            outlook = run_outlook(brief, rag=self._rag,
                                  run_tournament=self._run_tournament)
            self.outlook_ready.emit(outlook)
        except Exception as e:
            self.error_occurred.emit(f'{type(e).__name__}: {e}')
