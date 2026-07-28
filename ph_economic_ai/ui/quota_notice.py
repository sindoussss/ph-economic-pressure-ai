"""A visible notice when a provider's quota runs out, and when it comes back.

A run that dies on HTTP 429 currently tells the user nothing. The traceback goes
to the console, the swarm raises "Group 1 failed", and the screen shows a stalled
progress bar. The one fact the user needs — which ceiling was hit and how long
until it clears — exists in the response headers and never reaches them.

Not a modal dialog, on purpose. The event arrives on a worker thread mid-run,
and a modal would block the Qt event loop, freeze the live canvas, and demand a
click before the run could continue. This is a banner that appears over the top
of the window, counts the refill down, and dismisses itself.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout


class QuotaNotice(QFrame):
    """Amber banner: which limit, how long, and what the app is doing about it."""

    #: Emitted from `report()` so a worker thread can raise the notice safely.
    #: Qt widgets may only be touched on the GUI thread, and the quota event
    #: arrives from whichever agent call hit the ceiling.
    quota_reported = pyqtSignal(object)

    _AMBER_BG = '#FEF3C7'
    _AMBER_LINE = '#FDE68A'
    _AMBER_INK = '#92400E'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('quotaNotice')
        self.setStyleSheet(
            f'QFrame#quotaNotice{{background:{self._AMBER_BG};'
            f'border:1px solid {self._AMBER_LINE};border-radius:10px;}}'
            f'QFrame#quotaNotice QLabel{{background:transparent;border:none;}}'
        )
        self._remaining = 0
        self._build()
        self.hide()

        self.quota_reported.connect(self._on_reported)
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._countdown)

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 10, 10)
        row.setSpacing(12)

        col = QVBoxLayout()
        col.setSpacing(2)
        self._title = QLabel('Provider limit reached')
        self._title.setStyleSheet(
            f'font-size:11px;font-weight:700;color:{self._AMBER_INK};')
        self._detail = QLabel('')
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(f'font-size:9px;color:{self._AMBER_INK};')
        col.addWidget(self._title)
        col.addWidget(self._detail)
        row.addLayout(col, 1)

        self._dismiss = QPushButton('Dismiss')
        self._dismiss.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dismiss.setStyleSheet(
            'QPushButton{background:transparent;border:1px solid #FDE68A;'
            'border-radius:6px;padding:4px 10px;font-size:9px;color:#92400E;}'
            'QPushButton:hover{background:#FDE68A;}'
        )
        self._dismiss.clicked.connect(self.dismiss)
        row.addWidget(self._dismiss, 0, Qt.AlignmentFlag.AlignTop)

    # ── Public API ────────────────────────────────────────────────────────────

    def report(self, status) -> None:
        """Thread-safe entry point. Register this with `llm.on_quota_event`."""
        self.quota_reported.emit(status)

    def dismiss(self) -> None:
        self._tick.stop()
        self.hide()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _on_reported(self, status) -> None:
        blocked = bool(getattr(status, 'blocked', False))
        scope = getattr(status, 'scope', 'tokens')
        provider = getattr(status, 'provider', 'provider')
        tier = getattr(status, 'tier', '')
        window = 'per-minute tokens' if scope == 'tokens' else 'daily requests'

        self._title.setText(
            f'{provider} {window} exhausted' if blocked
            else f'{provider} {window} almost exhausted')

        # What the app is doing about it matters as much as what went wrong. A
        # fallback to local means the run continues on a weaker model, and the
        # user has to know that before reading the verdict.
        try:
            from ph_economic_ai.engine import llm
            backup = llm.fallback_provider(tier) if tier else None
        except Exception:
            backup = None
        action = (f'Falling back to {backup} for the {tier} tier, so this run '
                  f'will finish on a weaker model.' if (blocked and backup)
                  else 'Waiting for the window to refill.' if blocked
                  else 'Still running.')

        self._status = status
        self._remaining = int(getattr(status, 'reset_seconds', None) or 0)
        self._action = action
        self._render()
        self.show()
        self.raise_()
        if self._remaining > 0:
            self._tick.start()

    def _render(self) -> None:
        limit = getattr(self._status, 'limit', None)
        remaining = getattr(self._status, 'remaining', None)
        used = ''
        if limit is not None and remaining is not None:
            used = f'{remaining:,} of {limit:,} left. '
        when = (f'Refills in {self._remaining}s. ' if self._remaining > 0
                else 'Should have refilled. ')
        self._detail.setText(f'{used}{when}{self._action}')

    def _countdown(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            self._remaining = 0
            self._tick.stop()
            self._title.setText('Provider limit refilled')
            self._detail.setText('The window has reset. Runs can proceed at '
                                 'full capacity again.')
            # Leave it up briefly so a user who looked away still sees the
            # recovery, then clear it rather than accumulating stale banners.
            QTimer.singleShot(8000, self.dismiss)
            return
        self._render()
