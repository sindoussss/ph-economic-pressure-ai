"""Pressure Monitor view — the one-click, monitor-led panel.

Layout follows the design: the present-pressure read (the Pressure Brief) is the
HERO at the top; the bounded, monthly Outlook sits below as a quieter secondary.
The single "Run" button gathers the frozen data, runs the Forum debate (Monitor),
then the Tournament debate (Outlook) — all off the UI thread via MonitorThread.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ph_economic_ai.engine.monitor import MonitorThread

_INK = '#0F1115'
_T2 = '#525866'
_T3 = '#8B95A7'
_DIV = '#E5E7EB'
_RISE = '#B42318'   # rising pressure = red
_EASE = '#067647'   # easing = green
_FLAT = '#525866'

_DIR_COLOR = {'rising': _RISE, 'easing': _EASE, 'flat': _FLAT, 'unknown': _T3}


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


class PressureMonitorPanel(QWidget):
    def __init__(self, rag, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._thread: MonitorThread | None = None
        self.setStyleSheet(f'background:#FAFAF8;')

        outer = QScrollArea(self)
        outer.setWidgetResizable(True)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.setStyleSheet('QScrollArea{border:none;background:#FAFAF8;}')
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(outer)

        body = QWidget()
        outer.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(48, 36, 48, 48)
        col.setSpacing(0)

        eyebrow = QLabel('PRESSURE MONITOR')
        eyebrow.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;letter-spacing:3px;'
            f'color:{_T3};')
        col.addWidget(eyebrow)

        title = QLabel('Current pressure on gas, food, and electricity')
        title.setStyleSheet(f'font-size:22px;font-weight:700;color:{_INK};'
                            f'margin-top:6px;')
        title.setWordWrap(True)
        col.addWidget(title)

        row = QHBoxLayout()
        self._run_btn = QPushButton('Run')
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setStyleSheet(
            f'QPushButton{{background:{_INK};color:#fff;border:none;border-radius:6px;'
            f'padding:9px 22px;font-size:13px;font-weight:600;}}'
            f'QPushButton:disabled{{background:#C7CBD1;}}')
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)
        self._status = QLabel('Reads the frozen snapshot; senses the present, then '
                              'forecasts humbly.')
        self._status.setStyleSheet(f'color:{_T2};font-size:12px;margin-left:14px;')
        row.addWidget(self._status)
        row.addStretch()
        row.setContentsMargins(0, 18, 0, 24)
        col.addLayout(row)

        # ── HERO: present-pressure cards ─────────────────────────────────────
        self._hero_label = QLabel('PRESENT PRESSURE')
        self._hero_label.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;letter-spacing:2px;'
            f'color:{_T3};margin-bottom:10px;')
        col.addWidget(self._hero_label)
        self._cards = QVBoxLayout()
        self._cards.setSpacing(12)
        col.addLayout(self._cards)
        self._narrative = QLabel('')
        self._narrative.setWordWrap(True)
        self._narrative.setStyleSheet(f'color:{_T2};font-size:13px;margin-top:14px;'
                                      f'line-height:1.5;')
        col.addWidget(self._narrative)

        # ── SECONDARY: bounded monthly outlook ───────────────────────────────
        self._outlook_label = QLabel('NEXT-MONTH OUTLOOK  ·  bounded')
        self._outlook_label.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;letter-spacing:2px;'
            f'color:{_T3};margin-top:34px;margin-bottom:10px;')
        col.addWidget(self._outlook_label)
        self._outlook = QVBoxLayout()
        self._outlook.setSpacing(10)
        col.addLayout(self._outlook)
        self._outlook_label.setVisible(False)

        col.addStretch()

    # ── run ──────────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._thread is not None and self._thread.isRunning():
            return
        _clear(self._cards)
        _clear(self._outlook)
        self._narrative.setText('')
        self._outlook_label.setVisible(False)
        self._run_btn.setEnabled(False)
        self._status.setText('Gathering current pressure and running the Forum debate…')

        self._thread = MonitorThread(self._rag, window='this_week', rounds=2)
        self._thread.monitor_ready.connect(self._on_monitor_ready)
        self._thread.outlook_ready.connect(self._on_outlook_ready)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

    def _on_error(self, msg: str):
        self._status.setText(f'Run failed: {msg}')
        self._run_btn.setEnabled(True)

    def _on_monitor_ready(self, brief):
        self._status.setText(f'Present read as of {brief.as_of} ({brief.window}). '
                             'Forecasting…')
        _clear(self._cards)
        for r in brief.readings:
            self._cards.addWidget(self._sector_card(r))
        if brief.narrative:
            self._narrative.setText(brief.narrative)

    def _on_outlook_ready(self, outlook):
        self._outlook_label.setVisible(True)
        _clear(self._outlook)
        for s in outlook.sectors:
            self._outlook.addWidget(self._outlook_row(s))
        self._status.setText(f'Done. Present read + bounded {outlook.horizon} outlook.')
        self._run_btn.setEnabled(True)

    # ── rendering ─────────────────────────────────────────────────────────────

    def _sector_card(self, r) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px solid {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)

        top = QHBoxLayout()
        name = QLabel(r.sector.upper())
        name.setStyleSheet(f'font-size:13px;font-weight:700;color:{_INK};'
                           f'letter-spacing:1px;')
        top.addWidget(name)
        top.addStretch()
        est = '—' if r.estimate is None else f'{r.estimate:+.2f} {r.unit}'
        val = QLabel(est)
        val.setStyleSheet(f'font-size:15px;font-weight:700;'
                          f'color:{_DIR_COLOR.get(r.direction, _T3)};')
        top.addWidget(val)
        lay.addLayout(top)

        meta = QLabel(f"{r.direction}  ·  agreement {r.confidence}%  "
                      f"(not a probability)  ·  {', '.join(r.sources) or 'no sources'}")
        meta.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:4px;')
        meta.setWordWrap(True)
        lay.addWidget(meta)

        if r.drivers:
            drv = QLabel('· ' + '\n· '.join(d for d in r.drivers if d))
            drv.setStyleSheet(f'color:{_T2};font-size:12px;margin-top:8px;')
            drv.setWordWrap(True)
            lay.addWidget(drv)
        return card

    def _outlook_row(self, s) -> QFrame:
        row = QFrame()
        row.setStyleSheet(
            f'QFrame{{background:transparent;border:none;border-top:1px solid {_DIV};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(row)
        lay.setContentsMargins(0, 10, 0, 2)
        head = QHBoxLayout()
        name = QLabel(f'{s.sector.upper()}  ·  {s.basis}')
        name.setStyleSheet(f'font-size:12px;font-weight:600;color:{_INK};')
        head.addWidget(name)
        head.addStretch()
        if s.point is not None and s.interval is not None:
            pt = QLabel(f'{s.point:+.2f} {s.unit}  [{s.interval[0]:+.2f}, '
                        f'{s.interval[1]:+.2f}]')
        else:
            pt = QLabel('—')
        pt.setStyleSheet(f'font-size:12px;color:{_T2};')
        head.addWidget(pt)
        lay.addLayout(head)
        note = QLabel(s.note)
        note.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:2px;')
        note.setWordWrap(True)
        lay.addWidget(note)
        return row
