# ph_economic_ai/ui/agent_performance.py
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

_GREEN  = '#1a7f37'
_AMBER  = '#7d4e00'
_RED    = '#cf222e'
_GRAY   = '#57606a'
_BG_GREEN  = '#dafbe1'
_BG_AMBER  = '#fff8c5'
_BG_RED    = '#ffebe9'
_BG_GRAY   = '#f6f8fa'


def _tier_color(trust: float) -> tuple[str, str]:
    if trust > 0.70:
        return _GREEN, _BG_GREEN
    if trust < 0.30:
        return _RED, _BG_RED
    return _AMBER, _BG_AMBER


class AgentPerformancePanel(QWidget):
    def __init__(self, store: 'AgentTrustStore', parent=None):
        super().__init__(parent)
        self._store = store
        self.setStyleSheet('background:#ffffff;')
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Left: trust leaderboard ──────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(240)
        left.setStyleSheet('background:#f6f8fa;border-right:1px solid #d0d7de;')
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)

        header_lbl = QLabel('TRUST LEADERBOARD')
        header_lbl.setStyleSheet(
            'color:#57606a;font-size:10px;font-weight:700;letter-spacing:1px;'
        )
        left_layout.addWidget(header_lbl)

        self._leaderboard_area = QWidget()
        self._leaderboard_layout = QVBoxLayout(self._leaderboard_area)
        self._leaderboard_layout.setContentsMargins(0, 8, 0, 0)
        self._leaderboard_layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._leaderboard_area)
        scroll.setStyleSheet('QScrollArea{border:none;}')
        left_layout.addWidget(scroll)
        root.addWidget(left)

        # ── Right: run history ───────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(14, 14, 14, 14)

        run_lbl = QLabel('RUN HISTORY')
        run_lbl.setStyleSheet(
            'color:#57606a;font-size:10px;font-weight:700;letter-spacing:1px;'
        )
        right_layout.addWidget(run_lbl)

        self._run_table = QTableWidget(0, 6)
        self._run_table.setHorizontalHeaderLabels(
            ['Date', 'Predicted', 'Actual', 'Error', 'Quality', 'Status']
        )
        self._run_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._run_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._run_table.setStyleSheet(
            'QTableWidget{border:1px solid #d0d7de;border-radius:6px;background:#fff;}'
            'QHeaderView::section{background:#f6f8fa;color:#57606a;font-weight:600;'
            '  border:none;border-bottom:1px solid #d0d7de;padding:4px;}'
        )
        right_layout.addWidget(self._run_table)

        self._doe_status = QLabel('DOE Checker: initializing...')
        self._doe_status.setStyleSheet(
            'background:#ddf4ff;border:1px solid #80ccff;border-radius:6px;'
            'color:#0550ae;font-size:11px;padding:6px 10px;'
        )
        right_layout.addWidget(self._doe_status)
        root.addWidget(right, stretch=1)

    def refresh(self) -> None:
        self._refresh_leaderboard()
        self._refresh_run_table()

    def update_doe_status(self, last_checked: str, next_check: str,
                          pending_count: int) -> None:
        self._doe_status.setText(
            f'DOE Checker active · Last checked: {last_checked} · '
            f'Next check: {next_check} · {pending_count} run(s) pending grade'
        )

    def _refresh_leaderboard(self) -> None:
        rows = self._store.get_all_trust_rows()
        while self._leaderboard_layout.count():
            item = self._leaderboard_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for row in rows[:15]:
            trust = row['trust_score']
            tc, bc = _tier_color(trust)
            name = row['agent_name']
            abbrev = ''.join(w[0] for w in name.split()[:2]).upper()

            entry = QWidget()
            entry.setStyleSheet('background:transparent;')
            h = QHBoxLayout(entry)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)

            badge = QLabel(abbrev)
            badge.setFixedSize(34, 34)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(
                f'background:{bc};color:{tc};border:1.5px solid {tc};'
                f'border-radius:17px;font-size:9px;font-weight:700;'
            )
            h.addWidget(badge)

            meta = QVBoxLayout()
            meta.setSpacing(2)
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet('color:#24292f;font-size:11px;font-weight:500;')
            meta.addWidget(name_lbl)

            bar_outer = QFrame()
            bar_outer.setFixedHeight(5)
            bar_outer.setStyleSheet('background:#eaeef2;border-radius:2px;')
            bar_inner = QFrame(bar_outer)
            bar_inner.setFixedHeight(5)
            bar_inner.setFixedWidth(max(4, int(trust * 140)))
            bar_inner.setStyleSheet(f'background:{tc};border-radius:2px;')
            meta.addWidget(bar_outer)
            h.addLayout(meta)

            score_lbl = QLabel(f'{trust:.2f}')
            score_lbl.setStyleSheet(
                f'color:{tc};font-family:monospace;font-size:11px;font-weight:700;'
            )
            h.addWidget(score_lbl)
            self._leaderboard_layout.addWidget(entry)

        self._leaderboard_layout.addStretch()

    def _refresh_run_table(self) -> None:
        from datetime import datetime
        runs = self._store.get_recent_runs(limit=20)
        self._run_table.setRowCount(len(runs))
        for i, run in enumerate(runs):
            ts = run.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(ts)
                date_str = dt.strftime('%b %d')
            except Exception:
                date_str = ts[:10]

            pred = run.get('final_estimate')
            actual = run.get('actual_price_change')
            error = run.get('accuracy_error')
            quality = run.get('internal_quality')

            pred_str = f'+₱{pred:.2f}/L' if pred is not None else '—'
            actual_str = f'+₱{actual:.2f}/L' if actual is not None else '—'
            error_str = f'₱{error:.2f}' if error is not None else '—'
            quality_str = f'{quality:.2f}' if quality is not None else '—'

            if actual is not None:
                status = 'Graded ✓'
                status_color = _GREEN
                status_bg = _BG_GREEN
            else:
                status = '⏳ Pending DOE'
                status_color = _GRAY
                status_bg = _BG_GRAY

            for col, val in enumerate([date_str, pred_str, actual_str,
                                        error_str, quality_str, status]):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 5:
                    item.setForeground(QColor(status_color))
                    item.setBackground(QColor(status_bg))
                self._run_table.setItem(i, col, item)
