from __future__ import annotations

from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
)
from PyQt6.QtCore import Qt

_LEVER_COLORS: dict[str, str] = {
    'OPSF':     '#EA580C',
    'NFA':      '#16A34A',
    'BSP RATE': '#2563EB',
    'BSP':      '#2563EB',
    'EXCISE':   '#7C3AED',
    'DSWD':     '#0891B2',
    'ERC':      '#D97706',
    'DBM':      '#6D28D9',
    'DTI SRP':  '#0369A1',
    'DTI':      '#0369A1',
    'TARIFF':   '#B45309',
}
_DEFAULT_LEVER_COLOR = '#374151'

_URGENCY_LABELS = ['IMMEDIATE', 'SHORT-TERM', 'MEDIUM-TERM']
_URGENCY_COLORS = ['#DC2626', '#D97706', '#2563EB']


def _lever_color(lever: str) -> str:
    upper = lever.upper()
    for key, color in _LEVER_COLORS.items():
        if key in upper:
            return color
    return _DEFAULT_LEVER_COLOR


class PolicyRecoWidget(QFrame):
    """Displays AI-generated policy recommendations as styled cards."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            'QFrame#recoRoot{background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;}'
        )
        self.setObjectName('recoRoot')

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        # Header row
        header = QHBoxLayout()
        title = QLabel('Policy Recommendations')
        title.setStyleSheet(
            'font-size:11px;font-weight:700;color:#1C1E26;'
            'border:none;background:transparent;'
        )
        ai_badge = QLabel('AI ADVISER')
        ai_badge.setStyleSheet(
            'background:#1C1E26;color:#FFFFFF;font-size:7px;font-weight:700;'
            'border-radius:4px;padding:2px 6px;border:none;'
        )
        header.addWidget(title)
        header.addStretch()
        header.addWidget(ai_badge)
        root.addLayout(header)

        # Cards container — cleared and rebuilt on each set_recos() call
        self._cards = QVBoxLayout()
        self._cards.setSpacing(8)
        root.addLayout(self._cards)

        # Initial placeholder
        self._show_placeholder()

    # ── public API ────────────────────────────────────────────────────────────

    def set_recos(self, recos: list) -> None:
        self._clear_cards()
        if not recos:
            self._show_placeholder()
            return
        for idx, reco in enumerate(recos):
            self._cards.addWidget(self._make_card(reco, idx))

    # ── internals ─────────────────────────────────────────────────────────────

    def _clear_cards(self) -> None:
        while self._cards.count():
            item = self._cards.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_placeholder(self) -> None:
        lbl = QLabel('Awaiting sector verdicts…')
        lbl.setStyleSheet(
            'font-size:9px;color:#9EA3AE;border:none;background:transparent;'
        )
        self._cards.addWidget(lbl)

    def _make_card(self, reco, idx: int) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            'QFrame{background:#F8F9FB;border:1px solid #E5E7EB;border-radius:10px;}'
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 10)
        cl.setSpacing(6)

        # ── Top row: urgency chip + lever badge + action ──────────────────────
        top = QHBoxLayout()
        top.setSpacing(6)

        urgency_color = _URGENCY_COLORS[idx] if idx < len(_URGENCY_COLORS) else '#374151'
        urgency_lbl = _URGENCY_LABELS[idx] if idx < len(_URGENCY_LABELS) else ''

        urgency_chip = QLabel(urgency_lbl)
        urgency_chip.setFixedHeight(16)
        urgency_chip.setStyleSheet(
            f'background:transparent;color:{urgency_color};font-size:7px;'
            f'font-weight:700;border:1px solid {urgency_color};'
            f'border-radius:3px;padding:0px 5px;'
        )

        lever_chip = QLabel(reco.lever)
        lever_chip.setFixedHeight(16)
        lever_chip.setStyleSheet(
            f'background:{_lever_color(reco.lever)};color:#FFFFFF;'
            f'font-size:7px;font-weight:700;border-radius:3px;'
            f'padding:0px 6px;border:none;'
        )

        top.addWidget(urgency_chip)
        top.addWidget(lever_chip)
        top.addStretch()
        cl.addLayout(top)

        # ── Action ───────────────────────────────────────────────────────────
        action_lbl = QLabel(reco.action)
        action_lbl.setWordWrap(True)
        action_lbl.setStyleSheet(
            'font-size:10px;font-weight:600;color:#111827;'
            'border:none;background:transparent;'
        )
        cl.addWidget(action_lbl)

        # ── Impact badge ──────────────────────────────────────────────────────
        impact_lbl = QLabel(f'⟶  {reco.impact}')
        impact_lbl.setWordWrap(True)
        impact_lbl.setStyleSheet(
            'background:#DCFCE7;color:#15803D;font-size:9px;font-weight:600;'
            'border-radius:5px;padding:4px 9px;border:none;'
        )
        cl.addWidget(impact_lbl)

        # ── Timeline + Risk ───────────────────────────────────────────────────
        bot = QHBoxLayout()
        bot.setSpacing(10)

        tl_frame = QFrame()
        tl_frame.setStyleSheet(
            'QFrame{background:#F1F5F9;border-radius:6px;border:none;}'
        )
        tl_l = QVBoxLayout(tl_frame)
        tl_l.setContentsMargins(8, 5, 8, 5)
        tl_l.setSpacing(2)
        tl_header = QLabel('TIMELINE')
        tl_header.setStyleSheet(
            'font-size:7px;font-weight:700;color:#94A3B8;'
            'border:none;background:transparent;'
        )
        tl_val = QLabel(reco.timeline)
        tl_val.setWordWrap(True)
        tl_val.setStyleSheet(
            'font-size:8px;color:#374151;border:none;background:transparent;'
        )
        tl_l.addWidget(tl_header)
        tl_l.addWidget(tl_val)

        risk_frame = QFrame()
        risk_frame.setStyleSheet(
            'QFrame{background:#FEF9EE;border-radius:6px;border:none;}'
        )
        risk_l = QVBoxLayout(risk_frame)
        risk_l.setContentsMargins(8, 5, 8, 5)
        risk_l.setSpacing(2)
        risk_header = QLabel('RISK')
        risk_header.setStyleSheet(
            'font-size:7px;font-weight:700;color:#94A3B8;'
            'border:none;background:transparent;'
        )
        risk_val = QLabel(reco.risk)
        risk_val.setWordWrap(True)
        risk_val.setStyleSheet(
            'font-size:8px;color:#92400E;border:none;background:transparent;'
        )
        risk_l.addWidget(risk_header)
        risk_l.addWidget(risk_val)

        bot.addWidget(tl_frame, stretch=1)
        bot.addWidget(risk_frame, stretch=1)
        cl.addLayout(bot)

        return card
