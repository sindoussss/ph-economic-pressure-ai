"""Causal chain visualization widget for the PH Economic Simulation.

Displays the cross-sector causal chain (oil shock → fuel → food → electricity
→ household CPI → BSP signal) as a vertical waterfall with colored nodes,
connecting arrows, and magnitude badges.

Also contains BSPAlertBanner — a horizontal banner shown in Stage 4 when
the projected CPI breaches the BSP 2-4% target.
"""
from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPainterPath, QLinearGradient,
)

# ── Color map for chain node types ────────────────────────────────────────────
_NODE_COLORS = {
    'trigger':     '#6366F1',   # indigo  — input shock
    'oil':         '#B45309',   # amber   — crude / landed cost
    'fuel':        '#DC2626',   # red     — pump price
    'transport':   '#D97706',   # orange  — logistics
    'food':        '#16A34A',   # green   — food sector
    'electricity': '#0284C7',   # blue    — electricity sector
    'household':   '#475569',   # slate   — CPI / household
    'policy':      '#7C3AED',   # violet  — BSP / policy
    'default':     '#64748B',   # gray    — fallback
}

_SEVERITY_COLORS = {
    'STABLE':   ('#059669', '#ECFDF5', '#D1FAE5'),  # text, bg, border
    'WATCH':    ('#D97706', '#FFFBEB', '#FDE68A'),
    'ALERT':    ('#DC2626', '#FEF2F2', '#FECACA'),
    'CRITICAL': ('#7F1D1D', '#FEF2F2', '#F87171'),
}

_SEVERITY_ICONS = {
    'STABLE':   'BSP TARGET: WITHIN RANGE',
    'WATCH':    'BSP TARGET: APPROACHING UPPER BOUND',
    'ALERT':    'BSP TARGET: BREACHED — RATE ACTION LIKELY',
    'CRITICAL': 'BSP TARGET: CRITICALLY EXCEEDED — TIGHTENING EXPECTED',
}


def _classify_node(label: str) -> str:
    label_lower = label.lower()
    if any(k in label_lower for k in ('shock', 'input', 'trigger', 'opec', 'scenario')):
        return 'trigger'
    if any(k in label_lower for k in ('brent', 'crude', 'wti', 'oil', 'landed', 'import')):
        return 'oil'
    if any(k in label_lower for k in ('pump', 'fuel', 'gasoline', 'diesel', 'retail', 'doe')):
        return 'fuel'
    if any(k in label_lower for k in ('transport', 'logistics', 'freight', 'distribution')):
        return 'transport'
    if any(k in label_lower for k in ('food', 'agri', 'rice', 'nfa', 'harvest', 'cpi food')):
        return 'food'
    if any(k in label_lower for k in ('electric', 'meralco', 'power', 'grid', 'kwh', 'erc')):
        return 'electricity'
    if any(k in label_lower for k in ('household', 'consumer', 'basket', 'cpi', 'inflation')):
        return 'household'
    if any(k in label_lower for k in ('bsp', 'policy', 'rate', 'monetary', 'signal')):
        return 'policy'
    return 'default'


# ── BSP Alert Banner ──────────────────────────────────────────────────────────

class BSPAlertBanner(QFrame):
    """Plain page-header stat shown at the top of Stage 4 when BSP target is
    at risk. No colored box -- severity is conveyed by the eyebrow wording
    and a small severity-colored dot, not a bordered/tinted alert card."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._header = None

    def set_alert(self, alert: dict):
        severity    = alert.get('severity', 'STABLE')
        projected   = alert.get('projected_cpi', 0.0)
        current     = alert.get('current_cpi', 0.0)
        # No hardcoded fallback CPI here: an unrelated constant duplicating
        # `live_data._CURRENT_CPI_PCT` is exactly the "one quantity, two
        # constructions" shape this project's own audits keep finding, and it
        # would decay out of sync with the real one the moment either changes.
        cpi_as_of   = alert.get('cpi_as_of', 'unknown vintage')
        impact      = alert.get('sector_cpi_impact', 0.0)
        breakdown   = alert.get('breakdown', {})

        text_c, _bg_c, _border_c = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS['STABLE'])

        if self._header is not None:
            self._layout.removeWidget(self._header)
            self._header.deleteLater()

        parts = []
        if 'fuel' in breakdown:
            parts.append(f'Fuel: +{breakdown["fuel"]:.2f}ppt')
        if 'food' in breakdown:
            parts.append(f'Food: +{breakdown["food"]:.2f}ppt')
        if 'electricity' in breakdown:
            parts.append(f'Elec: +{breakdown["electricity"]:.2f}ppt')
        caption = (f'Baseline {current:.1f}% ({cpi_as_of}) + sector impact {impact:+.2f}ppt'
                  + ('  ·  ' + '  ·  '.join(parts) if parts else ''))

        icon = {'STABLE': '●', 'WATCH': '◆', 'ALERT': '▲', 'CRITICAL': '■'}.get(severity, '●')
        eyebrow_text = f'{icon} {_SEVERITY_ICONS.get(severity, "")}'

        from ph_economic_ai.ui import theme as _theme
        self._header = _theme.page_header(
            eyebrow_text, ' ',
            right_eyebrow='PROJECTED CPI',
            right_value=f'{projected:.2f}%',
            right_caption=caption)
        # The severity color still shows, just on the eyebrow icon/text rather
        # than as a page-wide tint -- find the eyebrow label and recolor it.
        for lbl in self._header.findChildren(QLabel):
            if lbl.text() == eyebrow_text:
                lbl.setStyleSheet(lbl.styleSheet() + f';color:{text_c};')
                break
        self._layout.addWidget(self._header)
        self.show()


# ── Chain Node Painter ────────────────────────────────────────────────────────

class _ChainNode(QWidget):
    """One node in the causal chain waterfall."""

    def __init__(self, step_idx: int, label: str, mechanism: str,
                 magnitude: str, is_last: bool = False, parent=None):
        super().__init__(parent)
        self._step_idx  = step_idx
        self._label     = label
        self._mechanism = mechanism
        self._magnitude = magnitude
        self._is_last   = is_last
        self._node_type = _classify_node(label)
        self._color     = QColor(_NODE_COLORS.get(self._node_type, _NODE_COLORS['default']))
        self._phase     = 0.0
        self._anim      = False

        self.setMinimumHeight(72)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def start_anim(self):
        self._anim = True
        self._phase = 0.0

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w, h = self.width(), self.height()

        # ── Layout constants ──────────────────────────────────────────────────
        node_x    = 20          # left edge of node circle
        node_r    = 10          # circle radius
        node_cx   = node_x + node_r
        node_cy   = 28          # vertical center of node
        text_x    = node_x + node_r * 2 + 12  # start of text block
        badge_r   = 5

        # ── Vertical connector line (above node) ──────────────────────────────
        if self._step_idx > 0:
            lc = QColor(self._color); lc.setAlpha(60)
            painter.setPen(QPen(lc, 1.5, Qt.PenStyle.SolidLine))
            painter.drawLine(node_cx, 0, node_cx, node_cy - node_r - 1)

        # ── Arrowhead ─────────────────────────────────────────────────────────
        if self._step_idx > 0:
            ac = QColor(self._color); ac.setAlpha(100)
            painter.setPen(QPen(ac, 1.2))
            painter.setBrush(QBrush(ac))
            arrow = QPainterPath()
            ax, ay = node_cx, node_cy - node_r - 1
            arrow.moveTo(ax, ay)
            arrow.lineTo(ax - 4, ay - 7)
            arrow.lineTo(ax + 4, ay - 7)
            arrow.closeSubpath()
            painter.drawPath(arrow)

        # ── Node circle ───────────────────────────────────────────────────────
        # Outer ring (glow)
        gc = QColor(self._color); gc.setAlpha(30)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gc))
        painter.drawEllipse(QPointF(node_cx, node_cy), node_r + 5, node_r + 5)

        # Main fill
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(QPointF(node_cx, node_cy), node_r, node_r)

        # Step number inside circle
        f = QFont(); f.setPixelSize(8); f.setBold(True); painter.setFont(f)
        painter.setPen(QPen(QColor('#FFFFFF')))
        painter.drawText(
            QRectF(node_cx - node_r, node_cy - node_r, node_r * 2, node_r * 2),
            Qt.AlignmentFlag.AlignCenter,
            str(self._step_idx + 1)
        )

        # ── Connector line (below node, to next) ──────────────────────────────
        if not self._is_last:
            lc2 = QColor(self._color); lc2.setAlpha(40)
            painter.setPen(QPen(lc2, 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(node_cx, node_cy + node_r + 1, node_cx, h)

        # ── Label text ────────────────────────────────────────────────────────
        lf = QFont(); lf.setPixelSize(10); lf.setBold(True); painter.setFont(lf)
        painter.setPen(QPen(self._color))
        painter.drawText(
            QRectF(text_x, 14, w - text_x - 10, 16),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._label
        )

        # ── Mechanism text ────────────────────────────────────────────────────
        mf = QFont(); mf.setPixelSize(8); painter.setFont(mf)
        painter.setPen(QPen(QColor('#6B7280')))
        painter.drawText(
            QRectF(text_x, 32, w - text_x - 90, 28),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
            self._mechanism
        )

        # ── Magnitude badge ───────────────────────────────────────────────────
        if self._magnitude:
            badge_text = self._magnitude[:18]
            bf = QFont(); bf.setPixelSize(9); bf.setBold(True); painter.setFont(bf)
            fm_width = 70  # fixed badge width

            badge_x = w - fm_width - 10
            badge_y = node_cy - 9
            bh      = 18

            # Badge background
            pos = '+' in badge_text and '-' not in badge_text.split('+')[0]
            badge_bg = QColor('#DCFCE7' if pos else '#FEE2E2')
            badge_border = QColor('#16A34A' if pos else '#DC2626')
            badge_text_c = QColor('#166534' if pos else '#991B1B')

            painter.setPen(QPen(badge_border, 1.0))
            painter.setBrush(QBrush(badge_bg))
            painter.drawRoundedRect(badge_x, badge_y, fm_width, bh, 4, 4)
            painter.setPen(QPen(badge_text_c))
            painter.drawText(
                QRectF(badge_x, badge_y, fm_width, bh),
                Qt.AlignmentFlag.AlignCenter,
                badge_text
            )

        painter.end()


# ── Main Causal Chain Widget ──────────────────────────────────────────────────

class CausalChainWidget(QFrame):
    """Scrollable vertical causal chain display.

    Call set_chain(steps) with a list of CausalChainStep objects.
    Call set_alert(alert_dict) to show the BSP banner inside this widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('CCW')
        self.setStyleSheet(
            'QFrame#CCW{background:#FFFFFF;border:1px solid #E5E7EB;border-radius:12px;}'
            'QFrame#CCW QLabel{background:transparent;border:none;}'
        )
        self._steps: list = []
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setFixedHeight(38)
        hdr.setStyleSheet('background:#F8F9FB;border-bottom:1px solid #E5E7EB;border-radius:12px 12px 0 0;')
        hl = QHBoxLayout(hdr); hl.setContentsMargins(14, 0, 14, 0)

        title = QLabel('MACRO CAUSAL CHAIN')
        title.setStyleSheet('font-size:8px;font-weight:700;letter-spacing:1.5px;color:#9CA3AF;')

        self._badge = QLabel('Pending…')
        self._badge.setStyleSheet(
            'font-size:7px;font-weight:600;color:#6B7280;background:#F3F4F6;'
            'border-radius:3px;padding:1px 7px;border:1px solid #E5E7EB;')

        hl.addWidget(title); hl.addStretch(); hl.addWidget(self._badge)
        root.addWidget(hdr)

        # BSP inline alert strip (inside the card, below header)
        self._bsp_strip = BSPAlertBanner()
        root.addWidget(self._bsp_strip)

        # Scrollable chain body
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            'QScrollBar:vertical{width:3px;background:transparent;}'
            'QScrollBar::handle:vertical{background:#D1D5DB;border-radius:1px;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}'
        )

        self._body = QWidget()
        self._body.setStyleSheet('background:transparent;')
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 8, 0, 16)
        self._body_layout.setSpacing(0)

        self._placeholder = QLabel('Chain generates after all three sector debates complete.')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet('font-size:9px;color:#9CA3AF;padding:24px;')
        self._body_layout.addWidget(self._placeholder)
        self._body_layout.addStretch()

        scroll.setWidget(self._body)
        root.addWidget(scroll, stretch=1)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_chain(self, steps: list):
        """Update display with a list of CausalChainStep objects."""
        # Clear
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._steps = steps

        if not steps:
            self._placeholder = QLabel('No chain data received.')
            self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._placeholder.setStyleSheet('font-size:9px;color:#9CA3AF;padding:24px;')
            self._body_layout.addWidget(self._placeholder)
            self._body_layout.addStretch()
            return

        self._badge.setText(f'{len(steps)} steps')
        self._badge.setStyleSheet(
            'font-size:7px;font-weight:600;color:#1D4ED8;background:#DBEAFE;'
            'border-radius:3px;padding:1px 7px;border:1px solid #BFDBFE;')

        for i, step in enumerate(steps):
            node = _ChainNode(
                step_idx=i,
                label=step.label,
                mechanism=step.mechanism,
                magnitude=step.magnitude,
                is_last=(i == len(steps) - 1),
            )
            node.setMinimumHeight(72)
            self._body_layout.addWidget(node)

        self._body_layout.addStretch()

    def set_alert(self, alert: dict):
        """Show BSP alert banner inside this widget."""
        self._bsp_strip.set_alert(alert)
