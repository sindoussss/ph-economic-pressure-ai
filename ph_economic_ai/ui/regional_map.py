from __future__ import annotations
import math
from typing import Optional

from PyQt6.QtWidgets import QFrame, QToolTip
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QLinearGradient

from ph_economic_ai.engine.swarm import ALL_REGIONS

# Island group bands: isle_code → (y_start, y_end, label)
_ISLE_BANDS: dict[str, tuple[float, float, str]] = {
    'L': (0.00, 0.48, 'LUZON'),
    'V': (0.48, 0.65, 'VISAYAS'),
    'M': (0.65, 1.00, 'MINDANAO'),
}

_BAND_COLORS: dict[str, QColor] = {
    'L': QColor(248, 250, 253),
    'V': QColor(244, 247, 252),
    'M': QColor(248, 250, 253),
}

_DOT_R     = 11.0
_NCR_DOT_R = 14.0
_SAT_AT    = 3.0   # ₱/L at which color fully saturates


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(a.red()   + t * (b.red()   - a.red())),
        int(a.green() + t * (b.green() - a.green())),
        int(a.blue()  + t * (b.blue()  - a.blue())),
    )


_NEUTRAL = QColor('#EEF0F4')
_RED_SAT  = QColor('#DC2626')
_RED_BDR  = QColor('#B91C1C')
_GRN_SAT  = QColor('#059669')
_GRN_BDR  = QColor('#047857')
_GRAY_BDR = QColor('#CBD5E1')


def _dot_colors(val: Optional[float]) -> tuple[QColor, QColor]:
    """Returns (fill, border) for a region dot given its price-change estimate."""
    if val is None:
        return QColor('#E2E8F0'), _GRAY_BDR
    t = min(abs(val) / _SAT_AT, 1.0)
    if val > 0:
        return _lerp_color(_NEUTRAL, _RED_SAT, t), _lerp_color(_GRAY_BDR, _RED_BDR, t)
    return _lerp_color(_NEUTRAL, _GRN_SAT, t), _lerp_color(_GRAY_BDR, _GRN_BDR, t)


class RegionalMapWidget(QFrame):
    """Schematic dot-map of all 17 PH regions, color-coded by price-change estimate."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._estimates: dict[str, Optional[float]] = {}
        self._hover: Optional[str] = None
        self.setMouseTracking(True)
        self.setMinimumHeight(370)
        self.setStyleSheet(
            'QFrame{background:#F8F9FB;border-radius:10px;border:1px solid #EAECF0;}'
        )

    def set_estimates(self, estimates: dict[str, Optional[float]]) -> None:
        self._estimates = estimates
        self.update()

    # ── geometry helpers ──────────────────────────────────────────────────────

    def _bounds(self) -> tuple[float, float, float, float]:
        """left, top, right, bottom of the map drawing area."""
        r = self.rect()
        return (
            float(r.left()) + 28.0,
            float(r.top())  + 38.0,
            float(r.right()) - 108.0,
            float(r.bottom()) - 18.0,
        )

    def _center(self, reg: dict) -> QPointF:
        left, top, right, bottom = self._bounds()
        return QPointF(
            left + reg['nx'] * (right  - left),
            top  + reg['ny'] * (bottom - top),
        )

    # ── event handling ────────────────────────────────────────────────────────

    def mouseMoveEvent(self, ev):
        pos = QPointF(ev.position())
        old = self._hover
        self._hover = None
        for reg in ALL_REGIONS:
            c = self._center(reg)
            r = _NCR_DOT_R if reg['name'] == 'NCR' else _DOT_R
            if math.hypot(pos.x() - c.x(), pos.y() - c.y()) <= r + 3:
                self._hover = reg['name']
                est = self._estimates.get(reg['name'])
                if est is not None:
                    sign = '+' if est >= 0 else ''
                    tip = f"{reg['name']}\nEst. change: {sign}{est:.2f} ₱/L"
                else:
                    tip = f"{reg['name']}\nNo estimate yet"
                QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
                break
        if self._hover != old:
            self.update()
        super().mouseMoveEvent(ev)

    # ── painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        left, top, right, bottom = self._bounds()
        w, h = right - left, bottom - top

        # Title
        p.setFont(QFont('Segoe UI', 9, QFont.Weight.Bold))
        p.setPen(QColor('#374151'))
        p.drawText(
            QRectF(left, float(self.rect().top()) + 8.0, w, 22.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            'Regional Price Impact  —  All 17 Regions',
        )

        # Island group bands
        for isle, (y0, y1, label) in _ISLE_BANDS.items():
            by0 = top + y0 * h
            by1 = top + y1 * h
            p.setBrush(QBrush(_BAND_COLORS[isle]))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(QRectF(left, by0, w, by1 - by0))

            if y0 > 0:
                p.setPen(QPen(QColor('#E2E8F0'), 1.0, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(left, by0), QPointF(right, by0))

            # Rotated island group label in left margin
            p.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
            p.setPen(QColor('#C4CADB'))
            mid_y = top + (y0 + y1) / 2.0 * h
            p.save()
            p.translate(left - 14.0, mid_y)
            p.rotate(-90.0)
            p.drawText(QRectF(-24.0, -6.0, 48.0, 12.0), Qt.AlignmentFlag.AlignCenter, label)
            p.restore()

        # Region dots
        for reg in ALL_REGIONS:
            c = self._center(reg)
            est = self._estimates.get(reg['name'])
            fill, border = _dot_colors(est)
            r = _NCR_DOT_R if reg['name'] == 'NCR' else _DOT_R
            hovered = self._hover == reg['name']

            # Hover glow
            if hovered:
                glow = QColor(fill)
                glow.setAlpha(70)
                p.setBrush(QBrush(glow))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(c, r + 6.0, r + 6.0)

            # Dot body
            p.setBrush(QBrush(fill))
            p.setPen(QPen(border, 2.0 if hovered else 1.3))
            p.drawEllipse(c, r, r)

            # Code label inside dot (white when color is saturated, dark otherwise)
            p.setFont(QFont('Segoe UI', 5, QFont.Weight.Bold))
            sat = min(abs(est) / _SAT_AT, 1.0) if est is not None else 0.0
            text_col = QColor('#FFFFFF') if sat > 0.55 else QColor('#1E293B')
            p.setPen(text_col)
            p.drawText(
                QRectF(c.x() - r, c.y() - r, r * 2, r * 2),
                Qt.AlignmentFlag.AlignCenter,
                reg['code'],
            )

            # Show estimate below hovered dot
            if hovered and est is not None:
                p.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
                sign = '+' if est >= 0 else ''
                tag = f'{sign}{est:.2f}₱'
                p.setPen(QColor('#1E293B'))
                p.setBrush(QBrush(QColor(255, 255, 255, 210)))
                p.setPen(Qt.PenStyle.NoPen)
                tw = 40.0
                th = 13.0
                tx = c.x() - tw / 2.0
                ty = c.y() + r + 3.0
                p.drawRoundedRect(QRectF(tx, ty, tw, th), 3, 3)
                p.setPen(QColor('#111827'))
                p.drawText(QRectF(tx, ty, tw, th), Qt.AlignmentFlag.AlignCenter, tag)

        # Legend
        self._paint_legend(p, right + 14.0, top, bottom)

        p.end()

    def _paint_legend(self, p: QPainter, lx: float, top: float, bottom: float) -> None:
        h = bottom - top
        lh = min(180.0, h * 0.58)
        ly = top + (h - lh) * 0.32

        # Gradient bar
        grad = QLinearGradient(QPointF(lx, ly), QPointF(lx, ly + lh))
        grad.setColorAt(0.0, _RED_SAT)
        grad.setColorAt(0.5, _NEUTRAL)
        grad.setColorAt(1.0, _GRN_SAT)
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor('#E2E8F0'), 1.0))
        p.drawRoundedRect(QRectF(lx, ly, 11.0, lh), 3, 3)

        # Ticks
        p.setFont(QFont('Segoe UI', 7))
        p.setPen(QColor('#6B7280'))
        for label, frac in [(f'+₱{_SAT_AT:.0f}+', 0.0), ('   0', 0.5), (f'-₱{_SAT_AT:.0f}+', 1.0)]:
            ty = ly + frac * lh
            p.drawText(
                QRectF(lx + 15.0, ty - 6.0, 70.0, 12.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

        # Title
        p.setFont(QFont('Segoe UI', 7, QFont.Weight.Bold))
        p.setPen(QColor('#374151'))
        p.drawText(
            QRectF(lx - 4.0, ly - 20.0, 86.0, 14.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            '₱/L change',
        )

        # Footnote
        p.setFont(QFont('Segoe UI', 6))
        p.setPen(QColor('#9CA3AF'))
        p.drawText(
            QRectF(lx - 6.0, ly + lh + 12.0, 96.0, 28.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            'Derived via DOE\nfreight multipliers\nper region',
        )
