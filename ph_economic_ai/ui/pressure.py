import math
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath


_BAND_CONFIG = [
    ('Stable',   '0 – 30',   '#EBF4FF', '#4A90E2'),
    ('Rising',   '31 – 60',  '#FFF8EE', '#E0A84A'),
    ('High',     '61 – 80',  '#FFF3EE', '#E07A4A'),
    ('Critical', '81 – 100', '#FFEFEE', '#E05040'),
]


class PressureGauge(QWidget):
    def __init__(self, size: int = 120, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._size = size
        self.setFixedSize(size, size)

    def set_value(self, value: float):
        self._value = float(max(0.0, min(100.0, value)))
        self.update()

    def _arc_color(self) -> str:
        if self._value <= 30:
            return '#4A90E2'
        elif self._value <= 60:
            return '#E0A84A'
        elif self._value <= 80:
            return '#E07A4A'
        return '#E05040'

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        m = 10
        side = self._size - 2 * m
        rect = QRectF(m + 6, m + 6, side - 12, side - 12)
        pw = max(8, self._size // 10)

        # Background track
        pen = QPen(QColor('#EAEAEA'), pw, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        # Value arc
        pen.setColor(QColor(self._arc_color()))
        painter.setPen(pen)
        span = int(-270.0 * self._value / 100.0 * 16)
        if span != 0:
            painter.drawArc(rect, 225 * 16, span)

        # Value text (large)
        painter.setPen(QPen(QColor('#111111')))
        f = QFont()
        f.setPointSize(max(10, self._size // 6))
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(
            QRectF(m, m + side * 0.15, side, side * 0.45),
            Qt.AlignmentFlag.AlignCenter,
            f'{int(self._value)}'
        )

        # Sub-label
        f2 = QFont()
        f2.setPointSize(max(7, self._size // 14))
        painter.setFont(f2)
        painter.setPen(QPen(QColor('#BBBBBB')))
        painter.drawText(
            QRectF(m, m + side * 0.55, side, side * 0.25),
            Qt.AlignmentFlag.AlignCenter,
            '/ 100'
        )


def _band_card(label: str, rng: str, bg: str, color: str, active: bool) -> QFrame:
    card = QFrame()
    border = f'border: 1.5px solid {color};' if active else 'border: 1px solid #EAEAEA;'
    card.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(1)
    lbl = QLabel(('⚠ ' if active else '') + label)
    lbl.setStyleSheet(f'font-size:9px; font-weight:700; color:{color};'
                      f'text-transform:uppercase; letter-spacing:0.5px;')
    rng_lbl = QLabel(rng)
    rng_lbl.setStyleSheet(f'font-size:9px; color:{color};')
    if active:
        now = QLabel('← NOW')
        now.setStyleSheet(f'font-size:8px; font-weight:700; color:{color};')
        layout.addWidget(now)
    layout.addWidget(lbl)
    layout.addWidget(rng_lbl)
    return card


class PressureGaugePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gauge = PressureGauge(size=160)
        self._band_cards: list[QFrame] = []
        self._history_labels: list[QLabel] = []
        self._history: list[float] = []
        self._current_band = 'Stable'
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        title = QLabel('Economic Pressure Index')
        title.setStyleSheet('font-size:20px; font-weight:700; color:#111111;')
        layout.addWidget(title)

        sub = QLabel('Weighted composite of oil price, USD/PHP rate, and fuel demand')
        sub.setStyleSheet('font-size:12px; color:#888888;')
        layout.addWidget(sub)

        # Gauge centered
        gauge_row = QHBoxLayout()
        gauge_row.addStretch()
        gauge_row.addWidget(self._gauge)
        gauge_row.addStretch()
        layout.addLayout(gauge_row)

        # Bands grid (2×2)
        bands_row = QHBoxLayout()
        bands_row.setSpacing(8)
        for label, rng, bg, color in _BAND_CONFIG:
            card = _band_card(label, rng, bg, color, active=(label == self._current_band))
            self._band_cards.append(card)
            bands_row.addWidget(card)
        layout.addLayout(bands_row)

        # History
        hist_title = QLabel('RECENT INDEX VALUES')
        hist_title.setStyleSheet(
            'font-size:10px; font-weight:700; color:#BBBBBB; letter-spacing:0.8px;'
        )
        layout.addWidget(hist_title)

        self._hist_layout = QVBoxLayout()
        self._hist_layout.setSpacing(4)
        layout.addLayout(self._hist_layout)

        layout.addStretch()

    def refresh(self, result: dict):
        index = result['pressure_index']
        band = result['pressure_band']
        self._gauge.set_value(index)
        self._current_band = band

        for i, (label, rng, bg, color) in enumerate(_BAND_CONFIG):
            active = label == band
            card = self._band_cards[i]
            border = f'border: 1.5px solid {color};' if active else 'border: 1px solid #EAEAEA;'
            card.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')

        # Update history
        self._history.append(index)
        self._history = self._history[-5:]
        while self._hist_layout.count():
            item = self._hist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for val in reversed(self._history):
            b = pressure_band_color(val)
            lbl = QLabel(f'  {val:.1f} / 100  —  {result["pressure_band"]}')
            lbl.setStyleSheet(f'font-size:12px; color:{b}; padding:4px 0;')
            self._hist_layout.addWidget(lbl)


def pressure_band_color(index: float) -> str:
    if index <= 30:
        return '#4A90E2'
    elif index <= 60:
        return '#E0A84A'
    elif index <= 80:
        return '#E07A4A'
    return '#E05040'
