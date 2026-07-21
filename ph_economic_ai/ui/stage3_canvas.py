import re
import math
from collections import defaultdict

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGraphicsView, QGraphicsScene, QGraphicsObject,
    QSizePolicy, QScrollArea, QTextEdit, QGraphicsTextItem,
    QGraphicsPathItem, QGraphicsItem, QGraphicsEllipseItem,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QFontMetricsF,
)

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.engine.debate import Agent, DebateEngine, DebateThread, AgentResponse
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml

_AGENT_COLORS = ['#6366F1', '#0EA5E9', '#F59E0B', '#10B981',
                 '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6',
                 '#F97316', '#84CC16']

_SOURCE_DISPLAY: dict[str, str] = {
    'YahooFinanceCrude': 'Crude Oil',
    'YahooFinanceForex': 'PHP/USD',
    'ManilaBulletin':    'Manila Bul.',
    'BusinessWorld':     'BizWorld',
    'neda_2024_2026':    'NEDA',
}

# ── Canvas ────────────────────────────────────────────────────────────────────
_CANVAS_BG   = '#F0F2F7'
_DOT_COLOR   = '#C8CCDA'
_DOT_SPACING = 24

# ── Node radii ────────────────────────────────────────────────────────────────
_AR      = 22    # main agent circle radius
_AR_MINI = 15    # mini agent circle radius
_ER      = 30    # engine circle radius
_RR      = 12    # RAG source radius

# ── Ring layout ───────────────────────────────────────────────────────────────
_RING_A      = 240
_RING_A_MINI = 380
_RING_R      = 540


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _edge_pt(ax, ay, tx, ty, r):
    dx, dy = tx - ax, ty - ay
    dist = math.hypot(dx, dy)
    if dist < 1e-6: return ax, ay
    return ax + dx / dist * r, ay + dy / dist * r


def _qcurve(x1, y1, x2, y2, bend=0.0):
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    path = QPainterPath(QPointF(x1, y1))
    path.quadTo(QPointF(mx - dy * bend, my + dx * bend), QPointF(x2, y2))
    return path


# ── Ripple ────────────────────────────────────────────────────────────────────

class _RippleItem(QGraphicsObject):
    R_END = 60.0; DURATION = 0.5

    def __init__(self, r_start, color, parent=None):
        super().__init__(parent)
        self._r_start = r_start; self._color = color; self._t = 0.0
        self.setZValue(-1)

    def boundingRect(self):
        r = self.R_END + 4
        return QRectF(-r, -r, r * 2, r * 2)

    def paint(self, painter, option, widget=None):
        t = min(self._t / self.DURATION, 1.0)
        r = self._r_start + (self.R_END - self._r_start) * (1 - (1 - t) ** 2)
        alpha = int(200 * (1 - t))
        if alpha <= 0: return
        c = QColor(self._color); c.setAlpha(alpha)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(c, max(0.5, 2.0 * (1 - t))))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, 0), r, r)

    def advance(self, dt):
        self._t += dt; self.update()
        return self._t >= self.DURATION


class _TravelDot(QGraphicsObject):
    def __init__(self, path, color, parent=None):
        super().__init__(parent)
        self._path = path; self._color = color; self._prog = 0.0
        self.setPos(path.pointAtPercent(0.0)); self.setZValue(10)

    def boundingRect(self): return QRectF(-7, -7, 14, 14)

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fade = 1.0 - self._prog * 0.4
        c = QColor(self._color); c.setAlpha(int(70 * fade))
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(c))
        painter.drawEllipse(QPointF(0, 0), 6, 6)
        c2 = QColor(self._color); c2.setAlpha(int(220 * fade))
        painter.setBrush(QBrush(c2))
        painter.drawEllipse(QPointF(0, 0), 2.8, 2.8)

    def advance(self, dt):
        self._prog = min(1.0, self._prog + dt * 0.8)
        self.setPos(self._path.pointAtPercent(self._prog))
        self.update(); return self._prog >= 1.0


# ── Agent circle node ─────────────────────────────────────────────────────────

class _AgentCircle(QGraphicsObject):
    node_clicked     = pyqtSignal(str)
    hover_enter      = pyqtSignal(str)
    hover_leave      = pyqtSignal(str)
    position_changed = pyqtSignal(str, float, float)

    def __init__(self, agent, color, radius=_AR, parent=None):
        super().__init__(parent)
        self._agent = agent; self._color = QColor(color); self._colors = color
        self._r = radius; self._active = False; self._hovered = False
        self._statement = ''; self._phase = 0.0; self._scale_v = 1.0
        self._press_scene_pos = QPointF(0, 0)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        r = self._r + 18
        return QRectF(-r, -r, r * 2, r * 2)

    def shape(self):
        p = QPainterPath()
        p.addEllipse(QPointF(0, 0), self._r, self._r)
        return p

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        R = self._r

        # Active pulse ring
        if self._active:
            pulse = (math.sin(self._phase) + 1) / 2
            ring_r = R + 6 + pulse * 8
            rc = QColor(self._color); rc.setAlpha(int(90 * pulse))
            painter.setPen(QPen(rc, 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), ring_r, ring_r)
        elif self._hovered:
            hc = QColor(self._color); hc.setAlpha(40)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(hc))
            painter.drawEllipse(QPointF(0, 0), R + 8, R + 8)

        # Shadow
        sc = QColor(0, 0, 0, 30)
        painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(sc))
        painter.drawEllipse(QPointF(1, 3), R - 1, R - 1)

        # Filled circle
        painter.setBrush(QBrush(self._color))
        if self._active:
            pulse = (math.sin(self._phase) + 1) / 2
            painter.setPen(QPen(QColor('#FFFFFF'), 2.0 + pulse))
        else:
            painter.setPen(QPen(QColor('#FFFFFF'), 2.0))
        painter.drawEllipse(QPointF(0, 0), R, R)

        # Mini dot indicator
        if self._agent.is_mini:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
            painter.drawEllipse(QPointF(0, 0), R * 0.35, R * 0.35)

    def anim_update(self, dt):
        target = 1.04 if self._hovered else 1.0
        self._scale_v += (target - self._scale_v) * 0.22
        self.setScale(self._scale_v)
        if self._active:
            self._phase = (self._phase + dt * 5.0) % (2 * math.pi)
            self.update()
        elif self._hovered:
            self.update()

    def hoverEnterEvent(self, event):
        self._hovered = True
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.hover_enter.emit(self._agent.name); event.accept()

    def hoverLeaveEvent(self, event):
        self._hovered = False
        self.unsetCursor()
        self.hover_leave.emit(self._agent.name); event.accept()

    def mousePressEvent(self, event):
        self._press_scene_pos = self.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if (self.pos() - self._press_scene_pos).manhattanLength() < 6:
            self.node_clicked.emit(self._agent.name)
        super().mouseReleaseEvent(event); event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed.emit(self._agent.name, value.x(), value.y())
        return super().itemChange(change, value)

    def set_active(self):   self._active = True;  self.update()
    def set_done(self, statement, estimate=None):
        self._statement = statement; self._active = False; self.update()

    @property
    def color(self):     return self._colors
    @property
    def statement(self): return self._statement


# ── Engine circle ─────────────────────────────────────────────────────────────

class _EngineCircle(QGraphicsObject):
    clicked = pyqtSignal(); position_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._price = '₱---'; self._live = False
        self._hb_phase = 0.0; self._hovered = False; self._scale_v = 1.0
        self._press_scene_pos = QPointF(0, 0)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        r = _ER + 20; return QRectF(-r, -r, r * 2, r * 2)

    def shape(self):
        p = QPainterPath(); p.addEllipse(QPointF(0, 0), _ER, _ER); return p

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Glow when live
        if self._live:
            hb = (math.sin(self._hb_phase) + 1) / 2
            for expand, af in [(16, 0.15), (10, 0.30), (5, 0.50)]:
                gc = QColor(34, 197, 94, int(hb * 28 * af))
                painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(gc))
                painter.drawEllipse(QPointF(0, 0), _ER + expand, _ER + expand)

        # Shadow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 35)))
        painter.drawEllipse(QPointF(1, 4), _ER - 1, _ER - 1)

        # Dark circle
        painter.setBrush(QBrush(QColor('#0F172A')))
        border_c = QColor('#22C55E') if self._live else QColor('#334155')
        painter.setPen(QPen(border_c, 2.0))
        painter.drawEllipse(QPointF(0, 0), _ER, _ER)

        # Text
        def _t(text, px, col, rect):
            f = QFont(); f.setPixelSize(px)
            painter.setFont(f); painter.setPen(QPen(QColor(col)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        _t('HIST', 6, '#475569', QRectF(-_ER, -_ER + 6, _ER * 2, 12))
        _t(self._price, 14, '#22C55E' if self._live else '#94A3B8',
           QRectF(-_ER, -8, _ER * 2, 18))
        _t('GBM', 6, '#475569', QRectF(-_ER, _ER - 18, _ER * 2, 12))

    def anim_update(self, dt):
        self._hb_phase = (self._hb_phase + dt * 1.6) % (2 * math.pi)
        target = 1.06 if self._hovered else 1.0
        self._scale_v += (target - self._scale_v) * 0.18
        self.setScale(self._scale_v); self.update()

    def hoverEnterEvent(self, event):
        self._hovered = True; self.setCursor(Qt.CursorShape.OpenHandCursor); event.accept()

    def hoverLeaveEvent(self, event):
        self._hovered = False; self.unsetCursor(); event.accept()

    def mousePressEvent(self, event):
        self._press_scene_pos = self.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor); super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if (self.pos() - self._press_scene_pos).manhattanLength() < 6: self.clicked.emit()
        super().mouseReleaseEvent(event); event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed.emit(value.x(), value.y())
        return super().itemChange(change, value)

    def set_price(self, price):
        self._price = f'₱{price:.0f}'; self._live = True; self.update()


# ── RAG circle ────────────────────────────────────────────────────────────────

class _RagCircle(QGraphicsObject):
    clicked = pyqtSignal(str, int); position_changed = pyqtSignal(str, float, float)

    def __init__(self, source, chunk_count, color='#94A3B8', parent=None):
        super().__init__(parent)
        self._source = source; self._count = chunk_count
        self._color = QColor(color); self._has = chunk_count > 0
        self._hovered = False; self._press_scene_pos = QPointF(0, 0)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

    def boundingRect(self):
        r = _RR + 8; return QRectF(-r, -r, r * 2, r * 2)

    def shape(self):
        p = QPainterPath(); p.addEllipse(QPointF(0, 0), _RR, _RR); return p

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = _RR + (2 if self._hovered else 0)
        if self._hovered:
            hc = QColor(self._color); hc.setAlpha(25)
            painter.setPen(Qt.PenStyle.NoPen); painter.setBrush(QBrush(hc))
            painter.drawEllipse(QPointF(0, 0), r + 6, r + 6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 18)))
        painter.drawEllipse(QPointF(1, 2), r, r)
        painter.setBrush(QBrush(QColor('#FFFFFF')))
        rc = QColor(self._color); rc.setAlpha(210 if self._has else 80)
        painter.setPen(QPen(rc, 2.0 if self._has else 1.2))
        painter.drawEllipse(QPointF(0, 0), r, r)
        if self._has:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(self._color))
            painter.drawEllipse(QPointF(0, 0), r * 0.4, r * 0.4)

    def hoverEnterEvent(self, event):
        self._hovered = True; self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.update(); event.accept()

    def hoverLeaveEvent(self, event):
        self._hovered = False; self.unsetCursor(); self.update(); event.accept()

    def mousePressEvent(self, event):
        self._press_scene_pos = self.pos()
        self.setCursor(Qt.CursorShape.ClosedHandCursor); super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        if (self.pos() - self._press_scene_pos).manhattanLength() < 6:
            self.clicked.emit(self._source, self._count)
        super().mouseReleaseEvent(event); event.accept()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.position_changed.emit(self._source, value.x(), value.y())
        return super().itemChange(change, value)


# ── Canvas view ───────────────────────────────────────────────────────────────

class _DotGridView(QGraphicsView):
    resized = pyqtSignal()

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setStyleSheet(f'border:none;background:{_CANVAS_BG};')
        self.setBackgroundBrush(QBrush(QColor(_CANVAS_BG)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def resizeEvent(self, event):
        super().resizeEvent(event); self.resized.emit()

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor); event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fitInView(self.scene().itemsBoundingRect().adjusted(-60, -60, 60, 60),
                       Qt.AspectRatioMode.KeepAspectRatio); event.accept()

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(_DOT_COLOR)))
        sp = _DOT_SPACING
        x = int(rect.left() / sp) * sp
        while x < rect.right():
            y = int(rect.top() / sp) * sp
            while y < rect.bottom():
                painter.drawEllipse(QPointF(x, y), 0.9, 0.9)
                y += sp
            x += sp


# ── Node Details Card (pinned beside right sidebar, Mirofish style) ───────────

class _NodeDetailCard(QFrame):
    """Appears beside the right sidebar when a node is clicked."""
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        # Only outer border, no internal borders at all
        self.setStyleSheet(
            'QFrame#nodeCard{background:#FFFFFF;border:1px solid #DDE1EC;border-radius:14px;}'
            'QFrame#nodeCard QWidget{background:transparent;border:none;}'
            'QFrame#nodeCard QLabel{background:transparent;border:none;}')
        self.setObjectName('nodeCard')
        # Scroll-control state
        self._user_scrolled  = False   # True when user has scrolled up during streaming
        self._displayed_text = ''      # last text pushed into _msg_area (for delta appends)
        self._build()
        self.hide()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setStyleSheet('background:transparent;border:none;')
        hh = QHBoxLayout(hdr)
        hh.setContentsMargins(18, 14, 12, 10)
        hh.setSpacing(8)

        self._title = QLabel('Node Details')
        self._title.setStyleSheet(
            'font-size:14px;font-weight:700;color:#0F172A;'
            'background:transparent;border:none;')

        self._badge = QLabel('AGENT')
        self._badge.setFixedHeight(24)
        self._badge.setStyleSheet(
            'font-size:10px;font-weight:700;color:#FFFFFF;'
            'background:#6366F1;border-radius:6px;padding:0 10px;border:none;')

        close_btn = QPushButton('✕')
        close_btn.setFixedSize(24, 24)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            'QPushButton{border:none;border-radius:12px;font-size:11px;'
            'color:#94A3B8;background:transparent;}'
            'QPushButton:hover{background:#F1F5F9;color:#475569;}')
        close_btn.clicked.connect(self.closed.emit)

        hh.addWidget(self._title)
        hh.addStretch()
        hh.addWidget(self._badge)
        hh.addWidget(close_btn)
        lay.addWidget(hdr)

        # ── Compact meta row (name · model · status in one line) ──────────
        meta_w = QWidget()
        meta_w.setStyleSheet('background:transparent;border:none;')
        meta_h = QHBoxLayout(meta_w)
        meta_h.setContentsMargins(18, 2, 18, 10)
        meta_h.setSpacing(12)

        self._meta_name = QLabel('—')
        self._meta_name.setStyleSheet(
            'font-size:11px;font-weight:700;color:#1E293B;'
            'background:transparent;border:none;')

        self._meta_model = QLabel('—')
        self._meta_model.setStyleSheet(
            'font-size:10px;color:#94A3B8;background:transparent;border:none;')

        self._meta_status = QLabel('Idle')
        self._meta_status.setStyleSheet(
            'font-size:10px;font-weight:600;color:#94A3B8;'
            'background:transparent;border:none;')

        meta_h.addWidget(self._meta_name)
        meta_h.addWidget(self._meta_model)
        meta_h.addStretch()
        meta_h.addWidget(self._meta_status)
        lay.addWidget(meta_w)

        # ── Extra info (round + price) in tiny chips ───────────────────────
        chips_w = QWidget()
        chips_w.setStyleSheet('background:transparent;border:none;')
        chips_h = QHBoxLayout(chips_w)
        chips_h.setContentsMargins(18, 0, 18, 10)
        chips_h.setSpacing(6)

        self._chip_round = QLabel('—')
        self._chip_price = QLabel('—')
        for chip in [self._chip_round, self._chip_price]:
            chip.setStyleSheet(
                'font-size:9px;color:#64748B;background:#F1F5F9;'
                'border-radius:4px;padding:2px 8px;border:none;')

        chips_h.addWidget(self._chip_round)
        chips_h.addWidget(self._chip_price)
        chips_h.addStretch()
        lay.addWidget(chips_w)

        # ── AI Message / response area — the main content ─────────────────
        # Subtle bg to distinguish from white card
        msg_w = QWidget()
        msg_w.setStyleSheet(
            'QWidget{background:#F8F9FC;border-bottom-left-radius:14px;'
            'border-bottom-right-radius:14px;border:none;}')
        mv = QVBoxLayout(msg_w)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(0)

        # tiny label inside the bg
        msg_lbl = QLabel('  Response')
        msg_lbl.setFixedHeight(26)
        msg_lbl.setStyleSheet(
            'font-size:9px;font-weight:700;letter-spacing:0.8px;'
            'color:#94A3B8;background:transparent;border:none;padding-left:18px;'
            'padding-top:8px;')
        mv.addWidget(msg_lbl)

        self._msg_area = QTextEdit()
        self._msg_area.setReadOnly(True)
        self._msg_area.setPlaceholderText('Click a node to see its response here…')
        self._msg_area.setMinimumHeight(220)
        self._msg_area.setMaximumHeight(400)
        self._msg_area.setStyleSheet(
            'QTextEdit{'
            '  background:transparent;'
            '  border:none;'
            '  font-size:11px;'
            '  color:#334155;'
            '  line-height:160%;'
            '  padding:0 18px 14px 18px;'
            '}'
            'QScrollBar:vertical{'
            '  width:4px;'
            '  background:#FFFFFF;'
            '  margin:0;'
            '  border:none;'
            '}'
            'QScrollBar::handle:vertical{'
            '  background:#D1D5DB;'
            '  border-radius:2px;'
            '  min-height:24px;'
            '  border:none;'
            '}'
            'QScrollBar::handle:vertical:hover{'
            '  background:#94A3B8;'
            '}'
            'QScrollBar::add-line:vertical,'
            'QScrollBar::sub-line:vertical{'
            '  height:0;width:0;background:none;border:none;'
            '}'
            'QScrollBar::add-page:vertical,'
            'QScrollBar::sub-page:vertical{'
            '  background:#FFFFFF;border:none;'
            '}')
        mv.addWidget(self._msg_area)
        lay.addWidget(msg_w, stretch=1)

        # Wire up smart auto-scroll: stop forcing the view to the bottom once
        # the user drags the scrollbar up; re-enable when they reach the bottom.
        sb = self._msg_area.verticalScrollBar()
        sb.sliderMoved.connect(self._on_slider_moved)
        sb.valueChanged.connect(self._on_scroll_value_changed)

    # ── Scroll intent tracking ────────────────────────────────────────────────

    def _on_slider_moved(self, value):
        """User dragged the scrollbar — stop forcing auto-scroll."""
        sb = self._msg_area.verticalScrollBar()
        self._user_scrolled = value < sb.maximum()

    def _on_scroll_value_changed(self, value):
        """Re-enable auto-scroll once the user reaches the bottom again."""
        sb = self._msg_area.verticalScrollBar()
        if value >= sb.maximum():
            self._user_scrolled = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_badge(self, text, color):
        self._badge.setText(text)
        self._badge.setStyleSheet(
            f'font-size:10px;font-weight:700;color:#FFFFFF;'
            f'background:{color};border-radius:6px;padding:0 10px;border:none;')

    def _set_meta(self, name, model, status, status_color, round_num, price):
        self._meta_name.setText(name)
        self._meta_model.setText(model)
        self._meta_status.setText(status)
        self._meta_status.setStyleSheet(
            f'font-size:10px;font-weight:600;color:{status_color};'
            'background:transparent;border:none;')
        self._chip_round.setText(f'Round {round_num}' if round_num else '—')
        self._chip_price.setText(f'₱{price:.2f}/L' if price is not None else '—')

    def _set_msg(self, text):
        """Full replace — used for non-streaming updates (initial state, done, rag, engine)."""
        self._displayed_text = text
        self._msg_area.setPlainText(text)
        if not self._user_scrolled:
            sb = self._msg_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _append_msg(self, full_text):
        """Delta-append — used while streaming so the viewport is never touched.

        Qt's insertPlainText() moves only the cursor, leaving the scrollbar
        exactly where the user left it.  We only nudge it to the bottom when
        the user hasn't scrolled away.
        """
        new_text = full_text or '(thinking…)'
        current   = self._displayed_text

        if new_text.startswith(current):
            delta = new_text[len(current):]
            if delta:
                cursor = self._msg_area.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self._msg_area.setTextCursor(cursor)
                self._msg_area.insertPlainText(delta)
                self._displayed_text = new_text
                if not self._user_scrolled:
                    sb = self._msg_area.verticalScrollBar()
                    sb.setValue(sb.maximum())
        else:
            # Text changed non-incrementally (thinking → visible transition etc.)
            self._set_msg(new_text)

    # ── Public API ────────────────────────────────────────────────────────────

    def show_agent(self, name, model, color, round_num, price, status):
        self._user_scrolled  = False          # fresh node → resume auto-scroll
        self._displayed_text = ''
        self._set_badge('AGENT', color)
        status_color = '#6366F1' if 'Respond' in status else '#94A3B8'
        self._set_meta(name, model, status, status_color, round_num, price)
        self._set_msg('(waiting for response…)')

    def update_live_text(self, text):
        self._meta_status.setText('● Responding')
        self._meta_status.setStyleSheet(
            'font-size:10px;font-weight:700;color:#6366F1;'
            'background:transparent;border:none;')
        self._append_msg(text or '(thinking…)')

    def update_done(self, statement, price, round_num):
        self._meta_status.setText('Done')
        self._meta_status.setStyleSheet(
            'font-size:10px;font-weight:600;color:#10B981;'
            'background:transparent;border:none;')
        self._chip_round.setText(f'Round {round_num}')
        if price is not None:
            self._chip_price.setText(f'₱{price:.2f}/L')
        clean = re.sub(r'<think>.*?</think>', '', statement or '', flags=re.DOTALL)
        clean = re.sub(r'\*+', '', clean).strip()
        self._set_msg(clean or '—')

    def show_rag(self, source, disp, count, agents):
        self._set_badge('DATA', '#64748B')
        self._set_meta(disp, source, 'Active', '#10B981', None, None)
        self._chip_round.setText(f'{count} chunks')
        self._set_msg('Used by:\n' + '\n'.join(f'  · {a}' for a in agents))

    def show_engine(self, price_str, cv_rmse, scenario):
        self._set_badge('ENGINE', '#22C55E')
        status = 'Live' if '---' not in price_str else 'Idle'
        self._set_meta('HISTGBM Engine', 'HistGradientBoostingRegressor',
                       status, '#22C55E', None, None)
        self._chip_price.setText(price_str)
        self._chip_round.setText(f'CV-RMSE: {cv_rmse:.4f}')
        self._set_msg('\n'.join(f'{k}: {v}' for k, v in scenario.items()) or '—')


# ── Right sidebar ─────────────────────────────────────────────────────────────

class _EntityLegend(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(150)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName('entityLegend')
        # Use a simple stylesheet — no border-radius so children don't get clipped
        self.setStyleSheet(
            'QFrame#entityLegend{'
            '  background:#FFFFFF;'
            '  border:1px solid #DDE1EC;'
            '  border-radius:10px;'
            '}')
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(14, 12, 16, 14)
        self._lay.setSpacing(6)
        self._built = False

    def rebuild(self, main_agents: list, mini_agents: list,
                rag_sources: list, colors: dict):
        # Clear
        while self._lay.count():
            it = self._lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()

        hdr = QLabel('ENTITY TYPES')
        hdr.setStyleSheet(
            'font-size:7px;font-weight:700;color:#94A3B8;'
            'letter-spacing:1.4px;background:transparent;border:none;')
        self._lay.addWidget(hdr)
        self._lay.addSpacing(5)

        def _row(label, color, dot_style='circle'):
            rw_widget = QWidget()
            rw_widget.setStyleSheet('background:transparent;border:none;')
            rw = QHBoxLayout(rw_widget)
            rw.setSpacing(7); rw.setContentsMargins(0, 1, 0, 1)
            dot = QLabel()
            dot.setFixedSize(12, 12)
            if dot_style == 'ring':
                dot.setStyleSheet(
                    f'border:2.5px solid {color};border-radius:6px;background:transparent;border:none;')
            else:
                dot.setStyleSheet(
                    f'background:{color};border-radius:6px;border:none;')
            nm = QLabel(label)
            nm.setStyleSheet(
                'font-size:10px;color:#334155;background:transparent;border:none;')
            rw.addWidget(dot); rw.addWidget(nm); rw.addStretch()
            self._lay.addWidget(rw_widget)

        if main_agents:
            _row('Agent', _AGENT_COLORS[0])
        if mini_agents:
            _row('Mini Agent', _AGENT_COLORS[1], 'ring')
        _row('Engine', '#22C55E')
        if rag_sources:
            _row('Data Source', '#94A3B8', 'ring')

        self.adjustSize()
        self._built = True


# ── Right sidebar ────────────────────────────────────────────────────────────

def _make_sidebar() -> tuple:
    """Build right sidebar. Returns (panel, refs_dict)."""
    SIDEBAR_W = 310
    panel = QFrame()
    panel.setFixedWidth(SIDEBAR_W)
    panel.setStyleSheet('QFrame{background:#FFFFFF;border-left:1px solid #E8ECF4;}')
    lay = QVBoxLayout(panel); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)

    # ── Top block: title + status + scenario chips, all in one cohesive unit ──
    top_block = QWidget()
    top_block.setStyleSheet('background:#FFFFFF;')
    tb = QVBoxLayout(top_block); tb.setContentsMargins(16, 14, 14, 0); tb.setSpacing(6)

    # Row 1: title left, stop button right
    row1 = QHBoxLayout(); row1.setSpacing(8)
    title_col = QVBoxLayout(); title_col.setSpacing(1)
    title = QLabel('PH Economic Simulation')
    title.setStyleSheet('font-size:13px;font-weight:700;color:#0F172A;background:transparent;')
    subtitle = QLabel('Multi-agent debate engine')
    subtitle.setStyleSheet('font-size:9px;color:#94A3B8;background:transparent;')
    title_col.addWidget(title); title_col.addWidget(subtitle)
    stop_btn = QPushButton('■ Stop')
    stop_btn.setEnabled(False); stop_btn.setFixedSize(60, 26)
    stop_btn.setStyleSheet(
        'QPushButton{border:1.5px solid #E2E8F0;border-radius:7px;'
        'font-size:9px;font-weight:600;color:#64748B;background:#FFFFFF;}'
        'QPushButton:hover{background:#FEF2F2;border-color:#FCA5A5;color:#EF4444;}'
        'QPushButton:disabled{color:#CBD5E1;border-color:#F1F5F9;}')
    row1.addLayout(title_col, stretch=1); row1.addWidget(stop_btn, alignment=Qt.AlignmentFlag.AlignTop)
    tb.addLayout(row1)

    # Row 2: live dot + status text + round badge, all inline
    row2 = QHBoxLayout(); row2.setSpacing(5)
    live_dot = QLabel('●')
    live_dot.setStyleSheet('font-size:7px;color:#D1D5DB;background:transparent;')
    status_lbl = QLabel('No simulation running')
    status_lbl.setStyleSheet('font-size:9px;color:#94A3B8;background:transparent;')
    round_lbl = QLabel('')
    round_lbl.setStyleSheet(
        'font-size:8px;font-weight:700;color:#6366F1;background:#EEF2FF;'
        'border-radius:4px;padding:1px 7px;border:none;')
    row2.addWidget(live_dot); row2.addWidget(status_lbl, stretch=1); row2.addWidget(round_lbl)
    tb.addLayout(row2)

    # Row 3: scenario chips (tight, compact)
    row3 = QHBoxLayout(); row3.setSpacing(4); row3.setContentsMargins(0, 2, 0, 12)
    chip_refs = []
    def _chip(icon):
        w = QLabel(icon)
        w.setStyleSheet(
            'font-size:8px;color:#475569;background:#F1F5F9;'
            'border-radius:4px;padding:2px 6px;border:none;')
        row3.addWidget(w); chip_refs.append(w)
    _chip('Oil —'); _chip('USD —'); _chip('BSP —'); _chip('Dem —')
    row3.addStretch()
    tb.addLayout(row3)

    lay.addWidget(top_block)

    # Thin separator before the list sections
    sep0 = QFrame(); sep0.setFrameShape(QFrame.Shape.HLine)
    sep0.setStyleSheet('border:none;border-top:1px solid #E8ECF4;background:transparent;')
    sep0.setFixedHeight(1)
    lay.addWidget(sep0)

    # ── AGENTS section — accent: indigo ──────────────────────────────────────
    ag_hdr = QWidget(); ag_hdr.setFixedHeight(34)
    ag_hdr.setStyleSheet('background:#FFFFFF;')
    agh = QHBoxLayout(ag_hdr); agh.setContentsMargins(0, 0, 14, 0); agh.setSpacing(0)

    # Coloured left-bar accent
    ag_bar = QWidget(); ag_bar.setFixedSize(3, 34)
    ag_bar.setStyleSheet('background:#6366F1;border-radius:0;')
    ag_title = QLabel('AGENTS')
    ag_title.setStyleSheet(
        'font-size:8px;font-weight:700;letter-spacing:1.4px;'
        'color:#6366F1;background:transparent;padding-left:11px;')
    ag_count = QLabel('')
    ag_count.setStyleSheet(
        'font-size:8px;font-weight:600;color:#6366F1;background:#EEF2FF;'
        'border-radius:4px;padding:1px 7px;border:none;')
    agh.addWidget(ag_bar); agh.addWidget(ag_title, stretch=1); agh.addWidget(ag_count)
    lay.addWidget(ag_hdr)

    agent_scroll = QScrollArea(); agent_scroll.setWidgetResizable(True)
    agent_scroll.setFrameShape(QFrame.Shape.NoFrame)
    agent_scroll.setFixedHeight(220)
    agent_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    agent_scroll.setStyleSheet(
        'QScrollArea{border:none;background:#FFFFFF;}'
        'QScrollBar:vertical{width:4px;background:#FFFFFF;margin:0;border:none;}'
        'QScrollBar::handle:vertical{background:#D1D5DB;border-radius:2px;min-height:32px;border:none;}'
        'QScrollBar::handle:vertical:hover{background:#94A3B8;}'
        'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;border:none;background:none;}'
        'QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:#FFFFFF;border:none;}')
    agent_inner = QWidget(); agent_inner.setStyleSheet('background:#FFFFFF;')
    agent_vbox = QVBoxLayout(agent_inner)
    agent_vbox.setContentsMargins(10, 4, 10, 6); agent_vbox.setSpacing(2)
    agent_vbox.addStretch()
    agent_scroll.setWidget(agent_inner)
    lay.addWidget(agent_scroll)

    # Separator with label baked in
    sep1 = QFrame(); sep1.setFrameShape(QFrame.Shape.HLine)
    sep1.setStyleSheet('border:none;border-top:1px solid #E8ECF4;background:transparent;')
    sep1.setFixedHeight(1)
    lay.addWidget(sep1)

    # ── DEBATE LOG section — accent: sky blue ────────────────────────────────
    log_hdr_w = QWidget(); log_hdr_w.setFixedHeight(34)
    log_hdr_w.setStyleSheet('background:#FFFFFF;')
    lhh = QHBoxLayout(log_hdr_w); lhh.setContentsMargins(0, 0, 14, 0); lhh.setSpacing(0)

    log_bar = QWidget(); log_bar.setFixedSize(3, 34)
    log_bar.setStyleSheet('background:#0EA5E9;border-radius:0;')
    log_title = QLabel('DEBATE LOG')
    log_title.setStyleSheet(
        'font-size:8px;font-weight:700;letter-spacing:1.4px;'
        'color:#0EA5E9;background:transparent;padding-left:11px;')
    log_count_lbl = QLabel('')
    log_count_lbl.setStyleSheet(
        'font-size:8px;font-weight:600;color:#0EA5E9;background:#E0F2FE;'
        'border-radius:4px;padding:1px 7px;border:none;')
    lhh.addWidget(log_bar); lhh.addWidget(log_title, stretch=1); lhh.addWidget(log_count_lbl)
    lay.addWidget(log_hdr_w)

    log_scroll = QScrollArea(); log_scroll.setWidgetResizable(True)
    log_scroll.setFrameShape(QFrame.Shape.NoFrame)
    log_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    log_scroll.setStyleSheet(
        'QScrollArea{border:none;background:#FFFFFF;}'
        'QScrollBar:vertical{width:4px;background:#FFFFFF;margin:0;border:none;}'
        'QScrollBar::handle:vertical{background:#D1D5DB;border-radius:2px;min-height:32px;border:none;}'
        'QScrollBar::handle:vertical:hover{background:#94A3B8;}'
        'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;border:none;background:none;}'
        'QScrollBar::add-page:vertical,QScrollBar::sub-page:vertical{background:#FFFFFF;border:none;}')
    log_inner = QWidget(); log_inner.setStyleSheet('background:#FFFFFF;')
    log_vbox = QVBoxLayout(log_inner)
    log_vbox.setContentsMargins(10, 4, 10, 6); log_vbox.setSpacing(2)
    log_vbox.addStretch()
    log_scroll.setWidget(log_inner)
    lay.addWidget(log_scroll, stretch=1)

    # ── Emerging output ───────────────────────────────────────────────────────
    sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
    sep2.setStyleSheet('border:none;border-top:1px solid #E8ECF4;background:transparent;')
    sep2.setFixedHeight(1)
    lay.addWidget(sep2)

    price_w = QWidget()
    price_w.setStyleSheet('background:#FFFFFF;')
    pv = QVBoxLayout(price_w); pv.setContentsMargins(16, 10, 16, 14); pv.setSpacing(1)
    out_lbl = QLabel('EMERGING OUTPUT')
    out_lbl.setStyleSheet(
        'font-size:7px;font-weight:700;color:#94A3B8;letter-spacing:1.4px;background:transparent;')
    price_lbl = QLabel('₱---.-- /L')
    price_lbl.setStyleSheet(
        'font-size:28px;font-weight:700;color:#0F172A;background:transparent;')
    pv.addWidget(out_lbl); pv.addWidget(price_lbl)
    lay.addWidget(price_w)

    refs = {
        'stop_btn': stop_btn, 'chip_refs': chip_refs,
        'live_dot': live_dot, 'status_lbl': status_lbl, 'round_lbl': round_lbl,
        'ag_count': ag_count, 'agent_vbox': agent_vbox,
        'log_vbox': log_vbox, 'log_scroll': log_scroll, 'log_count_lbl': log_count_lbl,
        'price_lbl': price_lbl,
    }
    return panel, refs


# ── Agent status row (sidebar agent list) ────────────────────────────────────

class _AgentStatusRow(QWidget):
    """One row in the sidebar Agents section showing live status."""
    clicked = pyqtSignal(str)

    def __init__(self, name: str, color: str, is_mini: bool, parent=None):
        super().__init__(parent)
        self._name = name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        h = QHBoxLayout(self); h.setContentsMargins(10, 8, 10, 8); h.setSpacing(10)

        # Colored dot
        self._dot = QLabel()
        self._dot.setFixedSize(12, 12)
        dot_style = (f'border:2.5px solid {color};border-radius:6px;background:transparent;'
                     if is_mini else
                     f'background:{color};border-radius:6px;border:none;')
        self._dot.setStyleSheet(dot_style)

        # Name
        nm = QLabel(name); nm.setStyleSheet(
            'font-size:11px;font-weight:600;color:#1E293B;background:transparent;')

        # Status badge
        self._status = QLabel('Idle')
        self._status.setStyleSheet(
            'font-size:10px;color:#94A3B8;background:transparent;font-weight:500;')

        h.addWidget(self._dot); h.addWidget(nm, stretch=1); h.addWidget(self._status)
        self.setStyleSheet(
            'QWidget{background:#F8FAFC;border-radius:7px;}'
            'QWidget:hover{background:#EEF2FF;}')

    def set_responding(self):
        self._status.setStyleSheet(
            'font-size:10px;color:#6366F1;background:transparent;font-weight:700;')
        self._status.setText('● Responding')

    def set_done(self, price=None):
        txt = f'₱{price:.2f}/L' if price is not None else 'Done'
        self._status.setStyleSheet(
            'font-size:10px;color:#10B981;background:transparent;font-weight:700;')
        self._status.setText(txt)

    def set_idle(self):
        self._status.setStyleSheet(
            'font-size:10px;color:#CBD5E1;background:transparent;font-weight:500;')
        self._status.setText('Idle')

    def mousePressEvent(self, event): self.clicked.emit(self._name)


# ── Log row ───────────────────────────────────────────────────────────────────

class _LogRow(QWidget):
    clicked = pyqtSignal(str)

    def __init__(self, response, color, parent=None):
        super().__init__(parent)
        self._name = response.agent_name
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(0)
        card = QWidget()
        card.setStyleSheet(
            'QWidget{background:#FFFFFF;border-radius:7px;}'
            'QWidget:hover{background:#F8FAFC;}')
        ch = QHBoxLayout(card); ch.setContentsMargins(0, 0, 0, 0); ch.setSpacing(0)
        stripe = QWidget(); stripe.setFixedWidth(3)
        stripe.setStyleSheet(
            f'background:{color};border-top-left-radius:7px;border-bottom-left-radius:7px;')
        txt = QWidget(); txt.setStyleSheet('background:transparent;')
        tv = QVBoxLayout(txt); tv.setContentsMargins(10, 8, 10, 8); tv.setSpacing(3)
        rh = QHBoxLayout(); rh.setSpacing(6)
        rnd = QLabel(f'R{response.round_num}')
        rnd.setStyleSheet('font-size:9px;color:#94A3B8;background:transparent;')
        nm = QLabel(response.agent_name)
        nm.setStyleSheet('font-size:11px;font-weight:700;color:#1E293B;background:transparent;')
        est_txt = f'+₱{response.price_estimate:.2f}' if response.price_estimate is not None else '—'
        est = QLabel(est_txt)
        est.setStyleSheet(f'font-size:11px;font-weight:700;color:{color};background:transparent;')
        rh.addWidget(rnd); rh.addWidget(nm); rh.addStretch(); rh.addWidget(est)
        tv.addLayout(rh)
        if response.statement:
            raw = re.sub(r'\*+', '', response.statement.replace('\n', ' ')).strip()
            pl = QLabel(raw[:120] + ('…' if len(raw) > 120 else ''))
            pl.setWordWrap(True)
            pl.setStyleSheet('font-size:9px;color:#64748B;background:transparent;')
            tv.addWidget(pl)
        ch.addWidget(stripe); ch.addWidget(txt, stretch=1)
        outer.addWidget(card)

    def mousePressEvent(self, event): self.clicked.emit(self._name)


# ── Main panel ────────────────────────────────────────────────────────────────

class Stage3CanvasPanel(QWidget):
    simulation_complete = pyqtSignal(object)

    def __init__(self, rag, agents, regressor, df, cv_rmse, parent=None):
        super().__init__(parent)
        self._rag = rag; self._agents = list(agents)
        self._regressor = regressor; self._df = df; self._cv_rmse = cv_rmse
        self._thread = None; self._engine = None; self._scenario = {}

        self._circles:    dict[str, _AgentCircle]     = {}
        self._name_lbls:  dict[str, QGraphicsTextItem] = {}
        self._est_lbls:   dict[str, QGraphicsTextItem] = {}
        self._apos:       dict[str, tuple]             = {}
        self._lines:      dict[str, QGraphicsPathItem] = {}
        self._dot_paths:  dict[str, QPainterPath]      = {}
        self._eng_item:   _EngineCircle | None         = None
        self._engine_pos: tuple                        = (440, 330)
        self._agent_rag_lines: dict[str, list]         = {}
        self._rag_line_map:    dict[str, list]         = {}
        self._src_line_map:    dict[str, list]         = {}
        self._rag_circles:     dict[str, _RagCircle]   = {}
        self._rag_lbls:        dict[str, QGraphicsTextItem] = {}
        self._agent_radii:     dict[str, int]          = {}
        self._rpos:            dict[str, tuple]        = {}
        self._placed_srcs:     list                    = []
        self._rag_src_colors:  dict[str, str]          = {}

        self._tokens:           dict[str, str]          = {}
        self._agent_responses:  dict[str, AgentResponse] = {}
        self._active_set:   set            = set()
        self._ripples:      list           = []
        self._dots:         list           = []
        self._dot_timers:   dict           = {}
        self._dash_offsets: dict           = {}
        self._anim_t        = 0.0
        self._selected_name = ''
        self._selected_round: int = 0
        self._selected_price  = None

        self._build()

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_timer.start()

    # ── Build UI ──────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        body = QWidget()
        bl = QHBoxLayout(body); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(0)

        # Canvas
        self._scene = QGraphicsScene()
        self._view  = _DotGridView(self._scene)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._view.resized.connect(self._on_view_resized)
        bl.addWidget(self._view, stretch=1)

        # Right sidebar
        self._sidebar_panel, self._sb = _make_sidebar()
        self._sb['stop_btn'].clicked.connect(self._on_stop)
        bl.addWidget(self._sidebar_panel)
        root.addWidget(body, stretch=1)

        # ── Node Details Card — child of self, floats beside sidebar ──────
        self._detail_card = _NodeDetailCard(self)
        self._detail_card.closed.connect(self._on_detail_closed)

        # ── Entity legend — child of self so it is never clipped by the viewport ──
        self._entity_legend = _EntityLegend(self)

        # Agent status row registry
        self._agent_status_rows: dict[str, _AgentStatusRow] = {}
        self._log_count = 0

    def _on_detail_closed(self):
        self._detail_card.hide()

    # ── Legend / overlay positioning ──────────────────────────────────────────

    def _build_legend(self):
        main_agents = [a for a in self._agents if not a.is_mini]
        mini_agents = [a for a in self._agents if a.is_mini]
        self._entity_legend.rebuild(main_agents, mini_agents, self._placed_srcs, self._rag_src_colors)
        self._entity_legend.show()
        self._position_overlays()

    def _build_agent_rows(self):
        """Populate sidebar agent list."""
        vbox = self._sb['agent_vbox']
        while vbox.count() > 1:
            it = vbox.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._agent_status_rows.clear()

        main_agents = [a for a in self._agents if not a.is_mini]
        mini_agents = [a for a in self._agents if a.is_mini]
        ordered = main_agents + mini_agents

        for i, agent in enumerate(ordered):
            color = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            row = _AgentStatusRow(agent.name, color, agent.is_mini)
            row.clicked.connect(self._on_node_clicked)
            vbox.insertWidget(vbox.count() - 1, row)
            self._agent_status_rows[agent.name] = row

        total = len(ordered)
        self._sb['ag_count'].setText(f'{total} agent{"s" if total != 1 else ""}')

    def _position_overlays(self):
        margin = 14

        # Map the view's top-left into self's coordinate space so overlay
        # positions are always relative to self (never clipped by viewport).
        view_origin = self._view.mapTo(self, QPoint(0, 0))
        vx, vy = view_origin.x(), view_origin.y()
        vw, vh = self._view.width(), self._view.height()

        # Entity legend — bottom-left of canvas area
        self._entity_legend.adjustSize()
        lh = self._entity_legend.height()
        self._entity_legend.move(vx + margin, vy + vh - margin - lh)
        self._entity_legend.raise_()

        # Node detail card — pinned just left of sidebar, y held steady
        dc = self._detail_card
        if dc.isVisible():
            sw = self._sidebar_panel.width()
            dx = self.width() - sw - dc.width() - 12
            dc.move(dx, dc.y())          # keep whatever y was set by _show_detail_card
        dc.raise_()

    def _on_view_resized(self):
        self._position_overlays()

    def _show_detail_card(self, scene_x: float, scene_y: float):
        """Show detail card pinned beside the sidebar at a fixed, stable position."""
        # Horizontal: always just left of the sidebar
        sw = self._sidebar_panel.width()
        dc_w = self._detail_card.width()
        dx = self.width() - sw - dc_w - 12

        # Vertical: fixed — vertically centred in the canvas view area,
        # clamped to stay fully on-screen. Never follows the clicked node so
        # the card stays put when switching between nodes.
        self._detail_card.adjustSize()
        dc_h = self._detail_card.height()
        view_origin = self._view.mapTo(self, QPoint(0, 0))
        vy = view_origin.y()
        vh = self._view.height()
        dy = vy + max(12, (vh - dc_h) // 2)
        dy = max(vy + 12, min(dy, vy + vh - dc_h - 12))

        self._detail_card.move(dx, dy)
        self._detail_card.show()
        self._detail_card.raise_()

    # ── Simulation ────────────────────────────────────────────────────────────

    def start_simulation(self, scenario, agents=None):
        if agents: self._agents = list(agents)
        self._scenario = scenario.to_dict() if hasattr(scenario, 'to_dict') else dict(scenario)
        s = self._scenario

        # Update scenario chips
        chips = self._sb['chip_refs']
        vals = [
            f"Oil {s.get('oil_pct',0):+.0f}%",
            f"USD {s.get('usd_pct',0):+.0f}%",
            f"BSP {s.get('bsp_rate',6.5):.2f}%",
            f"Dem {s.get('demand_index',72):.0f}",
        ]
        for chip, txt in zip(chips, vals):
            chip.setText(txt)

        self._dots.clear(); self._ripples.clear()
        self._active_set.clear(); self._dot_timers.clear(); self._dash_offsets.clear()
        self._scene.clear()
        for d in [self._circles, self._name_lbls, self._est_lbls, self._apos,
                  self._lines, self._dot_paths, self._agent_rag_lines, self._rag_line_map,
                  self._src_line_map, self._rag_circles, self._rag_lbls, self._agent_radii,
                  self._rpos, self._tokens, self._agent_responses]:
            d.clear()
        self._placed_srcs.clear(); self._rag_src_colors.clear()
        self._eng_item = None; self._selected_name = ''
        self._detail_card.hide()
        self._log_count = 0
        self._sb['log_count_lbl'].setText('')

        while self._sb['log_vbox'].count() > 1:
            it = self._sb['log_vbox'].takeAt(0)
            if it.widget(): it.widget().deleteLater()

        self._place_nodes()
        self._build_legend()
        self._build_agent_rows()

        self._engine = DebateEngine(self._agents, self._rag, self._scenario)
        rounds = 3 if len(self._agents) <= 7 else 2
        self._thread = DebateThread(self._engine, rounds=rounds)
        self._thread.token_received.connect(self._on_token)
        self._thread.agent_done.connect(self._on_agent_done)
        self._thread.debate_complete.connect(self._on_debate_complete)
        self._thread.error_occurred.connect(self._on_error)

        self._sb['stop_btn'].setEnabled(True)
        self._sb['live_dot'].setStyleSheet('font-size:8px;color:#22C55E;background:transparent;')
        self._sb['status_lbl'].setText('Simulation running')
        self._sb['round_lbl'].setText('Round 1')
        self._update_price()
        self._thread.start()

    def _lbl(self, text, px=9, bold=False, color='#64748B'):
        item = QGraphicsTextItem(text)
        item.setDefaultTextColor(QColor(color))
        f = QFont(); f.setPixelSize(px)
        if bold: f.setBold(True)
        item.setFont(f); self._scene.addItem(item); return item

    def _place_nodes(self):
        cx, cy = 440, 330; self._engine_pos = (cx, cy)

        eng = _EngineCircle()
        self._eng_item = eng; eng.setPos(cx, cy)
        eng.clicked.connect(self._on_engine_clicked)
        eng.position_changed.connect(self._on_engine_moved)
        self._scene.addItem(eng)

        el = self._lbl('ENGINE', 7, True, '#94A3B8')
        el.setPos(cx - el.boundingRect().width() / 2, cy + _ER + 4)
        self._rag_lbls['__engine__'] = el

        main_agents = [a for a in self._agents if not a.is_mini]
        mini_agents = [a for a in self._agents if a.is_mini]
        n_main = len(main_agents); n_mini = len(mini_agents)
        ordered = main_agents + mini_agents
        apos = []; angs = []

        for i, agent in enumerate(ordered):
            color  = _AGENT_COLORS[i % len(_AGENT_COLORS)]
            radius = _AR_MINI if agent.is_mini else _AR

            if agent.is_mini:
                j = mini_agents.index(agent)
                ang = (2 * math.pi * j / max(n_mini, 1)) - math.pi / 2
                if n_main > 0: ang += math.pi / n_main
                ring = _RING_A_MINI
            else:
                j = main_agents.index(agent)
                ang = (2 * math.pi * j / max(n_main, 1)) - math.pi / 2
                ring = _RING_A

            angs.append(ang)
            ax = cx + ring * math.cos(ang)
            ay = cy + ring * math.sin(ang)
            apos.append((ax, ay))
            self._apos[agent.name] = (ax, ay)
            self._agent_radii[agent.name] = radius

            circ = _AgentCircle(agent, color, radius=radius)
            circ.node_clicked.connect(self._on_node_clicked)
            circ.hover_enter.connect(self._on_hover_enter)
            circ.hover_leave.connect(self._on_hover_leave)
            circ.position_changed.connect(self._on_agent_moved)
            self._circles[agent.name] = circ
            circ.setPos(ax, ay); self._scene.addItem(circ)

            # Name label below circle
            nl = self._lbl(agent.name, 8 if agent.is_mini else 9, True, '#374151')
            nl.setPos(ax - nl.boundingRect().width() / 2, ay + radius + 5)
            self._name_lbls[agent.name] = nl

            # Estimate label
            el2 = self._lbl('', 8, True, color)
            el2.setPos(ax - 20, ay + radius + 20)
            self._est_lbls[agent.name] = el2

            # Spoke
            a_ex, a_ey = _edge_pt(ax, ay, cx, cy, radius + 2)
            e_ex, e_ey = _edge_pt(cx, cy, ax, ay, _ER + 2)
            path = _qcurve(a_ex, a_ey, e_ex, e_ey)
            self._dot_paths[agent.name] = path
            pen = QPen(QColor(color), 1.0); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pi = QGraphicsPathItem(path); pi.setPen(pen); pi.setOpacity(0.20)
            self._scene.addItem(pi); self._lines[agent.name] = pi

        # RAG circles
        all_srcs = []; seen = set()
        for ag in ordered:
            for src in ag.rag_sources:
                if src not in seen: seen.add(src); all_srcs.append(src)

        counts = self._rag.source_chunk_counts
        s2a: dict[str, list[int]] = {}
        for i, ag in enumerate(ordered):
            for src in ag.rag_sources: s2a.setdefault(src, []).append(i)

        excl = defaultdict(list)
        for src in all_srcs:
            if len(s2a[src]) == 1: excl[s2a[src][0]].append(src)

        rpos = {}
        for ai, srcs in excl.items():
            base = angs[ai]; k = len(srcs)
            for j, src in enumerate(srcs):
                a = base + (j - (k - 1) / 2) * 0.20
                rpos[src] = (cx + _RING_R * math.cos(a), cy + _RING_R * math.sin(a))
        for src in all_srcs:
            if len(s2a[src]) > 1:
                idxs = s2a[src]
                sa = sum(math.sin(angs[i]) for i in idxs) / len(idxs)
                ca = sum(math.cos(angs[i]) for i in idxs) / len(idxs)
                rpos[src] = (cx + _RING_R * math.cos(math.atan2(sa, ca)),
                             cy + _RING_R * math.sin(math.atan2(sa, ca)))

        for src in all_srcs:
            if src not in rpos: continue
            rx, ry = rpos[src]
            cnt   = counts.get(src, 0)
            pcol  = _AGENT_COLORS[s2a[src][0] % len(_AGENT_COLORS)]
            self._placed_srcs.append(src); self._rag_src_colors[src] = pcol

            rc = _RagCircle(src, cnt, pcol); rc.setPos(rx, ry)
            rc.clicked.connect(self._on_rag_clicked)
            rc.position_changed.connect(self._on_rag_moved)
            self._scene.addItem(rc)
            self._rag_circles[src] = rc; self._rpos[src] = (rx, ry)

            disp = _SOURCE_DISPLAY.get(src, src)
            sl = self._lbl(disp, 8, True, '#64748B')
            lw, lh = sl.boundingRect().width(), sl.boundingRect().height()
            sl.setPos(rx - lw / 2, ry + _RR + 4 if ry >= cy else ry - _RR - lh - 2)
            self._rag_lbls[src] = sl

            for ai in s2a[src]:
                col = _AGENT_COLORS[ai % len(_AGENT_COLORS)]
                ax2, ay2 = apos[ai]
                ag_r = self._agent_radii.get(ordered[ai].name, _AR)
                rr_ex, rr_ey = _edge_pt(rx, ry, ax2, ay2, _RR + 2)
                ag_ex, ag_ey = _edge_pt(ax2, ay2, rx, ry, ag_r + 2)
                pen = QPen(QColor(col), 0.9)
                pen.setStyle(Qt.PenStyle.DotLine); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                rl = QGraphicsPathItem(_qcurve(rr_ex, rr_ey, ag_ex, ag_ey))
                rl.setPen(pen); rl.setOpacity(0.30); self._scene.addItem(rl)
                ag_name = ordered[ai].name
                self._agent_rag_lines.setdefault(ag_name, []).append(rl)
                self._rag_line_map.setdefault(ag_name, []).append({'src': src, 'item': rl})
                self._src_line_map.setdefault(src, []).append({'agent': ag_name, 'item': rl})

        self._view.fitInView(
            self._scene.itemsBoundingRect().adjusted(-80, -80, 80, 80),
            Qt.AspectRatioMode.KeepAspectRatio)

    # ── Drag handlers ─────────────────────────────────────────────────────────

    def _on_agent_moved(self, name, x, y):
        self._apos[name] = (x, y)
        cx, cy = self._engine_pos
        r = self._agent_radii.get(name, _AR)
        a_ex, a_ey = _edge_pt(x, y, cx, cy, r + 2)
        e_ex, e_ey = _edge_pt(cx, cy, x, y, _ER + 2)
        new_path = _qcurve(a_ex, a_ey, e_ex, e_ey)
        self._dot_paths[name] = new_path
        li = self._lines.get(name)
        if li: li.setPath(new_path)
        nl = self._name_lbls.get(name)
        if nl: nl.setPos(x - nl.boundingRect().width() / 2, y + r + 5)
        el = self._est_lbls.get(name)
        if el: el.setPos(x - el.boundingRect().width() / 2, y + r + 20)
        for entry in self._rag_line_map.get(name, []):
            rx, ry = self._rpos.get(entry['src'], (0, 0))
            rr_ex, rr_ey = _edge_pt(rx, ry, x, y, _RR + 2)
            ag_ex, ag_ey = _edge_pt(x, y, rx, ry, r + 2)
            entry['item'].setPath(_qcurve(rr_ex, rr_ey, ag_ex, ag_ey))

    def _on_rag_moved(self, source, x, y):
        self._rpos[source] = (x, y)
        for entry in self._src_line_map.get(source, []):
            ax, ay = self._apos.get(entry['agent'], (0, 0))
            ag_r = self._agent_radii.get(entry['agent'], _AR)
            rr_ex, rr_ey = _edge_pt(x, y, ax, ay, _RR + 2)
            ag_ex, ag_ey = _edge_pt(ax, ay, x, y, ag_r + 2)
            entry['item'].setPath(_qcurve(rr_ex, rr_ey, ag_ex, ag_ey))
        sl = self._rag_lbls.get(source)
        if sl:
            lw, lh = sl.boundingRect().width(), sl.boundingRect().height()
            cy = self._engine_pos[1]
            sl.setPos(x - lw / 2, y + _RR + 4 if y >= cy else y - _RR - lh - 2)

    def _on_engine_moved(self, x, y):
        self._engine_pos = (x, y)
        el = self._rag_lbls.get('__engine__')
        if el: el.setPos(x - el.boundingRect().width() / 2, y + _ER + 4)
        for name, (ax, ay) in self._apos.items():
            r = self._agent_radii.get(name, _AR)
            a_ex, a_ey = _edge_pt(ax, ay, x, y, r + 2)
            e_ex, e_ey = _edge_pt(x, y, ax, ay, _ER + 2)
            new_path = _qcurve(a_ex, a_ey, e_ex, e_ey)
            self._dot_paths[name] = new_path
            li = self._lines.get(name)
            if li: li.setPath(new_path)

    # ── Animation ─────────────────────────────────────────────────────────────

    def _anim_tick(self):
        dt = 0.033; self._anim_t += dt
        done = [r for r in self._ripples if r.advance(dt)]
        for r in done: self._scene.removeItem(r); self._ripples.remove(r)
        done_d = [d for d in self._dots if d.advance(dt)]
        for d in done_d: self._scene.removeItem(d); self._dots.remove(d)

        for name in list(self._active_set):
            self._dot_timers[name] = self._dot_timers.get(name, 999) + dt
            if self._dot_timers[name] >= 0.45:
                self._dot_timers[name] = 0; self._spawn_dot(name)
            pi = self._lines.get(name); circ = self._circles.get(name)
            if pi and circ:
                off = self._dash_offsets.get(name, 0) + dt * 90
                self._dash_offsets[name] = off
                pen = QPen(QColor(circ.color), 1.6)
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([5.0, 4.0])
                pen.setDashOffset(-(off % 18))
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pi.setPen(pen); pi.setOpacity(0.85)

        for circ in self._circles.values(): circ.anim_update(dt)
        if self._eng_item: self._eng_item.anim_update(dt)

    def _spawn_dot(self, name):
        path = self._dot_paths.get(name); circ = self._circles.get(name)
        if path and circ:
            dot = _TravelDot(path, QColor(circ.color))
            self._scene.addItem(dot); self._dots.append(dot)

    def _spawn_ripple(self, x, y, r, color):
        rip = _RippleItem(r, QColor(color)); rip.setPos(x, y)
        self._scene.addItem(rip); self._ripples.append(rip)

    # ── Hover ─────────────────────────────────────────────────────────────────

    def _on_hover_enter(self, name):
        li = self._lines.get(name)
        if li and name not in self._active_set: li.setOpacity(0.65)
        for rl in self._agent_rag_lines.get(name, []): rl.setOpacity(0.80)

    def _on_hover_leave(self, name):
        li = self._lines.get(name)
        if li and name not in self._active_set: li.setOpacity(0.20)
        for rl in self._agent_rag_lines.get(name, []): rl.setOpacity(0.30)

    # ── Click handlers ────────────────────────────────────────────────────────

    def _on_node_clicked(self, name):
        self._selected_name = name
        circ = self._circles.get(name)
        if not circ: return
        c = circ.color
        ax, ay = self._apos.get(name, (0, 0))
        r = self._agent_radii.get(name, _AR)
        self._spawn_ripple(ax, ay, r, c)
        agent = next((a for a in self._agents if a.name == name), None)
        model = llm.describe_model(agent.tier) if agent else '—'

        resp = self._agent_responses.get(name)
        if name in self._active_set:
            status = '● Responding'
        elif resp:
            status = 'Done'
        else:
            status = 'Idle'

        round_num = resp.round_num if resp else self._selected_round
        price     = resp.price_estimate if resp else self._selected_price

        self._detail_card.show_agent(name, model, c, round_num, price, status)

        # ── Restore existing content so clicking never wipes a finished response ──
        if resp:
            # Agent already finished — show their full response immediately
            self._detail_card.update_done(resp.statement, resp.price_estimate, resp.round_num)
        elif name in self._active_set:
            # Agent is mid-stream — restore whatever tokens we have so far
            raw = self._tokens.get(name, '')
            if raw:
                vis = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
                vis = re.sub(r'<think>.*', '', vis, flags=re.DOTALL).strip()
                m   = re.search(r'<think>(.*)', raw, re.DOTALL)
                think = re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
                self._detail_card.update_live_text(vis or think or '…')

        self._show_detail_card(ax, ay)

    def _on_engine_clicked(self):
        cx, cy = self._engine_pos
        self._spawn_ripple(cx, cy, _ER, '#22C55E')
        price_str = self._sb['price_lbl'].text()
        self._detail_card.show_engine(price_str, self._cv_rmse, self._scenario)
        self._show_detail_card(cx, cy)

    def _on_rag_clicked(self, source, count):
        disp = _SOURCE_DISPLAY.get(source, source)
        agents_using = [ag.name for ag in self._agents if source in ag.rag_sources]
        rx, ry = self._rpos.get(source, (0, 0))
        self._spawn_ripple(rx, ry, _RR, self._rag_src_colors.get(source, '#94A3B8'))
        self._detail_card.show_rag(source, disp, count, agents_using)
        self._show_detail_card(rx, ry)

    # ── Simulation signals ────────────────────────────────────────────────────

    def _on_token(self, name, token):
        self._tokens[name] = self._tokens.get(name, '') + token
        raw = self._tokens[name]
        vis = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
        vis = re.sub(r'<think>.*', '', vis, flags=re.DOTALL).strip()

        circ = self._circles.get(name)
        if circ: circ.set_active()
        self._active_set.add(name)

        # Update agent status row
        row = self._agent_status_rows.get(name)
        if row: row.set_responding()

        # Update sidebar round label
        m = re.search(r'<think>(.*)', raw, re.DOTALL)
        think = re.sub(r'\s+', ' ', m.group(1)).strip() if m else ''
        text_to_show = vis or think or '…'

        # Update floating detail card if it's this agent
        if self._detail_card.isVisible() and self._selected_name == name:
            self._detail_card.update_live_text(vis or think or '…')

    def _on_agent_done(self, resp):
        name = resp.agent_name
        circ = self._circles.get(name)
        if circ: circ.set_done(resp.statement, resp.price_estimate)
        self._tokens[name] = ''
        self._agent_responses[name] = resp          # ← persist for later re-clicks
        self._active_set.discard(name)
        self._dot_timers.pop(name, None)

        pi = self._lines.get(name)
        color = circ.color if circ else '#94A3B8'
        if pi:
            pen = QPen(QColor(color), 1.0); pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pi.setPen(pen); pi.setOpacity(0.20)

        self._selected_round = resp.round_num
        self._selected_price = resp.price_estimate
        self._sb['round_lbl'].setText(f'Round {resp.round_num}')
        self._update_price()

        # Update agent status row
        row = self._agent_status_rows.get(name)
        if row: row.set_done(resp.price_estimate)

        # Price estimate label on canvas
        el = self._est_lbls.get(name)
        if el and resp.price_estimate is not None:
            el.setPlainText(f'+₱{resp.price_estimate:.2f}/L')
            el.setDefaultTextColor(QColor(color))
            ax, ay = self._apos.get(name, (0, 0))
            r = self._agent_radii.get(name, _AR)
            el.setPos(ax - el.boundingRect().width() / 2, ay + r + 20)

        # Update floating detail card if showing this agent
        if self._detail_card.isVisible() and self._selected_name == name:
            self._detail_card.update_done(resp.statement, resp.price_estimate, resp.round_num)

        # Add to debate log
        self._log_count += 1
        self._sb['log_count_lbl'].setText(str(self._log_count))
        row_w = _LogRow(resp, color)
        row_w.clicked.connect(self._on_node_clicked)
        self._sb['log_vbox'].insertWidget(self._sb['log_vbox'].count() - 1, row_w)
        QTimer.singleShot(0, lambda: self._sb['log_scroll'].verticalScrollBar().setValue(
            self._sb['log_scroll'].verticalScrollBar().maximum()))

    def _update_price(self):
        try:
            X, y, feature_cols, df_feat = build_features(self._df)
            last = df_feat.iloc[-1]
            feats = []
            for col in feature_cols:
                if col == 'prev_gas_price': feats.append(float(last['gas_price']))
                elif col == 'oil_price':
                    feats.append(float(last[col]) * (1 + self._scenario.get('oil_pct', 0) / 100))
                elif col == 'usd_php':
                    feats.append(float(last[col]) * (1 + self._scenario.get('usd_pct', 0) / 100))
                else: feats.append(float(last[col]))
            price = float(ml.predict(self._regressor, np.array(feats))[0])
            if self._eng_item: self._eng_item.set_price(price)
            self._sb['price_lbl'].setText(f'₱{price:.2f} /L')
        except Exception as e:
            import logging; logging.warning('_update_price: %s', e)

    def _on_debate_complete(self, responses):
        self._sb['stop_btn'].setEnabled(False)
        self._sb['live_dot'].setStyleSheet('font-size:8px;color:#10B981;background:transparent;')
        self._sb['status_lbl'].setText('Simulation complete')
        self._sb['round_lbl'].setText('Done')
        cx, cy = self._engine_pos
        self._spawn_ripple(cx, cy, _ER, '#22C55E')
        QTimer.singleShot(200, lambda: self._spawn_ripple(cx, cy, _ER, '#6366F1'))
        QTimer.singleShot(400, lambda: self._spawn_ripple(cx, cy, _ER, '#0EA5E9'))
        for li in self._lines.values(): li.setOpacity(0.40)
        self.simulation_complete.emit(responses)

    def _on_error(self, msg):
        self._sb['stop_btn'].setEnabled(False)
        self._sb['live_dot'].setStyleSheet('font-size:8px;color:#EF4444;background:transparent;')
        self._sb['status_lbl'].setText(f'Error: {msg}')

    def _on_stop(self):
        if self._thread and self._thread.isRunning():
            self._thread.terminate(); self._thread.wait()
        self._sb['stop_btn'].setEnabled(False)
        self._sb['live_dot'].setStyleSheet('font-size:8px;color:#D1D5DB;background:transparent;')
        self._sb['status_lbl'].setText('Stopped')

    @property
    def engine(self): return self._engine

    def scenario(self): return self._scenario