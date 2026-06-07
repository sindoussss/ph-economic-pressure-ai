"""Left navigation — vertical journey/timeline.

Each stage is a dot on a connected spine. The dots tell you where you are;
the spine tells you the simulation moves linearly: Home → Overview → run →
Report → Interact, with a side-channel for Agent Performance insights.

Visual states:
  ●  active (filled black circle)
  ○  inactive (outlined gray circle, clickable)
  ◌  locked  (dashed outline, dimmed, not clickable)

Behavior preserved:
  - `stage_changed` signal emits 0-based stage index on click
  - `set_active(idx)` selects programmatically
  - `unlock_stages(indices)` un-disables stages locked at startup
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor


# ── Palette ───────────────────────────────────────────────────────────────────
BG          = '#FFFFFF'
INK         = '#0F1115'
TEXT_INACT  = '#525866'
TEXT_HOVER  = '#1F2937'
TEXT_LOCKED = '#CBD2DC'
SPINE       = '#E5E7EB'
SPINE_ACT   = '#0F1115'      # spine segment behind the active dot turns ink
DIVIDER     = '#E5E7EB'
HAIRLINE    = '#EFF0F3'


# Nav structure:
#   (stage_index, label, has_connector_below)
# `has_connector_below=False` is the visual section break — no spine
# drawn from this row down to the next.
_NAV: list[tuple[int, str, bool]] = [
    (0, 'Home',              False),   # break after Home
    (1, 'Overview',          True),
    (2, 'Simulation',        True),
    (3, 'Report',            True),
    (4, 'Interact',          False),   # break after Interact
    (5, 'Agent performance', False),   # section item, no connector
    (6, 'Methodology & accuracy', False),   # last row, no connector
]


# Geometry constants
_DOT_RADIUS       = 5      # filled/outlined circle radius
_DOT_LEFT_PADDING = 22     # x-position of dot center
_LABEL_LEFT       = 44     # x-position where label starts
_ROW_HEIGHT       = 44     # tall enough that connector line looks proportioned
_SPINE_WIDTH      = 1.5    # thickness of the connecting spine line


# ─────────────────────────────────────────────────────────────────────────────
class _TimelineRow(QFrame):
    """A single timeline node — dot + label + optional connector line below."""

    clicked = pyqtSignal(int)

    def __init__(self, stage_index: int, label: str,
                 connector_below: bool, parent=None):
        super().__init__(parent)
        self._idx = stage_index
        self._label_text = label
        self._connector_below = connector_below
        self._active = False
        self._locked = False
        self._hover = False
        self.setMouseTracking(True)
        self.setFixedHeight(_ROW_HEIGHT)
        self.setStyleSheet(f'background:{BG};border:none;')
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build()

    def _build(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(_LABEL_LEFT, 0, 18, 0)
        h.setSpacing(0)
        self._label = QLabel(self._label_text)
        self._label.setStyleSheet('background:transparent;border:none;')
        h.addWidget(self._label)
        h.addStretch()

    # ── Painter (dot + connector line) ───────────────────────────────────────
    def paintEvent(self, event):
        # First, paint the background (matches stylesheet behavior)
        super().paintEvent(event)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        cx = _DOT_LEFT_PADDING
        cy = self.height() / 2

        # ── Connector line below this dot ──
        if self._connector_below:
            line_color = QColor(SPINE_ACT if self._active else SPINE)
            pen = QPen(line_color, _SPINE_WIDTH)
            pen.setCosmetic(True)
            p.setPen(pen)
            # Draw from dot bottom edge to row bottom
            p.drawLine(QPointF(cx, cy + _DOT_RADIUS + 1),
                       QPointF(cx, self.height()))

        # ── Dot ──
        if self._locked:
            # Dashed outline circle
            pen = QPen(QColor(TEXT_LOCKED), 1.2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(cx, cy), _DOT_RADIUS, _DOT_RADIUS)
        elif self._active:
            # Filled ink circle with a halo ring
            halo = QColor(INK)
            halo.setAlpha(28)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(halo))
            p.drawEllipse(QPointF(cx, cy), _DOT_RADIUS + 4, _DOT_RADIUS + 4)
            p.setBrush(QBrush(QColor(INK)))
            p.drawEllipse(QPointF(cx, cy), _DOT_RADIUS, _DOT_RADIUS)
        else:
            # Outlined gray circle
            color = QColor(TEXT_HOVER if self._hover else SPINE)
            pen = QPen(color, 1.4)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(BG)))
            p.drawEllipse(QPointF(cx, cy), _DOT_RADIUS, _DOT_RADIUS)

        p.end()

    # ── State management ─────────────────────────────────────────────────────
    def _apply_label_style(self):
        if self._locked:
            self._label.setStyleSheet(
                f'color:{TEXT_LOCKED};font-size:13px;font-weight:400;'
                f'background:transparent;border:none;'
            )
            return
        if self._active:
            color, weight = INK, '600'
        elif self._hover:
            color, weight = TEXT_HOVER, '500'
        else:
            color, weight = TEXT_INACT, '400'
        self._label.setStyleSheet(
            f'color:{color};font-size:13px;font-weight:{weight};'
            f'background:transparent;border:none;'
        )

    def set_active(self, on: bool):
        if self._active == on:
            return
        self._active = on
        self._apply_label_style()
        self.update()

    def set_locked(self, on: bool):
        if self._locked == on:
            return
        self._locked = on
        self.setCursor(
            Qt.CursorShape.ArrowCursor if on
            else Qt.CursorShape.PointingHandCursor
        )
        self._apply_label_style()
        self.update()

    def is_locked(self) -> bool:
        return self._locked

    # ── Mouse events ─────────────────────────────────────────────────────────
    def enterEvent(self, event):
        if not self._locked:
            self._hover = True
            self._apply_label_style()
            self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._hover:
            self._hover = False
            self._apply_label_style()
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._locked:
            self.clicked.emit(self._idx)
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
class SidebarWidget(QWidget):
    stage_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(208)
        self.setStyleSheet(
            f'QWidget{{background:{BG};}}'
            f'SidebarWidget{{border-right:1px solid {DIVIDER};}}'
        )
        self._rows_by_idx: dict[int, _TimelineRow] = {}
        self._locked: set[int] = {2, 3, 4}
        self._active: int = 0
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Brand block ──────────────────────────────────────────────────────
        brand_wrap = QWidget()
        brand_wrap.setStyleSheet(f'background:{BG};')
        bv = QVBoxLayout(brand_wrap)
        bv.setContentsMargins(22, 22, 20, 18)
        bv.setSpacing(2)
        brand = QLabel('PH Economic AI')
        brand.setStyleSheet(
            f'font-size:14px;font-weight:700;color:{INK};'
            f'background:transparent;border:none;letter-spacing:-0.2px;'
        )
        ver = QLabel('v2.0  ·  local')
        ver.setStyleSheet(
            f'font-size:11px;color:{TEXT_INACT};background:transparent;'
            f'border:none;letter-spacing:0.2px;'
        )
        bv.addWidget(brand)
        bv.addWidget(ver)
        outer.addWidget(brand_wrap)

        hl = QFrame()
        hl.setFrameShape(QFrame.Shape.HLine)
        hl.setStyleSheet(f'background:{DIVIDER};border:none;')
        hl.setFixedHeight(1)
        outer.addWidget(hl)

        # ── Timeline rows ───────────────────────────────────────────────────
        nav_wrap = QWidget()
        nav_wrap.setStyleSheet(f'background:{BG};')
        nv = QVBoxLayout(nav_wrap)
        nv.setContentsMargins(0, 18, 0, 18)
        nv.setSpacing(0)

        for stage_idx, name, connector_below in _NAV:
            row = _TimelineRow(stage_idx, name, connector_below)
            row.clicked.connect(self._on_click)
            self._rows_by_idx[stage_idx] = row
            nv.addWidget(row)

        nv.addStretch()
        outer.addWidget(nav_wrap, stretch=1)

        # Initial state
        for idx in self._locked:
            row = self._rows_by_idx.get(idx)
            if row is not None:
                row.set_locked(True)
        self._refresh_active()

    # ── State ────────────────────────────────────────────────────────────────
    def _on_click(self, idx: int):
        if idx in self._locked:
            return
        self._active = idx
        self._refresh_active()
        self.stage_changed.emit(idx)

    def _refresh_active(self):
        for idx, row in self._rows_by_idx.items():
            row.set_active(idx == self._active and not row.is_locked())

    # ── Public API ───────────────────────────────────────────────────────────
    def unlock_stages(self, indices: list[int]):
        for i in indices:
            self._locked.discard(i)
            row = self._rows_by_idx.get(i)
            if row is not None:
                row.set_locked(False)
        self._refresh_active()

    def set_active(self, idx: int):
        if idx in self._locked:
            return
        self._active = idx
        self._refresh_active()
