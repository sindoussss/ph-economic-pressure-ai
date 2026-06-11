"""Stage 3 Swarm Canvas — MiroFish-inspired scattered graph view.

White background, monospace labels, tiny dots, curved gray/red connections,
side node-details panel, terminal-style console.

Public interface (consumed by main_window.py):
    Stage3SwarmPanel(store=None, parent=None)
        .swarm_complete : pyqtSignal(MasterVerdict)
        .reset()
        .connect_thread(swarm_thread)
        .connect_food_thread(debate_thread)
        .connect_elec_thread(debate_thread)
        ._apply_trust_badges()
"""
from __future__ import annotations

import math
import random
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QTextEdit, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsObject, QSizePolicy,
    QGraphicsPathItem, QGraphicsItem, QPushButton, QScrollArea,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QPoint, pyqtSignal, QTimer,
    QPropertyAnimation, QEasingCurve, pyqtProperty, QVariantAnimation,
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QPainterPath, QFont, QFontMetrics,
)

from ph_economic_ai.engine.swarm import REGIONS, build_swarm_agents, _ROLE_RAG
from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
from ph_economic_ai.ui import kg_live as _kg_live
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder


# ════════════════════════════════════════════════════════════════════════════
#  Palette — MiroFish light, near-monochrome with red/amber accents
# ════════════════════════════════════════════════════════════════════════════
BG          = '#FCFCFD'       # near-white, slightly warmer than pure
TEXT_1      = '#0A0A0A'       # near-black for primary text
TEXT_2      = '#6B7280'       # medium gray for secondary
TEXT_3      = '#9CA3AF'       # lighter gray for tertiary
DIVIDER     = '#E5E7EB'
SOFT_BORDER = '#D1D5DB'
SURFACE     = '#FAFAFA'
DOT_BG      = '#E5E7EB'       # background dot color (very subtle)
DOT_SIZE    = 0.5             # radius of background dots — small pinpricks
DOT_SPACING = 16              # px between background dots

# Region cluster outlines — neutral, technical
HALO_BG    = '#F9FAFB'        # very subtle gray-white fill (vs pure bg)
HALO_LINE  = '#E5E7EB'        # cluster outline color
HALO_LINE_DARK = '#D1D5DB'    # axis lines

# Ambient noise dots (simulates dense network)
NOISE_DOT       = '#E5E7EB'
NOISE_DOT_RED   = '#FDA4AF'   # very pale red accents
NOISE_COUNT     = 800
NOISE_DOT_SIZE  = 0.6

# Cluster hash IDs (technical look)
CLUSTER_IDS = {
    'ncr': 'c_a47f3b',
    'luz': 'c_8e21cd',
    'vis': 'c_5d09a2',
    'dav': 'c_3b71f4',
    'food': 'c_food_61e',
    'elec': 'c_elec_92a',
    'master': 'c_mst_001',
}

# Node states
N_IDLE      = '#9CA3AF'       # slate gray for idle agents
N_ACTIVE    = '#EF4444'       # red — active (MiroFish-style)
N_DONE      = '#10B981'       # emerald — settled / consensus
N_DEAD      = '#E5E7EB'       # faded gray
N_REGIONAL  = '#1F2937'       # near-black regional judge
N_MASTER    = '#0A0A0A'       # pure black master
N_RAG       = '#3B82F6'       # blue chip for RAG sources

# Sector accents
N_FOOD      = '#16A34A'
N_ELEC      = '#D97706'

# Connection lines
LINE_IDLE   = '#E5E7EB'
LINE_ACTIVE = '#FCA5A5'       # light red — MiroFish-style
LINE_DONE   = '#D1D5DB'

# Trust tiers (light-mode pills)
T_PROMOTED  = ('#10B981', '#ECFDF5', '#A7F3D0')   # text, bg, border
T_DEMOTED   = ('#EF4444', '#FEF2F2', '#FECACA')
T_DEFAULT   = ('#92400E', '#FFFBEB', '#FDE68A')


# ════════════════════════════════════════════════════════════════════════════
#  Layout — scene coordinates and pseudo-random scatter
# ════════════════════════════════════════════════════════════════════════════
SCENE_W, SCENE_H = 1400.0, 880.0
HW, HH = SCENE_W / 2, SCENE_H / 2     # half-width, half-height

# Anchor positions (scene origin = center)
A_MASTER   = (0.0, 0.0)
A_REGIONS  = [(-260.0, -160.0), (260.0, -160.0), (-260.0, 160.0), (260.0, 160.0)]
A_FOOD     = (-560.0, -260.0)
A_ELEC     = (560.0, 260.0)
A_RAG_ROW  = (0.0, -370.0)           # horizontal row across top

R_AGENT_DOT      = 5.0
R_REGIONAL       = 13.0
R_MASTER         = 20.0
R_RAG_CHIP       = 7.0
R_SECTOR_AGENT   = 5.0
R_SECTOR_VERDICT = 14.0

AGENT_SCATTER    = 120.0   # max radius around regional anchor
SECTOR_SCATTER   = 65.0
RAG_SPACING      = 80.0


def _scatter(seed_key: str, anchor: tuple[float, float],
             max_r: float, min_r: float = 35.0) -> tuple[float, float]:
    """Deterministic pseudo-random position around an anchor."""
    rnd = random.Random(seed_key)
    angle = rnd.uniform(0, 2 * math.pi)
    r = rnd.uniform(min_r, max_r)
    return anchor[0] + r * math.cos(angle), anchor[1] + r * math.sin(angle)


def _curve(x1: float, y1: float, x2: float, y2: float,
           sag: float = 0.18) -> QPainterPath:
    """Soft bezier between two points; sag controls how much the curve bows."""
    p = QPainterPath()
    p.moveTo(x1, y1)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy) or 1
    nx, ny = -dy / length, dx / length
    cx, cy = mx + nx * length * sag, my + ny * length * sag
    p.quadTo(cx, cy, x2, y2)
    return p


# ════════════════════════════════════════════════════════════════════════════
#  Connection — a thin curved line between two nodes that can change state
# ════════════════════════════════════════════════════════════════════════════
class _Edge(QGraphicsPathItem):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setZValue(-5)
        self._state = 'idle'
        self.set_state('idle')

    def set_path_between(self, x1, y1, x2, y2):
        self.setPath(_curve(x1, y1, x2, y2))

    def set_state(self, state: str):
        self._state = state
        if state == 'active':
            pen = QPen(QColor(LINE_ACTIVE), 1.2)
            pen.setCosmetic(True)
            self.setZValue(-3)
        elif state == 'done':
            pen = QPen(QColor(LINE_DONE), 0.8)
            pen.setCosmetic(True)
            self.setZValue(-5)
        elif state == 'dead':
            pen = QPen(QColor(LINE_IDLE), 0.6)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DotLine)
            self.setZValue(-6)
        else:  # idle
            pen = QPen(QColor(LINE_IDLE), 0.7)
            pen.setCosmetic(True)
            self.setZValue(-5)
        self.setPen(pen)


# ════════════════════════════════════════════════════════════════════════════
#  AgentNode — tiny dot with optional trust badge and active pulse
# ════════════════════════════════════════════════════════════════════════════
class _AgentNode(QGraphicsObject):
    clicked = pyqtSignal(str)

    def __init__(self, name: str, role: str, group_id: int,
                 region: str, rag_sources: list[str] | None = None, parent=None):
        super().__init__(parent)
        self._name = name
        self._role = role
        self._group_id = group_id
        self._region = region
        self._rag_sources = rag_sources or []
        self._state = 'idle'       # idle | active | done | dead
        self._pulse = 0.0           # 0..1 active pulse phase
        self._message = ''
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(5)

        # Pulse animation (only runs when state == 'active')
        self._anim = QVariantAnimation()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(900)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_pulse)

    def boundingRect(self) -> QRectF:
        # Tall enough to include role label below
        r = R_AGENT_DOT + 6
        return QRectF(-r - 22, -r - 12, (r + 22) * 2, (r + 12) * 2 + 12)

    def _on_pulse(self, v):
        self._pulse = float(v)
        self.update()

    @property
    def _role_abbrev(self) -> str:
        """Short uppercase role tag for the under-dot label."""
        r = self._role or ''
        if r == 'Forecaster':       return 'FCST'
        if r == 'DataExtractor':    return 'DATA'
        if r == 'Synthesizer':      return 'SYN'
        if r == 'Critic':           return 'CRIT'
        if r == 'ConfidenceScorer': return 'CONF'
        return r[:4].upper()

    def set_state(self, state: str):
        if self._state == state:
            return
        self._state = state
        if state == 'active':
            self._anim.start()
        else:
            self._anim.stop()
            self._pulse = 0.0
        self.update()

    def set_message(self, message: str):
        self._message = message or ''

    def set_trust(self, trust: float, tier: str) -> None:
        self._trust_score = trust
        self._trust_tier = tier
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Outer glow during active pulse
        if self._state == 'active':
            glow_r = R_AGENT_DOT + 4 + self._pulse * 8
            glow_alpha = int(80 * (1.0 - self._pulse))
            glow = QColor(N_ACTIVE)
            glow.setAlpha(glow_alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(0, 0), glow_r, glow_r)

        # Dot
        color = {
            'idle':   N_IDLE,
            'active': N_ACTIVE,
            'done':   N_DONE,
            'dead':   N_DEAD,
        }.get(self._state, N_IDLE)

        # Dead state: dashed thin ring instead of filled dot
        if self._state == 'dead':
            pen = QPen(QColor(color), 1.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT, R_AGENT_DOT)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT, R_AGENT_DOT)
            # White inner highlight for filled dots — gives the MiroFish "ring" feel
            inner = QColor(255, 255, 255, 220)
            painter.setBrush(QBrush(inner))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT - 2, R_AGENT_DOT - 2)
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT - 3, R_AGENT_DOT - 3)

        # Role abbreviation below the dot
        painter.setPen(QColor(TEXT_2))
        f_role = QFont('Consolas', 8)
        f_role.setBold(True)
        painter.setFont(f_role)
        painter.drawText(QRectF(-24, R_AGENT_DOT + 2, 48, 12),
                         Qt.AlignmentFlag.AlignCenter, self._role_abbrev)

        # Trust badge — tiny mono pill, top-right
        if hasattr(self, '_trust_tier'):
            self._draw_trust_badge(painter)

    def _draw_trust_badge(self, painter: QPainter):
        tier = self._trust_tier
        trust = self._trust_score
        if tier == 'promoted':
            text_c, bg_c, border_c = T_PROMOTED
            indicator = '▲'
        elif tier == 'demoted':
            text_c, bg_c, border_c = T_DEMOTED
            indicator = '▼'
        else:
            text_c, bg_c, border_c = T_DEFAULT
            indicator = '●'

        bw, bh = 30, 11
        bx, by = R_AGENT_DOT + 1, -bh / 2 - R_AGENT_DOT
        painter.save()
        painter.setBrush(QBrush(QColor(bg_c)))
        pen = QPen(QColor(border_c), 0.6)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(QRectF(bx, by, bw, bh), 3, 3)
        painter.setPen(QColor(text_c))
        font = QFont('Consolas', 6)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(bx, by, bw, bh),
                         Qt.AlignmentFlag.AlignCenter, f'{indicator}{trust:.2f}')
        painter.restore()

    def hoverEnterEvent(self, event):
        self.setScale(1.4)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self._name)
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  RegionalNode — medium dark diamond
# ════════════════════════════════════════════════════════════════════════════
class _RegionalNode(QGraphicsObject):
    clicked = pyqtSignal(int)

    def __init__(self, judge_id: int, region: str, parent=None):
        super().__init__(parent)
        self._judge_id = judge_id
        self._region = region
        self._state = 'idle'
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(7)

    def boundingRect(self) -> QRectF:
        # Wide enough for the longest region label ('CENTRAL LUZON' etc.) and
        # tall enough for the label area below the diamond. Extra padding so
        # the hover-scale (1.15x) doesn't clip anything.
        return QRectF(-90, -R_REGIONAL - 6, 180, R_REGIONAL + 32)

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._state == 'active':
            fill, border = N_ACTIVE, N_ACTIVE
        elif self._state == 'done':
            fill, border = N_DONE, N_DONE
        else:
            fill, border = '#FFFFFF', N_REGIONAL

        # Rotated square (diamond)
        painter.save()
        painter.rotate(45)
        side = R_REGIONAL * 1.4
        rect = QRectF(-side / 2, -side / 2, side, side)
        painter.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border), 1.6)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 2, 2)
        painter.restore()

        # Region label below
        painter.setPen(QColor(TEXT_1))
        f = QFont('Consolas', 9)
        f.setBold(True)
        painter.setFont(f)
        label_rect = QRectF(-88, R_REGIONAL + 4, 176, 14)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter,
                         self._region.upper()[:18])

    def hoverEnterEvent(self, event):
        self.setScale(1.15)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self._judge_id)
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  MasterNode — large black square
# ════════════════════════════════════════════════════════════════════════════
class _MasterNode(QGraphicsObject):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = 'idle'
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(10)

    def boundingRect(self) -> QRectF:
        # Wide enough for 'master judge' label and 1.08x hover scale.
        return QRectF(-80, -R_MASTER - 6, 160, R_MASTER + 28)

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._state == 'done':
            fill, border = N_DONE, N_DONE
        elif self._state == 'active':
            fill, border = N_ACTIVE, N_ACTIVE
        else:
            fill, border = N_MASTER, N_MASTER

        side = R_MASTER * 1.6
        rect = QRectF(-side / 2, -side / 2, side, side)
        painter.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 3, 3)

        # Inner "MJ" label in white
        painter.setPen(QColor('#FFFFFF'))
        f = QFont('Consolas', 11)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, 'MJ')

        # Compact subtitle below (smaller, less prominent)
        painter.setPen(QColor(TEXT_3))
        f2 = QFont('Consolas', 8)
        f2.setBold(False)
        painter.setFont(f2)
        painter.drawText(QRectF(-78, R_MASTER + 4, 156, 12),
                         Qt.AlignmentFlag.AlignCenter, 'master judge')

    def hoverEnterEvent(self, event):
        self.setScale(1.08)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  RagNode — tiny labeled chip in MiroFish top-row style
# ════════════════════════════════════════════════════════════════════════════
_RAG_SHORT: dict[str, str] = {
    'YahooFinanceCrude': 'CRUDE',
    'YahooFinanceForex': 'FOREX',
    'ManilaBulletin':    'MB',
    'neda_2024_2026':    'NEDA',
    'BusinessWorld':     'BW',
    'DOEBulletin':       'DOE',
    'PHRetailFuel':      'RTL',
    'PAGASAWeather':     'WX',
    'Inquirer':          'INQ',
    'GoogleNewsCrude':   'GNCR',
    'GoogleNewsForex':   'GNFX',
    'GoogleNewsBSP':     'GNBSP',
    'GoogleNewsRetail':  'GNRTL',
    'GoogleNewsCPI':     'GNCPI',
    'GoogleNewsTransport': 'GNTR',
    # Structured JSON APIs
    'OpenMeteoManila':   'WX7D',
    'WBPhilFood':        'WBFP',
    'EIAElectricity':    'EIA',
    # News-RSS feeds for food / electricity sectors
    'NFARiceRetail':     'NFA',
    'MeralcoCharge':     'MRLC',
    'WESMSpot':          'WESM',
}


class _RagNode(QGraphicsObject):
    clicked = pyqtSignal(str)

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self._source = source
        self._label = _RAG_SHORT.get(source, source[:5].upper())
        self._highlight = False
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(4)

    def boundingRect(self) -> QRectF:
        return QRectF(-34, -13, 68, 26)

    def set_highlight(self, on: bool):
        self._highlight = on
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bg = N_RAG if self._highlight else '#FFFFFF'
        text_c = '#FFFFFF' if self._highlight else N_RAG
        border = N_RAG
        rect = QRectF(-32, -11, 64, 22)
        painter.setBrush(QBrush(QColor(bg)))
        pen = QPen(QColor(border), 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 4, 4)
        painter.setPen(QColor(text_c))
        f = QFont('Consolas', 9)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._label)

    def hoverEnterEvent(self, event):
        self.setScale(1.1)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self._source)
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  EvidenceNode — tiny faint satellite dot = one real retrieved RAG chunk
# ════════════════════════════════════════════════════════════════════════════
class _EvidenceNode(QGraphicsObject):
    """A small soft-blue dot = one real retrieved RAG chunk. Hover grows it
    (animated, like the other nodes); click -> source + text."""
    clicked = pyqtSignal(str, str)            # (source, text)
    _R = 4.0
    _FILL = '#93A4C4'                          # soft slate-blue = "evidence"
    _RING = '#5C6E94'                          # slightly darker edge for definition

    def __init__(self, source: str, text: str, parent=None):
        super().__init__(parent)
        self._source = source or '?'
        self._text = text or ''
        self.setZValue(1)                     # below agents (z=5), above edges
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f'{self._source}: {self._text[:80]}')
        # animated hover scale (matches the grow-on-hover of the other nodes)
        self._scale_anim = QPropertyAnimation(self, b'scale')
        self._scale_anim.setDuration(140)
        self._scale_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def boundingRect(self) -> QRectF:
        r = self._R + 1.5
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(self._RING), 0.8))
        p.setBrush(QBrush(QColor(self._FILL)))
        p.drawEllipse(QRectF(-self._R, -self._R, 2 * self._R, 2 * self._R))

    def _animate_scale(self, target: float):
        self._scale_anim.stop()
        self._scale_anim.setStartValue(self.scale())
        self._scale_anim.setEndValue(target)
        self._scale_anim.start()

    def hoverEnterEvent(self, event):
        self._animate_scale(1.9)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._animate_scale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, ev):
        self.clicked.emit(self._source, self._text)
        ev.accept()


# ════════════════════════════════════════════════════════════════════════════
#  SectorAgentNode — dot for food/electricity agents
# ════════════════════════════════════════════════════════════════════════════
_SECTOR_ROLE_ABBREV = {
    'Agri Analyst':         'AGRI',
    'Supply Chain Expert':  'SUPP',
    'Weather Interpreter':  'WTHR',
    'Trade Policy Critic':  'TRDE',
    'Energy Economist':     'ECON',
    'Grid Analyst':         'GRID',
    'Regulatory Expert':    'REGL',
    'Demand Forecaster':    'DMND',
}


class _SectorAgentNode(QGraphicsObject):
    """Same visual language as _AgentNode — filled dot, white inner highlight,
    role abbreviation below, pulse on active. The only differences from
    _AgentNode: sector accent color when done, no per-agent rag_sources list."""
    clicked = pyqtSignal(str)

    def __init__(self, name: str, sector: str, parent=None):
        super().__init__(parent)
        self._name = name
        self._sector = sector              # 'food' | 'elec'
        self._state = 'idle'
        self._pulse = 0.0
        self._estimate: Optional[float] = None
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(5)

        # Active-state pulse animation
        self._anim = QVariantAnimation()
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(900)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_pulse)

    def boundingRect(self) -> QRectF:
        # Match _AgentNode bounding so role label fits below the dot
        r = R_AGENT_DOT + 6
        return QRectF(-r - 22, -r - 12, (r + 22) * 2, (r + 12) * 2 + 12)

    def _on_pulse(self, v):
        self._pulse = float(v)
        self.update()

    @property
    def _role_abbrev(self) -> str:
        return _SECTOR_ROLE_ABBREV.get(self._name, self._name[:4].upper())

    def set_state(self, state: str, estimate: Optional[float] = None):
        if estimate is not None:
            self._estimate = estimate
        if self._state == state:
            return
        self._state = state
        if state == 'active':
            self._anim.start()
        else:
            self._anim.stop()
            self._pulse = 0.0
        self.update()

    def set_trust(self, trust: float, tier: str) -> None:
        self._trust_score = trust
        self._trust_tier = tier
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = N_FOOD if self._sector == 'food' else N_ELEC

        # Active pulse glow (red, like _AgentNode)
        if self._state == 'active':
            glow_r = R_AGENT_DOT + 4 + self._pulse * 8
            glow_alpha = int(80 * (1.0 - self._pulse))
            glow = QColor(N_ACTIVE)
            glow.setAlpha(glow_alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(0, 0), glow_r, glow_r)

        # Dot color by state — idle uses sector accent so cluster identity stays
        if self._state == 'active':
            color = N_ACTIVE
        elif self._state == 'done':
            color = accent           # green for food / amber for elec
        elif self._state == 'dead':
            color = N_DEAD
        else:
            color = accent           # idle dots take the sector tint

        if self._state == 'dead':
            pen = QPen(QColor(color), 1.0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT, R_AGENT_DOT)
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT, R_AGENT_DOT)
            inner = QColor(255, 255, 255, 220)
            painter.setBrush(QBrush(inner))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT - 2, R_AGENT_DOT - 2)
            painter.setBrush(QBrush(QColor(color)))
            painter.drawEllipse(QPointF(0, 0), R_AGENT_DOT - 3, R_AGENT_DOT - 3)

        # Role abbreviation below dot
        painter.setPen(QColor(TEXT_2))
        f_role = QFont('Consolas', 8)
        f_role.setBold(True)
        painter.setFont(f_role)
        painter.drawText(QRectF(-24, R_AGENT_DOT + 2, 48, 12),
                         Qt.AlignmentFlag.AlignCenter, self._role_abbrev)

        # Trust badge (same as _AgentNode)
        if hasattr(self, '_trust_tier'):
            _AgentNode._draw_trust_badge(self, painter)

    def hoverEnterEvent(self, event):
        self.setScale(1.4)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self._name)
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  SectorVerdictNode — small labeled square for food/elec verdicts
# ════════════════════════════════════════════════════════════════════════════
class _SectorVerdictNode(QGraphicsObject):
    """Sector verdict — a diamond (rotated square) matching the regional judge
    style, with sector name + (optional) estimate label below."""
    clicked = pyqtSignal(str)

    def __init__(self, sector: str, parent=None):
        super().__init__(parent)
        self._sector = sector
        self._state = 'idle'
        self._estimate: str = '—'
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(8)

    def boundingRect(self) -> QRectF:
        # Wide enough for 'ELECTRICITY' label + estimate row + hover scale.
        return QRectF(-90, -R_REGIONAL - 6, 180, R_REGIONAL + 50)

    def set_complete(self, estimate: str):
        self._state = 'done'
        self._estimate = estimate
        self.update()

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        accent = N_FOOD if self._sector == 'food' else N_ELEC

        if self._state == 'active':
            fill, border = N_ACTIVE, N_ACTIVE
        elif self._state == 'done':
            fill, border = accent, accent
        else:
            fill, border = '#FFFFFF', accent

        # Rotated square (diamond), same construction as _RegionalNode
        painter.save()
        painter.rotate(45)
        side = R_REGIONAL * 1.4
        rect = QRectF(-side / 2, -side / 2, side, side)
        painter.setBrush(QBrush(QColor(fill)))
        pen = QPen(QColor(border), 1.6)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 2, 2)
        painter.restore()

        # Sector label below the diamond — bold, sector accent color
        label = 'FOOD' if self._sector == 'food' else 'ELECTRICITY'
        painter.setPen(QColor(accent))
        f = QFont('Consolas', 9)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(QRectF(-88, R_REGIONAL + 4, 176, 14),
                         Qt.AlignmentFlag.AlignCenter, label)

        # Compact estimate below the sector label, only when set
        if self._estimate and self._estimate != '—':
            painter.setPen(QColor(TEXT_2))
            f2 = QFont('Consolas', 8)
            f2.setBold(False)
            painter.setFont(f2)
            painter.drawText(QRectF(-88, R_REGIONAL + 20, 176, 14),
                             Qt.AlignmentFlag.AlignCenter, self._estimate)

    def hoverEnterEvent(self, event):
        self.setScale(1.15)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setScale(1.0)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit(self._sector)
        super().mousePressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
#  Region label callout (tiny floating text in upper-left of a cluster)
# ════════════════════════════════════════════════════════════════════════════
class _ClusterLabel(QGraphicsObject):
    def __init__(self, text: str, color: str = TEXT_2, parent=None):
        super().__init__(parent)
        self._text = text.upper()
        self._color = color
        self.setZValue(2)

    def boundingRect(self) -> QRectF:
        return QRectF(-80, -10, 160, 20)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QColor(self._color))
        f = QFont('Consolas', 10)
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(self.boundingRect(),
                         Qt.AlignmentFlag.AlignCenter, self._text)


# ════════════════════════════════════════════════════════════════════════════
#  SwarmCanvas — QGraphicsView holding all nodes + edges
# ════════════════════════════════════════════════════════════════════════════
class _SwarmCanvas(QGraphicsView):
    node_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._scene.setSceneRect(-HW, -HH, SCENE_W, SCENE_H)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QBrush(QColor(BG)))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._round_counter = 0  # increments each time we observe a new round

        # Indices for fast lookup
        self._agents: dict[str, _AgentNode] = {}
        self._edges_agent_master: dict[str, _Edge] = {}     # agent_name -> edge to master
        self._edges_agent_regional: dict[str, _Edge] = {}   # agent_name -> edge to regional
        self._edges_agent_rag: dict[str, list[_Edge]] = {}  # agent_name -> list of edges to RAG nodes
        self._regionals: list[_RegionalNode] = []
        self._master: Optional[_MasterNode] = None
        self._rag_nodes: dict[str, _RagNode] = {}
        self._sector_agents: dict[str, _SectorAgentNode] = {}
        self._food_verdict: Optional[_SectorVerdictNode] = None
        self._elec_verdict: Optional[_SectorVerdictNode] = None
        self._food_edges: dict[str, _Edge] = {}
        self._elec_edges: dict[str, _Edge] = {}
        self._responses: dict[str, str] = {}     # last response text per agent
        self._food_typing: set[str] = set()
        self._elec_typing: set[str] = set()

        self._build()

    def _build(self):
        # ─── Cluster boundary outlines (neutral gray, dashed) ─────────────────
        # No pastel fills — just subtle gray rings. Each cluster gets a small
        # ID code (just the hash) since the region name is shown under the
        # diamond judge node already.
        cluster_keys = ['ncr', 'luz', 'vis', 'dav']
        for gid in range(len(REGIONS)):
            boundary_pen = QPen(QColor(HALO_LINE), 0.6, Qt.PenStyle.DashLine)
            boundary_pen.setCosmetic(True)
            outline = self._scene.addEllipse(
                A_REGIONS[gid][0] - 150, A_REGIONS[gid][1] - 118, 300, 236,
                boundary_pen, QBrush(QColor(HALO_BG)))
            outline.setZValue(-18)

            # Just the cluster hash code — small, in corner of cluster
            cluster_id = CLUSTER_IDS[cluster_keys[gid]]
            cx, cy = A_REGIONS[gid]
            self._add_cluster_tag(cx - 145, cy - 110, cluster_id)

        # Food + Elec cluster outlines (same neutral style, with own tag)
        for sector_key, anchor in (('food', A_FOOD), ('elec', A_ELEC)):
            pen = QPen(QColor(HALO_LINE), 0.6, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            outline = self._scene.addEllipse(
                anchor[0] - 105, anchor[1] - 90, 210, 180,
                pen, QBrush(QColor(HALO_BG)))
            outline.setZValue(-18)
            self._add_cluster_tag(anchor[0] - 100, anchor[1] - 83,
                                   CLUSTER_IDS[sector_key])

        # ─── Axis crosshair (very subtle, through origin) ─────────────────────
        ax_pen = QPen(QColor(HALO_LINE_DARK), 0.5, Qt.PenStyle.DotLine)
        ax_pen.setCosmetic(True)
        h_axis = self._scene.addLine(-HW, 0, HW, 0, ax_pen)
        h_axis.setZValue(-19)
        v_axis = self._scene.addLine(0, -HH, 0, HH, ax_pen)
        v_axis.setZValue(-19)

        # ─── Ambient noise particles (simulates dense network) ────────────────
        noise_rnd = random.Random(42)
        for _ in range(NOISE_COUNT):
            nx = noise_rnd.uniform(-HW + 40, HW - 40)
            ny = noise_rnd.uniform(-HH + 60, HH - 60)
            # Skip points too close to actual nodes
            tx, ty = nx, ny
            color = NOISE_DOT_RED if noise_rnd.random() < 0.08 else NOISE_DOT
            dot = self._scene.addEllipse(
                tx - NOISE_DOT_SIZE, ty - NOISE_DOT_SIZE,
                NOISE_DOT_SIZE * 2, NOISE_DOT_SIZE * 2,
                QPen(Qt.PenStyle.NoPen), QBrush(QColor(color)))
            dot.setZValue(-15)

        # ─── Master at center ─────────────────────────────────────────────────
        self._master = _MasterNode()
        self._master.setPos(*A_MASTER)
        self._master.clicked.connect(lambda: self._emit_master_click())
        self._scene.addItem(self._master)

        # ─── 4 regional judges ────────────────────────────────────────────────
        # REGION_PAIRS = [(0,1), (2,3)] → only 2 judges in actual code, but visualize 4
        weight_font = QFont('Consolas', 8)
        weight_font.setBold(True)
        for gid in range(len(REGIONS)):
            r = _RegionalNode(gid, REGIONS[gid])
            r.setPos(*A_REGIONS[gid])
            r.clicked.connect(self._emit_regional_click)
            self._scene.addItem(r)
            self._regionals.append(r)
            # Edge from regional to master
            edge = _Edge()
            edge.set_path_between(*A_REGIONS[gid], *A_MASTER)
            self._scene.addItem(edge)
            # Edge weight label (midpoint, tiny mono)
            mx, my = (A_REGIONS[gid][0] + A_MASTER[0]) / 2, (A_REGIONS[gid][1] + A_MASTER[1]) / 2
            weights = ['0.34', '0.27', '0.19', '0.20']
            wl = self._scene.addText(f'w={weights[gid]}', weight_font)
            wl.setDefaultTextColor(QColor(TEXT_3))
            wl.setPos(mx - 14, my - 12)
            wl.setZValue(-2)

        # ─── 20 swarm agents — 5 per region, scattered around their judge ─────
        agents = build_swarm_agents()
        for ag in agents:
            anchor = A_REGIONS[ag.group_id]
            x, y = _scatter(ag.name, anchor, AGENT_SCATTER, min_r=30)
            node = _AgentNode(ag.name, ag.role, ag.group_id,
                              ag.region_name, ag.rag_sources)
            node.setPos(x, y)
            node.clicked.connect(self._emit_agent_click)
            self._scene.addItem(node)
            self._agents[ag.name] = node
            # Connection to regional judge
            e_reg = _Edge()
            e_reg.set_path_between(x, y, *A_REGIONS[ag.group_id])
            self._scene.addItem(e_reg)
            self._edges_agent_regional[ag.name] = e_reg

        # ─── RAG sources — wrapped two-row band near top ──────────────────────
        # Build a unique source list — gas swarm (_ROLE_RAG) + food/electricity
        # sector agents (their per-agent rag_sources). Order: gas first, then any
        # new sector-only sources appended.
        from ph_economic_ai.engine.debate import FOOD_AGENTS, ELECTRICITY_AGENTS as _ELEC_AGENTS_FOR_RAG
        all_sources: list[str] = []
        for s_list in _ROLE_RAG.values():
            for s in s_list:
                if s not in all_sources:
                    all_sources.append(s)
        for ag in list(FOOD_AGENTS) + list(_ELEC_AGENTS_FOR_RAG):
            for s in (ag.rag_sources or []):
                if s not in all_sources:
                    all_sources.append(s)

        # Wrap into 2 rows if more than 8 sources — keeps row width manageable
        # so the rightmost chips don't slide under the top-right corner card.
        rag_positions: dict[str, tuple[float, float]] = {}
        per_row = max(1, math.ceil(len(all_sources) / 2)) if len(all_sources) > 8 else len(all_sources)
        row_count = math.ceil(len(all_sources) / per_row)
        for i, src in enumerate(all_sources):
            row_idx = i // per_row
            col_idx = i % per_row
            t = (col_idx / max(1, per_row - 1)) - 0.5    # -0.5 to 0.5
            base_x = t * (per_row * RAG_SPACING)
            base_y = A_RAG_ROW[1] + row_idx * 36         # 36px between rows
            jitter = _scatter(src, (0, 0), 12, min_r=0)
            x, y = base_x + jitter[0], base_y + jitter[1]
            rag_positions[src] = (x, y)
            rn = _RagNode(src)
            rn.setPos(x, y)
            rn.clicked.connect(self._emit_rag_click)
            self._scene.addItem(rn)
            self._rag_nodes[src] = rn

        # RAG row header label — sits above the first row of chips
        rag_label_font = QFont('Consolas', 10)
        rag_label_font.setBold(True)
        rag_hdr_text = self._scene.addText('RAG SOURCES', rag_label_font)
        rag_hdr_text.setDefaultTextColor(QColor(TEXT_1))
        if all_sources:
            first_x, first_y = rag_positions[all_sources[0]]
            rag_hdr_text.setPos(first_x - 30, first_y - 36)
        rag_hdr_text.setZValue(2)

        # ─── Agent → RAG faint edges (ALL RAG sources per agent) ──────────────
        # All start as 'dead' style (very faint dotted) to keep clutter low.
        for ag in agents:
            edges_for_agent: list[_Edge] = []
            ax, ay = self._agents[ag.name].pos().x(), self._agents[ag.name].pos().y()
            for src in ag.rag_sources:   # ALL RAG sources, not just first
                if src not in rag_positions:
                    continue
                rx, ry = rag_positions[src]
                rag_edge = _Edge()
                rag_edge.set_path_between(ax, ay, rx, ry)
                rag_edge.set_state('dead')   # uses dotted style — very faint
                self._scene.addItem(rag_edge)
                edges_for_agent.append(rag_edge)
            self._edges_agent_rag[ag.name] = edges_for_agent

        # ─── Intra-region FULL MESH (every agent connects to every other) ─────
        # Replaces the ring topology with a complete graph per region.
        agents_by_group: dict[int, list[_AgentNode]] = {}
        for ag_node in self._agents.values():
            agents_by_group.setdefault(ag_node._group_id, []).append(ag_node)
        for gid, group_nodes in agents_by_group.items():
            for i in range(len(group_nodes)):
                for j in range(i + 1, len(group_nodes)):
                    a, b = group_nodes[i], group_nodes[j]
                    edge = _Edge()
                    edge.set_path_between(a.pos().x(), a.pos().y(),
                                          b.pos().x(), b.pos().y())
                    edge.set_state('dead')
                    self._scene.addItem(edge)

        # ─── Cross-region agent bridges (deterministic random sampling) ───────
        # Each agent gets 1-2 connections to agents in OTHER regions.
        all_agents_list = list(self._agents.values())
        bridge_rnd = random.Random(7331)
        for ag_node in all_agents_list:
            other_agents = [a for a in all_agents_list
                            if a._group_id != ag_node._group_id]
            if not other_agents:
                continue
            # 1-2 cross-region neighbors per agent
            num_bridges = bridge_rnd.choice([1, 2])
            partners = bridge_rnd.sample(other_agents, min(num_bridges, len(other_agents)))
            for partner in partners:
                e = _Edge()
                e.set_path_between(ag_node.pos().x(), ag_node.pos().y(),
                                   partner.pos().x(), partner.pos().y())
                e.set_state('dead')
                self._scene.addItem(e)

        # ─── Cross-region judge mesh ──────────────────────────────────────────
        for i in range(len(A_REGIONS)):
            for j in range(i + 1, len(A_REGIONS)):
                e = _Edge()
                e.set_path_between(*A_REGIONS[i], *A_REGIONS[j])
                e.set_state('idle')
                self._scene.addItem(e)

        # ─── RAG row interconnections (each RAG ↔ next 2 RAGs along the row) ─
        for i, src_a in enumerate(all_sources):
            for j in (i + 1, i + 2):
                if j >= len(all_sources):
                    continue
                src_b = all_sources[j]
                e = _Edge()
                e.set_path_between(*rag_positions[src_a],
                                   *rag_positions[src_b])
                e.set_state('dead')
                self._scene.addItem(e)

        # ─── Cross-cluster sector → all regional judges ───────────────────────
        # Both sector verdicts link to all 4 regional judges (not just one each)
        for gid in range(len(A_REGIONS)):
            for anchor in (A_FOOD, A_ELEC):
                e = _Edge()
                e.set_path_between(*anchor, *A_REGIONS[gid])
                e.set_state('dead')
                self._scene.addItem(e)

        # ─── Food cluster (left side) ─────────────────────────────────────────
        from ph_economic_ai.engine.debate import FOOD_AGENTS, ELECTRICITY_AGENTS
        self._food_verdict = _SectorVerdictNode('food')
        self._food_verdict.setPos(*A_FOOD)
        self._food_verdict.clicked.connect(self._emit_sector_verdict_click)
        self._scene.addItem(self._food_verdict)

        sector_rnd = random.Random(91)
        rag_list = list(rag_positions.items())
        for ag in FOOD_AGENTS:
            x, y = _scatter(ag.name, A_FOOD, SECTOR_SCATTER, min_r=30)
            node = _SectorAgentNode(ag.name, 'food')
            node.setPos(x, y)
            node.clicked.connect(self._emit_sector_agent_click)
            self._scene.addItem(node)
            self._sector_agents[ag.name] = node
            edge = _Edge()
            edge.set_path_between(x, y, *A_FOOD)
            self._scene.addItem(edge)
            self._food_edges[ag.name] = edge
            # 2 random RAG links per food agent
            for src, (rx, ry) in sector_rnd.sample(rag_list, min(2, len(rag_list))):
                re = _Edge()
                re.set_path_between(x, y, rx, ry)
                re.set_state('dead')
                self._scene.addItem(re)

        # Edge from food verdict to master
        food_to_master = _Edge()
        food_to_master.set_path_between(*A_FOOD, *A_MASTER)
        self._scene.addItem(food_to_master)
        self._food_to_master_edge = food_to_master

        # ─── Electricity cluster (right side) ─────────────────────────────────
        self._elec_verdict = _SectorVerdictNode('elec')
        self._elec_verdict.setPos(*A_ELEC)
        self._elec_verdict.clicked.connect(self._emit_sector_verdict_click)
        self._scene.addItem(self._elec_verdict)

        for ag in ELECTRICITY_AGENTS:
            x, y = _scatter(ag.name, A_ELEC, SECTOR_SCATTER, min_r=30)
            node = _SectorAgentNode(ag.name, 'elec')
            node.setPos(x, y)
            node.clicked.connect(self._emit_sector_agent_click)
            self._scene.addItem(node)
            self._sector_agents[ag.name] = node
            edge = _Edge()
            edge.set_path_between(x, y, *A_ELEC)
            self._scene.addItem(edge)
            self._elec_edges[ag.name] = edge
            # 2 random RAG links per elec agent
            for src, (rx, ry) in sector_rnd.sample(rag_list, min(2, len(rag_list))):
                re = _Edge()
                re.set_path_between(x, y, rx, ry)
                re.set_state('dead')
                self._scene.addItem(re)

        elec_to_master = _Edge()
        elec_to_master.set_path_between(*A_ELEC, *A_MASTER)
        self._scene.addItem(elec_to_master)
        self._elec_to_master_edge = elec_to_master

        # (Region cluster labels removed — region name already appears
        #  under each regional diamond. Eliminating duplicate text.)

    # ── Cluster tag helper (single-line hash code in corner of cluster) ──────
    def _add_cluster_tag(self, x: float, y: float, cluster_id: str):
        f = QFont('Consolas', 9)
        t = self._scene.addText(cluster_id, f)
        t.setDefaultTextColor(QColor(TEXT_3))
        t.setPos(x, y)
        t.setZValue(-10)

    # ── Click emitters ────────────────────────────────────────────────────────
    def _emit_agent_click(self, name: str):
        node = self._agents.get(name)
        if not node:
            return
        self.node_clicked.emit({
            'type': 'agent',
            'name': name,
            'role': node._role,
            'group_id': node._group_id,
            'region': node._region,
            'status': node._state,
            'rag': node._rag_sources,
            'message': self._responses.get(name, node._message),
            'color': N_IDLE,
        })

    def _emit_regional_click(self, judge_id: int):
        if judge_id >= len(self._regionals):
            return
        node = self._regionals[judge_id]
        self.node_clicked.emit({
            'type': 'regional',
            'judge_id': judge_id,
            'region': node._region,
            'status': node._state,
        })

    def _emit_master_click(self):
        if self._master is None:
            return
        self.node_clicked.emit({
            'type': 'master',
            'status': self._master._state,
        })

    def _emit_rag_click(self, source: str):
        self.node_clicked.emit({
            'type': 'rag',
            'source': source,
        })

    def _emit_evidence_click(self, source: str, text: str):
        self.node_clicked.emit({
            'kind': 'evidence',
            'label': source,
            'payload': {'source': source, 'text': text},
        })

    def add_evidence_layer(self, rag, scenario: dict, top_k: int = 3):
        """Hang each agent's REAL retrieved chunks off it as satellite dots.
        Re-callable: clears any prior evidence first. Guarded — never raises."""
        from ph_economic_ai.engine.kg_swarm_adapter import _scenario_text
        # clear prior evidence (idempotent)
        for it in getattr(self, '_evidence_items', []):
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._evidence_items = []
        if rag is None:
            return
        try:
            text = _scenario_text(scenario or {})
        except Exception:
            text = ''
        for node in list(self._agents.values()):
            try:
                chunks = rag.query(text, top_k=top_k,
                                   sources=getattr(node, '_rag_sources', None)) or []
            except Exception:
                continue
            ax, ay = node.pos().x(), node.pos().y()
            seen = set()
            n = len(chunks)
            for i, c in enumerate(chunks):
                src, txt = c.get('source', '?'), c.get('text', '')
                if (src, txt) in seen:
                    continue
                seen.add((src, txt))
                ang = (2 * math.pi * i / max(n, 1)) - math.pi / 2
                ex, ey = ax + 26.0 * math.cos(ang), ay + 26.0 * math.sin(ang)
                edge = _Edge()
                edge.set_path_between(ax, ay, ex, ey)
                edge.set_state('dead')                 # faint dotted line
                self._scene.addItem(edge)
                self._evidence_items.append(edge)
                ev = _EvidenceNode(src, txt)
                ev.setPos(ex, ey)
                ev.clicked.connect(self._emit_evidence_click)
                self._scene.addItem(ev)
                self._evidence_items.append(ev)

    def _emit_sector_agent_click(self, name: str):
        node = self._sector_agents.get(name)
        if not node:
            return
        self.node_clicked.emit({
            'type': 'sector_agent',
            'name': name,
            'sector': node._sector,
            'status': node._state,
            'estimate': node._estimate,
            'message': self._responses.get(name, ''),
        })

    def _emit_sector_verdict_click(self, sector: str):
        verdict_node = self._food_verdict if sector == 'food' else self._elec_verdict
        self.node_clicked.emit({
            'type': 'sector_verdict',
            'sector': sector,
            'status': verdict_node._state if verdict_node else 'idle',
            'estimate': verdict_node._estimate if verdict_node else '—',
        })

    # ── State updates (called from Stage3SwarmPanel slots) ────────────────────
    def mark_active(self, agent_name: str):
        node = self._agents.get(agent_name)
        if node:
            node.set_state('active')
            edge = self._edges_agent_regional.get(agent_name)
            if edge:
                edge.set_state('active')
            # Light up agent → RAG edges
            for rag_edge in self._edges_agent_rag.get(agent_name, []):
                rag_edge.set_state('active')
            # Highlight RAG sources used by this agent
            for src in node._rag_sources:
                rn = self._rag_nodes.get(src)
                if rn:
                    rn.set_highlight(True)

    def mark_idle(self, agent_name: str):
        node = self._agents.get(agent_name)
        if node and node._state == 'active':
            node.set_state('idle')
            edge = self._edges_agent_regional.get(agent_name)
            if edge:
                edge.set_state('idle')
            for rag_edge in self._edges_agent_rag.get(agent_name, []):
                rag_edge.set_state('dead')
            for src in node._rag_sources:
                rn = self._rag_nodes.get(src)
                if rn:
                    rn.set_highlight(False)

    def mark_eliminated(self, agent_name: str):
        node = self._agents.get(agent_name)
        if node:
            node.set_state('dead')
            edge = self._edges_agent_regional.get(agent_name)
            if edge:
                edge.set_state('dead')
            for rag_edge in self._edges_agent_rag.get(agent_name, []):
                rag_edge.set_state('dead')

    def mark_survivor(self, agent_name: str, group_id: int):
        node = self._agents.get(agent_name)
        if node:
            node.set_state('done')
            edge = self._edges_agent_regional.get(agent_name)
            if edge:
                edge.set_state('done')
            for rag_edge in self._edges_agent_rag.get(agent_name, []):
                rag_edge.set_state('done')

    def mark_regional_active(self, judge_id: int):
        if judge_id < len(self._regionals):
            self._regionals[judge_id].set_state('active')

    def mark_regional_done(self, judge_id: int):
        if judge_id < len(self._regionals):
            self._regionals[judge_id].set_state('done')

    def mark_master_done(self):
        if self._master:
            self._master.set_state('done')

    def store_response(self, agent_name: str, statement: str):
        self._responses[agent_name] = statement

    # ── Sector helpers ────────────────────────────────────────────────────────
    def mark_sector_agent_typing(self, name: str, sector: str):
        node = self._sector_agents.get(name)
        if node:
            node.set_state('active')
            edge = (self._food_edges if sector == 'food' else self._elec_edges).get(name)
            if edge:
                edge.set_state('active')
            (self._food_typing if sector == 'food' else self._elec_typing).add(name)

    def mark_sector_agent_done(self, name: str, sector: str,
                               estimate: Optional[float]):
        node = self._sector_agents.get(name)
        if node:
            node.set_state('done', estimate)
            edge = (self._food_edges if sector == 'food' else self._elec_edges).get(name)
            if edge:
                edge.set_state('done')
            (self._food_typing if sector == 'food' else self._elec_typing).discard(name)

    def mark_sector_complete(self, sector: str, estimate_str: str):
        if sector == 'food' and self._food_verdict:
            self._food_verdict.set_complete(estimate_str)
            self._food_to_master_edge.set_state('done')
        elif sector == 'elec' and self._elec_verdict:
            self._elec_verdict.set_complete(estimate_str)
            self._elec_to_master_edge.set_state('done')

    def reset(self):
        for node in self._agents.values():
            node.set_state('idle')
        for edge in self._edges_agent_regional.values():
            edge.set_state('idle')
        for edge_list in self._edges_agent_rag.values():
            for e in edge_list:
                e.set_state('dead')
        for r in self._regionals:
            r.set_state('idle')
        if self._master:
            self._master.set_state('idle')
        for rn in self._rag_nodes.values():
            rn.set_highlight(False)
        for node in self._sector_agents.values():
            node.set_state('idle')
        for edge in self._food_edges.values():
            edge.set_state('idle')
        for edge in self._elec_edges.values():
            edge.set_state('idle')
        if self._food_verdict:
            self._food_verdict._state = 'idle'
            self._food_verdict._estimate = '—'
            self._food_verdict.update()
        if self._elec_verdict:
            self._elec_verdict._state = 'idle'
            self._elec_verdict._estimate = '—'
            self._elec_verdict.update()
        self._food_to_master_edge.set_state('idle')
        self._elec_to_master_edge.set_state('idle')
        self._food_typing.clear()
        self._elec_typing.clear()
        self._responses.clear()

    # ── Fit view ──────────────────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # ── Dotted background ─────────────────────────────────────────────────────
    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(rect, QColor(BG))

        # Background dot grid
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(DOT_BG)))
        left = int(rect.left()) - (int(rect.left()) % DOT_SPACING)
        top  = int(rect.top())  - (int(rect.top())  % DOT_SPACING)
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawEllipse(QPointF(x, y), DOT_SIZE, DOT_SIZE)
                y += DOT_SPACING
            x += DOT_SPACING

    # ── Foreground overlay: corner UI drawn in viewport pixels ─────────────────
    # This avoids being scaled by fitInView — text stays sharp and at consistent
    # readable sizes regardless of scene zoom.
    def drawForeground(self, painter: QPainter, rect: QRectF):
        super().drawForeground(painter, rect)
        painter.save()
        painter.resetTransform()      # switch from scene coords → viewport pixels
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        vw = self.viewport().width()
        vh = self.viewport().height()

        f_title = QFont('Consolas', 10); f_title.setBold(True)
        f_lbl   = QFont('Consolas', 9);  f_lbl.setBold(True)
        f_meta  = QFont('Consolas', 9)
        f_small = QFont('Consolas', 8)

        # Card helper — soft shadow + white fill + thin border
        def _card(x, y, w, h):
            # Subtle shadow
            shadow = QColor(0, 0, 0, 16)
            painter.setBrush(QBrush(shadow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(x + 2, y + 3, w, h), 8, 8)
            # White card
            painter.setBrush(QBrush(QColor('#FFFFFF')))
            painter.setPen(QPen(QColor(DIVIDER), 1))
            painter.drawRoundedRect(QRectF(x, y, w, h), 8, 8)

        # ─── Top-left card: title + metadata ─────────────────────────────────
        TL_X, TL_Y, TL_W, TL_H = 12, 12, 268, 124
        _card(TL_X, TL_Y, TL_W, TL_H)

        painter.setPen(QColor(TEXT_1))
        painter.setFont(f_title)
        painter.drawText(QPointF(TL_X + 14, TL_Y + 22), 'GRAPH VISUALIZATION')

        meta_rows = [
            ('LAYOUT',   'force-directed-v2'),
            ('NODES',    '37'),
            ('EDGES',    '110+'),
            ('CLUSTERS', '6'),
        ]
        for i, (k, v) in enumerate(meta_rows):
            row_y = TL_Y + 48 + i * 17
            painter.setPen(QColor(TEXT_3))
            painter.setFont(f_lbl)
            painter.drawText(QPointF(TL_X + 14, row_y), k)
            painter.setPen(QColor(TEXT_2))
            painter.setFont(f_meta)
            painter.drawText(QPointF(TL_X + 100, row_y), v)

        # ─── Top-right card: session info ────────────────────────────────────
        TR_W, TR_H = 280, 124
        TR_X, TR_Y = vw - TR_W - 12, 12
        _card(TR_X, TR_Y, TR_W, TR_H)

        painter.setPen(QColor(TEXT_1))
        painter.setFont(f_title)
        painter.drawText(QRectF(TR_X + 14, TR_Y + 8, TR_W - 28, 18),
                         Qt.AlignmentFlag.AlignRight, 'SWARM SESSION · LIVE')

        painter.setPen(QColor(TEXT_3))
        painter.setFont(f_small)
        painter.drawText(QRectF(TR_X + 14, TR_Y + 26, TR_W - 28, 14),
                         Qt.AlignmentFlag.AlignRight, 'session_8618a2043ae9')

        session_rows = [
            ('ENGINE',   'swarm.v2'),
            ('AGENTS',   '20/20 alive'),
            ('PHASE',    'group_arena'),
            ('PARALLEL', '4'),
        ]
        for i, (k, v) in enumerate(session_rows):
            row_y = TR_Y + 56 + i * 17
            painter.setPen(QColor(TEXT_3))
            painter.setFont(f_lbl)
            painter.drawText(QRectF(TR_X + 14, row_y, 100, 14),
                             Qt.AlignmentFlag.AlignRight, k)
            painter.setPen(QColor(TEXT_2))
            painter.setFont(f_meta)
            painter.drawText(QRectF(TR_X + 120, row_y, TR_W - 134, 14),
                             Qt.AlignmentFlag.AlignRight, v)

        # ─── Bottom-left card: legend ────────────────────────────────────────
        BL_W, BL_H = 232, 108
        BL_X, BL_Y = 12, vh - BL_H - 12
        _card(BL_X, BL_Y, BL_W, BL_H)

        painter.setPen(QColor(TEXT_1))
        painter.setFont(f_lbl)
        painter.drawText(QPointF(BL_X + 14, BL_Y + 22), 'LEGEND')

        items = [
            (N_IDLE,     'idle'),
            (N_ACTIVE,   'active'),
            (N_DONE,     'survivor'),
            (N_DEAD,     'eliminated'),
        ]
        for i, (color, label) in enumerate(items):
            col = i % 2
            row = i // 2
            row_y = BL_Y + 44 + row * 18
            col_x = BL_X + 22 + col * 100
            painter.setBrush(QBrush(QColor(color)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(col_x, row_y - 4), 3, 3)
            painter.setPen(QColor(TEXT_2))
            painter.setFont(f_meta)
            painter.drawText(QPointF(col_x + 10, row_y), label)

        # Scale bar inside the card
        scale_y = BL_Y + BL_H - 16
        painter.setPen(QPen(QColor(TEXT_3), 1.2))
        painter.drawLine(QPointF(BL_X + 18, scale_y), QPointF(BL_X + 70, scale_y))
        painter.drawLine(QPointF(BL_X + 18, scale_y - 3), QPointF(BL_X + 18, scale_y + 3))
        painter.drawLine(QPointF(BL_X + 70, scale_y - 3), QPointF(BL_X + 70, scale_y + 3))
        painter.setPen(QColor(TEXT_3))
        painter.setFont(f_small)
        painter.drawText(QPointF(BL_X + 78, scale_y + 3), '62u')

        # ─── Bottom-right card: live metrics ─────────────────────────────────
        BR_W, BR_H = 232, 108
        BR_X, BR_Y = vw - BR_W - 12, vh - BR_H - 12
        _card(BR_X, BR_Y, BR_W, BR_H)

        painter.setPen(QColor(TEXT_1))
        painter.setFont(f_lbl)
        painter.drawText(QRectF(BR_X + 14, BR_Y + 8, BR_W - 28, 18),
                         Qt.AlignmentFlag.AlignRight, 'METRICS')
        metric_rows = [
            ('density',  '0.087'),
            ('avg_deg',  '4.21'),
            ('clusters', '6'),
            ('diameter', '4'),
        ]
        for i, (k, v) in enumerate(metric_rows):
            row_y = BR_Y + 44 + i * 16
            painter.setPen(QColor(TEXT_3))
            painter.setFont(f_meta)
            painter.drawText(QRectF(BR_X + 14, row_y, 130, 14),
                             Qt.AlignmentFlag.AlignRight, k)
            painter.setPen(QColor(TEXT_2))
            painter.setFont(f_meta)
            painter.drawText(QRectF(BR_X + 150, row_y, BR_W - 164, 14),
                             Qt.AlignmentFlag.AlignRight, v)

        painter.restore()


# ════════════════════════════════════════════════════════════════════════════
#  NodeDetailsCard — floating side panel with node info
# ════════════════════════════════════════════════════════════════════════════
class _NodeDetailsCard(QFrame):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(360)
        self.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1px solid #E5E7EB;border-radius:6px;}'
            'QFrame QLabel{background:transparent;border:none;}'
        )
        self.hide()
        self._build()

    def _build(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(0)

        # Header row
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        self._tag = QLabel('NODE')
        self._tag.setStyleSheet(
            'background:#0A0A0A;color:#FFFFFF;font-family:Consolas,monospace;'
            'font-size:8px;font-weight:700;letter-spacing:1.2px;padding:3px 7px;'
            'border-radius:2px;')
        hdr.addWidget(self._tag)
        hdr.addStretch()
        close = QPushButton('×')
        close.setFixedSize(20, 20)
        close.setStyleSheet(
            'QPushButton{background:transparent;color:#9CA3AF;border:none;'
            'font-size:18px;font-weight:300;}'
            'QPushButton:hover{color:#0A0A0A;}'
        )
        close.clicked.connect(self.closed.emit)
        hdr.addWidget(close)
        v.addLayout(hdr)
        v.addSpacing(10)

        # Name
        self._name_lbl = QLabel('')
        self._name_lbl.setStyleSheet(
            'font-size:17px;font-weight:700;color:#0A0A0A;letter-spacing:-0.3px;'
        )
        self._name_lbl.setWordWrap(True)
        v.addWidget(self._name_lbl)
        v.addSpacing(2)

        self._subtitle_lbl = QLabel('')
        self._subtitle_lbl.setStyleSheet('font-size:11px;color:#6B7280;')
        self._subtitle_lbl.setWordWrap(True)
        v.addWidget(self._subtitle_lbl)
        v.addSpacing(14)

        # Divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('border:none;border-top:1px solid #E5E7EB;')
        sep.setFixedHeight(1)
        v.addWidget(sep)
        v.addSpacing(10)

        # Properties section header
        prop_hdr = QLabel('PROPERTIES')
        prop_hdr.setStyleSheet(
            'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
            'color:#9CA3AF;letter-spacing:1.3px;'
        )
        v.addWidget(prop_hdr)
        v.addSpacing(6)

        self._props_widget = QWidget()
        self._props_layout = QVBoxLayout(self._props_widget)
        self._props_layout.setContentsMargins(0, 0, 0, 0)
        self._props_layout.setSpacing(4)
        v.addWidget(self._props_widget)
        v.addSpacing(12)

        # Summary section
        sum_hdr = QLabel('SUMMARY')
        sum_hdr.setStyleSheet(
            'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
            'color:#9CA3AF;letter-spacing:1.3px;'
        )
        v.addWidget(sum_hdr)
        v.addSpacing(6)

        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        self._summary.setStyleSheet(
            'QTextEdit{border:none;font-size:11px;color:#374151;background:transparent;'
            'line-height:1.5;}'
            'QScrollBar:vertical{width:3px;background:transparent;border:none;}'
            'QScrollBar::handle:vertical{background:#D1D5DB;border-radius:1.5px;min-height:20px;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}'
        )
        self._summary.setMinimumHeight(140)
        self._summary.setMaximumHeight(240)
        v.addWidget(self._summary)

    def _clear_props(self):
        while self._props_layout.count():
            item = self._props_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _add_prop(self, key: str, val: str):
        row = QHBoxLayout()
        row.setSpacing(8)
        k = QLabel(key.upper())
        k.setStyleSheet(
            'font-family:Consolas,monospace;font-size:9px;color:#9CA3AF;'
            'letter-spacing:0.5px;'
        )
        k.setFixedWidth(80)
        vlb = QLabel(val)
        vlb.setStyleSheet('font-size:11px;color:#0A0A0A;')
        vlb.setWordWrap(True)
        row.addWidget(k)
        row.addWidget(vlb, stretch=1)
        container = QWidget()
        container.setLayout(row)
        self._props_layout.addWidget(container)

    def show_agent(self, name, role, group_id, region, status, color,
                   rag_sources=None, message=''):
        self._tag.setText('SWARM AGENT')
        self._name_lbl.setText(name)
        self._subtitle_lbl.setText(f'{role}  ·  {region}')
        self._clear_props()
        self._add_prop('Role', role)
        self._add_prop('Region', region)
        self._add_prop('Group', f'#{group_id}')
        self._add_prop('Status', status.upper())
        if rag_sources:
            self._add_prop('Sources', ', '.join(_RAG_SHORT.get(s, s[:5]) for s in rag_sources))
        self._summary.setPlainText(message or '— no response yet —')
        self.show()

    def show_regional(self, judge_id, region, status):
        self._tag.setText('REGIONAL JUDGE')
        self._name_lbl.setText(f'Regional Judge {judge_id + 1}')
        self._subtitle_lbl.setText(region)
        self._clear_props()
        self._add_prop('Region', region)
        self._add_prop('Status', status.upper())
        self._summary.setPlainText('Aggregates survivors from this region to '
                                   'produce a regional verdict.')
        self.show()

    def show_master(self, status, estimate):
        self._tag.setText('MASTER JUDGE')
        self._name_lbl.setText('Master Judge')
        self._subtitle_lbl.setText('Cross-regional consensus')
        self._clear_props()
        self._add_prop('Status', status.upper())
        self._add_prop('Estimate', estimate)
        self._summary.setPlainText('Combines regional verdicts and sector inputs '
                                   'to produce the final cross-regional estimate.')
        self.show()

    def show_sector_agent(self, name, sector, status, rag_sources, estimate, color):
        self._tag.setText(f'{sector.upper()} AGENT')
        self._name_lbl.setText(name)
        self._subtitle_lbl.setText(f'{sector.title()} sector')
        self._clear_props()
        self._add_prop('Sector', sector.upper())
        self._add_prop('Status', status.upper())
        est_str = f'{estimate:+.2f}' if isinstance(estimate, (int, float)) else (estimate or '—')
        self._add_prop('Estimate', est_str)
        self._summary.setPlainText(f'{sector.title()} sector agent contributing to '
                                   f'the cross-domain verdict.')
        self.show()

    def show_sector_verdict(self, sector, status, estimate, color):
        self._tag.setText(f'{sector.upper()} VERDICT')
        self._name_lbl.setText(f'{sector.title()} Sector Verdict')
        self._subtitle_lbl.setText(f'Consensus of {sector} agents')
        self._clear_props()
        self._add_prop('Sector', sector.upper())
        self._add_prop('Status', status.upper())
        self._add_prop('Estimate', estimate)
        self._summary.setPlainText(f'Aggregated estimate from all {sector} sector '
                                   f'agents.')
        self.show()

    def show_rag(self, source):
        self._tag.setText('RAG SOURCE')
        self._name_lbl.setText(_RAG_SHORT.get(source, source))
        self._subtitle_lbl.setText(source)
        self._clear_props()
        self._add_prop('Source', source)
        self._add_prop('Type', 'External feed')
        self._summary.setPlainText('Retrieval-augmented context source. Agents query '
                                   'this feed for domain-specific evidence.')
        self.show()

    def update_message(self, agent_name: str, message: str):
        # Only update if currently showing this agent
        if self.isVisible() and self._name_lbl.text() == agent_name:
            self._summary.setPlainText(message or '— no response yet —')


# ════════════════════════════════════════════════════════════════════════════
#  Stage3SwarmPanel — public widget
# ════════════════════════════════════════════════════════════════════════════
class Stage3SwarmPanel(QWidget):
    swarm_complete = pyqtSignal(object)
    view_report_requested = pyqtSignal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self.setStyleSheet(f'background:{BG};')
        self._groups_done = 0
        self._regional_done_count = 0
        self._master_estimate = '—'
        self._gas_est: Optional[float] = None
        self._food_est: Optional[float] = None
        self._elec_est: Optional[float] = None
        self._active_agents = 0
        self._elim_count = 0
        self._alive_count = 20
        self._elapsed_s = 0
        self._clock_running = False
        self._console_count = 0
        self._build()

        self._clock_tmr = QTimer(self)
        self._clock_tmr.setInterval(1000)
        self._clock_tmr.timeout.connect(self._on_clock_tick)

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())
        outer.addLayout(self._build_main_row(), stretch=1)
        outer.addWidget(self._build_console())

        # Completion toast — an achievement-style card that slides in from the top
        self._build_completion_toast()

        # Live KG state
        self._kg_builder = KnowledgeGraphBuilder()
        self._agent_meta = {}
        self._rag = None
        self._scenario = {}
        self._kg_dirty = False
        self._kg_refresh = QTimer(self)
        self._kg_refresh.setInterval(1500)
        self._kg_refresh.timeout.connect(self._flush_kg)

        # Node details card — floats over canvas
        self._details_card = _NodeDetailsCard(self)
        self._details_card.closed.connect(self._details_card.hide)

    # ── Completion toast (achievement-style, slides in from the top) ──────────
    def _build_completion_toast(self):
        self._toast = QFrame(self)
        self._toast.setObjectName('completeToast')
        self._toast.setStyleSheet(
            '#completeToast{background:#16181F;border:1px solid #2C2F3A;'
            'border-radius:12px;}')
        lay = QHBoxLayout(self._toast)
        lay.setContentsMargins(14, 10, 12, 10)
        lay.setSpacing(12)

        badge = QLabel('✓')
        badge.setFixedSize(30, 30)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setStyleSheet(
            'background:#15A150;color:#FFFFFF;border-radius:15px;'
            'font-family:Consolas,monospace;font-size:16px;font-weight:700;')
        lay.addWidget(badge)

        col = QVBoxLayout(); col.setSpacing(1); col.setContentsMargins(0, 0, 0, 0)
        eyebrow = QLabel('SIMULATION COMPLETE')
        eyebrow.setStyleSheet(
            'color:#FFFFFF;font-family:Consolas,monospace;font-size:11px;'
            'font-weight:700;letter-spacing:1.4px;')
        sub = QLabel('master verdict ready')
        sub.setStyleSheet('color:#9AA0AA;font-family:Consolas,monospace;font-size:9px;')
        col.addWidget(eyebrow); col.addWidget(sub)
        lay.addLayout(col)
        lay.addSpacing(10)

        btn = QPushButton('View report →')
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            'QPushButton{background:#FFFFFF;color:#16181F;border:none;border-radius:8px;'
            'padding:7px 14px;font-family:Consolas,monospace;font-size:11px;font-weight:700;}'
            'QPushButton:hover{background:#E9ECF2;}')
        btn.clicked.connect(self.view_report_requested.emit)
        lay.addWidget(btn)
        self._toast_btn = btn

        shadow = QGraphicsDropShadowEffect(self._toast)
        shadow.setBlurRadius(30); shadow.setColor(QColor(0, 0, 0, 130)); shadow.setOffset(0, 9)
        self._toast.setGraphicsEffect(shadow)

        self._toast_anim = QPropertyAnimation(self._toast, b'pos')
        self._toast_anim.setDuration(560)
        self._toast_anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._toast.hide()

    def _show_completion_toast(self):
        self._toast.adjustSize()
        cx = max(12, (self.width() - self._toast.width()) // 2)
        start = QPoint(cx, -self._toast.height() - 6)
        end = QPoint(cx, 16)
        self._toast.move(start)
        self._toast.show()
        self._toast.raise_()
        self._toast_anim.stop()
        self._toast_anim.setStartValue(start)
        self._toast_anim.setEndValue(end)
        self._toast_anim.start()

    def _hide_completion_toast(self):
        if hasattr(self, '_toast'):
            self._toast.hide()

    def _build_header(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(56)
        bar.setStyleSheet(
            f'background:{BG};border-bottom:1px solid {DIVIDER};'
        )
        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 0, 24, 0)
        h.setSpacing(20)

        # Brand
        brand = QLabel('SWARM ENGINE')
        brand.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:11px;font-weight:700;'
            f'letter-spacing:3px;color:{TEXT_1};'
        )
        h.addWidget(brand)

        version = QLabel('/ V2.0')
        version.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_3};'
            f'letter-spacing:1px;'
        )
        h.addWidget(version)
        h.addStretch()

        # Phase indicator (text, no pill)
        self._phase_lbl = QLabel('Step 0/4  ·  Initializing')
        self._phase_lbl.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_2};'
            f'letter-spacing:1px;'
        )
        h.addWidget(self._phase_lbl)
        h.addStretch()

        # Stat — ACTIVE
        def _stat(label: str, val_color: str) -> tuple:
            box = QFrame()
            box.setStyleSheet('QFrame{background:transparent;border:none;}'
                              'QFrame QLabel{background:transparent;border:none;}')
            bv = QVBoxLayout(box)
            bv.setContentsMargins(0, 0, 0, 0)
            bv.setSpacing(0)
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
                f'color:{TEXT_3};letter-spacing:1.4px;'
            )
            val = QLabel('0')
            val.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:14px;font-weight:700;'
                f'color:{val_color};letter-spacing:0.5px;'
            )
            bv.addWidget(lbl)
            bv.addWidget(val)
            return box, val

        active_box, self._active_val = _stat('ACTIVE', N_ACTIVE)
        alive_box,  self._alive_val  = _stat('ALIVE', N_DONE)
        elim_box,   self._elim_val   = _stat('ELIM', TEXT_2)
        time_box,   self._time_val   = _stat('ELAPSED', TEXT_1)
        self._alive_val.setText('20')
        self._time_val.setText('00:00')

        h.addWidget(active_box)
        h.addSpacing(12)
        h.addWidget(alive_box)
        h.addSpacing(12)
        h.addWidget(elim_box)
        h.addSpacing(12)
        h.addWidget(time_box)
        h.addSpacing(8)

        # Status badge — far right
        self._status_badge = QLabel('● PENDING')
        self._status_badge.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
            f'color:{TEXT_2};letter-spacing:1.2px;'
        )
        h.addWidget(self._status_badge)
        return bar

    def _build_main_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        # Canvas
        self._canvas = _SwarmCanvas()
        self._canvas.node_clicked.connect(self._on_node_clicked)
        row.addWidget(self._canvas, stretch=1)

        # Knowledge-graph canvas — shown after swarm completes, hidden initially
        self._kg_canvas = KnowledgeGraphCanvas()
        self._kg_canvas.setVisible(False)
        row.addWidget(self._kg_canvas, stretch=1)

        # Right sidebar — verdicts
        right = self._build_verdict_sidebar()
        row.addWidget(right)
        return row

    def _build_verdict_sidebar(self) -> QFrame:
        side = QFrame()
        side.setFixedWidth(260)
        side.setStyleSheet(
            f'QFrame{{background:{BG};border-left:1px solid {DIVIDER};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )
        v = QVBoxLayout(side)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(0)

        # Header
        hdr = QLabel('SECTOR VERDICTS')
        hdr.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1.5px;'
        )
        v.addWidget(hdr)
        v.addSpacing(12)

        # Verdict cards
        def _verdict_card(tag: str, accent: str) -> tuple:
            card = QFrame()
            card.setStyleSheet(
                f'QFrame{{background:#FFFFFF;border:1px solid {DIVIDER};'
                f'border-radius:4px;}}'
                f'QFrame QLabel{{background:transparent;border:none;}}'
            )
            cv = QVBoxLayout(card)
            cv.setContentsMargins(12, 10, 12, 10)
            cv.setSpacing(2)
            top = QHBoxLayout()
            top.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(6, 6)
            dot.setStyleSheet(f'background:{accent};border-radius:3px;')
            label_w = QLabel(tag)
            label_w.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
                f'color:{TEXT_2};letter-spacing:1.2px;'
            )
            top.addWidget(dot)
            top.addWidget(label_w)
            top.addStretch()
            cv.addLayout(top)
            val = QLabel('—')
            val.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:16px;font-weight:700;'
                f'color:{TEXT_1};'
            )
            sub = QLabel('pending')
            sub.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:9px;color:{TEXT_3};'
                f'letter-spacing:0.3px;'
            )
            cv.addWidget(val)
            cv.addWidget(sub)
            return card, val, sub

        gas_card, self._gas_val, self._gas_sub = _verdict_card('GAS', '#475569')
        food_card, self._food_val, self._food_sub = _verdict_card('FOOD', N_FOOD)
        elec_card, self._elec_val, self._elec_sub = _verdict_card('ELECTRICITY', N_ELEC)

        v.addWidget(gas_card)
        v.addSpacing(8)
        v.addWidget(food_card)
        v.addSpacing(8)
        v.addWidget(elec_card)
        v.addSpacing(20)

        # Phase progress
        ph_hdr = QLabel('PHASE PROGRESS')
        ph_hdr.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1.5px;'
        )
        v.addWidget(ph_hdr)
        v.addSpacing(8)

        self._phase_rows: list[tuple[QLabel, QLabel]] = []
        for label in ('Initialize', 'Group arena', 'Regional judges', 'Master verdict'):
            row = QHBoxLayout()
            row.setSpacing(8)
            dot = QLabel('○')
            dot.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:11px;color:{TEXT_3};'
            )
            txt = QLabel(label)
            txt.setStyleSheet(
                f'font-size:11px;color:{TEXT_2};'
            )
            row.addWidget(dot)
            row.addWidget(txt)
            row.addStretch()
            container = QWidget()
            container.setLayout(row)
            v.addWidget(container)
            self._phase_rows.append((dot, txt))

        v.addStretch()
        return side

    def _build_console(self) -> QWidget:
        cw = QWidget()
        cw.setFixedHeight(100)
        cw.setStyleSheet('background:#0A0A0A;')
        cv = QVBoxLayout(cw)
        cv.setContentsMargins(20, 8, 20, 8)
        cv.setSpacing(2)

        # Header strip
        head = QHBoxLayout()
        head.setSpacing(0)
        title = QLabel('CONSOLE OUTPUT')
        title.setStyleSheet(
            'font-family:Consolas,monospace;font-size:8px;font-weight:700;'
            'color:#6B7280;letter-spacing:1.5px;background:transparent;'
        )
        head.addWidget(title)
        head.addStretch()
        self._console_id = QLabel('swarm_session_001')
        self._console_id.setStyleSheet(
            'font-family:Consolas,monospace;font-size:8px;color:#4B5563;'
            'letter-spacing:0.8px;background:transparent;'
        )
        head.addWidget(self._console_id)
        cv.addLayout(head)

        self._console = QTextEdit()
        self._console.setReadOnly(True)
        self._console.setStyleSheet(
            'QTextEdit{background:transparent;border:none;color:#D1D5DB;'
            'font-family:Consolas,Monaco,monospace;font-size:10px;line-height:1.4;}'
            'QScrollBar:vertical{width:3px;background:transparent;}'
            'QScrollBar::handle:vertical{background:#374151;border-radius:1.5px;}'
            'QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}'
        )
        cv.addWidget(self._console)
        return cw

    # ── Resize handling for floating details card ─────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_details_card()

    def _position_details_card(self):
        dc = self._details_card
        if not dc.isVisible():
            return
        dc.adjustSize()
        # Sidebar is 260px on the right, console is 100px tall on the bottom
        right_panel_w = 260
        bottom_console_h = 100
        x = self.width() - right_panel_w - dc.width() - 16
        canvas_top_y = 56     # header height
        canvas_bot_y = self.height() - bottom_console_h
        canvas_mid = (canvas_top_y + canvas_bot_y) // 2
        y = max(canvas_top_y + 10, canvas_mid - dc.height() // 2)
        dc.move(x, y)
        dc.raise_()

    # ── Click handling ────────────────────────────────────────────────────────
    def _on_node_clicked(self, info: dict):
        # Knowledge-graph nodes carry 'payload' instead of 'type'
        if 'payload' in info:
            kind = info.get('kind', '')
            pl = info.get('payload', {})
            label = info.get('label', info.get('id', ''))
            if kind == 'evidence':
                detail = f"{pl.get('source', '')}: {pl.get('text', '')[:300]}"
                self._details_card.show_agent(
                    label, 'Evidence', 0, pl.get('source', ''),
                    'ready', '#3B6FD4', message=detail)
            elif kind == 'entity':
                prov = (pl.get('provenance') or [{}])[0]
                detail = (f"{label} ({pl.get('type', '')}) "
                          f"— from {prov.get('source', '')}")
                self._details_card.show_agent(
                    label, pl.get('type', 'Entity'), 0, prov.get('source', ''),
                    'ready', '#15A150', message=detail)
            else:
                detail = info.get('label', '')
                self._details_card.show_agent(
                    label, kind or 'node', 0, '', 'ready', '#9CA3AF',
                    message=detail)
            self._position_details_card()
            return
        ntype = info.get('type')
        if ntype == 'agent':
            self._details_card.show_agent(
                info['name'], info['role'], info['group_id'],
                info['region'], info['status'], info.get('color', N_IDLE),
                rag_sources=info.get('rag', []),
                message=info.get('message', ''))
        elif ntype == 'regional':
            self._details_card.show_regional(
                info['judge_id'], info['region'], info['status'])
        elif ntype == 'master':
            self._details_card.show_master(info['status'], self._master_estimate)
        elif ntype == 'sector_agent':
            self._details_card.show_sector_agent(
                info['name'], info['sector'], info['status'],
                [], info.get('estimate'),
                N_FOOD if info['sector'] == 'food' else N_ELEC)
        elif ntype == 'sector_verdict':
            self._details_card.show_sector_verdict(
                info['sector'], info['status'], info.get('estimate', '—'),
                N_FOOD if info['sector'] == 'food' else N_ELEC)
        elif ntype == 'rag':
            self._details_card.show_rag(info['source'])
        self._position_details_card()

    # ── Knowledge-graph integration ───────────────────────────────────────────
    def show_knowledge_graph(self, builder):
        """Swap the live arena for the MiroFish knowledge graph + start enrichment."""
        from ph_economic_ai.ui.kg_extract_worker import EntityExtractWorker
        self._kg_builder = builder
        nodes, edges = builder.snapshot()
        self._kg_canvas.set_snapshot(nodes, edges)
        self._canvas.setVisible(False)
        self._kg_canvas.setVisible(True)
        try:
            self._kg_canvas.node_clicked.connect(self._on_node_clicked)
        except Exception:
            pass
        self._log(
            f'KNOWLEDGE GRAPH  {len(nodes)} nodes  {len(edges)} edges',
            color='#3B6FD4',
        )
        self._kg_worker = EntityExtractWorker(builder)
        self._kg_worker.progress.connect(
            lambda _i: self._kg_canvas.set_snapshot(*builder.snapshot()))
        self._kg_worker.done.connect(
            lambda: self._log('entity extraction complete', color='#15A150'))
        self._kg_worker.start()

    # ── Live knowledge-graph helpers ──────────────────────────────────────────
    def _begin_live_graph(self, rag, scenario, agent_meta):
        """Start a fresh live knowledge graph: seed it and show the KG canvas."""
        self._rag, self._scenario, self._agent_meta = rag, scenario or {}, agent_meta or {}
        self._kg_builder = KnowledgeGraphBuilder()
        try:
            # Seed the FULL connected skeleton (agents + judges + master + data) so the
            # graph is rich and cohesive from t=0 instead of a few scattered dots.
            _kg_live.seed_skeleton(self._kg_builder, self._agent_meta.values(), self._scenario)
        except Exception:
            pass
        # Keep the OLD structured arena as the live view (the labelled "line vibe"
        # the user wants); the bare force-graph is not shown live.
        self._canvas.setVisible(True)
        self._kg_canvas.setVisible(False)
        self._hide_completion_toast()

    def _flush_kg(self):
        try:
            self._kg_canvas.set_snapshot(*self._kg_builder.snapshot())
        except Exception:
            pass
        self._kg_dirty = False

    def has_live_graph(self) -> bool:
        try:
            return len(self._kg_builder.snapshot()[0]) > 3
        except Exception:
            return False

    # ── Console ───────────────────────────────────────────────────────────────
    def _log(self, text: str, color: str = '#D1D5DB'):
        self._console_count += 1
        ts = f'{self._elapsed_s // 60:02d}:{self._elapsed_s % 60:02d}'
        line = (f'<span style="color:#4B5563;">[{ts}]</span> '
                f'<span style="color:{color};">{text}</span>')
        self._console.append(line)

    # ── Clock ─────────────────────────────────────────────────────────────────
    def _on_clock_tick(self):
        self._elapsed_s += 1
        m, s = divmod(self._elapsed_s, 60)
        self._time_val.setText(f'{m:02d}:{s:02d}')

    def _start_clock(self):
        if not self._clock_running:
            self._clock_running = True
            self._clock_tmr.start()

    # ── Phase row updates ─────────────────────────────────────────────────────
    def _set_phase(self, step: int, label: str):
        self._phase_lbl.setText(f'Step {step}/4  ·  {label}')
        for i, (dot, txt) in enumerate(self._phase_rows):
            if i < step:
                dot.setText('●')
                dot.setStyleSheet(
                    f'font-family:Consolas,monospace;font-size:11px;color:{N_DONE};'
                )
                txt.setStyleSheet(f'font-size:11px;color:{TEXT_1};font-weight:600;')
            elif i == step:
                dot.setText('●')
                dot.setStyleSheet(
                    f'font-family:Consolas,monospace;font-size:11px;color:{N_ACTIVE};'
                )
                txt.setStyleSheet(f'font-size:11px;color:{TEXT_1};font-weight:600;')
            else:
                dot.setText('○')
                dot.setStyleSheet(
                    f'font-family:Consolas,monospace;font-size:11px;color:{TEXT_3};'
                )
                txt.setStyleSheet(f'font-size:11px;color:{TEXT_2};')

    # ── SwarmThread signal handlers ───────────────────────────────────────────
    def _on_agent_typing(self, group_id: int, agent_name: str):
        self._start_clock()
        self._canvas.mark_active(agent_name)
        self._active_agents += 1
        self._active_val.setText(str(self._active_agents))
        if self._groups_done == 0 and self._elapsed_s < 2:
            self._set_phase(1, 'Group arena')
            self._status_badge.setText('● ARENA')
            self._status_badge.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
                f'color:{N_ACTIVE};letter-spacing:1.2px;'
            )

    def _on_agent_done_typing(self, group_id: int, agent_name: str):
        self._canvas.mark_idle(agent_name)
        self._active_agents = max(0, self._active_agents - 1)
        self._active_val.setText(str(self._active_agents))

    def _on_group_round_done(self, group_id: int, round_num: int, responses):
        for resp in responses:
            self._canvas.store_response(resp.agent_name, resp.statement)
            self._details_card.update_message(resp.agent_name, resp.statement)
        try:
            _kg_live.add_round(self._kg_builder, responses, self._agent_meta,
                               self._rag, self._scenario)
            self._kg_dirty = True
        except Exception:
            pass

    def _on_group_eliminated(self, group_id: int, agent_name: str,
                             score: float, round_num: int):
        region = REGIONS[group_id] if group_id < len(REGIONS) else f'G{group_id}'
        self._canvas.mark_eliminated(agent_name)
        self._elim_count += 1
        self._alive_count = max(0, self._alive_count - 1)
        self._elim_val.setText(str(self._elim_count))
        self._alive_val.setText(str(self._alive_count))
        self._log(f'ELIM  {region[:8]}  R{round_num}  {agent_name.split()[-1]}  '
                  f'score={score:.2f}', color='#F87171')

    def _on_group_survivor(self, group_id: int, survivor):
        region = REGIONS[group_id] if group_id < len(REGIONS) else f'G{group_id}'
        self._canvas.mark_survivor(survivor.response.agent_name, group_id=group_id)
        self._groups_done += 1
        self._log(f'SURVIVOR  {region[:8]}  {survivor.response.agent_name.split()[-1]}',
                  color='#34D399')
        if self._groups_done >= 4:
            self._set_phase(2, 'Regional judges')
            self._status_badge.setText('● JUDGING')

    def _on_regional_done(self, judge_id: int, verdict):
        self._canvas.mark_regional_active(judge_id)
        QTimer.singleShot(200, lambda jid=judge_id: self._canvas.mark_regional_done(jid))
        self._regional_done_count += 1
        est = f'+{verdict.estimate:.2f}/L' if verdict.estimate is not None else 'N/A'
        self._log(f'REGIONAL #{judge_id + 1}  estimate={est}', color='#A78BFA')
        if self._regional_done_count >= 2:
            self._set_phase(3, 'Master verdict')
            self._status_badge.setText('● MASTER')
        try:
            _kg_live.add_regional(self._kg_builder, getattr(verdict, 'region_pair', ()),
                                  getattr(verdict, 'estimate', None), self._agent_meta)
            self._kg_dirty = True
        except Exception:
            pass

    def _on_swarm_complete(self, master_verdict):
        self._canvas.mark_master_done()
        self._clock_tmr.stop()
        est = (f'+₱{master_verdict.final_estimate:.2f}/L'
               if master_verdict.final_estimate is not None else 'N/A')
        self._master_estimate = est
        self._gas_est = master_verdict.final_estimate
        self._set_phase(4, 'Complete')
        self._status_badge.setText('● COMPLETED')
        self._status_badge.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
            f'color:{N_DONE};letter-spacing:1.2px;'
        )
        self._gas_val.setText(est)
        self._gas_val.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:16px;font-weight:700;color:{N_DONE};'
        )
        self._gas_sub.setText(f'{master_verdict.confidence_pct}% confidence')
        self._gas_sub.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;color:{N_DONE};'
        )
        self._log(f'MASTER  gas={est}  conf={master_verdict.confidence_pct}%',
                  color='#34D399')
        self._apply_trust_badges()
        try:
            _kg_live.add_master(self._kg_builder, getattr(master_verdict, 'final_estimate', None))
            self._flush_kg()
            from ph_economic_ai.ui.kg_extract_worker import EntityExtractWorker
            self._kg_worker = EntityExtractWorker(self._kg_builder)
            self._kg_worker.progress.connect(
                lambda _i: self._kg_canvas.set_snapshot(*self._kg_builder.snapshot()))
            self._kg_worker.start()
            self._kg_refresh.stop()
        except Exception:
            pass
        self._show_completion_toast()
        self.swarm_complete.emit(master_verdict)

    # ── Trust badges ──────────────────────────────────────────────────────────
    def _apply_trust_badges(self) -> None:
        if self._store is None:
            return
        trust_map = self._store.get_all_trust()
        for item in self._canvas._scene.items():
            if hasattr(item, 'set_trust'):
                agent_name = getattr(item, '_name', None)
                if agent_name:
                    trust = trust_map.get(agent_name, 0.5)
                    tier = ('promoted' if trust > 0.70 else
                            ('demoted' if trust < 0.30 else 'default'))
                    item.set_trust(trust, tier)

    # ── Food sector ───────────────────────────────────────────────────────────
    def connect_food_thread(self, thread):
        thread.token_received.connect(self._on_food_token)
        thread.agent_done.connect(self._on_food_agent_done)
        thread.debate_complete.connect(self._on_food_canvas_complete)

    def _on_food_token(self, agent_name: str, token: str):
        self._start_clock()
        canvas = self._canvas
        if agent_name not in canvas._food_typing:
            self._active_agents += 1
            self._active_val.setText(str(self._active_agents))
        canvas.mark_sector_agent_typing(agent_name, 'food')

    def _on_food_agent_done(self, resp):
        est = resp.price_estimate
        self._canvas.mark_sector_agent_done(resp.agent_name, 'food', est)
        est_str = f'{est:+.2f}%' if est is not None else '?'
        self._active_agents = max(0, self._active_agents - 1)
        self._active_val.setText(str(self._active_agents))
        self._log(f'FOOD    {resp.agent_name.split()[0]}  {est_str}', color='#86EFAC')
        try:
            _kg_live.add_sector_agent(self._kg_builder, resp.agent_name, 'food',
                                      resp.price_estimate, getattr(resp, 'statement', ''))
            self._kg_dirty = True
        except Exception:
            pass

    def _on_food_canvas_complete(self, responses):
        estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
        avg = sum(estimates) / len(estimates) if estimates else None
        est_str = f'{avg:+.2f}%' if avg is not None else 'N/A'
        self._food_est = avg
        self._canvas.mark_sector_complete('food', est_str)
        self._food_val.setText(est_str)
        color = N_FOOD if (avg or 0) >= 0 else N_ACTIVE
        self._food_val.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:16px;font-weight:700;color:{color};'
        )
        self._food_sub.setText('consensus reached')
        self._food_sub.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;color:{color};'
        )
        self._log(f'FOOD CONSENSUS  {est_str}', color='#34D399')

    # ── Electricity sector ────────────────────────────────────────────────────
    def connect_elec_thread(self, thread):
        thread.token_received.connect(self._on_elec_token)
        thread.agent_done.connect(self._on_elec_agent_done)
        thread.debate_complete.connect(self._on_elec_canvas_complete)

    def _on_elec_token(self, agent_name: str, token: str):
        self._start_clock()
        canvas = self._canvas
        if agent_name not in canvas._elec_typing:
            self._active_agents += 1
            self._active_val.setText(str(self._active_agents))
        canvas.mark_sector_agent_typing(agent_name, 'elec')

    def _on_elec_agent_done(self, resp):
        est = resp.price_estimate
        self._canvas.mark_sector_agent_done(resp.agent_name, 'elec', est)
        est_str = f'+₱{est:.4f}/kWh' if est is not None else '?'
        self._active_agents = max(0, self._active_agents - 1)
        self._active_val.setText(str(self._active_agents))
        self._log(f'ELEC    {resp.agent_name.split()[0]}  {est_str}', color='#FBBF24')
        try:
            _kg_live.add_sector_agent(self._kg_builder, resp.agent_name, 'elec',
                                      resp.price_estimate, getattr(resp, 'statement', ''))
            self._kg_dirty = True
        except Exception:
            pass

    def _on_elec_canvas_complete(self, responses):
        estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
        avg = sum(estimates) / len(estimates) if estimates else None
        est_str = f'+₱{avg:.4f}/kWh' if avg is not None else 'N/A'
        self._elec_est = avg
        self._canvas.mark_sector_complete('elec', est_str)
        self._elec_val.setText(est_str)
        color = N_ELEC if (avg or 0) >= 0 else N_ACTIVE
        self._elec_val.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:16px;font-weight:700;color:{color};'
        )
        self._elec_sub.setText('consensus reached')
        self._elec_sub.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;color:{color};'
        )
        self._log(f'ELEC CONSENSUS  {est_str}', color='#FBBF24')

    # ── Thread wiring ─────────────────────────────────────────────────────────
    def connect_thread(self, thread):
        meta = {}
        try:
            price = (getattr(thread, '_scenario', {}) or {}).get('current_price', 0.0)
            meta = {a.name: a for a in build_swarm_agents(price)}
        except Exception:
            pass
        self._begin_live_graph(getattr(thread, '_rag', None),
                               getattr(thread, '_scenario', {}), meta)
        thread.agent_typing.connect(self._on_agent_typing)
        thread.agent_done_typing.connect(self._on_agent_done_typing)
        thread.group_round_done.connect(self._on_group_round_done)
        thread.group_eliminated.connect(self._on_group_eliminated)
        thread.group_survivor.connect(self._on_group_survivor)
        thread.regional_done.connect(self._on_regional_done)
        thread.swarm_complete.connect(self._on_swarm_complete)
        try:
            self._canvas.add_evidence_layer(getattr(thread, '_rag', None),
                                            getattr(thread, '_scenario', {}))
        except Exception:
            pass

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self._groups_done = 0
        self._regional_done_count = 0
        self._master_estimate = '—'
        self._gas_est = None
        self._food_est = None
        self._elec_est = None
        self._active_agents = 0
        self._elim_count = 0
        self._alive_count = 20
        self._elapsed_s = 0
        self._clock_running = False
        self._clock_tmr.stop()

        self._active_val.setText('0')
        self._alive_val.setText('20')
        self._elim_val.setText('0')
        self._time_val.setText('00:00')

        val_style = (f'font-family:Consolas,monospace;font-size:16px;'
                     f'font-weight:700;color:{TEXT_1};')
        sub_style = f'font-family:Consolas,monospace;font-size:9px;color:{TEXT_3};'
        for val_lbl in (self._gas_val, self._food_val, self._elec_val):
            val_lbl.setText('—')
            val_lbl.setStyleSheet(val_style)
        self._gas_sub.setText('pending')
        self._food_sub.setText('pending')
        self._elec_sub.setText('pending')
        for sub_lbl in (self._gas_sub, self._food_sub, self._elec_sub):
            sub_lbl.setStyleSheet(sub_style)

        self._set_phase(0, 'Initializing')
        self._status_badge.setText('● PENDING')
        self._status_badge.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
            f'color:{TEXT_2};letter-spacing:1.2px;'
        )
        self._canvas.reset()
        self._console.clear()
        self._console_count = 0
        self._details_card.hide()
        self._kg_builder = KnowledgeGraphBuilder()
        self._kg_dirty = False
        self._kg_refresh.stop()
        self._hide_completion_toast()
