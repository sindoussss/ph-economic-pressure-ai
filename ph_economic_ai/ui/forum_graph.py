"""A literal, labelled force-directed graph for the forum debate.

Draws a real network: named circular nodes with a soft glow, curved edges carrying
relationship labels (READS / ESTIMATES), colour-coded by kind — sector hubs,
agents, the RAG sources each agent reads, and their estimate claims. Node
positions glide smoothly between updates (a small timeline animation) so the graph
grows organically instead of jumping. Reuses the pure force layout in `kg_layout`;
drawn with QGraphicsScene — offline, no web view, no new deps.

Drop-in: same `set_snapshot(nodes, edges)` API as KnowledgeGraphCanvas.
"""
import math

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimeLine
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsView

from ph_economic_ai.ui.kg_layout import render_model

_W, _H = 900, 560
_SECTOR = {'gas': '#E5484D', 'food': '#15A150', 'electricity': '#E8920C'}
_AGENT = '#3B82F6'
_SOURCE = '#8B5CF6'
_CLAIM = '#8B93A1'
_EDGE = QColor('#9AA4B2')          # darker so edges read on real displays
_EDGE_LABEL = {'retrieved': 'READS', 'claims': 'ESTIMATES'}   # in_region: unlabelled

# kind -> (color, radius, font_pt, bold, text_color)
_STYLE = {
    'master': ('#3B82F6', 30, 14, True, '#0F1115'),   # colour overridden per sector
    'agent':  (_AGENT, 20, 12, False, '#0F1115'),
    'source': (_SOURCE, 14, 11, False, '#5B21B6'),
    'claim':  (_CLAIM, 11, 11, False, '#6B7280'),
}


class ForumGraphCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet('border:1px solid #C3CAD4;border-radius:10px;background:#FFFFFF;')
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._pos: dict = {}          # currently displayed positions
        self._nodes: list = []
        self._edges: list = []
        self._start: dict = {}
        self._target: dict = {}
        self._anim: QTimeLine | None = None

    # -- API (matches KnowledgeGraphCanvas) --
    def reset(self):
        if self._anim is not None:
            self._anim.stop()
            self._anim = None
        self._pos, self._nodes, self._edges = {}, [], []
        self._scene.clear()

    def set_snapshot(self, nodes, edges):
        self._nodes, self._edges = list(nodes), list(edges)
        if not nodes:
            self._scene.clear()
            return
        rm = render_model(nodes, edges, width=_W, height=_H, iterations=150)
        self._target = {n['id']: (n['x'], n['y']) for n in rm['nodes']}
        # New nodes start at their target; existing nodes glide from where they are.
        self._start = {nid: self._pos.get(nid, self._target[nid]) for nid in self._target}

        xs = [p[0] for p in self._target.values()]
        ys = [p[1] for p in self._target.values()]
        pad = 90
        rect = QRectF(min(xs) - pad, min(ys) - pad,
                      (max(xs) - min(xs)) + 2 * pad, (max(ys) - min(ys)) + 2 * pad)
        self._scene.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        self._draw(self._start)                     # draw immediately (also keeps tests sync)
        if self._anim is not None:
            self._anim.stop()
        self._anim = QTimeLine(430, self)
        self._anim.setUpdateInterval(20)
        self._anim.valueChanged.connect(self._on_anim)
        self._anim.finished.connect(lambda: self._draw(self._target))
        self._anim.start()

    def node_item_count(self) -> int:
        return len(self._scene.items())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # -- animation --
    def _on_anim(self, t: float):
        e = t * t * (3 - 2 * t)                     # smoothstep ease
        interp = {
            nid: (self._start[nid][0] + (self._target[nid][0] - self._start[nid][0]) * e,
                  self._start[nid][1] + (self._target[nid][1] - self._start[nid][1]) * e)
            for nid in self._target}
        self._draw(interp)

    def _draw(self, pos: dict):
        self._pos = pos
        self._scene.clear()
        for edge in self._edges:                    # edges under nodes
            a, b = pos.get(edge.src), pos.get(edge.dst)
            if a and b:
                self._edge(a, b, _EDGE_LABEL.get(edge.kind, ''))
        for n in self._nodes:
            if n.id in pos:
                self._node(n, *pos[n.id])

    # -- drawing --
    def _edge(self, a, b, label):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length          # unit normal for the curve bow
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        cx, cy = mx + nx * 26, my + ny * 26
        path = QPainterPath(QPointF(a[0], a[1]))
        path.quadTo(QPointF(cx, cy), QPointF(b[0], b[1]))
        item = self._scene.addPath(path, QPen(_EDGE, 1.4))
        item.setZValue(0)
        if label:
            self._edge_label(cx, cy, label)

    def _edge_label(self, x, y, text):
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        t = self._scene.addText(text, font)
        t.setDefaultTextColor(QColor('#6B7280'))
        br = t.boundingRect()
        w, h = br.width() + 12, br.height() + 2
        box = QPainterPath()
        box.addRoundedRect(QRectF(x - w / 2, y - h / 2, w, h), 5, 5)
        bg = self._scene.addPath(box, QPen(QColor('#BFC6D0'), 1.0), QBrush(QColor('#FFFFFF')))
        bg.setZValue(1)
        t.setPos(x - br.width() / 2, y - br.height() / 2)
        t.setZValue(2)

    def _node(self, n, x, y):
        color, r, fs, bold, tcol = _STYLE.get(n.kind, (_SOURCE, 12, 10, False, '#333333'))
        if n.kind == 'master':
            color = _SECTOR.get((n.payload or {}).get('sector'), '#3B82F6')

        halo = QColor(color)
        halo.setAlpha(45)
        glow = self._scene.addEllipse(x - r - 8, y - r - 8, 2 * (r + 8), 2 * (r + 8),
                                      QPen(Qt.PenStyle.NoPen), QBrush(halo))
        glow.setZValue(2)
        dot = self._scene.addEllipse(x - r, y - r, 2 * r, 2 * r,
                                     QPen(QColor('#FFFFFF'), 2.0), QBrush(QColor(color)))
        dot.setZValue(3)

        font = QFont()
        font.setPointSize(fs)
        font.setBold(bold)
        t = self._scene.addText(n.label or '', font)
        t.setDefaultTextColor(QColor(tcol))
        br = t.boundingRect()
        t.setPos(x - br.width() / 2, y + r + 2)     # label below the node
        t.setZValue(4)
