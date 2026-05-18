import math
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                              QGraphicsView, QGraphicsScene, QGraphicsItem)
from PyQt6.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPolygonF, QPainterPath


_NODES = [
    ('Oil Market',        80,  80),
    ('USD / PHP',         80, 190),
    ('Demand',            80, 300),
    ('ML Model',         280, 185),
    ('Prediction\nOutput', 490,  80),
    ('Economic\nIndex',  490, 300),
]

_EDGES = [
    (0, 3), (1, 3), (2, 3),   # inputs → ML Model
    (3, 4), (3, 5),            # ML Model → outputs
]

_W, _H = 130, 52


class _NodeItem(QGraphicsItem):
    def __init__(self, label: str, x: float, y: float):
        super().__init__()
        self._label = label
        self.setPos(x, y)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, _W, _H)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, _W, _H), 8, 8)
        painter.fillPath(path, QColor('#FFFFFF'))
        painter.setPen(QPen(QColor('#4A90E2'), 1.5))
        painter.drawPath(path)
        painter.setPen(QPen(QColor('#111111')))
        f = QFont()
        f.setPointSize(9)
        painter.setFont(f)
        painter.drawText(
            QRectF(0, 0, _W, _H),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )


class _EdgeItem:
    """Draws a line with an arrowhead from source node edge to target node edge."""

    def __init__(self, scene: QGraphicsScene, src_pos: QPointF, dst_pos: QPointF):
        # Offset to node centres
        sx = src_pos.x() + _W
        sy = src_pos.y() + _H / 2
        dx = dst_pos.x()
        dy = dst_pos.y() + _H / 2

        pen = QPen(QColor('#CCCCCC'), 1.5)
        pen.setStyle(Qt.PenStyle.SolidLine)
        line = scene.addLine(sx, sy, dx, dy, pen)
        line.setZValue(-1)

        # Arrowhead
        angle = math.atan2(dy - sy, dx - sx)
        size = 9
        p1 = QPointF(
            dx - size * math.cos(angle - math.pi / 6),
            dy - size * math.sin(angle - math.pi / 6),
        )
        p2 = QPointF(
            dx - size * math.cos(angle + math.pi / 6),
            dy - size * math.sin(angle + math.pi / 6),
        )
        arrow = QPolygonF([QPointF(dx, dy), p1, p2])
        scene.addPolygon(arrow, QPen(Qt.PenStyle.NoPen), QColor('#CCCCCC'))


class AgentGraphPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel('Agent Network')
        title.setStyleSheet('font-size:20px; font-weight:700; color:#111111;')
        layout.addWidget(title)

        sub = QLabel('Data flow from economic inputs through the ML model to outputs')
        sub.setStyleSheet('font-size:12px; color:#888888;')
        layout.addWidget(sub)

        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 700, 420)
        scene.setBackgroundBrush(QColor('#FAFAFA'))

        node_items = []
        for label, x, y in _NODES:
            item = _NodeItem(label, x, y)
            scene.addItem(item)
            node_items.append(QPointF(x, y))

        for src_idx, dst_idx in _EDGES:
            _EdgeItem(scene, node_items[src_idx], node_items[dst_idx])

        view = QGraphicsView(scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setStyleSheet('border: 1px solid #EAEAEA; border-radius: 10px; background: #FAFAFA;')
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        layout.addWidget(view)
