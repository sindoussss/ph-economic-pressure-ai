"""MiroFish-style knowledge-graph canvas: light organic cloud of tiny colour-coded
nodes, faint mass edges, a red relationship fan on the focused node. Renders a
snapshot (KGNode/KGEdge) via the pure kg_layout render model."""
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QBrush, QPen, QColor, QPainter
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem

from ph_economic_ai.ui.kg_layout import render_model

_EDGE = QColor('#D8DCE4')
_EDGE_HOT = QColor('#E5484D')
_W, _H = 1100, 760


class _NodeItem(QGraphicsEllipseItem):
    def __init__(self, nid, x, y, r, color):
        super().__init__(QRectF(x - r, y - r, 2 * r, 2 * r))
        self.nid = nid
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(2)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)


class KnowledgeGraphCanvas(QGraphicsView):
    node_clicked = pyqtSignal(dict)            # {'id','kind','label','payload'}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet('border:none;background:#FCFCFC;')
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._nodes: dict[str, dict] = {}      # id -> render node
        self._info: dict[str, dict] = {}       # id -> {kind,label,payload}
        self._edges: list[dict] = []
        self._node_items: dict[str, _NodeItem] = {}
        self._edge_items: list = []
        self._focused = None

    # -- API --
    def set_snapshot(self, nodes, edges):
        self._scene.clear()
        self._node_items.clear(); self._edge_items.clear()
        self._info = {n.id: {'id': n.id, 'kind': n.kind, 'label': n.label,
                             'payload': n.payload} for n in nodes}
        rm = render_model(nodes, edges, width=_W, height=_H)
        self._nodes = {n['id']: n for n in rm['nodes']}
        self._edges = rm['edges']
        for e in self._edges:                  # faint mass edges first
            a, b = self._nodes.get(e['src']), self._nodes.get(e['dst'])
            if not a or not b:
                continue
            line = self._scene.addLine(a['x'], a['y'], b['x'], b['y'], QPen(_EDGE, 0.5))
            line.setZValue(1); line.setData(0, (e['src'], e['dst']))
            self._edge_items.append(line)
        for n in self._nodes.values():         # nodes on top
            item = _NodeItem(n['id'], n['x'], n['y'], n['r'], n['color'])
            self._scene.addItem(item)
            self._node_items[n['id']] = item
        if self._nodes:
            self._scene.setSceneRect(self._scene.itemsBoundingRect())
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def focus(self, node_id):
        self._focused = node_id
        for line in self._edge_items:
            s, d = line.data(0)
            hot = node_id in (s, d)
            line.setPen(QPen(_EDGE_HOT if hot else _EDGE, 1.1 if hot else 0.5))
            line.setZValue(3 if hot else 1)

    def focused_edge_count(self) -> int:
        if not self._focused:
            return 0
        return sum(1 for e in self._edges if self._focused in (e['src'], e['dst']))

    def node_info(self, node_id) -> dict:
        return self._info.get(node_id, {})

    def node_item_count(self) -> int:
        return len(self._node_items)

    # -- interaction --
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, _NodeItem):
            self.focus(item.nid)
            self.node_clicked.emit(self._info.get(item.nid, {}))
        super().mousePressEvent(event)
