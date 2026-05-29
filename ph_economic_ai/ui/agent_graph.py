import math
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QWheelEvent


# ── Node type palette ─────────────────────────────────────────────────────────
_TYPES = {
    'DataSource': ('#4A90E2', 'Data Source'),
    'Processor':  ('#E0A84A', 'Processor'),
    'Feature':    ('#9B59B6', 'Feature'),
    'Model':      ('#27AE60', 'Model'),
    'Output':     ('#E74C3C', 'Output'),
    'UI':         ('#1ABC9C', 'UI Component'),
}

# (id, name, node_type, x, y, description, {props})
_NODES = [
    # ── Data Sources ──────────────────────────────────────────────────────────
    (0,  'Yahoo Finance',   'DataSource',  70,  230,
     'Unofficial v8 JSON API — no key required',
     {'auth': 'None', 'timeout': '8s', 'format': 'JSON'}),
    (1,  'BZ=F',            'DataSource', 210,   90,
     'Brent Crude front-month futures, monthly close',
     {'ticker': 'BZ=F', 'interval': '1mo', 'range': '5y', 'unit': 'USD/bbl'}),
    (2,  'PHP=X',           'DataSource', 210,  230,
     'USD/PHP spot exchange rate, monthly close',
     {'ticker': 'PHP=X', 'interval': '1mo', 'range': '5y', 'unit': 'PHP/USD'}),
    (3,  'RB=F',            'DataSource', 210,  370,
     'RBOB gasoline front-month futures, monthly close',
     {'ticker': 'RB=F', 'interval': '1mo', 'range': '5y', 'unit': 'USD/gal'}),

    # ── Processors ───────────────────────────────────────────────────────────
    (4,  'HTTP Fetcher',    'Processor',  380,  160,
     'Calls Yahoo Finance v8 chart endpoint with browser User-Agent',
     {'library': 'requests 2.32', 'user_agent': 'Chrome/120', 'raise_for_status': 'yes'}),
    (5,  'Deduplicator',    'Processor',  380,  290,
     'Removes duplicate YYYY-MM entries caused by contract rollovers',
     {'strategy': 'keep last', 'method': 'series[~index.duplicated(keep=last)]'}),
    (6,  'Gas Converter',   'Processor',  380,  420,
     'Converts RBOB USD/gal → estimated Philippine retail PHP/liter',
     {'formula': '(RBOB ÷ 3.785 × PHP/USD) × 1.35 + 12', 'calibrated': '2021–2025'}),
    (7,  'JSON Cache',      'Processor',  560,  450,
     '24-hour local JSON file cache — survives network outages',
     {'path': 'cache/data.json', 'ttl': '24h', 'states': 'Live / Cached / Cached·Stale'}),
    (8,  'Inner Joiner',    'Processor',  560,  230,
     'Aligns all three series on shared YYYY-MM dates via dropna()',
     {'rows': '~47 months', 'range': '2021-06 → 2026-05'}),
    (9,  'Demand Model',    'Processor',  380,  540,
     'Computes seasonal demand index from month number — no external data',
     {'formula': '65+17cos(2π(m-3)/12)+6cos(2π(m-12)/12)', 'range': '55–90'}),

    # ── Features ─────────────────────────────────────────────────────────────
    (10, 'oil_price',       'Feature',    720,  100,
     'Brent crude monthly close price',
     {'dtype': 'float64', 'unit': 'USD/bbl', 'range': '61–115'}),
    (11, 'usd_php',         'Feature',    720,  210,
     'USD/PHP monthly exchange rate',
     {'dtype': 'float64', 'unit': 'PHP/USD', 'range': '50–62'}),
    (12, 'gas_price',       'Feature',    720,  320,
     'Estimated RON 95 Metro Manila retail price',
     {'dtype': 'float64', 'unit': 'PHP/liter', 'range': '48–95'}),
    (13, 'demand_index',    'Feature',    720,  430,
     'Seasonal fuel demand index',
     {'dtype': 'float64', 'range': '55–90', 'peaks': 'Mar / Dec', 'trough': 'Jun–Jul'}),
    (14, 'prev_gas_price',  'Feature',    840,  265,
     'Lag-1 gas price — captures temporal price autocorrelation',
     {'dtype': 'float64', 'lag': '1 month'}),

    # ── Model ─────────────────────────────────────────────────────────────────
    (15, 'Preprocessor',    'Model',      960,  210,
     'Adds lag feature, drops NaN rows, splits 80/20 by time',
     {'features': '5 columns', 'target': 'gas_price', 'split': '80/20 time-ordered'}),
    (16, 'RandomForest',    'Model',     1090,  140,
     'Ensemble of 100 decision trees trained on monthly economic data',
     {'n_estimators': '100', 'random_state': '42', 'library': 'scikit-learn 1.7.2'}),
    (17, 'Confidence Est.', 'Model',     1090,  290,
     'Estimates prediction uncertainty from variance across individual trees',
     {'method': 'std of per-tree predictions', 'output': 'pred_std'}),
    (18, 'Pressure Calc',   'Model',      960,  380,
     'Weighted composite economic pressure score 0–100',
     {'oil_weight': '40%', 'fx_weight': '35%', 'demand_weight': '25%', 'bands': '4 levels'}),
    (19, 'Scenario Engine', 'Model',     1210,  100,
     'Re-runs the model under three hypothetical economic shocks',
     {'oil_shock': '+10%', 'usd_shock': '+5%', 'demand_drop': '−15%'}),
    (20, 'Explainer',       'Model',     1090,  420,
     'Rule-based narrative generator for pressure drivers',
     {'inputs': 'oil_delta, usd_delta, demand_norm, pressure_index', 'output': 'dict'}),

    # ── Outputs ───────────────────────────────────────────────────────────────
    (21, 'Predicted Price', 'Output',    1330,  100,
     'Forecast of next-month Philippine retail gasoline price',
     {'unit': 'PHP/liter', 'type': 'continuous regression'}),
    (22, 'Pressure Band',   'Output',    1330,  220,
     'Categorical pressure level derived from pressure index',
     {'values': 'Stable / Rising / High / Critical'}),
    (23, 'Explanation',     'Output',    1330,  360,
     'Natural language pressure narrative with advisory',
     {'fields': 'summary, advisory, risk_badge, drivers, expected_increase'}),
    (24, 'Scenarios',       'Output',    1330,  490,
     'Three what-if projected prices under different shock scenarios',
     {'count': '3 scenarios', 'format': 'dict[str, float]'}),

    # ── UI Components ─────────────────────────────────────────────────────────
    (25, 'Dashboard',       'UI',        1490,  160,
     'Main page: historical chart, stat cards, scenario simulation',
     {'chart': 'matplotlib FigureCanvasQTAgg', 'signals': 'recalculate, oil_shock'}),
    (26, 'Pressure Gauge',  'UI',        1490,  320,
     'Animated QPainter arc gauge displaying the pressure index',
     {'widget': 'PressureGaugePage', 'renderer': 'custom QPainter arcs'}),
    (27, 'Sidebar',         'UI',        1490,  460,
     'Navigation + color-coded data source status pill',
     {'pill': 'Live Data / Cached / Cached·Stale', 'nav_pages': '4'}),
]

_EDGES = [
    (0, 1), (0, 2), (0, 3),           # Yahoo Finance → tickers
    (1, 4), (2, 4), (3, 4),           # tickers → HTTP Fetcher
    (4, 5),                            # Fetcher → Deduplicator
    (3, 6), (2, 6),                    # RB=F + PHP=X → Gas Converter
    (6, 5),                            # Gas Converter → Deduplicator
    (5, 8), (9, 8),                    # Deduplicator + Demand → Inner Joiner
    (8, 7),                            # Inner Joiner → Cache
    (8, 10), (8, 11), (8, 12), (8, 13),  # Inner Joiner → Features
    (12, 14),                          # gas_price → lag feature
    (10, 15), (11, 15), (12, 15), (13, 15), (14, 15),  # Features → Preprocessor
    (15, 16),                          # Preprocessor → RandomForest
    (16, 17),                          # RandomForest → Confidence
    (16, 21),                          # RandomForest → Predicted Price
    (15, 18),                          # Preprocessor → Pressure Calc
    (18, 22),                          # Pressure Calc → Pressure Band
    (16, 19),                          # RandomForest → Scenario Engine
    (19, 24),                          # Scenario Engine → Scenarios
    (18, 20), (21, 20),                # Pressure Calc + Price → Explainer
    (20, 23),                          # Explainer → Explanation
    (21, 25), (22, 25), (23, 25), (24, 25), (17, 25),  # Outputs → Dashboard
    (22, 26),                          # Pressure Band → Pressure Gauge
    (7, 27),                           # Cache → Sidebar (status pill)
]

_NODE_RADIUS = 9
_EDGE_COLOR  = QColor('#FF4D8B')
_EDGE_ALPHA  = 140   # 0-255


# ── Graphics items ────────────────────────────────────────────────────────────

class _NodeItem(QGraphicsItem):
    def __init__(self, node_id: int, name: str, node_type: str,
                 x: float, y: float, on_click):
        super().__init__()
        self._id        = node_id
        self._name      = name
        self._type      = node_type
        self._on_click  = on_click
        self._color     = QColor(_TYPES[node_type][0])
        self._hover     = False
        self.setPos(x, y)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def boundingRect(self) -> QRectF:
        r = _NODE_RADIUS + 2
        return QRectF(-r, -r, r * 2, r * 2 + 22)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = _NODE_RADIUS
        color = self._color.lighter(115) if self._hover else self._color
        painter.setBrush(QBrush(color))
        pen_color = color.darker(140)
        painter.setPen(QPen(pen_color, 1.5))
        painter.drawEllipse(QRectF(-r, -r, r * 2, r * 2))

        # Label below circle
        f = QFont()
        f.setPointSize(7)
        painter.setFont(f)
        painter.setPen(QPen(QColor('#333333')))
        label_rect = QRectF(-40, r + 2, 80, 18)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                         self._name)

    def hoverEnterEvent(self, event):
        self._hover = True
        self.update()

    def hoverLeaveEvent(self, event):
        self._hover = False
        self.update()

    def mousePressEvent(self, event):
        self._on_click(self._id)
        super().mousePressEvent(event)


# ── Zoomable / pannable view ──────────────────────────────────────────────────

class _GraphView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setStyleSheet(
            'border: 1px solid #EAEAEA; border-radius:10px; background:#FAFAFA;'
        )
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self._zoom *= factor
        self._zoom = max(0.15, min(self._zoom, 5.0))
        self.scale(factor, factor)


# ── Detail panel ──────────────────────────────────────────────────────────────

class _DetailPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setStyleSheet(
            'QFrame { background:#FFFFFF; border:1px solid #EAEAEA;'
            ' border-radius:10px; }'
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(8)
        self._show_placeholder()

    def _clear(self):
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _show_placeholder(self):
        self._clear()
        lbl = QLabel('Click a node\nto see details')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet('color:#BBBBBB; font-size:12px;')
        self._layout.addWidget(lbl)
        self._layout.addStretch()

    def show_node(self, node_data: dict):
        self._clear()
        node_id, name, node_type, _, _, desc, props = node_data

        # Name
        name_lbl = QLabel(name)
        name_lbl.setWordWrap(True)
        name_lbl.setStyleSheet('font-size:14px; font-weight:700; color:#111111;')
        self._layout.addWidget(name_lbl)

        # Type badge
        color = _TYPES[node_type][0]
        label_text = _TYPES[node_type][1]
        badge = QLabel(f'  {label_text}  ')
        badge.setFixedHeight(22)
        badge.setStyleSheet(
            f'background:{color}; color:#FFFFFF; font-size:10px;'
            f' font-weight:600; border-radius:10px; padding:0 6px;'
        )
        badge.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._layout.addWidget(badge)

        # Divider
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet('color:#EAEAEA;')
        self._layout.addWidget(div)

        # Description
        desc_lbl = QLabel(desc)
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet('font-size:11px; color:#555555;')
        self._layout.addWidget(desc_lbl)

        # Properties
        if props:
            props_title = QLabel('Properties')
            props_title.setStyleSheet('font-size:10px; font-weight:700; color:#888888;'
                                      ' margin-top:6px; letter-spacing:1px;')
            self._layout.addWidget(props_title)

            for k, v in props.items():
                row = QWidget()
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 1, 0, 1)
                key_lbl = QLabel(k)
                key_lbl.setStyleSheet('font-size:10px; color:#999999;')
                val_lbl = QLabel(str(v))
                val_lbl.setWordWrap(True)
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
                val_lbl.setStyleSheet('font-size:10px; color:#333333; font-weight:600;')
                row_layout.addWidget(key_lbl)
                row_layout.addStretch()
                row_layout.addWidget(val_lbl)
                self._layout.addWidget(row)

        self._layout.addStretch()


# ── Legend ────────────────────────────────────────────────────────────────────

class _Legend(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 0)
        layout.setSpacing(18)

        title = QLabel('NODE TYPES')
        title.setStyleSheet('font-size:9px; font-weight:700; color:#AAAAAA; letter-spacing:1px;')
        layout.addWidget(title)

        for node_type, (color, label) in _TYPES.items():
            dot = QLabel('●')
            dot.setStyleSheet(f'color:{color}; font-size:14px;')
            txt = QLabel(label)
            txt.setStyleSheet('font-size:10px; color:#555555;')
            layout.addWidget(dot)
            layout.addWidget(txt)

        layout.addStretch()

        hint = QLabel('Scroll to zoom  ·  Drag to pan  ·  Click node for details')
        hint.setStyleSheet('font-size:10px; color:#BBBBBB;')
        layout.addWidget(hint)


# ── Main page ─────────────────────────────────────────────────────────────────

class AgentGraphPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._node_lookup = {n[0]: n for n in _NODES}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 16)
        root.setSpacing(10)

        # Header
        title = QLabel('Agent Network')
        title.setStyleSheet('font-size:20px; font-weight:700; color:#111111;')
        root.addWidget(title)

        sub = QLabel('Full data pipeline — from API sources through ML model to UI outputs')
        sub.setStyleSheet('font-size:12px; color:#888888;')
        root.addWidget(sub)

        # Graph + detail panel (horizontal split)
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)

        # Build scene
        scene = QGraphicsScene()
        scene.setBackgroundBrush(QColor('#FAFAFA'))

        # Draw edges first (z = -1)
        node_positions = {n[0]: QPointF(n[3], n[4]) for n in _NODES}
        edge_pen = QPen(_EDGE_COLOR, 1.0)
        edge_color_alpha = QColor(_EDGE_COLOR)
        edge_color_alpha.setAlpha(_EDGE_ALPHA)
        edge_pen.setColor(edge_color_alpha)

        for src_id, dst_id in _EDGES:
            sp = node_positions[src_id]
            dp = node_positions[dst_id]
            line = scene.addLine(sp.x(), sp.y(), dp.x(), dp.y(), edge_pen)
            line.setZValue(-1)

        # Draw nodes
        for node_data in _NODES:
            node_id, name, node_type, x, y, _, _ = node_data
            item = _NodeItem(node_id, name, node_type, x, y, self._on_node_click)
            item.setZValue(1)
            scene.addItem(item)

        # Expand scene rect with padding
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-40, -40, 40, 60))

        # View
        self._view = _GraphView(scene)
        self._view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Detail panel
        self._detail = _DetailPanel()

        body_layout.addWidget(self._view, stretch=1)
        body_layout.addWidget(self._detail)
        root.addWidget(body, stretch=1)

        # Legend
        root.addWidget(_Legend())

    def _on_node_click(self, node_id: int):
        self._detail.show_node(self._node_lookup[node_id])
