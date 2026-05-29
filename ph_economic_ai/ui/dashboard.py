import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QPushButton, QScrollArea, QSizePolicy,
                              QProgressBar, QTabWidget, QGridLayout)
from PyQt6.QtCore import pyqtSignal, Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from ph_economic_ai.ui.charts import PriceChart
from ph_economic_ai.ui.pressure import PressureGauge, pressure_band_color


# ── Indicator config (shared by build and update methods) ─────────────────────

_INDICATOR_SPECS = [
    ('oil_price',    'Oil Price',         'USD/bbl', '#4A90E2'),
    ('usd_php',      'USD/PHP Rate',      'PHP/USD', '#E0A84A'),
    ('demand_index', 'Demand Index',      'Index',   '#27AE60'),
    ('psei',         'PSEi',              'Points',  '#9B59B6'),
    ('cpi',          'CPI Inflation',     '% p.a.',  '#E74C3C'),
    ('bsp_rate',     'BSP Lending Rate',  '% p.a.',  '#1ABC9C'),
    ('remittances',  'OFW Remittances',   'USD bn',  '#E07A4A'),
]
_INDICATOR_META = {key: (title, ylabel, color) for key, title, ylabel, color in _INDICATOR_SPECS}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setStyleSheet('background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;')
    return f


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        'font-size:10px; font-weight:700; color:#BBBBBB;'
        'text-transform:uppercase; letter-spacing:0.9px;'
    )
    return lbl


# ── Mini summary card ─────────────────────────────────────────────────────────

class _MiniCard(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)
        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            'font-size:10px; color:#AAAAAA; letter-spacing:0.5px; border:none;'
        )
        self._value = QLabel('—')
        self._value.setStyleSheet('font-size:19px; font-weight:700; color:#111111; border:none;')
        self._badge = QLabel('')
        self._badge.setStyleSheet(
            'font-size:10px; font-weight:600; padding:2px 8px;'
            'border-radius:10px; border:none;'
        )
        layout.addWidget(self._label)
        layout.addWidget(self._value)
        layout.addWidget(self._badge)

    def update(self, value: str, badge: str, val_color: str, badge_bg: str, badge_color: str):
        self._value.setText(value)
        self._value.setStyleSheet(
            f'font-size:19px; font-weight:700; color:{val_color}; border:none;'
        )
        self._badge.setText(badge)
        self._badge.setStyleSheet(
            f'font-size:10px; font-weight:600; padding:2px 8px; border-radius:10px;'
            f'background:{badge_bg}; color:{badge_color}; border:none;'
        )


# ── If-Then Simulation panel ──────────────────────────────────────────────────

class _SimScenario(QFrame):
    def __init__(self, if_text: str, explanation: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet('border:none;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        if_lbl = QLabel('IF')
        if_lbl.setStyleSheet('font-size:9px; font-weight:700; color:#CCCCCC; border:none;')
        layout.addWidget(if_lbl)

        self._cond = QLabel(if_text)
        self._cond.setStyleSheet('font-size:12px; font-weight:600; color:#333333; border:none;')
        self._cond.setWordWrap(True)
        layout.addWidget(self._cond)

        arrow_row = QHBoxLayout()
        arrow_lbl = QLabel('→')
        arrow_lbl.setStyleSheet('font-size:16px; color:#DDDDDD; border:none;')
        arrow_row.addWidget(arrow_lbl)
        self._impact = QLabel('—')
        self._impact.setStyleSheet('font-size:20px; font-weight:700; border:none;')
        arrow_row.addWidget(self._impact)
        arrow_row.addStretch()
        layout.addLayout(arrow_row)

        self._unit = QLabel('per liter')
        self._unit.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        layout.addWidget(self._unit)

        self._bar = QProgressBar()
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        self._bar.setStyleSheet(
            'QProgressBar { background:#F0F0F0; border-radius:2px; border:none; }'
            'QProgressBar::chunk { border-radius:2px; }'
        )
        layout.addWidget(self._bar)

        self._desc = QLabel(explanation)
        self._desc.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

    def set_delta(self, delta: float, max_abs: float, up_color: str, down_color: str):
        color = up_color if delta >= 0 else down_color
        sign = '+' if delta >= 0 else '−'
        self._impact.setText(f'{sign}₱{abs(delta):.2f} / L')
        self._impact.setStyleSheet(f'font-size:20px; font-weight:700; color:{color}; border:none;')
        chunk_color = up_color if delta >= 0 else down_color
        self._bar.setStyleSheet(
            'QProgressBar { background:#F0F0F0; border-radius:2px; border:none; }'
            f'QProgressBar::chunk {{ border-radius:2px; background:{chunk_color}; }}'
        )
        pct = int(min(abs(delta) / max(max_abs, 0.01) * 100, 100))
        self._bar.setValue(pct)


class SimulationPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            'background:#FFFFFF; border:none; border-bottom:1px solid #EAEAEA;'
            'border-top-left-radius:10px; border-top-right-radius:10px;'
        )
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 10, 18, 10)
        icon = QLabel('🔮')
        icon.setStyleSheet('font-size:15px; border:none;')
        title = QLabel('If-Then Simulation')
        title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        badge = QLabel('Scenario impact on gas price')
        badge.setStyleSheet(
            'font-size:10px; font-weight:600; color:#888888;'
            'background:#F5F5F5; padding:3px 9px; border-radius:10px; border:none;'
        )
        hh.addWidget(icon)
        hh.addWidget(title)
        hh.addStretch()
        hh.addWidget(badge)
        layout.addWidget(header)

        # Three scenario columns
        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(0)

        self._oil_col = _SimScenario(
            'Oil prices rise +5%',
            'Higher crude input cost flows directly into refinery output pricing.'
        )
        self._usd_col = _SimScenario(
            'USD strengthens +2% vs PHP',
            'Dollar-denominated imports become more expensive in peso terms.'
        )
        self._dem_col = _SimScenario(
            'Demand index drops 10 pts',
            'Reduced consumption eases upward pressure, softening the price.'
        )

        for i, col in enumerate([self._oil_col, self._usd_col, self._dem_col]):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet('color:#F5F5F5; border:none; border-left:1px solid #F5F5F5;')
                cols.addWidget(sep)
            cols.addWidget(col)

        layout.addLayout(cols)

    def refresh(self, scenarios: dict):
        max_abs = max(abs(v) for v in scenarios.values()) or 1.0
        self._oil_col.set_delta(scenarios['oil_shock'],   max_abs, '#E07A4A', '#4AAE90')
        self._usd_col.set_delta(scenarios['usd_shock'],   max_abs, '#E07A4A', '#4AAE90')
        self._dem_col.set_delta(scenarios['demand_drop'], max_abs, '#E07A4A', '#4AAE90')


# ── Right panel (gauge + drivers + advisory) ──────────────────────────────────

class _RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet('background:#FFFFFF; border-left:1px solid #EAEAEA;')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea { border:none; } QScrollBar { width:0px; }')
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setStyleSheet('background:#FFFFFF;')
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        scroll.setWidget(inner)

        self._build_gauge_section()
        self._build_drivers_section()
        self._build_advisory_section()
        self._layout.addStretch()

    # ── Gauge + bands ─────────────────────────────────────────────────────────
    def _build_gauge_section(self):
        sec = QFrame()
        sec.setStyleSheet('background:#FFFFFF; border-bottom:1px solid #EAEAEA;')
        lyt = QVBoxLayout(sec)
        lyt.setContentsMargins(18, 16, 18, 14)
        lyt.setSpacing(8)
        lyt.addWidget(_section_title('Economic Pressure Index'))

        self._gauge = PressureGauge(size=100)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self._gauge)
        row.addStretch()
        lyt.addLayout(row)

        bands_grid = QHBoxLayout()
        bands_grid.setSpacing(4)
        self._band_frames: dict[str, QFrame] = {}
        self._band_labels: dict[str, QLabel] = {}
        configs = [
            ('Stable',   '0–30',   '#EBF4FF', '#4A90E2'),
            ('Rising',   '31–60',  '#FFF8EE', '#E0A84A'),
            ('High',     '61–80',  '#FFF3EE', '#E07A4A'),
            ('Critical', '81–100', '#FFEFEE', '#E05040'),
        ]
        for name, rng, bg, color in configs:
            f = QFrame()
            f.setStyleSheet(f'background:{bg}; border-radius:6px; border:1px solid #EAEAEA;')
            fl = QVBoxLayout(f)
            fl.setContentsMargins(6, 5, 6, 5)
            fl.setSpacing(1)
            nl = QLabel(name)
            nl.setStyleSheet(f'font-size:9px; font-weight:700; color:{color}; border:none;')
            rl = QLabel(rng)
            rl.setStyleSheet(f'font-size:9px; color:{color}; border:none;')
            fl.addWidget(nl)
            fl.addWidget(rl)
            bands_grid.addWidget(f)
            self._band_frames[name] = f
            self._band_labels[name] = nl

        self._now_label = QLabel('← NOW')
        lyt.addLayout(bands_grid)
        self._layout.addWidget(sec)

    # ── Drivers ───────────────────────────────────────────────────────────────
    def _build_drivers_section(self):
        sec = QFrame()
        sec.setStyleSheet('background:#FFFFFF; border-bottom:1px solid #EAEAEA;')
        lyt = QVBoxLayout(sec)
        lyt.setContentsMargins(18, 14, 18, 14)
        lyt.setSpacing(6)
        lyt.addWidget(_section_title('Key Drivers'))

        self._driver_rows: list[tuple[QLabel, QLabel, QLabel, QLabel, QHBoxLayout]] = []
        for _ in range(3):
            row = QHBoxLayout()
            row.setSpacing(8)
            icon_lbl = QLabel()
            icon_lbl.setStyleSheet('font-size:14px; border:none;')
            icon_lbl.setFixedWidth(20)
            text_col = QVBoxLayout()
            text_col.setSpacing(1)
            name_lbl = QLabel()
            name_lbl.setStyleSheet('font-size:12px; font-weight:600; color:#111111; border:none;')
            val_lbl = QLabel()
            val_lbl.setStyleSheet('font-size:10px; color:#888888; border:none;')
            text_col.addWidget(name_lbl)
            text_col.addWidget(val_lbl)
            status_lbl = QLabel()
            status_lbl.setStyleSheet('font-size:11px; font-weight:700; border:none;')
            row.addWidget(icon_lbl)
            row.addLayout(text_col)
            row.addStretch()
            row.addWidget(status_lbl)
            self._driver_rows.append((icon_lbl, name_lbl, val_lbl, status_lbl, row))
            lyt.addLayout(row)

        self._risk_badge = QLabel()
        self._risk_badge.setStyleSheet(
            'font-size:10px; font-weight:700; padding:4px 10px; border-radius:10px; border:none;'
        )
        lyt.addWidget(self._risk_badge)
        self._summary = QLabel()
        self._summary.setStyleSheet('font-size:11px; color:#555555; border:none;')
        self._summary.setWordWrap(True)
        lyt.addWidget(self._summary)
        self._layout.addWidget(sec)

    # ── Advisory ──────────────────────────────────────────────────────────────
    def _build_advisory_section(self):
        self._advisory_card = QFrame()
        self._advisory_card.setStyleSheet(
            'background:qlineargradient(x1:0,y1:0,x2:1,y2:1,'
            'stop:0 #EBF4FF, stop:1 #F0F7FF);'
            'border:1px solid #C8DEF5; border-radius:10px; margin:14px 18px 0px 18px;'
        )
        lyt = QVBoxLayout(self._advisory_card)
        lyt.setContentsMargins(14, 12, 14, 12)
        lyt.setSpacing(6)

        hdr = QHBoxLayout()
        self._adv_icon = QLabel('💡')
        self._adv_icon.setStyleSheet('font-size:14px; border:none;')
        adv_title = QLabel('Advisory Output')
        adv_title.setStyleSheet(
            'font-size:12px; font-weight:700; color:#4A90E2; border:none;'
        )
        hdr.addWidget(self._adv_icon)
        hdr.addWidget(adv_title)
        hdr.addStretch()
        lyt.addLayout(hdr)

        for attr, label in [
            ('_adv_increase', 'Expected change'),
            ('_adv_timing',   'Price adjustment'),
            ('_adv_window',   'Risk window'),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('font-size:11px; color:#666666; border:none;')
            val = QLabel('—')
            val.setStyleSheet('font-size:11px; font-weight:700; color:#111111; border:none;')
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            lyt.addLayout(row)
            setattr(self, attr, val)

        self._action_btn = QLabel()
        self._action_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._action_btn.setStyleSheet(
            'background:#4A90E2; color:#FFFFFF; border-radius:6px; border:none;'
            'font-size:11px; font-weight:600; padding:8px 12px; margin-top:4px;'
        )
        lyt.addWidget(self._action_btn)
        self._layout.addWidget(self._advisory_card)

    # ── Public refresh ────────────────────────────────────────────────────────
    def refresh(self, result: dict):
        self._gauge.set_value(result['pressure_index'])
        band = result['pressure_band']

        for name, frame in self._band_frames.items():
            active = name == band
            color = {'Stable':'#4A90E2','Rising':'#E0A84A','High':'#E07A4A','Critical':'#E05040'}[name]
            bg    = {'Stable':'#EBF4FF','Rising':'#FFF8EE','High':'#FFF3EE','Critical':'#FFEFEE'}[name]
            border = f'border:1.5px solid {color};' if active else 'border:1px solid #EAEAEA;'
            frame.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')

        expl = result['explanation']
        for i, d in enumerate(expl['drivers']):
            icon_lbl, name_lbl, val_lbl, status_lbl, _ = self._driver_rows[i]
            icon_lbl.setText(d['icon'])
            name_lbl.setText(d['name'])
            val_lbl.setText(d['value'])
            status_lbl.setText(d['status'])
            status_lbl.setStyleSheet(
                f'font-size:11px; font-weight:700; color:{d["color"]}; border:none;'
            )

        self._risk_badge.setText(expl['risk_badge'])
        c = expl['risk_color']
        bg = {
            '#E07A4A': '#FFF3EE', '#E0A84A': '#FFF8EE', '#4A90E2': '#EBF4FF'
        }.get(c, '#F5F5F5')
        self._risk_badge.setStyleSheet(
            f'font-size:10px; font-weight:700; padding:4px 10px; border-radius:10px;'
            f'background:{bg}; color:{c}; border:1px solid {c};'
        )
        self._summary.setText(expl['summary'])

        self._adv_icon.setText(expl['advisory_icon'])
        self._adv_increase.setText(expl['expected_increase'])
        idx = result['pressure_index']
        self._adv_timing.setText('~48–72 hours' if idx > 60 else ('~1 week' if idx > 30 else 'Stable'))
        self._adv_window.setText(
            'High (next 7 days)' if idx > 60 else ('Medium' if idx > 30 else 'Low')
        )
        self._action_btn.setText(f'{expl["advisory_icon"]}  Suggested: {expl["advisory"]}')


# ── Dashboard page ────────────────────────────────────────────────────────────

class DashboardPage(QWidget):
    recalculate_requested = pyqtSignal()
    oil_shock_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#FAFAFA;')

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(
            'QTabBar::tab { padding: 8px 22px; font-size: 12px; color: #888888; }'
            'QTabBar::tab:selected { color: #111111; font-weight: 700; '
            '  border-bottom: 2px solid #4A90E2; }'
            'QTabWidget::pane { border: none; }'
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._tabs)

        self._build_overview_tab()
        self._build_indicators_tab()
        self._build_model_insights_tab()

    def _build_overview_tab(self):
        overview = QWidget()
        overview.setStyleSheet('background:#FAFAFA;')
        main = QHBoxLayout(overview)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── Center column ─────────────────────────────────────────────────────
        center = QWidget()
        center.setStyleSheet('background:#FAFAFA;')
        clyt = QVBoxLayout(center)
        clyt.setContentsMargins(22, 20, 22, 20)
        clyt.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        pg_title = QLabel('Gasoline Price Dashboard')
        pg_title.setStyleSheet('font-size:18px; font-weight:700; color:#111111;')
        pg_sub = QLabel('Philippines · Live data · Trained on startup')
        pg_sub.setStyleSheet('font-size:11px; color:#AAAAAA;')
        title_col.addWidget(pg_title)
        title_col.addWidget(pg_sub)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._recalc_btn = QPushButton('↺  Recalculate')
        self._recalc_btn.setStyleSheet(
            'padding:7px 14px; font-size:11px; font-weight:600; border-radius:8px;'
            'border:1px solid #4A90E2; background:#FFFFFF; color:#4A90E2;'
        )
        self._recalc_btn.clicked.connect(self.recalculate_requested)
        self._shock_btn = QPushButton('⚡  Oil Shock +10%')
        self._shock_btn.setStyleSheet(
            'padding:7px 14px; font-size:11px; font-weight:600; border-radius:8px;'
            'border:1px solid #E07A4A; background:#FFFFFF; color:#E07A4A;'
        )
        self._shock_btn.clicked.connect(self.oil_shock_requested)
        btn_row.addWidget(self._recalc_btn)
        btn_row.addWidget(self._shock_btn)

        hdr.addLayout(title_col)
        hdr.addStretch()
        hdr.addLayout(btn_row)
        clyt.addLayout(hdr)

        self._chart = PriceChart()
        clyt.addWidget(self._chart)

        mini_row = QHBoxLayout()
        mini_row.setSpacing(10)
        self._price_card  = _MiniCard('Predicted Price')
        self._trend_card  = _MiniCard('Trend Direction')
        self._index_card  = _MiniCard('Pressure Index')
        for card in (self._price_card, self._trend_card, self._index_card):
            mini_row.addWidget(card)
        clyt.addLayout(mini_row)

        self._sim_panel = SimulationPanel()
        clyt.addWidget(self._sim_panel)

        self._right = _RightPanel()

        main.addWidget(center, stretch=1)
        main.addWidget(self._right)

        self._tabs.addTab(overview, 'Overview')

    def _build_indicators_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border:none; } QScrollBar { width:0px; }')

        container = QWidget()
        container.setStyleSheet('background:#FAFAFA;')
        grid = QGridLayout(container)
        grid.setContentsMargins(22, 20, 22, 20)
        grid.setSpacing(14)

        self._indicator_axes = {}
        self._indicator_canvases = {}

        for i, (key, title, ylabel, color) in enumerate(_INDICATOR_SPECS):
            fig = Figure(figsize=(4, 2), dpi=96)
            fig.patch.set_facecolor('#FFFFFF')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=10, fontweight='bold', color='#111111', pad=4)
            ax.set_ylabel(ylabel, fontsize=8, color='#888888')
            ax.tick_params(labelsize=7, colors='#888888')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)
            fig.tight_layout(pad=1.2)

            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(200)

            row_i, col_i = divmod(i, 2)
            grid.addWidget(canvas, row_i, col_i)

            self._indicator_axes[key] = ax
            self._indicator_canvases[key] = canvas

        scroll.setWidget(container)
        self._tabs.addTab(scroll, 'Indicators')

    def _build_model_insights_tab(self):
        container = QWidget()
        container.setStyleSheet('background:#FAFAFA;')
        lyt = QVBoxLayout(container)
        lyt.setContentsMargins(22, 20, 22, 20)
        lyt.setSpacing(16)

        # ── Feature importance card ───────────────────────────────────────────
        fi_frame = QFrame()
        fi_frame.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        fi_inner = QVBoxLayout(fi_frame)
        fi_inner.setContentsMargins(16, 14, 16, 14)
        fi_inner.setSpacing(8)

        fi_title = QLabel('Feature Importance')
        fi_title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        fi_inner.addWidget(fi_title)

        fi_sub = QLabel('Which signals drive the gas price model most?  Blue = original · Purple = new')
        fi_sub.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        fi_inner.addWidget(fi_sub)

        self._fi_fig = Figure(figsize=(6, 2.2), dpi=96)
        self._fi_fig.patch.set_facecolor('#FFFFFF')
        self._fi_ax = self._fi_fig.add_subplot(111)
        self._fi_fig.tight_layout(pad=1.2)
        self._fi_canvas = FigureCanvasQTAgg(self._fi_fig)
        self._fi_canvas.setFixedHeight(220)
        fi_inner.addWidget(self._fi_canvas)
        lyt.addWidget(fi_frame)

        # ── Forecast card ─────────────────────────────────────────────────────
        fc_frame = QFrame()
        fc_frame.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        fc_inner = QVBoxLayout(fc_frame)
        fc_inner.setContentsMargins(16, 14, 16, 14)
        fc_inner.setSpacing(8)

        fc_title = QLabel('6-Month Gas Price Forecast  (indicative)')
        fc_title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        fc_inner.addWidget(fc_title)

        self._fc_fig = Figure(figsize=(6, 2.2), dpi=96)
        self._fc_fig.patch.set_facecolor('#FFFFFF')
        self._fc_ax = self._fc_fig.add_subplot(111)
        self._fc_fig.tight_layout(pad=1.2)
        self._fc_canvas = FigureCanvasQTAgg(self._fc_fig)
        self._fc_canvas.setFixedHeight(220)
        fc_inner.addWidget(self._fc_canvas)

        self._rmse_label = QLabel()
        self._rmse_label.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        fc_inner.addWidget(self._rmse_label)

        lyt.addWidget(fc_frame)
        lyt.addStretch()

        self._tabs.addTab(container, 'Model Insights')

    def _update_model_insights(self, result: dict):
        _ORIGINAL = {'oil_price', 'usd_php', 'demand_index', 'prev_gas_price'}

        # ── Feature importance ────────────────────────────────────────────────
        fi = result.get('feature_importances', {})
        if fi:
            ax = self._fi_ax
            ax.clear()
            ax.set_facecolor('#FFFFFF')

            names = list(fi.keys())
            values = list(fi.values())
            colors = ['#4A90E2' if n in _ORIGINAL else '#9B59B6' for n in names]

            bars = ax.barh(names, values, color=colors, height=0.55)
            ax.set_xlabel('Normalized importance', fontsize=8, color='#888888')
            ax.tick_params(labelsize=8, colors='#333333')
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            ax.spines['left'].set_color('#EEEEEE')
            ax.set_xlim(0, max(values) * 1.18 if values else 1)
            for bar, val in zip(bars, values):
                ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=7, color='#888888')

            self._fi_canvas.draw()

        # ── Forecast ──────────────────────────────────────────────────────────
        forecast_prices = result.get('forecast_prices')
        cv_rmse = result.get('cv_rmse', 0.0)
        df = result.get('df')
        if forecast_prices is not None and df is not None and len(df) > 0:
            ax = self._fc_ax
            ax.clear()
            ax.set_facecolor('#FFFFFF')

            hist_dates = df['date'].tolist()[-12:]
            hist_prices = df['gas_price'].values[-12:].astype(float)
            n_hist = len(hist_dates)

            ax.plot(range(n_hist), hist_prices, color='#AAAAAA', linewidth=1.5,
                    label='Historical (last 12 mo)')

            fc_x = list(range(n_hist - 1, n_hist + len(forecast_prices)))
            fc_y = np.concatenate([[hist_prices[-1]], forecast_prices])
            ax.plot(fc_x, fc_y, color='#4A90E2', linewidth=2, label='Forecast')
            ax.fill_between(fc_x, fc_y - cv_rmse, fc_y + cv_rmse,
                            color='#4A90E2', alpha=0.15,
                            label=f'±₱{cv_rmse:.2f} CV-RMSE')

            tick_pos = list(range(0, n_hist, max(1, n_hist // 4)))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([hist_dates[i] for i in tick_pos], rotation=30, fontsize=7)
            ax.set_ylabel('₱ / liter', fontsize=8, color='#888888')
            ax.tick_params(colors='#888888', labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)
            ax.legend(fontsize=7, framealpha=0, loc='upper left')

            self._fc_canvas.draw()

            self._rmse_label.setText(
                f'CV-RMSE: ₱{cv_rmse:.2f}/liter  ·  '
                'Forecast uses flat projection of latest known input values'
            )
        else:
            self._rmse_label.setText('')

    def refresh(self, result: dict):
        # Chart
        self._chart.update_data(
            dates=result['df']['date'].tolist(),
            actuals=result['df']['gas_price'].values,
            train_means=result['train_means'],
            train_stds=result['train_stds'],
            predicted_price=result['predicted_price'],
            pred_std=result['pred_std'],
        )

        # Mini cards
        pp = result['predicted_price']
        cp = result['current_price']
        diff_pct = (pp - cp) / max(cp, 1) * 100
        sign = '+' if diff_pct >= 0 else ''
        self._price_card.update(
            f'₱{pp:.2f}', f'▲ {sign}{diff_pct:.1f}% vs current',
            '#4A90E2', '#FFF3EE', '#E07A4A'
        )

        trend = result['trend']
        trend_color = '#E07A4A' if trend == 'Rising' else ('#4AAE90' if trend == 'Falling' else '#888888')
        self._trend_card.update(
            f'{trend} {"▲" if trend=="Rising" else ("▼" if trend=="Falling" else "→")}',
            f'{result["confidence"]:.0f}% confidence',
            trend_color, '#EBF4FF', '#4A90E2'
        )

        idx = result['pressure_index']
        band = result['pressure_band']
        idx_color = pressure_band_color(idx)
        self._index_card.update(
            f'{idx:.0f} / 100', f'{band} Zone',
            idx_color,
            '#FFEFEE' if idx > 60 else '#FFF8EE',
            '#E05040' if idx > 80 else '#E07A4A'
        )

        # Simulation panel
        self._sim_panel.refresh(result['scenarios'])

        # Right panel
        self._right.refresh(result)
        self._update_indicators(result)
        self._update_model_insights(result)

    def _update_indicators(self, result: dict):
        df_raw = result.get('df_raw')
        if df_raw is None:
            return

        dates = df_raw['date'].tolist()
        tick_pos = list(range(0, len(dates), max(1, len(dates) // 5)))
        tick_labels = [dates[i] for i in tick_pos]

        for key, ax in self._indicator_axes.items():
            if key not in df_raw.columns:
                continue
            title, ylabel, color = _INDICATOR_META[key]
            values = df_raw[key].values.astype(float)
            x = list(range(len(values)))

            ax.clear()
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=10, fontweight='bold', color='#111111', pad=4)
            ax.set_ylabel(ylabel, fontsize=8, color='#888888')
            ax.plot(x, values, color=color, linewidth=1.5)
            ax.fill_between(x, float(values.min()), values, color=color, alpha=0.08)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)
            ax.tick_params(labelsize=7, colors='#888888')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)

            self._indicator_canvases[key].draw()
