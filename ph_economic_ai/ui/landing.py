"""Home / Landing — editorial-style entry view.

Layout:
  ┌──────────────────────────────────────────────────────────────────┐
  │  PH ECONOMIC AI  / v2.0                       FEEDS ● 5/6  03:12 │  thin nav bar
  ├──────────────────────────────────────────────────────────────────┤
  │                                                                  │
  │  Philippine economic         ┌─────────────────────────────┐     │
  │  forecasting,                │  BRENT     $72.40    +0.6%  │     │
  │  on autopilot.               │  USD/PHP   ₱57.82    -0.1%  │     │
  │                              │  PUMP AVG  ₱98.82          │     │
  │  Brief body paragraph.       │                             │     │
  │                              │  ──────────                 │     │
  │  [▶ RUN SWARM]               │  Manila     31.8°C          │     │
  │                              │  7d rain    134mm           │     │
  │                              └─────────────────────────────┘     │
  │                                                                  │
  │  20 agents · 14 RAG sources · 3 sectors                          │
  ├──────────────────────────────────────────────────────────────────┤
  │  RECENT WORK                                                     │
  │  #5  May 29  +₱0.42/L 78%       #4  May 28  +₱1.12/L 81%  ...    │
  ├──────────────────────────────────────────────────────────────────┤
  │  Local-first · Ollama · Open-Meteo · World Bank · EIA      MIT   │
  └──────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QSizePolicy, QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtGui import QFont, QFontDatabase, QColor


# ── Palette — light, refined, single accent ──────────────────────────────────
BG          = '#FAFAF8'        # warm near-white (paper)
PAPER       = '#FFFFFF'
INK         = '#0F1115'        # near-black headline ink
TEXT_2      = '#4A5568'
TEXT_3      = '#8B95A7'
DIVIDER     = '#E5E7EB'
HAIRLINE    = '#EFF0F3'
ACCENT      = '#0F1115'        # single accent — dark CTA
ACCENT_HOV  = '#2A2F38'
UP          = '#127145'        # subtle green for positive numbers
DOWN        = '#B0322A'        # subtle red for negative numbers


def _serif_font(size: int, bold: bool = True) -> QFont:
    """Pick the best serif available — Georgia is universal on Windows/Mac."""
    family = 'Georgia'
    f = QFont(family, size)
    f.setBold(bold)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


# ─────────────────────────────────────────────────────────────────────────────
class _DashboardFetchThread(QThread):
    ready = pyqtSignal(object, object)   # brief, pump_price

    def run(self):
        brief = None
        pump_price = None
        try:
            from ph_economic_ai.engine.live_data import LiveDataBrief
            brief = LiveDataBrief().fetch()
        except Exception as e:
            print(f'landing: live brief fetch failed: {e}')
        try:
            from ph_economic_ai.engine.swarm import fetch_live_retail_price
            pump_price = fetch_live_retail_price()
        except Exception:
            pass
        self.ready.emit(brief, pump_price)


# ─────────────────────────────────────────────────────────────────────────────
class LandingPanel(QWidget):
    run_requested = pyqtSignal()
    view_performance_requested = pyqtSignal()
    view_overview_requested = pyqtSignal()

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._store = store
        self.setStyleSheet(f'background:{BG};')
        self._build()

        self._fetch_thread: Optional[_DashboardFetchThread] = None
        QTimer.singleShot(80, self._refresh_async)
        self._clock = QTimer(self)
        self._clock.setInterval(30_000)
        self._clock.timeout.connect(self._tick_clock)
        self._clock.start()

    # ── Build ────────────────────────────────────────────────────────────────
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top nav is now global (lives in main_window), not embedded here.
        outer.addWidget(self._build_hero())
        outer.addWidget(self._build_recent_strip())
        outer.addWidget(self._build_how_it_works())
        outer.addWidget(self._build_footer())

    # ── Nav (top thin bar) ───────────────────────────────────────────────────
    def _build_nav(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(52)
        bar.setStyleSheet(
            f'QFrame{{background:{PAPER};border-bottom:1px solid {DIVIDER};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )
        h = QHBoxLayout(bar)
        h.setContentsMargins(40, 0, 40, 0)
        h.setSpacing(18)

        brand = QLabel('STRATA')
        brand.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:12px;font-weight:700;'
            f'color:{INK};letter-spacing:5px;'
        )
        h.addWidget(brand)

        ver = QLabel('/ v2.0')
        ver.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_3};'
            f'letter-spacing:1.5px;'
        )
        h.addWidget(ver)
        h.addStretch()

        # Right-side small navigation links
        for label, sig in (
            ('Overview', 'view_overview_requested'),
            ('Performance', 'view_performance_requested'),
        ):
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFlat(True)
            btn.setStyleSheet(
                f'QPushButton{{background:transparent;color:{TEXT_2};'
                f'border:none;font-size:11px;letter-spacing:0.3px;padding:6px 12px;}}'
                f'QPushButton:hover{{color:{INK};}}'
            )
            btn.clicked.connect(getattr(self, sig).emit)
            h.addWidget(btn)

        # Vertical divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f'color:{DIVIDER}; background:{DIVIDER};')
        sep.setFixedWidth(1)
        sep.setFixedHeight(20)
        h.addWidget(sep)

        # Feeds + clock
        self._feeds_lbl = QLabel('FEEDS  ○ ○ ○ ○ ○ ○')
        self._feeds_lbl.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:600;'
            f'color:{TEXT_2};letter-spacing:1.4px;padding-left:12px;'
        )
        h.addWidget(self._feeds_lbl)

        self._time_lbl = QLabel(datetime.now().strftime('%H:%M'))
        self._time_lbl.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:600;'
            f'color:{TEXT_2};letter-spacing:1.4px;padding-left:8px;'
        )
        h.addWidget(self._time_lbl)
        return bar

    def _tick_clock(self):
        if hasattr(self, '_time_lbl'):
            self._time_lbl.setText(datetime.now().strftime('%H:%M'))

    # ── Hero (the big editorial layout) ──────────────────────────────────────
    def _build_hero(self) -> QWidget:
        hero = QWidget()
        hero.setStyleSheet(f'background:{BG};')

        outer = QHBoxLayout(hero)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # The hero is centered in a max-width container, like a real website
        outer.addStretch(1)

        center = QWidget()
        center.setMaximumWidth(1180)
        center.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(60, 60, 60, 60)
        center_layout.setSpacing(64)

        # ─── Left half: editorial copy + CTA ────────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(0)
        left.setContentsMargins(0, 12, 0, 0)

        eyebrow = QLabel('SWARM INTELLIGENCE  ·  PHILIPPINES')
        eyebrow.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:3px;'
        )
        left.addWidget(eyebrow)
        left.addSpacing(22)

        # Serif headline — bold thesis statement
        headline = QLabel('Simulating the\nfuture of the\nPhilippine economy.')
        headline.setFont(_serif_font(40, bold=True))
        headline.setStyleSheet(
            f'color:{INK};letter-spacing:-1.4px;line-height:1.05;background:transparent;'
        )
        left.addWidget(headline)
        left.addSpacing(20)

        body = QLabel(
            'A 20-agent swarm pulls live Brent crude, USD/PHP, Manila '
            'weather, and Philippine retail prices to forecast next-month '
            'moves in fuel, food, and electricity. No inputs. No guesswork. '
            'Every run grounded in the most recent observable data.'
        )
        body.setWordWrap(True)
        body.setStyleSheet(
            f'font-size:14px;color:{TEXT_2};line-height:1.7;background:transparent;'
        )
        body.setMaximumWidth(520)
        left.addWidget(body)
        left.addSpacing(32)

        # CTA row — substantial black button with subtle elevation + arrow
        cta_row = QHBoxLayout()
        cta_row.setSpacing(24)
        self._run_btn = QPushButton('RUN SWARM  →')
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setFixedHeight(54)
        self._run_btn.setMinimumWidth(240)
        self._run_btn.setStyleSheet(
            f'QPushButton{{'
            f'background:{ACCENT};color:#FFFFFF;'
            f'border:1px solid {ACCENT};'
            f'border-top:1px solid #3A4150;'      # subtle top-edge highlight (3D ridge)
            f'font-family:Consolas,monospace;font-size:12px;font-weight:700;'
            f'letter-spacing:3.5px;padding:0 32px;border-radius:6px;'
            f'text-align:center;'
            f'}}'
            f'QPushButton:hover{{'
            f'background:{ACCENT_HOV};border-color:{ACCENT_HOV};'
            f'border-top:1px solid #525B6B;'
            f'}}'
            f'QPushButton:pressed{{background:#000000;border-color:#000000;}}'
            f'QPushButton:disabled{{'
            f'background:#9CA3AF;border-color:#9CA3AF;color:#F3F4F6;'
            f'}}'
        )
        # Soft elevation shadow — gives presence without going pill-shaped
        shadow = QGraphicsDropShadowEffect(self._run_btn)
        shadow.setBlurRadius(28)
        shadow.setColor(QColor(15, 17, 21, 80))
        shadow.setOffset(0, 8)
        self._run_btn.setGraphicsEffect(shadow)
        self._run_btn.clicked.connect(self.run_requested.emit)
        cta_row.addWidget(self._run_btn)

        run_caption = QLabel('~60 seconds  ·  live data inputs')
        run_caption.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:11px;color:{TEXT_3};'
            f'letter-spacing:1px;background:transparent;'
        )
        cta_row.addWidget(run_caption)
        cta_row.addStretch()
        left.addLayout(cta_row)

        left.addSpacing(40)

        # Compact stack line — agents · sources · sectors, separated by dots
        stack_row = QHBoxLayout()
        stack_row.setSpacing(0)
        for value, label, is_last in (
            ('20', 'AGENTS', False),
            ('14', 'RAG SOURCES', False),
            ('3', 'SECTORS', True),
        ):
            num = QLabel(value)
            num.setFont(_serif_font(20, bold=True))
            num.setStyleSheet(f'color:{INK};background:transparent;')
            lbl = QLabel(label)
            lbl.setStyleSheet(
                f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
                f'color:{TEXT_3};letter-spacing:1.6px;padding-left:8px;'
            )
            stack_row.addWidget(num)
            stack_row.addWidget(lbl)
            if not is_last:
                sep = QLabel('·')
                sep.setStyleSheet(
                    f'color:{TEXT_3};font-size:14px;padding:0 18px;'
                )
                stack_row.addWidget(sep)
        stack_row.addStretch()
        left.addLayout(stack_row)
        left.addStretch()

        center_layout.addLayout(left, stretch=3)

        # ─── Right half: live data card (the "product preview") ────────────
        center_layout.addWidget(self._build_live_card(), stretch=2)

        outer.addWidget(center, stretch=4)
        outer.addStretch(1)
        return hero

    def _build_live_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName('liveCard')
        card.setMinimumWidth(360)
        card.setMaximumWidth(420)
        card.setStyleSheet(
            f'QFrame#liveCard{{background:{PAPER};border:1px solid {DIVIDER};'
            f'border-radius:4px;}}'
            f'QFrame#liveCard QLabel{{background:transparent;border:none;}}'
        )
        v = QVBoxLayout(card)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(0)

        # Card header — small caps
        h = QHBoxLayout()
        h.setSpacing(8)
        tag = QLabel('LIVE  ·  PHILIPPINES')
        tag.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{INK};letter-spacing:1.8px;'
        )
        h.addWidget(tag)
        h.addStretch()
        self._card_status = QLabel('updating...')
        self._card_status.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_3};'
            f'letter-spacing:1.2px;'
        )
        h.addWidget(self._card_status)
        v.addLayout(h)
        v.addSpacing(16)

        # Hairline
        hl = QFrame()
        hl.setFrameShape(QFrame.Shape.HLine)
        hl.setStyleSheet(f'background:{HAIRLINE};border:none;')
        hl.setFixedHeight(1)
        v.addWidget(hl)
        v.addSpacing(18)

        # ── Markets ────────────────────────────────────────────────────────
        self._brent_row  = self._card_row('Brent crude',  '—', TEXT_3)
        self._usdphp_row = self._card_row('USD / PHP',    '—', TEXT_3)
        self._pump_row   = self._card_row('Retail pump',  '—', TEXT_3)
        for r in (self._brent_row, self._usdphp_row, self._pump_row):
            v.addWidget(r)
        v.addSpacing(20)

        hl2 = QFrame()
        hl2.setFrameShape(QFrame.Shape.HLine)
        hl2.setStyleSheet(f'background:{HAIRLINE};border:none;')
        hl2.setFixedHeight(1)
        v.addWidget(hl2)
        v.addSpacing(18)

        # ── Weather ────────────────────────────────────────────────────────
        wx_eyebrow = QLabel('MANILA  ·  7 DAYS')
        wx_eyebrow.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:9px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1.8px;'
        )
        v.addWidget(wx_eyebrow)
        v.addSpacing(10)
        self._wnow_row    = self._card_row('Now',          '—', TEXT_3)
        self._wrange_row  = self._card_row('Today range',  '—', TEXT_3)
        self._wweek_row   = self._card_row('7-day rain',   '—', TEXT_3)
        for r in (self._wnow_row, self._wrange_row, self._wweek_row):
            v.addWidget(r)

        v.addStretch()
        return card

    def _card_row(self, key: str, value: str, color: str) -> QWidget:
        row = QWidget()
        row.setStyleSheet('background:transparent;')
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 5, 0, 5)
        h.setSpacing(8)
        k = QLabel(key)
        k.setStyleSheet(f'font-size:12px;color:{TEXT_2};background:transparent;')
        h.addWidget(k)
        h.addStretch()
        v = QLabel(value)
        v.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:13px;font-weight:700;'
            f'color:{color};letter-spacing:0.3px;background:transparent;'
        )
        h.addWidget(v)
        return row

    # ── Recent strip ─────────────────────────────────────────────────────────
    def _build_recent_strip(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(
            f'QFrame{{background:{PAPER};border-top:1px solid {DIVIDER};'
            f'border-bottom:1px solid {DIVIDER};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )
        wrap.setMinimumHeight(190)

        # Center the strip
        outer = QHBoxLayout(wrap)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        center = QWidget()
        center.setMaximumWidth(1180)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(60, 18, 60, 18)
        cl.setSpacing(10)

        self._latest_head = QLabel('LATEST FORECAST  ·  exploratory')
        self._latest_head.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:2px;'
        )
        cl.addWidget(self._latest_head)
        self._latest_row = QHBoxLayout()
        self._latest_row.setSpacing(28)
        cl.addLayout(self._latest_row)

        head = QLabel('FUEL TRACK RECORD')
        head.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:2px;'
        )
        cl.addWidget(head)

        self._runs_row = QHBoxLayout()
        self._runs_row.setSpacing(28)
        cl.addLayout(self._runs_row)

        # Initial empty placeholder
        empty = QLabel('No simulations on record yet.')
        empty.setStyleSheet(f'color:{TEXT_3};font-size:12px;')
        self._runs_row.addWidget(empty)
        self._runs_row.addStretch()

        outer.addWidget(center, stretch=4)
        outer.addStretch()
        return wrap

    # ── How it works (3-column editorial) ───────────────────────────────────
    def _build_how_it_works(self) -> QWidget:
        wrap = QFrame()
        wrap.setStyleSheet(
            f'QFrame{{background:{BG};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )

        outer = QHBoxLayout(wrap)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addStretch()

        center = QWidget()
        center.setMaximumWidth(1180)
        cl = QVBoxLayout(center)
        cl.setContentsMargins(60, 80, 60, 80)
        cl.setSpacing(0)

        # Eyebrow + headline
        eyebrow = QLabel('HOW IT WORKS')
        eyebrow.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:3px;'
        )
        cl.addWidget(eyebrow)
        cl.addSpacing(20)

        title = QLabel('A swarm of small models,\nreading the country in real time.')
        title.setFont(_serif_font(28, bold=True))
        title.setStyleSheet(
            f'color:{INK};letter-spacing:-0.8px;line-height:1.15;'
        )
        cl.addWidget(title)
        cl.addSpacing(56)

        # Three-column grid
        cols = QHBoxLayout()
        cols.setSpacing(40)

        steps = [
            ('01',
             'Pull live data',
             'Brent crude, USD/PHP, Manila weather, retail pump prices, '
             'and Philippine-specific feeds (Meralco, WESM, NFA rice) load '
             'in parallel before each run.'),
            ('02',
             'Run the swarm',
             '20 role-specialised agents debate across 4 regional clusters. '
             'Round-by-round elimination produces a survivor per region; '
             'regional judges aggregate into a master verdict.'),
            ('03',
             'Score and evolve',
             'Each agent is graded immediately on citation quality and '
             'convergence; later, DOE pump prices retroactively score the '
             'forecast — promotions, demotions, and benching follow.'),
        ]
        for num, label, body_text in steps:
            col = QVBoxLayout()
            col.setSpacing(0)
            n = QLabel(num)
            n.setFont(_serif_font(48, bold=True))
            n.setStyleSheet(
                f'color:{INK};letter-spacing:-2px;background:transparent;'
            )
            col.addWidget(n)
            col.addSpacing(12)
            l = QLabel(label)
            l.setStyleSheet(
                f'font-size:16px;font-weight:700;color:{INK};'
                f'background:transparent;letter-spacing:-0.3px;'
            )
            col.addWidget(l)
            col.addSpacing(10)
            b = QLabel(body_text)
            b.setWordWrap(True)
            b.setStyleSheet(
                f'font-size:13px;color:{TEXT_2};line-height:1.65;'
                f'background:transparent;'
            )
            col.addWidget(b)
            col.addStretch()
            cw = QWidget()
            cw.setLayout(col)
            cols.addWidget(cw, stretch=1)

        cl.addLayout(cols)

        outer.addWidget(center, stretch=4)
        outer.addStretch()
        return wrap

    # ── Footer ───────────────────────────────────────────────────────────────
    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(48)
        bar.setStyleSheet(
            f'QFrame{{background:{BG};border:none;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}'
        )
        h = QHBoxLayout(bar)
        h.setContentsMargins(40, 0, 40, 0)
        h.setSpacing(16)
        left = QLabel(
            'Local-first  ·  Ollama  ·  Open-Meteo  ·  World Bank  ·  EIA'
        )
        left.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_3};'
            f'letter-spacing:1.4px;'
        )
        h.addWidget(left)
        h.addStretch()
        right = QLabel('MIT  ·  build 2026.05')
        right.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;color:{TEXT_3};'
            f'letter-spacing:1.4px;'
        )
        h.addWidget(right)
        return bar

    # ── Async live-data refresh ──────────────────────────────────────────────
    def _refresh_async(self):
        if self._fetch_thread is not None:
            return
        self._fetch_thread = _DashboardFetchThread()
        self._fetch_thread.ready.connect(self._on_data_ready)
        self._fetch_thread.start()

    def _on_data_ready(self, brief, pump_price):
        ready = 0
        total = 6
        if brief is not None:
            if brief.brent   is not None: ready += 1
            if brief.wti     is not None: ready += 1
            if brief.usd_php is not None: ready += 1
            if brief.psei    is not None: ready += 1
            if brief.weather_manila:      ready += 1
        if pump_price is not None:        ready += 1
        if hasattr(self, '_feeds_lbl'):
            self._feeds_lbl.setText(
                f'FEEDS  ' + '● ' * ready + '○ ' * (total - ready)
            )
        self._card_status.setText(
            datetime.now().strftime('updated %H:%M'))

        def _delta(series) -> tuple[str, str]:
            if not series or len(series) < 2:
                return '', TEXT_3
            try:
                a = series[0][1]; b = series[-1][1]
                if not a:
                    return '', TEXT_3
                pct = (b - a) / a * 100
                color = UP if pct >= 0 else DOWN
                return f'{pct:+.2f}%', color
            except Exception:
                return '', TEXT_3

        # Markets
        if brief is not None:
            if brief.brent is not None:
                pct, color = _delta(brief.brent_hist)
                text = f'${brief.brent:.2f}' + (f'  {pct}' if pct else '')
                self._replace_row(self._brent_row, 'Brent crude', text, color)
            if brief.usd_php is not None:
                pct, color = _delta(brief.fx_hist)
                text = f'₱{brief.usd_php:.4f}' + (f'  {pct}' if pct else '')
                self._replace_row(self._usdphp_row, 'USD / PHP', text, color)

            w = brief.weather_manila or {}
            if w.get('now_temp_c') is not None:
                self._replace_row(self._wnow_row, 'Now',
                                  f'{w["now_temp_c"]:.1f}°C', INK)
            if w.get('today_min') is not None and w.get('today_max') is not None:
                self._replace_row(self._wrange_row, 'Today range',
                                  f'{w["today_min"]:.1f}–{w["today_max"]:.1f}°C', INK)
            if w.get('week_rain_mm') is not None and w.get('week_wet_days') is not None:
                self._replace_row(self._wweek_row, '7-day rain',
                                  f'{w["week_rain_mm"]:.0f}mm  ·  '
                                  f'{w["week_wet_days"]} wet days', INK)

        if pump_price is not None:
            self._replace_row(self._pump_row, 'Retail pump',
                              f'₱{pump_price:.2f}/L', INK)

        self._fetch_thread = None
        self._refresh_recent_runs()

    def _replace_row(self, old: QWidget, key: str, value: str, color: str):
        new = self._card_row(key, value, color)
        parent_layout = old.parent().layout() if old.parent() else None
        if parent_layout is None:
            return
        idx = parent_layout.indexOf(old)
        if idx < 0:
            return
        parent_layout.removeWidget(old)
        old.deleteLater()
        parent_layout.insertWidget(idx, new)
        for name in ('_brent_row', '_usdphp_row', '_pump_row',
                     '_wnow_row', '_wrange_row', '_wweek_row'):
            if getattr(self, name, None) is old:
                setattr(self, name, new)
                break

    # ── Recent work strip refresh ────────────────────────────────────────────
    def _refresh_recent_runs(self):
        if self._store is None:
            return
        try:
            runs = self._store.get_recent_runs(limit=4)
        except Exception:
            runs = []

        # Latest 3-sector forecast (top of the card)
        while self._latest_row.count():
            it = self._latest_row.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if runs:
            from ph_economic_ai.ui.sector_forecast import sector_forecast_rows
            latest = runs[0]
            for r in sector_forecast_rows(
                gas=latest.get('final_estimate'),
                food=latest.get('food_estimate'),
                elec=latest.get('electricity_estimate'),
            ):
                self._latest_row.addWidget(self._build_sector_tile(r))
        self._latest_row.addStretch()

        if runs and runs[0].get('confidence_pct') is not None:
            self._latest_head.setText(
                f"LATEST FORECAST  ·  {self._fmt_date(runs[0].get('timestamp'))}"
                f"  ·  {runs[0]['confidence_pct']}% agreement  ·  exploratory")
        elif runs:
            self._latest_head.setText(
                f"LATEST FORECAST  ·  {self._fmt_date(runs[0].get('timestamp'))}  ·  exploratory")
        else:
            self._latest_head.setText('LATEST FORECAST  ·  exploratory')

        # Clear existing
        while self._runs_row.count():
            item = self._runs_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        prior = runs[1:4]
        if not prior:
            empty = QLabel('No simulations on record yet.')
            empty.setStyleSheet(f'color:{TEXT_3};font-size:12px;')
            self._runs_row.addWidget(empty)
            self._runs_row.addStretch()
            return

        for r in prior:
            self._runs_row.addWidget(self._build_run_tile(r))
        self._runs_row.addStretch()

    def _build_sector_tile(self, r: dict) -> QWidget:
        tile = QWidget()
        tile.setStyleSheet('background:transparent;')
        tile.setMaximumWidth(180)
        v = QVBoxLayout(tile)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        name = QLabel(r['label'])
        name.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1px;'
        )
        v.addWidget(name)
        arrows = {'up': '▲', 'down': '▼', 'flat': '■', 'na': '·'}
        color = {'up': UP, 'down': DOWN, 'flat': TEXT_3, 'na': TEXT_3}[r['direction']]
        val = QLabel(f"{arrows[r['direction']]}  {r['value_str']}")
        val.setStyleSheet(f'font-size:15px;font-weight:700;color:{color};')
        v.addWidget(val)
        return tile

    @staticmethod
    def _fmt_date(ts: str) -> str:
        ts = ts or ''
        try:
            return datetime.fromisoformat(ts).strftime('%b %d')
        except Exception:
            return ts[:10]

    def _build_run_tile(self, run: dict) -> QWidget:
        tile = QWidget()
        tile.setStyleSheet('background:transparent;')
        tile.setMaximumWidth(180)
        v = QVBoxLayout(tile)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)

        rid = run.get('run_id', '?')
        ts = run.get('timestamp', '') or ''
        try:
            dt = datetime.fromisoformat(ts)
            d_str = dt.strftime('%b %d')
        except Exception:
            d_str = ts[:10]

        head = QLabel(f'#{rid}  ·  {d_str}')
        head.setStyleSheet(
            f'font-family:Consolas,monospace;font-size:10px;font-weight:700;'
            f'color:{TEXT_3};letter-spacing:1.2px;'
        )
        v.addWidget(head)

        est = run.get('final_estimate')
        conf = run.get('confidence_pct')
        if est is not None:
            color = UP if est >= 0 else DOWN
            big = QLabel(f'{est:+.2f}₱/L')
            big.setFont(_serif_font(20, bold=True))
            big.setStyleSheet(f'color:{color};letter-spacing:-0.3px;')
            v.addWidget(big)
        else:
            v.addWidget(QLabel('—'))

        sub_parts = []
        if conf is not None:
            sub_parts.append(f'{conf}% agreement')
        actual = run.get('actual_price_change')
        if actual is not None:
            sub_parts.append('graded ✓')
        else:
            sub_parts.append('pending DOE')
        sub = QLabel('  ·  '.join(sub_parts))
        sub.setStyleSheet(f'font-size:10px;color:{TEXT_3};')
        v.addWidget(sub)
        return tile

    # ── External setters ─────────────────────────────────────────────────────
    def refresh_recent(self) -> None:
        """Re-read the store and repopulate the recent/latest forecasts.

        Public hook so the main window can refresh once the food/electricity
        sector debates finish (their estimates are written after the gas run is
        saved, so the post-gas refresh would otherwise show them as '—')."""
        self._refresh_recent_runs()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_recent_runs()

    def set_busy(self, busy: bool):
        if busy:
            self._run_btn.setText('RUNNING  ●')
            self._run_btn.setEnabled(False)
        else:
            self._run_btn.setText('RUN SWARM  →')
            self._run_btn.setEnabled(True)
            self._refresh_recent_runs()

    def update_live_data(self, brief) -> None:
        """Accept a LiveDataBrief pushed from main_window after its own fetch.
        Avoids a redundant second fetch from the landing page when the main
        window already has fresh data."""
        # Reuse _on_data_ready — pump_price is unavailable here so pass None;
        # the method already handles None safely.
        self._on_data_ready(brief, None)