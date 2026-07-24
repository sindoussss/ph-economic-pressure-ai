"""Pressure Monitor view — one-click, monitor-led, with a live Forum feed.

Left column: the present-pressure read (the hero) and the bounded monthly outlook.
Right column: the Forum debate LIVE — a named cast of agents whose chat cards
stream in as they analyse, with the moderator (the benchmark's voice) posting
between rounds. The single "Run" button drives it all off the UI thread.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ph_economic_ai.engine.monitor import MonitorThread
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.engine.kg_forum_adapter import add_forum_turn, seed_sectors
from ph_economic_ai.ui.forum_graph import ForumGraphCanvas

_INK = '#0F1115'
_T2 = '#4B5563'
_T3 = '#79828F'
_DIV = '#C3CAD4'          # darker so borders/dividers are visible on real displays,
                          # not only in screenshots (thin light lines vanish on HiDPI)
_RISE = '#B42318'
_EASE = '#067647'
_FLAT = '#525866'
_DIR_COLOR = {'rising': _RISE, 'easing': _EASE, 'flat': _FLAT, 'unknown': _T3}
_SECTOR_COLOR = {'gas': '#B42318', 'food': '#067647', 'electricity': '#B54708'}
_EYEBROW = ('font-family:Consolas,monospace;font-size:10px;letter-spacing:2px;'
            f'color:{_T3};')


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _initials(name: str) -> str:
    parts = [p for p in (name or '').split() if p]
    if not parts:
        return '?'
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else '')).upper()


class PressureMonitorPanel(QWidget):
    run_finished = pyqtSignal()          # emitted when a Monitor run ends (ok or error)

    def __init__(self, rag, parent=None):
        super().__init__(parent)
        self._rag = rag
        self._thread: MonitorThread | None = None
        self.setStyleSheet('background:#EDEFF3;')       # greyer page so white cards pop

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ══ LEFT: monitor + outlook ══════════════════════════════════════════
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet('QScrollArea{border:none;background:#EDEFF3;}')
        body = QWidget()
        left_scroll.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(48, 36, 40, 48)
        col.setSpacing(0)

        eyebrow = QLabel('PRESSURE MONITOR')
        eyebrow.setStyleSheet(f'{_EYEBROW}letter-spacing:3px;')
        col.addWidget(eyebrow)
        title = QLabel('Current pressure on gas, food, and electricity')
        title.setStyleSheet(f'font-size:22px;font-weight:700;color:{_INK};margin-top:6px;')
        title.setWordWrap(True)
        col.addWidget(title)

        row = QHBoxLayout()
        self._run_btn = QPushButton('Run')
        self._run_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_btn.setStyleSheet(
            f'QPushButton{{background:{_INK};color:#fff;border:none;border-radius:6px;'
            f'padding:9px 22px;font-size:13px;font-weight:600;}}'
            f'QPushButton:disabled{{background:#C7CBD1;}}')
        self._run_btn.clicked.connect(self._on_run)
        row.addWidget(self._run_btn)
        self._status = QLabel('Senses the present, then forecasts humbly.')
        self._status.setStyleSheet(f'color:{_T2};font-size:12px;margin-left:14px;')
        row.addWidget(self._status)
        row.addStretch()
        row.setContentsMargins(0, 18, 0, 24)
        col.addLayout(row)

        # ── DEBATE MAP: knowledge graph, grows turn by turn ──────────────────
        self._kg_label = QLabel('DEBATE MAP')
        self._kg_label.setStyleSheet(f'{_EYEBROW}margin-bottom:8px;')
        self._kg_label.setVisible(False)
        col.addWidget(self._kg_label)
        self._kg = ForumGraphCanvas()
        self._kg.setMinimumHeight(360)
        self._kg.setVisible(False)
        col.addWidget(self._kg)
        self._kg_builder: KnowledgeGraphBuilder | None = None
        self._feed_has_real = False

        self._hero_label = QLabel('PRESENT PRESSURE')
        self._hero_label.setStyleSheet(f'{_EYEBROW}margin-bottom:10px;')
        col.addWidget(self._hero_label)
        self._cards = QVBoxLayout()
        self._cards.setSpacing(12)
        col.addLayout(self._cards)
        self._narrative = QLabel('')
        self._narrative.setWordWrap(True)
        self._narrative.setStyleSheet(f'color:{_T2};font-size:13px;margin-top:14px;')
        col.addWidget(self._narrative)

        self._outlook_label = QLabel('NEXT-MONTH OUTLOOK  ·  bounded')
        self._outlook_label.setStyleSheet(f'{_EYEBROW}margin-top:34px;margin-bottom:10px;')
        self._outlook_label.setVisible(False)
        col.addWidget(self._outlook_label)
        self._outlook = QVBoxLayout()
        self._outlook.setSpacing(10)
        col.addLayout(self._outlook)
        col.addStretch()
        root.addWidget(left_scroll, stretch=1)

        # ══ RIGHT: live forum feed ═══════════════════════════════════════════
        right = QFrame()
        right.setFixedWidth(410)
        right.setStyleSheet(
            f'QFrame{{background:#F4F6F8;border-left:2px solid {_DIV};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        rcol = QVBoxLayout(right)
        rcol.setContentsMargins(22, 26, 22, 14)
        rcol.setSpacing(0)
        self._feed_head = QLabel('FORUM  ·  idle')
        self._feed_head.setStyleSheet(f'{_EYEBROW}margin-bottom:14px;')
        rcol.addWidget(self._feed_head)

        feed_scroll = QScrollArea()
        feed_scroll.setWidgetResizable(True)
        feed_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        feed_scroll.setStyleSheet('QScrollArea{border:none;background:#F4F6F8;}')
        self._feed_scroll = feed_scroll
        feed_body = QWidget()
        feed_scroll.setWidget(feed_body)
        self._feed = QVBoxLayout(feed_body)
        self._feed.setContentsMargins(0, 0, 6, 0)
        self._feed.setSpacing(10)
        self._feed.addStretch()
        rcol.addWidget(feed_scroll, stretch=1)

        self._typing = QLabel('')
        self._typing.setWordWrap(True)
        self._typing.setStyleSheet(f'color:{_T3};font-size:11px;font-style:italic;'
                                   f'margin-top:8px;')
        rcol.addWidget(self._typing)
        root.addWidget(right)

        self._add_feed(self._hint('Click Run to convene the forum.'))

    # ── feed helpers ──────────────────────────────────────────────────────────

    def _add_feed(self, w: QWidget) -> None:
        self._feed.insertWidget(self._feed.count() - 1, w)   # before the trailing stretch
        bar = self._feed_scroll.verticalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))

    def _clear_feed(self) -> None:
        for i in reversed(range(self._feed.count())):
            w = self._feed.itemAt(i).widget()
            if w is not None:
                self._feed.takeAt(i)
                w.deleteLater()

    def _feed_count(self) -> int:
        return sum(1 for i in range(self._feed.count())
                   if self._feed.itemAt(i).widget() is not None)

    def _hint(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f'color:{_T3};font-size:12px;')
        return lbl

    # ── run ────────────────────────────────────────────────────────────────────

    def start(self):
        """Programmatic entry (e.g. chained from the Home Run button)."""
        self._on_run()

    def _on_run(self):
        if self._thread is not None and self._thread.isRunning():
            return
        _clear(self._cards)
        _clear(self._outlook)
        self._narrative.setText('')
        self._outlook_label.setVisible(False)

        # Placeholder sector cards so PRESENT PRESSURE isn't a void while it runs.
        for s in ('gas', 'food', 'electricity'):
            self._cards.addWidget(self._placeholder_card(s))

        # Seed the debate map with the three sector hubs immediately, so the graph
        # is on screen from the first second and simply grows as agents connect.
        self._kg.reset()
        self._kg_builder = KnowledgeGraphBuilder()
        seed_sectors(self._kg_builder, ('gas', 'food', 'electricity'))
        try:
            self._kg.set_snapshot(*self._kg_builder.snapshot())
        except Exception:
            pass
        self._kg.setVisible(True)
        self._kg_label.setVisible(True)

        # A convening message in the feed instead of a blank white panel.
        self._clear_feed()
        self._add_feed(self._hint('The forum is convening — agents are gathering the '
                                  'frozen snapshot and reading the market…'))
        self._feed_has_real = False
        self._feed_head.setText('FORUM  ·  live')
        self._typing.setText('Waking the agents…')
        self._run_btn.setEnabled(False)
        self._status.setText('Gathering current pressure and running the Forum debate…')

        # Budget: keep the multi-round debate + moderator (rounds=2) but drop the
        # 39-call gas swarm — the benchmark says it can't beat naive, so the
        # bounded Outlook uses naive persistence + interval (fast and honest).
        self._thread = MonitorThread(self._rag, window='this_week', rounds=2,
                                     run_tournament=False)
        self._thread.forum_event.connect(self._on_forum_event)
        self._thread.monitor_ready.connect(self._on_monitor_ready)
        self._thread.outlook_ready.connect(self._on_outlook_ready)
        self._thread.error_occurred.connect(self._on_error)
        self._thread.start()

    def _on_forum_event(self, kind: str, data: object):
        d = data if isinstance(data, dict) else {}
        if kind == 'agent_start':
            self._typing.setText(
                f"✎ {d.get('name', '')} · {d.get('occupation', '')} "
                f"is reading {d.get('sector', '')}…")
        elif kind == 'agent_message':
            self._typing.setText('')
            if not self._feed_has_real:            # drop the convening placeholder
                self._clear_feed()
                self._feed_has_real = True
            self._add_feed(self._chat_card(d))
            self._update_graph(d)
        elif kind == 'moderator':
            self._add_feed(self._moderator_card(d))

    def _update_graph(self, d: dict):
        if self._kg_builder is None:
            return
        try:
            add_forum_turn(self._kg_builder, d.get('name', ''), d.get('occupation', ''),
                           d.get('sector', ''), d.get('estimate'), d.get('message', ''),
                           d.get('sources'))
            self._kg.set_snapshot(*self._kg_builder.snapshot())
            self._kg.setVisible(True)
            self._kg_label.setVisible(True)
        except Exception:
            pass

    def _on_error(self, msg: str):
        self._status.setText(f'Run failed: {msg}')
        self._typing.setText('')
        self._feed_head.setText('FORUM  ·  error')
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    def _on_monitor_ready(self, brief):
        self._feed_head.setText('FORUM  ·  done')
        self._typing.setText('')
        self._status.setText(f'Present read as of {brief.as_of} ({brief.window}). '
                             'Forecasting…')
        _clear(self._cards)
        for r in brief.readings:
            self._cards.addWidget(self._sector_card(r))
        if brief.narrative:
            self._narrative.setText(brief.narrative)

    def _on_outlook_ready(self, outlook):
        self._outlook_label.setVisible(True)
        _clear(self._outlook)
        for s in outlook.sectors:
            self._outlook.addWidget(self._outlook_row(s))
        self._status.setText(f'Done. Present read + bounded {outlook.horizon} outlook.')
        self._run_btn.setEnabled(True)
        self.run_finished.emit()

    # ── forum chat cards (right column) ────────────────────────────────────────

    def _chat_card(self, d: dict) -> QFrame:
        sector = d.get('sector', '')
        color = _SECTOR_COLOR.get(sector, _T3)
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px solid {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        h = QHBoxLayout(card)
        h.setContentsMargins(12, 12, 12, 12)
        h.setSpacing(10)

        avatar = QLabel(_initials(d.get('name', '?')))
        avatar.setFixedSize(34, 34)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setStyleSheet(f'background:{color};color:#fff;border-radius:17px;'
                             f'font-size:12px;font-weight:700;')
        h.addWidget(avatar, alignment=Qt.AlignmentFlag.AlignTop)

        v = QVBoxLayout()
        v.setSpacing(3)
        head = QHBoxLayout()
        head.setSpacing(6)
        nm = QLabel(d.get('name', ''))
        nm.setStyleSheet(f'font-size:12px;font-weight:700;color:{_INK};')
        head.addWidget(nm)
        occ = QLabel('· ' + d.get('occupation', ''))
        occ.setStyleSheet(f'font-size:11px;color:{_T3};')
        head.addWidget(occ)
        head.addStretch()
        chip = QLabel(sector.upper())
        chip.setStyleSheet(f'font-size:9px;font-weight:700;color:{color};letter-spacing:1px;')
        head.addWidget(chip)
        v.addLayout(head)

        msg = ' '.join((d.get('message') or '').split())
        if len(msg) > 320:
            msg = msg[:320] + '…'
        body = QLabel(msg or '(no reading)')
        body.setWordWrap(True)
        body.setStyleSheet(f'font-size:12px;color:{_T2};')
        v.addWidget(body)

        est = d.get('estimate')
        if est is not None:
            badge = QLabel(f'estimate  {est:+.2f} {d.get("unit", "")}')
            badge.setStyleSheet(f'font-size:11px;font-weight:600;color:{color};margin-top:2px;')
            v.addWidget(badge)
        h.addLayout(v, stretch=1)
        return card

    def _moderator_card(self, d: dict) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#F4F5F7;border:1px solid {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        lbl = QLabel(f"MODERATOR · {d.get('sector', '').upper()} · the benchmark's voice")
        lbl.setStyleSheet(f'{_EYEBROW}font-size:9px;')
        v.addWidget(lbl)
        txt = ' '.join((d.get('text') or '').split())
        if len(txt) > 320:
            txt = txt[:320] + '…'
        body = QLabel(txt or '(steering the next round)')
        body.setWordWrap(True)
        body.setStyleSheet(f'font-size:12px;color:{_T2};font-style:italic;')
        v.addWidget(body)
        return card

    # ── present-pressure & outlook cards (left column) ─────────────────────────

    def _placeholder_card(self, sector: str) -> QFrame:
        """A dashed 'analysing…' card shown per sector while the forum runs."""
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px dashed {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        name = QLabel(sector.upper())
        name.setStyleSheet(f'font-size:13px;font-weight:700;letter-spacing:1px;'
                           f'color:{_SECTOR_COLOR.get(sector, _T3)};')
        top.addWidget(name)
        top.addStretch()
        dots = QLabel('· · ·')
        dots.setStyleSheet(f'color:{_T3};font-size:15px;font-weight:700;')
        top.addWidget(dots)
        lay.addLayout(top)
        sub = QLabel('analysing present pressure…')
        sub.setStyleSheet(f'color:{_T3};font-size:12px;margin-top:4px;')
        lay.addWidget(sub)
        return card

    def _sector_card(self, r) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px solid {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        top = QHBoxLayout()
        name = QLabel(r.sector.upper())
        name.setStyleSheet(f'font-size:13px;font-weight:700;color:{_INK};letter-spacing:1px;')
        top.addWidget(name)
        top.addStretch()
        est = '—' if r.estimate is None else f'{r.estimate:+.2f} {r.unit}'
        val = QLabel(est)
        val.setStyleSheet(f'font-size:15px;font-weight:700;'
                          f'color:{_DIR_COLOR.get(r.direction, _T3)};')
        top.addWidget(val)
        lay.addLayout(top)
        meta = QLabel(f"{r.direction}  ·  agreement {r.confidence}% (not a probability)  ·  "
                      f"{', '.join(r.sources) or 'no sources'}")
        meta.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:4px;')
        meta.setWordWrap(True)
        lay.addWidget(meta)
        if r.drivers:
            drv = QLabel('· ' + '\n· '.join(d for d in r.drivers if d))
            drv.setStyleSheet(f'color:{_T2};font-size:12px;margin-top:8px;')
            drv.setWordWrap(True)
            lay.addWidget(drv)
        return card

    def _outlook_row(self, s) -> QFrame:
        row = QFrame()
        row.setStyleSheet(
            f'QFrame{{background:transparent;border:none;border-top:1px solid {_DIV};}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(row)
        lay.setContentsMargins(0, 10, 0, 2)
        head = QHBoxLayout()
        name = QLabel(f'{s.sector.upper()}  ·  {s.basis}')
        name.setStyleSheet(f'font-size:12px;font-weight:600;color:{_INK};')
        head.addWidget(name)
        head.addStretch()
        if s.point is not None and s.interval is not None:
            pt = QLabel(f'{s.point:+.2f} {s.unit}  [{s.interval[0]:+.2f}, {s.interval[1]:+.2f}]')
        else:
            pt = QLabel('—')
        pt.setStyleSheet(f'font-size:12px;color:{_T2};')
        head.addWidget(pt)
        lay.addLayout(head)
        note = QLabel(s.note)
        note.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:2px;')
        note.setWordWrap(True)
        lay.addWidget(note)
        return row
