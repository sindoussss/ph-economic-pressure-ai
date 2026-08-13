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

from ph_economic_ai.engine import interval as _interval
from ph_economic_ai.engine import price_calendar as _calendar
from ph_economic_ai.engine.monitor import MonitorThread
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.engine.kg_forum_adapter import add_forum_turn, seed_sectors
from ph_economic_ai.ui import honesty as _honesty
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

#: Display order/labels for the Food card's per-sub-category breakdown row.
#: Keys match `SectorReading.subcategories` (Task 1) -- the PSA sub-baskets
#: the food judge parses (Task 3). A category absent from `subcategories`
#: had no parseable read this cycle and shows an em-dash, never 0.0%.
_CATEGORY_DISPLAY_LABELS = {
    'rice': 'Rice', 'meat': 'Meat', 'fish': 'Fish', 'dairy_eggs': 'Dairy & Eggs',
    'vegetables': 'Vegetables', 'sugar': 'Sugar',
}


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


#: Characters shown before a statement is collapsed. Long enough to carry the
#: agent's opening claim, short enough that fifty cards stay scannable.
_PREVIEW_CHARS = 320


class _ExpandableCard(QFrame):
    """An agent card whose statement opens in full when clicked.

    The statement used to be truncated at `_PREVIEW_CHARS` and the remainder
    thrown away before it reached a widget, so a reader could not check a claim
    against the rest of what the agent said. Since these statements are exactly
    where a fabricated figure would hide, the full text has to be reachable.

    The whole card is the hit target rather than a small "more" link: the cards
    are dense, the panel is scrolled with a mouse, and a precise click target in
    a scrolling list is a fight.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._body = None
        self._hint = None
        self._full = ''
        self._expanded = False

    def attach_body(self, body, hint, full_text: str) -> None:
        self._body = body
        self._hint = hint
        self.set_full_message(full_text)

    def set_full_message(self, full_text: str) -> None:
        """Set the statement, keeping it whole regardless of what is displayed."""
        self._full = full_text or ''
        if self._body is None:
            return
        if not self._full:
            # An agent that said nothing already has its placeholder in the label
            # ('(no reading)', or blank while streaming). Rendering an empty
            # string over it would replace an explanation with a gap.
            if self._hint is not None:
                self._hint.hide()
            return
        if len(self._full) > _PREVIEW_CHARS:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip('Click to read the full statement')
        else:
            self.unsetCursor()
            self.setToolTip('')
        self._render()

    def _render(self) -> None:
        if self._body is None:
            return
        long_enough = len(self._full) > _PREVIEW_CHARS
        if not long_enough:
            self._body.setText(self._full)
            if self._hint is not None:
                self._hint.hide()
            return
        if self._expanded:
            self._body.setText(self._full)
            hint = 'Show less'
        else:
            self._body.setText(self._full[:_PREVIEW_CHARS].rstrip() + '…')
            remaining = len(self._full) - _PREVIEW_CHARS
            hint = f'Show more  ({remaining:,} more characters)'
        if self._hint is not None:
            self._hint.setText(hint)
            self._hint.show()

    def toggle(self) -> None:
        if len(self._full) <= _PREVIEW_CHARS:
            return
        self._expanded = not self._expanded
        self._render()

    def mouseReleaseEvent(self, event):     # noqa: N802 - Qt naming
        # Release rather than press, so a click that began as a drag-scroll does
        # not expand a card the user was only scrolling past.
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()
        super().mouseReleaseEvent(event)


class PressureMonitorPanel(QWidget):
    run_finished = pyqtSignal()          # emitted when a Monitor run ends (ok or error)

    def __init__(self, rag, store=None, parent=None):
        super().__init__(parent)
        self._rag = rag
        # Optional: without it the band still renders, but reports itself as an
        # uncalibrated prior rather than silently looking calibrated.
        self._store = store
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
            # Open the card immediately so its text can fill in as tokens arrive. A
            # 50-agent run is minutes long on one GPU; watching a card write itself is
            # the difference between "working" and "frozen".
            if not self._feed_has_real:            # drop the convening placeholder
                self._clear_feed()
                self._feed_has_real = True
            self._live_card = self._chat_card(d, live=True)
            self._live_text = ''
            self._add_feed(self._live_card)
        elif kind == 'agent_token':
            self._append_token(d.get('text', ''))
        elif kind == 'agent_message':
            self._typing.setText('')
            if not self._feed_has_real:
                self._clear_feed()
                self._feed_has_real = True
            # Swap the streaming card for the finished one, which adds the estimate
            # badge and the honest source list. If no card was opened (a headless
            # replay, or tokens never arrived) just append the finished card.
            self._drop_live_card()
            self._add_feed(self._chat_card(d))
            self._update_graph(d)
        elif kind == 'moderator':
            self._drop_live_card()
            self._add_feed(self._moderator_card(d))
        elif kind == 'judge':
            self._drop_live_card()
            self._add_feed(self._judge_card(d))

    def _append_token(self, text: str) -> None:
        """Append streamed text to the open card, keeping it readable.

        Tokens arrive faster than a human reads, so the body is trimmed to the same
        320-character window the finished card uses — the tail is what is being
        written, so the tail is what is shown.
        """
        card = getattr(self, '_live_card', None)
        body = getattr(card, '_body', None) if card is not None else None
        if body is None or not text:
            return
        # Accumulate RAW and normalise only for display. Collapsing whitespace on the
        # accumulator strips each chunk's trailing space before the next one arrives,
        # so "prices " + "are" renders as "pricesare".
        self._live_text = getattr(self, '_live_text', '') + text
        shown = ' '.join(self._live_text.split())
        if len(shown) > 320:
            shown = '…' + shown[-320:]
        try:
            body.setText(shown)
        except RuntimeError:
            self._live_card = None       # widget already deleted (feed cleared mid-run)

    def _drop_live_card(self) -> None:
        card = getattr(self, '_live_card', None)
        self._live_card = None
        self._live_text = ''
        if card is None:
            return
        try:
            idx = self._feed.indexOf(card)
            if idx >= 0:
                self._feed.takeAt(idx)
            card.deleteLater()
        except RuntimeError:
            pass

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
        # One track-record strip above the cards, not a copy on each of the
        # three. It is a fact about the app, not about a sector, and repeating
        # it turned the honest sentence into wallpaper a reader scrolls past.
        self._cards.addWidget(self._record_strip())
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

    def _chat_card(self, d: dict, live: bool = False) -> QFrame:
        """One agent's card. `live` returns it mid-stream: the body label is exposed as
        `card._body` so arriving tokens can be appended, and the estimate badge is
        omitted until the agent has actually produced a number.

        Long statements are previewed and expand on click. The preview used to be
        the whole story: the text was cut at `_PREVIEW_CHARS` and the remainder
        discarded before it ever reached a widget, so an agent's reasoning was
        unreadable past the first few lines and the reader could not check a claim
        against the rest of the statement.
        """
        sector = d.get('sector', '')
        color = _SECTOR_COLOR.get(sector, _T3)
        card = _ExpandableCard()
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
        body = QLabel(msg if msg else ('' if live else '(no reading)'))
        body.setWordWrap(True)
        body.setStyleSheet(f'font-size:12px;color:{_T2};')
        v.addWidget(body)

        hint = QLabel('')
        hint.setStyleSheet(f'font-size:10px;font-weight:600;color:{color};')
        hint.hide()
        v.addWidget(hint)
        # The full text lives on the card, so expanding is a re-render rather than
        # a re-fetch and nothing is lost on the way to the widget.
        card.attach_body(body, hint, msg)

        est = d.get('estimate')
        rejected = d.get('rejected_estimate')
        if est is not None:
            badge = QLabel(f'estimate  {est:+.2f} {d.get("unit", "")}')
            badge.setStyleSheet(f'font-size:11px;font-weight:600;color:{color};margin-top:2px;')
            v.addWidget(badge)
        elif rejected is not None:
            # Show that a number was produced and discarded. Dropping it silently
            # would make a magnitude hallucination look identical to an agent that
            # simply had nothing to say.
            badge = QLabel(f'estimate discarded  ({rejected:+.2f} {d.get("unit", "")} '
                           f'— implausible for one month)')
            badge.setStyleSheet(f'font-size:11px;font-weight:600;color:{_T3};'
                                f'margin-top:2px;font-style:italic;')
            badge.setWordWrap(True)
            v.addWidget(badge)
        h.addLayout(v, stretch=1)
        if live:
            card._body = body          # the token sink for this agent's turn
        return card

    def _judge_card(self, d: dict) -> QFrame:
        sector = d.get('sector', '')
        color = _SECTOR_COLOR.get(sector, _INK)
        card = QFrame()
        card.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px solid {color};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        v = QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(3)
        lbl = QLabel(f"JUDGE · {sector.upper()} · the verdict")
        lbl.setStyleSheet(f'{_EYEBROW}font-size:9px;color:{color};')
        v.addWidget(lbl)
        txt = ' '.join((d.get('text') or '').split())
        if len(txt) > 340:
            txt = txt[:340] + '…'
        body = QLabel(txt or '(synthesising the read)')
        body.setWordWrap(True)
        body.setStyleSheet(f'font-size:12px;color:{_INK};font-weight:500;')
        v.addWidget(body)
        est = d.get('estimate')
        if est is not None:
            badge = QLabel(f'verdict  {est:+.2f} {d.get("unit", "")}')
            badge.setStyleSheet(f'font-size:11px;font-weight:700;color:{color};margin-top:2px;')
            v.addWidget(badge)
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

    # Which dated event each sector's next move actually happens on. Fuel moves on
    # the weekly DOE adjustment; the CPI-derived sectors move on the PSA release.
    # This is a schedule lookup, not a forecast, which is why the app can state a
    # date at all.
    _SECTOR_EVENT = {'gas': 'fuel', 'food': 'cpi', 'electricity': 'cpi'}

    def _next_event(self, sector: str) -> dict:
        if self._SECTOR_EVENT.get(sector, 'cpi') == 'fuel':
            return _calendar.describe_next_fuel_adjustment()
        return _calendar.describe_next_cpi_release()

    def _graded_errors(self) -> list:
        """Past absolute errors, for calibrating the band. Empty is fine: the band
        then reports itself as uncalibrated rather than pretending."""
        store = self._store
        if store is None:
            return []
        try:
            return store.get_graded_errors()
        except Exception:
            return []

    def _graded_count(self) -> int:
        """How many runs carry a grade. NOT `len(self._graded_errors())`, which
        is the conformal calibration window and is capped."""
        store = self._store
        if store is None:
            return 0
        try:
            return int(store.count_graded_runs())
        except Exception:
            return 0

    def _run_count(self) -> int:
        """How many runs are stored, graded or not.

        Shown beside the graded count so "0 graded" reads as "not settled yet"
        rather than "this app has never run". They are different facts and the
        first without the second overstates the problem.
        """
        store = self._store
        if store is None:
            return 0
        try:
            return int(store.count_runs())
        except Exception:
            return 0

    def _chain_integrity_text(self) -> str:
        """Whether the hash-chained prediction log still verifies, for the strip.

        Separate from `_graded_count`/`_run_count`, which read `trust.db` --
        the mutable, correctable store. This reads `track_record.jsonl`, the
        append-only counterpart, so a reader gets both: how many forecasts
        have been graded, and whether the log of what was predicted can prove
        it has not been rewritten since.
        """
        store = self._store
        if store is None or not hasattr(store, 'track_record'):
            return ''
        try:
            return _honesty.chain_integrity_line(store.track_record)
        except Exception:
            return ''

    def _record_strip(self) -> QFrame:
        """What this app's forecasts have actually been checked against.

        The question a reader has before they can read anything below it, and
        the one the screen never answered: the cards showed a number, a range
        and an agreement percentage, all of which look like evidence, and
        nothing said whether a single forecast had ever been compared to a
        price. The answer is currently none, with the reasons enumerated, which
        is a stronger statement than silence in either direction.
        """
        strip = QFrame()
        strip.setStyleSheet(
            f'QFrame{{background:#FFFFFF;border:1px solid {_DIV};border-radius:10px;}}'
            f'QFrame QLabel{{background:transparent;border:none;}}')
        lay = QVBoxLayout(strip)
        lay.setContentsMargins(18, 12, 18, 12)
        head = QLabel('TRACK RECORD')
        head.setStyleSheet(_EYEBROW)
        lay.addWidget(head)
        for text, strong in ((_honesty.track_record_line(
                                 self._graded_count(), self._run_count()), True),
                             (self._chain_integrity_text(), False),
                             (_honesty.grade_backlog_line(self._grade_backlog()), False)):
            if not text:
                continue
            lbl = QLabel(text)
            lbl.setStyleSheet(f'color:{_INK if strong else _T3};font-size:11px;'
                              f'font-weight:{600 if strong else 400};margin-top:4px;')
            lbl.setWordWrap(True)
            lay.addWidget(lbl)
        return strip

    def _grade_backlog(self) -> dict:
        """Why each ungraded run is ungraded, counted by reason.

        Asked of `grade_verdict`, the same function that decides whether to
        grade, so the screen cannot describe a rule the grader no longer uses.
        """
        store = self._store
        if store is None:
            return {}
        counts: dict = {}
        try:
            from ph_economic_ai.engine.ground_truth import grade_verdict
            due = store.get_due_runs()
            for run in due:
                key = grade_verdict(store, run)['obstacle']
                if key is not None:
                    counts[key] = counts.get(key, 0) + 1
            # Runs whose forecast week has not closed yet. Without them the
            # counts do not add up to the stored total, and a reader who
            # subtracts finds runs the screen declined to account for.
            pending = store.count_runs() - len(due) - store.count_graded_runs()
            if pending > 0:
                counts['not_due'] = pending
        except Exception:
            return {}
        return counts

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

        # WHEN. A single magnitude floating free of a date was the gap users hit:
        # "is it tomorrow, or the 27th?". The date is a published schedule, so it is
        # stated plainly and never presented as a prediction.
        event = self._next_event(r.sector)
        when = QLabel(f"next change  ·  {event['when']}  ·  {event['label']}")
        when.setStyleSheet(f'color:{_INK};font-size:12px;font-weight:600;margin-top:6px;')
        lay.addWidget(when)

        # RANGE. One number reads as more precise than this system can justify. The
        # 50% band leads because it is narrow enough to act on; the 90% band is the
        # honest full picture, one click away rather than hidden.
        if r.estimate is not None:
            errors = self._graded_errors()
            b50 = _interval.band(r.estimate, errors, _interval.DEFAULT_LEVEL, r.sector)
            b90 = _interval.band(r.estimate, errors, _interval.EXPANDED_LEVEL, r.sector)

            rng = QLabel(f"{b50['low']:+.2f} to {b50['high']:+.2f} {r.unit}  ·  50% range")
            rng.setStyleSheet(f'color:{_T2};font-size:12px;margin-top:2px;')
            lay.addWidget(rng)

            # No `source` here any more: the provenance line below is always
            # present, so repeating it inside the toggle said the same thing
            # twice and made the honest sentence look like boilerplate.
            wide = QLabel(f"{b90['low']:+.2f} to {b90['high']:+.2f} {r.unit}  ·  90% range")
            wide.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:2px;')
            wide.setWordWrap(True)
            wide.setVisible(False)
            lay.addWidget(wide)

            # PROVENANCE, always. This line used to appear only when the band was
            # uncalibrated, which is backwards: a reader who never sees it cannot
            # tell a calibrated band from a screen that forgot to warn them, so
            # its absence carried the message. Now only its content changes.
            prov = QLabel(_honesty.band_provenance(b50))
            prov.setStyleSheet(
                f'color:{_INK if not b50["calibrated"] else _T2};font-size:11px;'
                f'font-weight:{600 if not b50["calibrated"] else 400};margin-top:4px;')
            prov.setWordWrap(True)
            lay.addWidget(prov)

            more = QPushButton('show 90% range')
            more.setCursor(Qt.CursorShape.PointingHandCursor)
            more.setFlat(True)
            more.setStyleSheet(
                f'QPushButton{{border:none;background:transparent;color:{_T3};'
                f'font-size:11px;text-align:left;padding:0;margin-top:2px;}}'
                f'QPushButton:hover{{color:{_INK};text-decoration:underline;}}')

            def _toggle(_=False, w=wide, btn=more):
                shown = not w.isVisible()
                w.setVisible(shown)
                btn.setText('hide 90% range' if shown else 'show 90% range')

            more.clicked.connect(_toggle)
            lay.addWidget(more)

            # The one measured accuracy result the app owns, with the caveat that
            # has to travel with it. Fuel only: the anchor backtest is a fuel
            # backtest, and reusing it under food or electricity would be the
            # cross-sector borrowing the band already refuses.
            if r.sector == 'gas':
                anchor = _honesty.anchor_record_line()
                if anchor:
                    al = QLabel(anchor)
                    al.setStyleSheet(f'color:{_T2};font-size:11px;margin-top:2px;')
                    al.setWordWrap(True)
                    lay.addWidget(al)

        # WHAT THE AGENTS SAID. The distinct values and their span lead; the
        # percentage follows as a summary of them. A percentage cannot separate
        # agents who independently agreed from agents who copied, and the
        # distinct count is the part a reader can check against the agent cards
        # on the same screen.
        spread = QLabel(_honesty.spread_line(
            getattr(r, 'estimates', None), r.unit, r.confidence))
        spread.setStyleSheet(f'color:{_T2};font-size:11px;margin-top:6px;')
        spread.setWordWrap(True)
        lay.addWidget(spread)

        caveat = _honesty.agreement_caveat(
            len(getattr(r, 'estimates', None) or []),
            distinct=len({round(float(e), 2)
                          for e in (getattr(r, 'estimates', None) or [])}))
        if caveat:
            cav = QLabel(caveat)
            cav.setStyleSheet(f'color:{_INK};font-size:11px;margin-top:2px;')
            cav.setWordWrap(True)
            lay.addWidget(cav)

        da = getattr(r, 'direction_agreement', 0)
        agree_bit = f"direction agreement {da}%" if da else 'direction unmeasured'
        meta = QLabel(f"{r.direction}  ·  {agree_bit} (not a probability)  ·  "
                      f"{', '.join(r.sources) or 'no sources'}")
        meta.setStyleSheet(f'color:{_T3};font-size:11px;margin-top:6px;')
        meta.setWordWrap(True)
        lay.addWidget(meta)
        if r.drivers:
            drv = QLabel('· ' + '\n· '.join(d for d in r.drivers if d))
            drv.setStyleSheet(f'color:{_T2};font-size:12px;margin-top:8px;')
            drv.setWordWrap(True)
            lay.addWidget(drv)

        # PER-CATEGORY BREAKDOWN, food only. Gated on the dict actually being
        # populated (not merely present, since `subcategories` defaults to `{}`
        # rather than `None` -- see Task 1) so gas/electricity, and a food read
        # that for some reason carries no category data, get no row at all
        # rather than six em-dashes pretending to be information.
        if r.sector == 'food' and getattr(r, 'subcategories', None):
            parts = []
            for category, label in _CATEGORY_DISPLAY_LABELS.items():
                value = r.subcategories.get(category)
                text = f'{label} {value:+.1f}%' if value is not None else f'{label} —'
                parts.append(text)
            breakdown = QLabel('  ·  '.join(parts))
            breakdown.setStyleSheet(f'color:{_T2};font-size:11px;margin-top:6px;')
            breakdown.setWordWrap(True)
            lay.addWidget(breakdown)

            # Without this, the six category values sit directly under the
            # headline estimate with nothing telling a reader they are the
            # judge's own separate per-category reads -- not components that
            # sum or average to the number above. One shade more muted than
            # the breakdown row itself (_T3 vs _T2), matching how this file
            # already de-emphasises secondary/caveat text relative to content.
            note = QLabel('Separate per-category reads from the same debate '
                          '— not components of the figure above.')
            note.setStyleSheet(f'color:{_T3};font-size:10px;font-style:italic;margin-top:2px;')
            note.setWordWrap(True)
            lay.addWidget(note)
            # The peso-anchor strip (₱ prices + projection) that used to render
            # here now lives on the Report page's Food card instead -- Monitor
            # keeps only the percentage breakdown above.

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
