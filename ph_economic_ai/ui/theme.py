"""Editorial design tokens + widget helpers — the single source of truth for the
app's look. Screens use these instead of hand-coding stylesheets."""
from PyQt6.QtWidgets import QLabel, QFrame, QVBoxLayout

# -- palette --
SURFACE = '#FBFBFA'
CARD = '#FFFFFF'
INK = '#1C1E26'
MUTED = '#6B7280'
FAINT = '#9AA0AA'
HAIRLINE = '#E5E7EB'
UP = '#B3261E'        # price up = red (bad for consumers)
DOWN = '#15803D'      # price down = green (good)
NEUTRAL = '#3B6FD4'
WARNING = '#B45309'   # amber -- collapsed-room and unscored-survivor caveats

# -- fonts --
SERIF = 'Georgia'
MONO = 'Consolas'

_DIR = {'up': UP, 'down': DOWN, 'flat': MUTED, 'na': FAINT}


def direction_color(direction: str) -> str:
    return _DIR.get(direction, MUTED)


def eyebrow(text) -> QLabel:
    lbl = QLabel(str(text).upper())
    lbl.setStyleSheet(
        f'font-family:{MONO},monospace;font-size:10px;font-weight:700;'
        f'letter-spacing:1.4px;color:{FAINT};background:transparent;')
    return lbl


def serif_number(text, color: str = INK, size: int = 24) -> QLabel:
    lbl = QLabel(str(text))
    lbl.setStyleSheet(
        f'font-family:{SERIF},serif;font-size:{size}px;font-weight:700;'
        f'color:{color};letter-spacing:-0.5px;background:transparent;')
    return lbl


def muted(text, size: int = 9, color: str = MUTED, upper: bool = False) -> QLabel:
    lbl = QLabel(str(text).upper() if upper else str(text))
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f'font-size:{size}px;color:{color};background:transparent;')
    return lbl


def hairline() -> QFrame:
    fr = QFrame()
    fr.setFixedHeight(1)
    fr.setStyleSheet(f'background:{HAIRLINE};border:none;')
    return fr


def confidence_bar(low_frac: float, width_frac: float) -> QFrame:
    """A 5px horizontal track with a filled segment from `low_frac` to
    `low_frac + width_frac` (both 0-1, fractions of the track's own width).
    Purely a visual echo of numbers already shown in the caption beside it --
    not a new statistic."""
    track = QFrame()
    track.setFixedHeight(5)
    track.setStyleSheet(f'background:{FAINT};border-radius:2px;')
    fill = QFrame(track)
    fill.setStyleSheet(f'background:{INK};border-radius:2px;')

    def _position():
        w = track.width()
        fill.setGeometry(int(w * low_frac), 0, max(2, int(w * width_frac)), 5)
    track.resizeEvent = lambda e: _position()
    return track


def card(title=None):
    """Editorial white card. Returns (frame, content_layout). Title -> eyebrow."""
    frame = QFrame()
    frame.setStyleSheet(
        f'QFrame{{background:{CARD};border:1px solid {HAIRLINE};border-radius:12px;}}')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(8)
    if title is not None:
        layout.addWidget(eyebrow(title))
    return frame, layout


def tag(kind: str = 'exploratory') -> QLabel:
    """Tiny muted/italic pill for the exploratory/validated honesty markers."""
    from ph_economic_ai.ui import honesty
    text = honesty.VALIDATED if kind == 'validated' else honesty.EXPLORATORY
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f'font-family:{MONO},monospace;font-size:8px;font-style:italic;'
        f'color:{FAINT};background:transparent;')
    return lbl


def stat_card(eyebrow_text: str, value: str, color: str = INK, meta: str = '',
              tag_kind: str | None = 'exploratory',
              confidence_frac: tuple | None = None):
    """One sector-forecast card: eyebrow, a serif value, an optional
    confidence-bar row, a muted meta caption, and an optional exploratory/
    validated tag. Returns (frame, layout) like `card()`."""
    frame, layout = card()
    layout.addWidget(eyebrow(eyebrow_text))
    layout.addWidget(serif_number(value, color=color, size=28))
    if confidence_frac is not None:
        low_frac, width_frac = confidence_frac
        layout.addWidget(confidence_bar(low_frac, width_frac))
    if meta:
        layout.addWidget(muted(meta, size=11))
    if tag_kind is not None:
        layout.addWidget(tag(tag_kind))
    return frame, layout
