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
