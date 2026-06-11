# Editorial Theme + Report Reference (SP2d-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `ui/theme.py` (the approved editorial design tokens + widget helpers) and migrate the Report (`stage4_report.py`) to it as the reference, so the rest of SP2d is a mechanical roll-out.

**Architecture:** `theme.py` is the single source of truth (palette/fonts + QLabel/QFrame factories). The Report's existing `_card`/`_muted` helpers delegate to `theme`; the consensus block uses `theme.serif_number`/`eyebrow`/`hairline`; stray legacy colours are retired. Styling only — no text/behaviour/layout change; SP2a honesty notes + SP2b bars + SP2c trajectories stay intact.

**Tech Stack:** Python 3.10, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-11-theme-and-report-restyle-design.md`.

**Confirmed anchors (`ui/stage4_report.py`):**
- `_card(self, title)` (~445): `frame = QFrame()` with `'QFrame{background:#FFFFFF;border:1px solid #EAECF0;border-radius:12px;}'`, a `QVBoxLayout(frame)` (margins 16,14,16,14; spacing 8), a title `QLabel(title)` styled `font-size:11px;font-weight:700;color:#1C1E26;`, `layout.addWidget(lbl)`, `return frame, layout`.
- `_muted(self, text)` (~458): `QLabel`, `'font-size:8px;color:#9EA3AE;text-transform:uppercase;'`, returns the label.
- `_build_swarm_left` consensus value: `val_lbl = QLabel(val_str)` + `val_lbl.setStyleSheet('font-size:24px;font-weight:700;color:#1C1E26;')`; `avg = consensus.get('weighted_avg')`; later `if _cal:` adds a `_cal_lbl`. `_build_left` has the twin pattern (`Weighted average`).
- `QFrame, QVBoxLayout, QLabel` imported in this file; `ui/honesty.py` exists (SP2a).

**Conventions:** Tests in `ph_economic_ai/tests/`, offscreen Qt. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/editorial-theme`

---

## Task 1: `ui/theme.py` design tokens + helpers

**Files:** Create `ph_economic_ai/ui/theme.py`; Test `ph_economic_ai/tests/test_theme.py`

- [ ] **Step 1: Failing test** — create `ph_economic_ai/tests/test_theme.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QFrame


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_tokens():
    from ph_economic_ai.ui import theme
    for c in (theme.SURFACE, theme.CARD, theme.INK, theme.MUTED, theme.FAINT,
              theme.HAIRLINE, theme.UP, theme.DOWN, theme.NEUTRAL):
        assert isinstance(c, str) and c.startswith('#')
    assert theme.direction_color('up') == theme.UP
    assert theme.direction_color('down') == theme.DOWN
    assert theme.direction_color('na') == theme.FAINT


def test_helpers(app):
    from ph_economic_ai.ui import theme
    assert isinstance(theme.eyebrow('hi'), QLabel) and theme.eyebrow('hi').text() == 'HI'
    assert theme.serif_number('1.8').text() == '1.8'
    assert isinstance(theme.muted('x'), QLabel)
    assert isinstance(theme.hairline(), QFrame)
    frame, layout = theme.card('Title')
    assert isinstance(frame, QFrame)
    assert 'TITLE' in [c.text() for c in frame.findChildren(QLabel)]
    assert theme.tag('validated').text() == 'validated'
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_theme.py -v`).

- [ ] **Step 3: Implement** — create `ph_economic_ai/ui/theme.py`:
```python
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
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`theme.py` + test) — `feat(ui): editorial design tokens + widget helpers (theme.py)`.

---

## Task 2: Migrate the Report to `theme`

**Files:** Modify `ph_economic_ai/ui/stage4_report.py`; Test `ph_economic_ai/tests/test_stage4_swarm.py` (append a content-survival guard)

- [ ] **Step 1: Append the guard test** to `ph_economic_ai/tests/test_stage4_swarm.py` (reuse the `MasterVerdict` fixture from `test_populate_swarm_updates_detail` / `test_consensus_marked_exploratory` — read & mirror it):
```python
def test_restyle_keeps_consensus_content(app):
    from PyQt6.QtWidgets import QLabel
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict
    from unittest.mock import MagicMock
    import numpy as np, pandas as pd
    panel = Stage4ReportPanel()
    rv = RegionalVerdict(judge_id=0, region_pair=('NCR', 'CAR'), estimate=1.5,
                         confidence=0.8, reasoning='', survivor_names=('a', 'b'))
    mv = MasterVerdict(final_estimate=1.5, confidence_pct=80, dissenting_regions=[],
                       reasoning='', regional_verdicts=[rv])
    df = pd.DataFrame({'date': pd.date_range('2024-01', periods=3, freq='M'),
                       'gas_price': [58., 59., 60.], 'oil_price': [80., 81., 82.],
                       'usd_php': [56., 56.5, 57.], 'cpi': [120., 121., 122.],
                       'remittances': [2.5, 2.6, 2.7], 'demand_index': [70., 71., 72.]})
    reg = MagicMock(); reg.predict.return_value = np.array([60.])
    reg.feature_importances_ = np.array([.5, .3, .2])
    panel.populate_swarm(mv, reg, df, 0.5, {'oil_pct': 5.0, 'usd_pct': 2.0})
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'varies per run' in texts          # SP2a honesty note survived the restyle
    assert 'SWARM CONSENSUS' in texts          # card title (now an eyebrow, uppercased)
```

- [ ] **Step 2: Run → fails** (`SWARM CONSENSUS` not uppercased yet — current `_card` title is `Swarm Consensus`): `python -m pytest ph_economic_ai/tests/test_stage4_swarm.py::test_restyle_keeps_consensus_content -v`.

- [ ] **Step 3: Implement** — in `ph_economic_ai/ui/stage4_report.py`:

(a) Add a top import near the others: `from ph_economic_ai.ui import theme as _theme`.

(b) Replace `_card` body with a delegate:
```python
    def _card(self, title: str):
        return _theme.card(title)
```

(c) Replace `_muted` body with a delegate (preserves the tiny-uppercase look via tokens):
```python
    def _muted(self, text: str) -> QLabel:
        return _theme.muted(text, size=8, color=_theme.FAINT, upper=True)
```

(d) In `_build_swarm_left`, replace the consensus value label:
```python
        val_str = f'+₱{avg:.2f}/L' if avg is not None else 'No consensus'
        val_lbl = QLabel(val_str)
        val_lbl.setStyleSheet('font-size:24px;font-weight:700;color:#1C1E26;')
```
with:
```python
        val_str = f'+₱{avg:.2f}/L' if avg is not None else 'No consensus'
        _vd = 'down' if (avg or 0) < 0 else ('up' if (avg or 0) > 0 else 'flat')
        val_lbl = _theme.serif_number(val_str, color=_theme.direction_color(_vd), size=26)
```
and just before the `if _cal:` block (which adds `_cal_lbl`), insert a hairline:
```python
        cf_layout.addWidget(_theme.hairline())
```

(e) In `_build_left` (debate mode), apply the SAME two changes to its twin consensus value label (`Weighted average` block): swap to `_theme.serif_number(val_str, color=_theme.direction_color(_vd), size=26)` (compute `_vd` from its `avg`) and add `cf_layout.addWidget(_theme.hairline())` before its `_cal` block.

(f) Replace remaining legacy/stray panel colours in this file with tokens where unambiguous: `'#F7F8FA'` panel backgrounds → `_theme.SURFACE`, consensus-frame border `'#EAECF0'` → `_theme.HAIRLINE`. (Leave chart-internal matplotlib hexes from SP2b/c as-is — those are already editorial.) Keep ALL label text, the SP2a `_note`, the SP2b sector bars, and the SP2c trajectory panel unchanged.

- [ ] **Step 4: Run → passes**: `python -m pytest ph_economic_ai/tests/test_stage4_swarm.py ph_economic_ai/tests/test_stage4_sector.py ph_economic_ai/tests/test_stage4_trajectories.py -v` (all pass — Report builds, content survives).

- [ ] **Step 5: Commit** (`stage4_report.py` + `test_stage4_swarm.py`) — `feat(ui): migrate Report to the editorial theme (reference screen)`.

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): run a sim → the Report's cards use the editorial card/eyebrow style; the consensus number is serif and coloured by direction; a hairline sits above the validated line; the SP2a "varies per run" note, the SP2b bars, and the SP2c trajectories all still render.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 theme tokens + helpers (eyebrow/serif_number/muted/hairline/card/tag/direction_color) → Task 1; §3.2 Report migration (_card/_muted delegate, consensus serif number + hairline, legacy colours → tokens, content intact) → Task 2; §4 testing (token/helper test + stage4 regression + the "varies per run" survival guard) → Tasks 1–2.
**Placeholder scan:** none — `theme.py` complete; Task 2 gives exact before/after for `_card`, `_muted`, the consensus value, and the hairline; (e)/(f) instruct mirroring + token swaps with concrete targets.
**Type consistency:** `theme.card(title) -> (QFrame, QVBoxLayout)` matches `_card`'s `return frame, layout` contract (call-sites `card, cl = self._card(...)` unchanged). `theme.muted(...) -> QLabel` matches `_muted`'s return. `serif_number`/`eyebrow`/`hairline` return QLabel/QFrame added to existing layouts. `direction_color` keys (`up/down/flat/na`) match the report's `direction` vocabulary. `theme.tag` lazily imports `honesty` (no circular import — honesty doesn't import theme). The guard test reuses the proven `MasterVerdict` fixture, and `SWARM CONSENSUS` is uppercased because `_card`→`theme.card`→`eyebrow` upcases the title.
````
