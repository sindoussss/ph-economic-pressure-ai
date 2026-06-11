# ph_economic_ai — Editorial Theme + Report Reference (SP2d-1 Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** First slice of SP2d (full editorial restyle). The codebase has **no theme module** — ~465 inline `setStyleSheet` calls with scattered/legacy palettes. SP2d-1 introduces the design-token module and proves it on one reference screen (the Report); SP2d-2…n roll it out screen-by-screen.

---

## 1. Problem & Goal

The app's look is inconsistent because every widget hand-codes colours/fonts (legacy strays like `#4A90E2`, `#888888`, `#EAEAEA` sit next to the intended palette). There's no single source of truth, so "make it editorial everywhere" can't be done coherently or maintainably one screen at a time.

**Goal:** create **`ui/theme.py`** — the approved editorial design tokens (palette, fonts) + small widget helpers — and migrate the **Report** (`stage4_report.py`) to it as the reference implementation, so the rest of SP2d is a mechanical roll-out against a proven system.

**Approved tokens (from the companion):** surface `#FBFBFA`, card `#FFFFFF`, ink `#1C1E26`, muted `#6B7280`, faint `#9AA0AA`, hairline `#E5E7EB`; accents price-up `#B3261E` (red), price-down `#15803D` (green), neutral `#3B6FD4`. Eyebrows = mono (Consolas) uppercase + letter-spacing; headline numbers = serif (Georgia) bold; body = sans. Legacy `#4A90E2`/`#888888`/`#EAEAEA` retired.

---

## 2. Scope

### In scope
- `ui/theme.py` — token constants + helper factories (pure-ish, offscreen-Qt testable).
- Migrate `stage4_report.py` core styling to `theme` (its `_card`/`_muted` helpers, the consensus block's serif number + eyebrow + hairline, and replace stray/legacy colours with tokens).

### Out of scope
- Every other screen (SP2d-2…n: landing, overview, simulation canvases, agent-perf, accuracy, setup, shared widgets).
- Dead files (`dashboard.py`, `pressure.py`, `stage1_rag.py`, `sidebar.py`) — not touched (candidates for separate deletion).
- Any behaviour/data/layout change — **styling only**; all existing labels/honesty wording (SP2a) and charts (SP2b/c) keep their text + structure.

### Non-negotiable
- No regressions: the Report still builds and all `stage4` tests stay green.
- The honesty tags (exploratory/validated) and the SP2b/SP2c content render unchanged in substance — theme changes their *style*, not their text or presence.

---

## 3. Components

### 3.1 `ui/theme.py`
**Token constants:**
```python
SURFACE, CARD = '#FBFBFA', '#FFFFFF'
INK, MUTED, FAINT, HAIRLINE = '#1C1E26', '#6B7280', '#9AA0AA', '#E5E7EB'
UP, DOWN, NEUTRAL = '#B3261E', '#15803D', '#3B6FD4'   # price up=red, down=green
SERIF, MONO = 'Georgia', 'Consolas'
```
**Helper factories (return styled QWidgets so screens don't hand-roll stylesheets):**
- `eyebrow(text) -> QLabel` — mono, ~10px, 700, letter-spacing, colour `FAINT`, uppercased.
- `serif_number(text, color=INK, size=24) -> QLabel` — `SERIF`, bold, given size/colour.
- `muted(text, size=9, color=MUTED) -> QLabel` — small sans muted label, word-wrap on.
- `hairline() -> QFrame` — 1px high, background `HAIRLINE`.
- `card(title=None) -> tuple[QFrame, QVBoxLayout]` — white card (`CARD`, 1px `HAIRLINE`, radius, padding); if `title`, adds an `eyebrow(title)` at top. Returns `(frame, content_layout)`.
- `tag(text, kind='exploratory') -> QLabel` — tiny muted/italic pill reusing `honesty.EXPLORATORY`/`VALIDATED`.
- `direction_color(direction) -> str` — `{'up':UP,'down':DOWN,'flat':MUTED,'na':FAINT}`.
- `qss_*` string helpers only if a raw stylesheet is genuinely needed (prefer the QLabel factories).

All helpers are tolerant (no external state); importing `theme` must not require a running app beyond what QWidget construction needs (tests use offscreen Qt).

### 3.2 `stage4_report.py` migration (the reference)
- `self._card(title)` → delegate to `theme.card(title)` (keep the method name/signature so call-sites are unchanged).
- `self._muted(text)` → delegate to `theme.muted(text)`.
- Consensus block (`_build_swarm_left` / `_build_left`): the big value label → `theme.serif_number(val_str, color=theme.direction_color(...), size=24)`; section labels → `theme.eyebrow(...)`; the calibrated/validated line keeps its text but uses token colours; insert a `theme.hairline()` before the validated line.
- Replace stray/legacy hexes in `stage4_report.py` with the matching token (`#F7F8FA`→`SURFACE` where it's a panel bg, ink/muted/hairline to tokens). Sector-card colours map via `theme.direction_color`.
- Keep all text, the SP2a notes, the SP2b bars, the SP2c trajectory panel exactly as-is (style only).

## 4. Testing
- `test_theme.py` (offscreen Qt): token constants exist + are hex/str; `eyebrow('x')`/`serif_number('1')`/`muted('y')` return `QLabel` with the given text; `hairline()` returns a `QFrame`; `card('T')` returns `(QFrame, layout)` and the frame contains an eyebrow label with `'T'`; `direction_color('up')==UP`.
- `stage4` regression: existing `test_stage4_sector.py`, `test_stage4_swarm.py`, `test_stage4_trajectories.py` stay green (the Report still builds; honesty/bars/trajectories intact). Add one assertion that a consensus label still contains the SP2a "varies per run" note after the restyle (guards against losing content during the migration).
- Full suite green.

## 5. Deliverables (definition of done)
1. `ui/theme.py` with the approved tokens + helper factories; tested.
2. `stage4_report.py` migrated to `theme` (card/muted/eyebrow/serif/hairline/colours); stray legacy colours in this file retired.
3. Report looks like the approved reference (editorial), all SP2a/b/c content intact; tests + full suite green.

## 6. Why it matters
It turns "restyle everything" from 465 ad-hoc edits into a coherent system: one approved vocabulary, proven on the most-seen screen, with the rest of SP2d a mechanical roll-out. Consistency, maintainability, and a credible polished look — without touching behaviour or the honesty surface.
