# Report Screen Visual Hierarchy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish SP2d-1's editorial-theme migration on the Report screen (`stage4_report.py` + `causal_chain_widget.py`), replace the BSP CPI banner's boxed-alert pattern with a plain header stat, and wire the already-computed food/electricity agreement percentages into the sector-forecast cards.

**Architecture:** Three new small factory functions in `ui/theme.py` (`confidence_bar`, `stat_card`, `page_header`), consumed by a structural rebuild of `Stage4ReportPanel.set_sector_forecasts()` and `BSPAlertBanner`, plus a mechanical token-substitution pass (raw `setStyleSheet` calls → `theme.*` calls) over the larger consensus/output panels that must not change their content. One small data-wiring change in `main_window.py` threads two already-computed values (`self._food_agreement`, `self._elec_agreement`) into a call that doesn't currently receive them.

**Tech Stack:** PyQt6, pytest (offscreen Qt via `QT_QPA_PLATFORM=offscreen`), the existing `ui/theme.py` editorial token module.

## Global Constraints

- No change to any computed value: `honesty.py` caveat/marker logic, `DebateEngine.consensus()`, conformal interval math, and the CPI projection formula are untouched. (Spec §2, non-negotiable.)
- Every honesty caveat, agreement percentage, interval, and "exploratory/not validated" tag that renders today must still render with the same text after the restyle. (Spec §4.)
- `BSPAlertBanner` keeps its class name and public interface (`set_alert(alert: dict)`); only its internals change. (Spec §3.4.)
- The one deliberate scope addition — food/electricity agreement % — only threads through already-computed values (`self._food_agreement`, `self._elec_agreement`); nothing new is computed. (Spec §2, "One small, deliberate scope addition.")
- Full suite green before this is done; launch the real app afterward and visually compare against the approved mockup (`.superpowers/brainstorm/1304-1786231728/content/full-mockup-v3.html`).

---

## Task 1: `theme.py` — `confidence_bar()`

**Files:**
- Modify: `ph_economic_ai/ui/theme.py`
- Test: `ph_economic_ai/tests/test_theme.py`

**Interfaces:**
- Produces: `confidence_bar(low_frac: float, width_frac: float) -> QFrame` — a 5px track (`FAINT`-tinted background) with a dark-`INK` filled child `QFrame` positioned from `low_frac` to `low_frac + width_frac` (both 0.0-1.0, fractions of the track's own width). Purely decorative; callers compute the fractions from data they already have.
- Produces: `WARNING: str` — a new token constant, `'#B45309'`. The existing caveat/warning labels across `stage4_report.py` already use this exact hex consistently (Tasks 7-8 below); this names it so it stops being a magic value not covered by SP2d-1's original approved token set.

- [ ] **Step 1: Write the failing test**

Add to `ph_economic_ai/tests/test_theme.py`:

```python
def test_confidence_bar(app):
    from ph_economic_ai.ui import theme
    bar = theme.confidence_bar(0.3, 0.2)
    assert isinstance(bar, QFrame)
    fills = [c for c in bar.findChildren(QFrame)]
    assert len(fills) == 1
    assert bar.height() == 5


def test_warning_token():
    from ph_economic_ai.ui import theme
    assert theme.WARNING == '#B45309'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_confidence_bar ph_economic_ai/tests/test_theme.py::test_warning_token -v`
Expected: both FAIL -- `test_confidence_bar` with `AttributeError: module 'ph_economic_ai.ui.theme' has no attribute 'confidence_bar'`, `test_warning_token` with `AttributeError: module 'ph_economic_ai.ui.theme' has no attribute 'WARNING'`

- [ ] **Step 3: Write the implementation**

Add `WARNING = '#B45309'` to `ph_economic_ai/ui/theme.py`'s existing token block (next to `UP`, `DOWN`, `NEUTRAL`):

```python
UP = '#B3261E'        # price up = red (bad for consumers)
DOWN = '#15803D'      # price down = green (good)
NEUTRAL = '#3B6FD4'
WARNING = '#B45309'   # amber -- collapsed-room and unscored-survivor caveats
```

Then add `confidence_bar` to `ph_economic_ai/ui/theme.py` (after `hairline()`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_confidence_bar ph_economic_ai/tests/test_theme.py::test_warning_token -v`
Expected: both PASS

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/theme.py ph_economic_ai/tests/test_theme.py
git commit -m "feat(theme): add confidence_bar() and WARNING token"
```

---

## Task 2: `theme.py` — `stat_card()`

**Files:**
- Modify: `ph_economic_ai/ui/theme.py`
- Test: `ph_economic_ai/tests/test_theme.py`

**Interfaces:**
- Consumes: `confidence_bar()` from Task 1; `card()`, `eyebrow()`, `serif_number()`, `muted()`, `tag()` (all pre-existing in `theme.py`).
- Produces: `stat_card(eyebrow_text: str, value: str, color: str = INK, meta: str = '', tag_kind: str | None = 'exploratory', confidence_frac: tuple[float, float] | None = None) -> tuple[QFrame, QVBoxLayout]` — same `(frame, layout)` return shape as `card()`. `confidence_frac`, when given, is `(low_frac, width_frac)` passed straight to `confidence_bar()`; when `None`, no bar is added (needed for the sector-forecast cards, where confidence data may be absent). `tag_kind=None` omits the tag entirely.

- [ ] **Step 1: Write the failing test**

Add to `ph_economic_ai/tests/test_theme.py`:

```python
def test_stat_card_full(app):
    from ph_economic_ai.ui import theme
    frame, layout = theme.stat_card(
        'GAS / FUEL', '-P0.08/L', color=theme.DOWN,
        meta='70% agent agreement', tag_kind='exploratory',
        confidence_frac=(0.38, 0.24))
    labels = [c.text() for c in frame.findChildren(QLabel)]
    assert 'GAS / FUEL' in labels
    assert '-P0.08/L' in labels
    assert '70% agent agreement' in labels
    assert 'exploratory' in labels
    assert len(frame.findChildren(QFrame)) >= 1  # the confidence bar's track


def test_stat_card_without_confidence_or_tag(app):
    from ph_economic_ai.ui import theme
    frame, layout = theme.stat_card('FOOD', '-0.21%', tag_kind=None)
    labels = [c.text() for c in frame.findChildren(QLabel)]
    assert 'FOOD' in labels
    assert '-0.21%' in labels
    assert 'exploratory' not in labels and 'validated' not in labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_stat_card_full ph_economic_ai/tests/test_theme.py::test_stat_card_without_confidence_or_tag -v`
Expected: FAIL with `AttributeError: module 'ph_economic_ai.ui.theme' has no attribute 'stat_card'`

- [ ] **Step 3: Write the implementation**

Add to `ph_economic_ai/ui/theme.py` (after `tag()`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_stat_card_full ph_economic_ai/tests/test_theme.py::test_stat_card_without_confidence_or_tag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/theme.py ph_economic_ai/tests/test_theme.py
git commit -m "feat(theme): add stat_card() editorial helper"
```

---

## Task 3: `theme.py` — `page_header()`

**Files:**
- Modify: `ph_economic_ai/ui/theme.py`
- Test: `ph_economic_ai/tests/test_theme.py`

**Interfaces:**
- Consumes: `eyebrow()`, `serif_number()`, `muted()` (pre-existing).
- Produces: `page_header(eyebrow_text: str, title: str, right_eyebrow: str | None = None, right_value: str | None = None, right_caption: str | None = None) -> QFrame` — a `QHBoxLayout` row: left side is `eyebrow_text` above `title` (as a bold serif label, not `serif_number` -- title is prose, not a number); right side, when `right_value` is given, is `right_eyebrow` above a serif number (`right_value`) above `right_caption`. No border, no background fill.

- [ ] **Step 1: Write the failing test**

Add to `ph_economic_ai/tests/test_theme.py`:

```python
def test_page_header_with_right_stat(app):
    from ph_economic_ai.ui import theme
    hdr = theme.page_header(
        'SIMULATION REPORT', 'Next-month sector forecast',
        right_eyebrow='PROJECTED CPI', right_value='6.03%',
        right_caption='baseline 6.2%')
    labels = [c.text() for c in hdr.findChildren(QLabel)]
    assert 'SIMULATION REPORT' in labels
    assert 'Next-month sector forecast' in labels
    assert 'PROJECTED CPI' in labels
    assert '6.03%' in labels
    assert 'baseline 6.2%' in labels


def test_page_header_without_right_stat(app):
    from ph_economic_ai.ui import theme
    hdr = theme.page_header('EYEBROW', 'Title only')
    labels = [c.text() for c in hdr.findChildren(QLabel)]
    assert 'Title only' in labels
    assert '6.03%' not in labels
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_page_header_with_right_stat ph_economic_ai/tests/test_theme.py::test_page_header_without_right_stat -v`
Expected: FAIL with `AttributeError: module 'ph_economic_ai.ui.theme' has no attribute 'page_header'`

- [ ] **Step 3: Write the implementation**

`theme.py` today imports `from PyQt6.QtWidgets import QLabel, QFrame, QVBoxLayout` and nothing from `QtCore`. Change that import line to also bring in `QHBoxLayout`, and add a new import for `Qt`:

```python
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QFrame, QHBoxLayout, QVBoxLayout


def page_header(eyebrow_text: str, title: str, right_eyebrow: str = None,
                 right_value: str = None, right_caption: str = None) -> QFrame:
    """Plain page-header row: title on the left, an optional stat on the
    right. No border, no fill -- replaces a boxed-alert pattern."""
    frame = QFrame()
    row = QHBoxLayout(frame)
    row.setContentsMargins(0, 0, 0, 10)

    left = QVBoxLayout()
    left.addWidget(eyebrow(eyebrow_text))
    title_lbl = QLabel(title)
    title_lbl.setStyleSheet(
        f'font-family:{SERIF},serif;font-size:20px;font-weight:700;color:{INK};')
    left.addWidget(title_lbl)
    row.addLayout(left)
    row.addStretch()

    if right_value is not None:
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignRight)
        if right_eyebrow:
            right.addWidget(eyebrow(right_eyebrow))
        right.addWidget(serif_number(right_value, size=26))
        if right_caption:
            right.addWidget(muted(right_caption, size=9))
        row.addLayout(right)

    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_theme.py::test_page_header_with_right_stat ph_economic_ai/tests/test_theme.py::test_page_header_without_right_stat -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/theme.py ph_economic_ai/tests/test_theme.py
git commit -m "feat(theme): add page_header() editorial helper"
```

---

## Task 4: Wire food/electricity agreement into `set_sector_forecasts`'s call site

**Files:**
- Modify: `ph_economic_ai/ui/main_window.py:762-763`
- Test: `ph_economic_ai/tests/test_main_window.py`

**Interfaces:**
- Consumes: `self._gas_agreement`, `self._food_agreement`, `self._elec_agreement` (all pre-existing `int` attributes on `MainWindow`, already populated by `_on_food_complete`/`_on_elec_complete`/the swarm-complete handler).
- Produces: `Stage4ReportPanel.set_sector_forecasts` now called with three extra keyword args; Task 5 defines the new parameter names this call must match exactly: `gas_agreement`, `food_agreement`, `elec_agreement`.

- [ ] **Step 1: Write the failing test**

Add to `ph_economic_ai/tests/test_main_window.py` (match the existing test module's fixture/import style used by the other `test_*` functions in that file):

```python
def test_push_sector_forecasts_passes_agreement_values(app, qtbot=None):
    from ph_economic_ai.ui.main_window import MainWindow
    win = MainWindow()
    win._gas_estimate, win._food_estimate, win._elec_estimate = -0.08, -0.21, -0.10
    win._gas_agreement, win._food_agreement, win._elec_agreement = 70, 62, 81
    captured = {}
    win._stage4.set_sector_forecasts = lambda **kw: captured.update(kw)
    win._push_sector_forecasts()
    assert captured['gas_agreement'] == 70
    assert captured['food_agreement'] == 62
    assert captured['elec_agreement'] == 81
```

Check `test_main_window.py`'s existing tests for the exact `app`/`qtbot` fixture names already in use in that file (e.g. `test_main_window_has_swarm_panel(app)`) and match them -- don't introduce a new fixture.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_main_window.py::test_push_sector_forecasts_passes_agreement_values -v`
Expected: FAIL with `KeyError: 'gas_agreement'` (the lambda never receives it, since the real call doesn't pass it yet)

- [ ] **Step 3: Write the implementation**

In `ph_economic_ai/ui/main_window.py`, change:

```python
            self._stage4.set_sector_forecasts(
                self._gas_estimate, self._food_estimate, self._elec_estimate)
```

to:

```python
            self._stage4.set_sector_forecasts(
                self._gas_estimate, self._food_estimate, self._elec_estimate,
                gas_agreement=self._gas_agreement, food_agreement=self._food_agreement,
                elec_agreement=self._elec_agreement)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_main_window.py::test_push_sector_forecasts_passes_agreement_values -v`
Expected: PASS (Task 5 gives `set_sector_forecasts` these keyword params; until Task 5 lands, this test passes against the stub lambda regardless of the real method's signature, so task order here doesn't block this step.)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/main_window.py ph_economic_ai/tests/test_main_window.py
git commit -m "feat(report): thread food/electricity agreement into sector forecasts"
```

---

## Task 5: Migrate `set_sector_forecasts()` to `stat_card()`

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py:340-388`
- Test: `ph_economic_ai/tests/test_stage4_swarm.py` (or wherever `set_sector_forecasts` is currently exercised -- grep first, see Step 0)

**Interfaces:**
- Consumes: `theme.stat_card()` (Task 2); `sector_forecast_rows()` from `ui/sector_forecast.py` (unchanged); the three new keyword args from Task 4.
- Produces: `set_sector_forecasts(self, gas=None, food=None, elec=None, gas_agreement=0, food_agreement=0, elec_agreement=0)` -- same three positional params as today (existing callers besides `main_window.py` must keep working), three new keyword-only-by-convention params with defaults so old call sites without them don't break.

- [ ] **Step 0: Find every existing caller/test of `set_sector_forecasts`**

Run: `grep -rn "set_sector_forecasts" ph_economic_ai/`

Confirm the only production caller is `main_window.py:762` (updated in Task 4) and note any test file calling it directly with positional args only -- those must keep passing since the three new params have defaults.

- [ ] **Step 1: Write the failing test**

Step 0 found `ph_economic_ai/tests/test_stage4_sector.py` already exercises this method. Add these tests there, alongside the existing ones -- do not create a new file (this project has a documented recurring defect pattern of coverage scattered across near-duplicate test files):

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_sector_forecasts_show_agreement_for_all_three(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10,
                               gas_agreement=70, food_agreement=62, elec_agreement=81)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert any('70%' in t for t in labels)
    assert any('62%' in t for t in labels)
    assert any('81%' in t for t in labels)


def test_sector_forecasts_omits_agreement_when_zero(app):
    """Old callers/tests still calling positionally-only must not crash, and a
    0 agreement (the pre-Task-4 default, or a genuinely unmeasured run) must
    not print a nonsense '0% agreement' caption."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10)
    labels = [c.text() for c in panel._sector_holder.findChildren(QLabel)]
    assert not any('% agreement' in t for t in labels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: FAIL -- `TypeError: set_sector_forecasts() got an unexpected keyword argument 'gas_agreement'`

- [ ] **Step 3: Write the implementation**

Replace `set_sector_forecasts` in `ph_economic_ai/ui/stage4_report.py` (lines 340-388) with:

```python
    def set_sector_forecasts(self, gas=None, food=None, elec=None,
                             gas_agreement=0, food_agreement=0, elec_agreement=0):
        """Render the gas/food/electricity next-month forecasts as a card grid."""
        from ph_economic_ai.ui.sector_forecast import sector_forecast_rows
        agreements = {'gas': gas_agreement, 'food': food_agreement, 'elec': elec_agreement}
        try:
            while self._sector_holder_layout.count():
                it = self._sector_holder_layout.takeAt(0)
                w = it.widget()
                if w is not None:
                    w.deleteLater()
            self._sector_holder_layout.addWidget(_theme.eyebrow('NEXT-MONTH SECTOR FORECAST'))
            self._sector_holder_layout.addWidget(_theme.muted('exploratory — not validated', size=9))
            # A QWidget wrapper, not a bare sub-layout: the clear loop above
            # only ever calls .deleteLater() on it.widget() results, so a
            # layout added via addLayout() (whose QLayoutItem.widget() is
            # always None) would never be cleaned up on the next render --
            # main_window.py calls this method from 5 call sites, so every
            # re-render would silently orphan the previous set of cards.
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for r in sector_forecast_rows(gas, food, elec):
                agreement = agreements[r['key']]
                meta = f'{agreement}% agent agreement' if agreement else ''
                confidence_frac = (0.0, r['bar']) if r['bar'] > 0 else None
                frame, layout = _theme.stat_card(
                    r['label'].upper(), r['value_str'],
                    color=_theme.direction_color(r['direction']),
                    meta=meta, tag_kind='exploratory',
                    confidence_frac=confidence_frac)
                row_layout.addWidget(frame)
                # The row key is 'elec'; the engine's sector name is 'electricity'.
                block = self._explanation_block(_ROW_KEY_TO_SECTOR.get(r['key'], r['key']))
                if block is not None:
                    layout.addWidget(block)
            self._sector_holder_layout.addWidget(row_widget)
            self._sector_holder.setVisible(True)
            self._build_sector_trajectories(gas, food, elec)
        except Exception:
            pass
```

This section header stays a plain `eyebrow()` + `muted()` pair rather than `page_header()` -- `page_header`'s `title` slot is a bold 20px serif heading (Task 3), and "exploratory — not validated" is caption text, not a heading. Reusing `page_header` here would render a caveat as if it were the page's main title.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: PASS

- [ ] **Step 5: Write a regression test for the widget-leak fix, and run it**

Add to `ph_economic_ai/tests/test_stage4_sector.py`:

```python
def test_calling_set_sector_forecasts_twice_does_not_leak_the_old_cards(app):
    """Re-rendering (main_window.py calls this from 5 sites) must actually
    remove the previous cards, not just lose track of them."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    panel.set_sector_forecasts(-0.08, -0.21, -0.10, gas_agreement=70)
    first_children = set(panel._sector_holder.findChildren(QLabel))
    panel.set_sector_forecasts(0.15, 0.30, 0.05, gas_agreement=40)
    second_children = set(panel._sector_holder.findChildren(QLabel))
    # None of the first render's labels should still be attached -- they may
    # not be destroyed yet (deleteLater is deferred), but they must no
    # longer be children of the holder.
    assert not (first_children & second_children)
    assert any('40%' in c.text() for c in second_children)
    assert not any('70%' in c.text() for c in second_children)
```

Run: `pytest ph_economic_ai/tests/test_stage4_sector.py -v`
Expected: PASS (fails against the pre-fix `addLayout(row_layout)` version -- the first render's labels stay attached as orphaned children of `_sector_holder`, so `first_children & second_children` would be empty only coincidentally; the stronger check is that `second_children` would contain both '70%' and '40%' labels simultaneously since the first row never gets removed -- confirm this test actually distinguishes the two implementations before trusting it, per this plan's own testing discipline)

- [ ] **Step 6: Run the file's full existing test coverage to confirm no other regression**

Run: `pytest ph_economic_ai/tests/test_stage4_sector.py ph_economic_ai/tests/test_stage4_trajectories.py -v`
Expected: all PASS (the two files Step 0's own grep found actually exercise `set_sector_forecasts`/the sector-forecast section -- not `test_stage4_swarm.py`/`test_stage4_classic_honesty.py`, which don't touch this method at all)

- [ ] **Step 7: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_sector.py
git commit -m "feat(report): migrate sector-forecast row list to stat_card grid"
```

---

## Task 6: Migrate `BSPAlertBanner` to `page_header()`

**Files:**
- Modify: `ph_economic_ai/ui/causal_chain_widget.py:75-152`
- Test: create `ph_economic_ai/tests/test_bsp_alert_banner.py`

**Interfaces:**
- Consumes: `theme.page_header()` (Task 3).
- Produces: `BSPAlertBanner.set_alert(alert: dict)` -- unchanged signature and `alert` dict shape (`severity`, `projected_cpi`, `current_cpi`, `cpi_as_of`, `sector_cpi_impact`, `breakdown`). Callers (`main_window.py::set_bsp_alert`, line 985) need no change.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_bsp_alert_banner.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_set_alert_shows_plain_header_no_colored_box(app):
    from ph_economic_ai.ui.causal_chain_widget import BSPAlertBanner
    banner = BSPAlertBanner()
    banner.set_alert({
        'severity': 'CRITICAL', 'projected_cpi': 6.03, 'current_cpi': 6.2,
        'cpi_as_of': 'PSA, Jul 2026', 'sector_cpi_impact': -0.17,
        'breakdown': {'fuel': -0.01, 'food': -0.08, 'electricity': -0.07},
    })
    assert not banner.isHidden()
    labels = [c.text() for c in banner.findChildren(QLabel)]
    assert any('6.03%' in t for t in labels)
    assert any('CRITICALLY EXCEEDED' in t.upper() for t in labels)
    # No QFrame in the banner (besides the banner itself) should carry a
    # colored background/border fill -- the boxed-alert pattern is gone.
    style = banner.styleSheet()
    assert 'background' not in style or '#FEF2F2' not in style


def test_set_alert_stable_severity_still_shows(app):
    from ph_economic_ai.ui.causal_chain_widget import BSPAlertBanner
    banner = BSPAlertBanner()
    banner.set_alert({
        'severity': 'STABLE', 'projected_cpi': 3.1, 'current_cpi': 3.0,
        'cpi_as_of': 'PSA, Jul 2026', 'sector_cpi_impact': 0.1, 'breakdown': {},
    })
    assert not banner.isHidden()
    labels = [c.text() for c in banner.findChildren(QLabel)]
    assert any('3.10%' in t for t in labels)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_bsp_alert_banner.py -v`
Expected: FAIL on the `'#FEF2F2' not in style` assertion (today's implementation sets exactly that background for `CRITICAL` severity via `_SEVERITY_COLORS`)

- [ ] **Step 3: Write the implementation**

Replace `BSPAlertBanner` in `ph_economic_ai/ui/causal_chain_widget.py` (lines 75-152) with:

```python
class BSPAlertBanner(QFrame):
    """Plain page-header stat shown at the top of Stage 4 when BSP target is
    at risk. No colored box -- severity is conveyed by the eyebrow wording
    and a small severity-colored dot, not a bordered/tinted alert card."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hide()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._header = None

    def set_alert(self, alert: dict):
        severity    = alert.get('severity', 'STABLE')
        projected   = alert.get('projected_cpi', 0.0)
        current     = alert.get('current_cpi', 0.0)
        cpi_as_of   = alert.get('cpi_as_of', 'unknown vintage')
        impact      = alert.get('sector_cpi_impact', 0.0)
        breakdown   = alert.get('breakdown', {})

        text_c, _bg_c, _border_c = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS['STABLE'])

        if self._header is not None:
            self._layout.removeWidget(self._header)
            self._header.deleteLater()

        parts = []
        if 'fuel' in breakdown:
            parts.append(f'Fuel: +{breakdown["fuel"]:.2f}ppt')
        if 'food' in breakdown:
            parts.append(f'Food: +{breakdown["food"]:.2f}ppt')
        if 'electricity' in breakdown:
            parts.append(f'Elec: +{breakdown["electricity"]:.2f}ppt')
        caption = (f'Baseline {current:.1f}% ({cpi_as_of}) + sector impact {impact:+.2f}ppt'
                  + ('  ·  ' + '  ·  '.join(parts) if parts else ''))

        icon = {'STABLE': '●', 'WATCH': '◆', 'ALERT': '▲', 'CRITICAL': '■'}.get(severity, '●')
        eyebrow_text = f'{icon} {_SEVERITY_ICONS.get(severity, "")}'

        from ph_economic_ai.ui import theme as _theme
        self._header = _theme.page_header(
            eyebrow_text, ' ',
            right_eyebrow='PROJECTED CPI',
            right_value=f'{projected:.2f}%',
            right_caption=caption)
        # The severity color still shows, just on the eyebrow icon/text rather
        # than as a page-wide tint -- find the eyebrow label and recolor it.
        for lbl in self._header.findChildren(QLabel):
            if lbl.text() == eyebrow_text:
                lbl.setStyleSheet(lbl.styleSheet() + f';color:{text_c};')
                break
        self._layout.addWidget(self._header)
        self.show()
```

This drops the `_icon_lbl`/`_title_lbl`/`_detail_lbl`/`_cpi_lbl` instance attributes the old implementation kept. Grep for any external reader of those attributes before deleting them:

Run: `grep -rn "_icon_lbl\|_title_lbl\|_detail_lbl\|_cpi_lbl" ph_economic_ai/ --include="*.py"`

If anything outside this class reads them, keep equivalent public accessors (e.g. a `banner.cpi_text` property) rather than breaking that caller silently.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest ph_economic_ai/tests/test_bsp_alert_banner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/causal_chain_widget.py ph_economic_ai/tests/test_bsp_alert_banner.py
git commit -m "feat(report): replace BSPAlertBanner's boxed alert with a plain header stat"
```

---

## Task 7: Token-restyle `_build_swarm_left()` and its Regional Verdicts table

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py:555-731`
- Test: create `ph_economic_ai/tests/test_stage4_swarm_left_restyle.py`

**Interfaces:**
- Consumes: `theme.eyebrow`, `theme.muted`, `theme.hairline`, `theme.card` (all pre-existing).
- Produces: no change to `_build_swarm_left(self, master_verdict, consensus: dict)`'s signature or the text content of any label it builds -- this task is a 1:1 style-call substitution, not a restructure.

This method carries eight independently-visible honesty labels (`sub_lbl`, `basis_lbl`, `caveat_lbl`, `cross_lbl`, `synth_lbl`, `bracket_lbl`, `unscored_lbl`, the `_outside`/anchor/`_note` conditionals). **None of their text or `.setVisible(...)` conditions may change.** Only their `setStyleSheet(...)` calls change, per this exact mapping (the Regional Verdicts card below this table is a real restructure, not a token swap -- handled separately after the table):

| Current `setStyleSheet(...)` string | Replace with |
|---|---|
| `'font-size:9px;color:#6B7280;'` (line 584, `sub_lbl`) | delete the call; use `_theme.muted(text, size=9)` to build the label instead of a bare `QLabel(text)` |
| `'font-size:8px;color:#9CA3AF;'` (lines 590, 608, 614, 621, 627 -- `basis_lbl`, `cross_lbl`, `synth_lbl`, `bracket_lbl`) | `_theme.muted(text, size=8, color=_theme.FAINT)` |
| `'font-size:9px;font-weight:600;color:#B45309;'` (lines 599, 628 -- `caveat_lbl`, `unscored_lbl`) | `f'font-size:9px;font-weight:600;color:{_theme.WARNING};'` |
| `f'background:{_theme.SURFACE};border-radius:10px;border:1px solid {_theme.HAIRLINE};'` (line 568, `consensus_frame`) | already tokenized; leave as-is |
| `'font-size:11px;font-weight:600;color:#1C1E26;'` (line 645, range-row values) | `f'font-size:11px;font-weight:600;color:{_theme.INK};'` |

For every row in the table where the replacement is "use `_theme.muted(...)` to build the label," change the code from:
```python
sub_lbl = QLabel('Master judge estimate · agent agreement not measurable this run')
sub_lbl.setStyleSheet('font-size:9px;color:#6B7280;')
```
to:
```python
sub_lbl = _theme.muted('Master judge estimate · agent agreement not measurable this run', size=9)
```
(same pattern for `basis_lbl`, `cross_lbl`, `synth_lbl`, `bracket_lbl` -- each keeps its exact existing text argument, only the construction call changes).

**Regional Verdicts: real restructure, not a token swap.** Replace the whole per-region `QFrame` loop (current lines 688-731) with a compact table -- same card (`rv_card`), same column (`self._left`), just denser. Add `QGridLayout` to the existing `PyQt6.QtWidgets` import line at the top of the file (it currently imports `QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QPushButton, QFileDialog, QStackedWidget` but not `QGridLayout`). Replace:

```python
        # Regional verdicts table
        rv_card, rvcl = self._card('Regional Verdicts')
        for rv in master_verdict.regional_verdicts:
            rvf = QFrame()
            rvf.setStyleSheet(f'background:{_theme.SURFACE};border-radius:8px;border:1px solid {_theme.HAIRLINE};')
            rvfl = QVBoxLayout(rvf)
            rvfl.setContentsMargins(10, 8, 10, 8)
            rvfl.setSpacing(3)

            head_row = QHBoxLayout()
            pair_str = ' & '.join(rv.region_pair)
            name_lbl = QLabel(pair_str[:50])
            name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')
            rejected = getattr(rv, 'rejected_estimate', None)
            if rv.estimate is not None:
                est_str = f'{rv.estimate:+.2f} ₱/L'
            elif rejected is not None:
                est_str = 'discarded'
            else:
                est_str = 'no estimate'
            est_lbl = QLabel(est_str)
            est_lbl.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
            head_row.addWidget(name_lbl)
            head_row.addStretch()
            head_row.addWidget(est_lbl)
            rvfl.addLayout(head_row)

            conf_lbl = QLabel(f'Agent agreement: {rv.confidence:.0%}')
            conf_lbl.setStyleSheet('font-size:8px;color:#9EA3AE;')
            rvfl.addWidget(conf_lbl)

            # A blank estimate reads as a crash unless we say what happened.
            note = _missing_estimate_note(rv.estimate, rejected)
            if note:
                note_lbl = QLabel(note)
                note_lbl.setWordWrap(True)
                note_lbl.setStyleSheet('font-size:8px;color:#B45309;')
                rvfl.addWidget(note_lbl)

            rvcl.addWidget(rvf)

        self._left.addWidget(card)
        self._left.addWidget(rv_card)
        self._left.addStretch()
```

with:

```python
        # Regional verdicts: one compact table row per region, replacing the
        # old one-bordered-QFrame-per-region layout. Every value the old
        # layout showed (name, estimate/discarded/no-estimate, agreement%,
        # the missing-estimate honesty note) still renders -- the note just
        # moves from inside each region's box to its own line below the
        # table, since a table cell can't word-wrap a variable-length caveat.
        rv_card, rvcl = self._card('Regional Verdicts')
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(4)
        notes = []
        for row_i, rv in enumerate(master_verdict.regional_verdicts):
            pair_str = ' & '.join(rv.region_pair)
            name_lbl = QLabel(pair_str[:50])
            name_lbl.setStyleSheet(f'font-size:10px;font-weight:600;color:{_theme.INK};')
            rejected = getattr(rv, 'rejected_estimate', None)
            if rv.estimate is not None:
                est_str = f'{rv.estimate:+.2f} ₱/L'
            elif rejected is not None:
                est_str = 'discarded'
            else:
                est_str = 'no estimate'
            est_lbl = QLabel(est_str)
            est_lbl.setStyleSheet(f'font-size:10px;font-weight:700;color:{_theme.INK};')
            conf_lbl = _theme.muted(f'{rv.confidence:.0%}', size=9, color=_theme.FAINT)
            grid.addWidget(name_lbl, row_i, 0)
            grid.addWidget(est_lbl, row_i, 1)
            grid.addWidget(conf_lbl, row_i, 2)

            note = _missing_estimate_note(rv.estimate, rejected)
            if note:
                notes.append(f'{pair_str}: {note}')
        rvcl.addLayout(grid)
        for note_text in notes:
            note_lbl = QLabel(note_text)
            note_lbl.setWordWrap(True)
            note_lbl.setStyleSheet(f'font-size:8px;color:{_theme.WARNING};')
            rvcl.addWidget(note_lbl)

        self._left.addWidget(card)
        self._left.addWidget(rv_card)
        self._left.addStretch()
```

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_stage4_swarm_left_restyle.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _mv(**overrides):
    from types import SimpleNamespace as NS
    base = dict(final_estimate=-0.08, confidence_pct=70, dissenting_regions=[],
               regional_verdicts=[], physical_anchor=None)
    base.update(overrides)
    return NS(**base)


def test_every_honesty_caveat_survives_the_restyle(app):
    """Pins every caveat string _build_swarm_left can render, so a future
    styling change can't silently drop one -- the exact regression shape
    this project's own audits keep finding (RSK-038 and its recurrences)."""
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'agreement_n': 3, 'agreement_distinct': 2, 'agreement_regions': (2, 2),
        'agreement_echo_n': 1, 'agreement_diversity': 0.3,
        'agreement_models': {}, 'unscored_regions': 1, 'outside_regional': True,
    }
    panel._build_swarm_left(_mv(confidence_pct=100), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    joined = ' '.join(texts)
    assert '100% agent agreement' in joined
    assert 'outside the regional range above' in joined
    assert any('unscored' in t.lower() or 'tie-break' in t.lower() for t in texts)


def test_narrow_room_caveat_visible_when_collapsed(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.ui import honesty as _honesty
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'agreement_n': 3, 'agreement_distinct': 1, 'agreement_regions': (2, 2),
        'agreement_echo_n': 0, 'agreement_diversity': 0.0,
        'agreement_models': {}, 'unscored_regions': 0,
    }
    panel._build_swarm_left(_mv(confidence_pct=100), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    expected_caveat = _honesty.agreement_caveat(3, 1, 0.0)
    assert expected_caveat in texts


def test_regional_table_keeps_every_value_and_the_missing_estimate_note(app):
    """Pins the Regional Verdicts restructure specifically: name, estimate
    (or 'discarded'), agreement%, and the missing-estimate honesty note all
    have to survive moving from one-box-per-region to a compact table."""
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel, _missing_estimate_note
    panel = Stage4ReportPanel()
    regions = [
        NS(region_pair=('NCR', 'Central Luzon'), estimate=-0.07, confidence=0.63,
           rejected_estimate=None),
        NS(region_pair=('Western Visayas', 'Davao'), estimate=None, confidence=0.0,
           rejected_estimate=99.0),
    ]
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 70, 'low': -0.10, 'high': -0.07,
        'agreement_n': 2, 'agreement_distinct': 2, 'agreement_regions': (2, 2),
        'agreement_echo_n': 0, 'agreement_diversity': 0.5,
        'agreement_models': {}, 'unscored_regions': 0,
    }
    panel._build_swarm_left(_mv(confidence_pct=70, regional_verdicts=regions), consensus)
    texts = [c.text() for c in panel._left.parentWidget().findChildren(QLabel)]
    assert any('NCR & Central Luzon' in t for t in texts)
    assert any('-0.07' in t for t in texts)
    assert any('63%' in t for t in texts)
    assert any('discarded' in t for t in texts)
    expected_note = _missing_estimate_note(None, 99.0)
    assert any(expected_note in t for t in texts)
```

Check the real `_left`/panel widget-tree attribute names by reading `_build()` (lines 116-241) before finalizing `panel._left.parentWidget()` above -- use whatever container actually holds `self._left`'s widgets so `findChildren(QLabel)` reaches them (may need `panel` itself rather than a sub-widget, depending on how `_left` is attached during `_build()`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest ph_economic_ai/tests/test_stage4_swarm_left_restyle.py -v`
Expected: FAIL if the widget-tree lookup is wrong (fix the lookup first against the *current*, pre-restyle code so you know the test itself is correct) -- then, once the lookup is right, it should PASS against current code (this test is a **pin**, not a new-behavior test: write it, confirm it passes today, before touching any styling, so Step 3's refactor is what's actually being verified).

- [ ] **Step 3: Apply the token substitutions from the table above**

Edit `ph_economic_ai/ui/stage4_report.py` lines 555-731 per the mapping table. Do not change any string literal that isn't a `setStyleSheet(...)` color/font argument.

- [ ] **Step 4: Run test to verify it still passes**

Run: `pytest ph_economic_ai/tests/test_stage4_swarm_left_restyle.py -v`
Expected: PASS (same as Step 2 -- if it now fails, the restyle dropped or renamed a caveat; fix the restyle, not the test)

- [ ] **Step 5: Run the file's full existing coverage**

Run: `pytest ph_economic_ai/tests/test_stage4_swarm.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_swarm_left_restyle.py
git commit -m "style(report): token-restyle the swarm consensus panel and regional verdicts"
```

---

## Task 8: Token-restyle `_build_left()` (classic mode)

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py:739-825`
- Test: `ph_economic_ai/tests/test_stage4_classic_honesty.py` (extend, don't replace)

**Interfaces:**
- Consumes: same `theme.*` helpers as Task 7.
- Produces: no change to `_build_left(self, consensus: dict, responses: list)`'s signature or label text -- same 1:1 substitution discipline as Task 7.

`_build_left` is the classic-mode counterpart to `_build_swarm_left`, built from the same `consensus` dict shape but for a single-engine debate (no regional verdicts, no swarm-specific caveats). Apply the identical mapping table from Task 7 to this method's matching lines:

| Line (approx) | Current | Replace with |
|---|---|---|
| 772 (`sub_lbl`) | `'font-size:9px;color:#6B7280;'` | build via `_theme.muted(text, size=9)` |
| 776 (`basis_lbl`) | `'font-size:8px;color:#9CA3AF;'` | `_theme.muted(text, size=8, color=_theme.FAINT)` |
| 780 (`caveat_lbl`) | `'font-size:9px;font-weight:600;color:#B45309;'` | `f'font-size:9px;font-weight:600;color:{_theme.WARNING};'` |
| 792 (range-row values) | `'font-size:11px;font-weight:600;color:#1C1E26;'` | `f'font-size:11px;font-weight:600;color:{_theme.INK};'` |

- [ ] **Step 1: Write the pin test**

Add to `ph_economic_ai/tests/test_stage4_classic_honesty.py` (read the file first to match its existing fixture/helper conventions):

```python
def test_classic_left_caveats_survive_the_restyle(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    panel = Stage4ReportPanel()
    consensus = {
        'weighted_avg': -0.08, 'confidence_pct': 100, 'low': -0.10, 'high': -0.07,
        'verdicts': [{'estimate': -0.08}, {'estimate': -0.08}, {'estimate': -0.09}],
    }
    panel._build_left(consensus, responses=[])
    texts = [c.text() for c in panel.findChildren(QLabel)]
    assert any('100% agent agreement' in t for t in texts)
```

- [ ] **Step 2: Run test to verify it passes against current code (pin, not new behavior)**

Run: `pytest ph_economic_ai/tests/test_stage4_classic_honesty.py::test_classic_left_caveats_survive_the_restyle -v`
Expected: PASS (adjust the `consensus`/`responses` fixture shape until it does, checking `_build_left`'s actual field reads at lines 739-825 if the shape above doesn't match)

- [ ] **Step 3: Apply the token substitutions from the table above**

Edit lines 739-825 per the mapping.

- [ ] **Step 4: Run test to verify it still passes**

Run: `pytest ph_economic_ai/tests/test_stage4_classic_honesty.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_classic_honesty.py
git commit -m "style(report): token-restyle the classic-mode consensus panel"
```

---

## Task 9: Token-restyle Final Outputs metrics grid and Validated Accuracy card

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py:843-980`
- Test: `ph_economic_ai/tests/test_stage4_swarm.py` (extend) or a new `ph_economic_ai/tests/test_stage4_outputs_restyle.py`

**Interfaces:**
- Consumes: `theme.card`, `theme.muted`, `theme.eyebrow`, `theme.hairline`.
- Produces: no change to `_build_right`'s signature, the four metric labels/values, or `_hs.validated_summary_lines(_report)`'s rendered text.

Two substitutions in `_build_right` (matplotlib chart internals are out of scope -- leave `ax.*`/`fig.*` calls untouched):

1. Metrics grid (lines 880-891): each `mf` frame's style
   `f'background:{_theme.SURFACE};border:1px solid {_theme.HAIRLINE};border-radius:9px;'`
   is already tokenized -- leave it. Only its value label needs a change:
   `'font-size:16px;font-weight:700;color:#1C1E26;'` → `f'font-size:16px;font-weight:700;color:{_theme.INK};'`

2. Validated Accuracy card (lines 968-974): each summary-line label's
   `'font-size:12px;color:#475467;'` → `_theme.muted(_line, size=12)` (build via the helper instead of a bare `QLabel` + `setStyleSheet`).

- [ ] **Step 1: Write the pin test**

```python
def test_final_outputs_and_accuracy_text_survive_the_restyle(app):
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    import numpy as np, pandas as pd
    panel = Stage4ReportPanel()
    df = pd.DataFrame({'gas_price': np.linspace(70, 80, 30)})
    consensus = {'weighted_avg': -0.08}
    panel._build_right(regressor=None, df=df, cv_rmse=1.5,
                       scenario={'current_price': 80.0}, consensus=consensus)
    texts = [c.text() for c in panel.findChildren(QLabel)]
    assert any('Next week (AI est.)' in t or '-0.02' in t for t in texts)
```

Read `_make_features`/`ml.forecast`'s actual behavior with `regressor=None` before trusting this fixture -- if it raises inside the `try/except` at lines 863-869, the test still validates the metrics-grid labels (which render regardless, per lines 874-891), just not the chart. Adjust the assertion to match what actually renders with a minimal/mocked regressor if `None` doesn't exercise the grid at all.

- [ ] **Step 2: Run test to verify it passes against current code (pin)**

Run: `pytest ph_economic_ai/tests/test_stage4_outputs_restyle.py -v`
Expected: PASS

- [ ] **Step 3: Apply the two substitutions above**

- [ ] **Step 4: Run test to verify it still passes**

Run: `pytest ph_economic_ai/tests/test_stage4_outputs_restyle.py -v`
Expected: PASS

- [ ] **Step 5: Run the file's full existing coverage**

Run: `pytest ph_economic_ai/tests/test_stage4_swarm.py ph_economic_ai/tests/test_stage4_classic_honesty.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py ph_economic_ai/tests/test_stage4_outputs_restyle.py
git commit -m "style(report): token-restyle final outputs grid and validated accuracy card"
```

---

## Task 10: Full-suite regression and visual confirmation

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: Run the full test suite**

Run: `pytest ph_economic_ai/ -x -q`
Expected: all tests pass (same total count as before this plan, plus the ~14 new tests added across Tasks 1-9)

- [ ] **Step 2: Grep for any remaining reference to the deleted BSPAlertBanner internals**

Run: `grep -rn "_icon_lbl\|_title_lbl\|_detail_lbl\b" ph_economic_ai/ --include="*.py"`
Expected: no matches outside `causal_chain_widget.py`'s own (now-removed) old code -- if `_detail_lbl` collides with `Stage4ReportPanel`'s own unrelated `_detail_lbl` (line 476 uses `self._detail_lbl.setText(...)` in `populate()` -- a *different* class, not `BSPAlertBanner`), confirm the grep result is that unrelated attribute, not a stray reference to the deleted one.

- [ ] **Step 3: Launch the real app and visually compare to the approved mockup**

Follow this project's own convention for UI changes: run the actual PyQt6 app (check for a `run`-skill or launch script; if none, `python -m ph_economic_ai.main` from the repo root per `main.py`'s role as the entry point) and navigate to the Report screen with a completed run. Compare side-by-side against `.superpowers/brainstorm/1304-1786231728/content/full-mockup-v3.html` (open it directly in a browser -- the companion server is already stopped, but the static HTML file still renders standalone). Confirm: no boxed/tinted BSP banner remains, all three sector cards show an agreement percentage, the grid reads as one consistent card system rather than mixed ad-hoc styling.

- [ ] **Step 4: Report findings**

If the visual comparison surfaces a mismatch (spacing, a missed hex, a card that doesn't match the mockup's proportions), fix it in a new small commit -- do not fold silent fixes into an earlier task's commit.

---

## Post-implementation note (scope actually delivered)

This plan's Goal line says "finish SP2d-1's editorial-theme migration on the Report screen" -- read after the fact, that overstates what Tasks 1-9 actually did. What was migrated is exactly what each task named: the sector-forecast cards (Task 5), the swarm and classic consensus panels and the Regional Verdicts table (Tasks 7-8), the final-outputs metrics grid and validated-accuracy card (Task 9, two specific `setStyleSheet` substitutions in `_build_right` -- not a sweep), and `BSPAlertBanner` (Task 6). `stage4_report.py` still carries 42 `setStyleSheet` calls outside those sections -- the top status bar, the export button, the recall banner, `_physics_anchor_label`, `_explanation_block`, classic-mode per-response cards (`name_lbl`/`est_lbl`/`stmt_lbl`), the dissenting-regions label, and the map basis/unvalidatable captions. None of that was ever in scope for this plan's tasks; it remains an acknowledged, un-migrated gap for a future slice, not something this branch quietly finished.
