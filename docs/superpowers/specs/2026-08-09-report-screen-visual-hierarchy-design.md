# ph_economic_ai — Report Screen Visual Hierarchy Restyle (SP2d continuation)

**Date:** 2026-08-09
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** Continuation of `2026-06-11-theme-and-report-restyle-design.md` (SP2d-1), which introduced `ui/theme.py`'s editorial tokens and did a first, partial migration of `stage4_report.py`. That pass covered the consensus block's number/eyebrow/hairline; most of the screen (sector cards, the final-outputs table, validated-accuracy panel) and all of `causal_chain_widget.py`'s CPI banner were never migrated — confirmed directly: `stage4_report.py` has 58 `setStyleSheet` calls against 14 `theme.*` references, `causal_chain_widget.py` has 2 against 397 lines. This spec finishes that migration on the Report screen and fixes two structural problems the tokens alone didn't solve.

---

## 1. Problem & Goal

The user's own words, screenshot in hand: "looks good, it works" on layout direction, but the BSP CPI callout specifically read as "AI slop" through two iterations before the actual cause was found — not the accent color, the pattern itself: a bordered, colored "alert card" announcing a headline number is a generic AI-dashboard trope regardless of which color fills it. Separately, the screen is dense and inconsistently boxed: sector numbers, agreement stats, and calibrated intervals each live in differently-shaped containers with no shared visual grammar, which is what SP2d-1 set out to fix but didn't finish.

**Goal:** finish the SP2d-1 migration on the Report screen (Direction B from brainstorming: "aligned terminal" — keep the information-dense, all-at-once view, but put every element on one disciplined grid, type scale, and card component) and eliminate the boxed-alert pattern from the CPI callout.

**Explicitly not a goal:** making the forecast read as more certain than it is. Every confidence number, interval, and caveat is displayed exactly as computed today — this is a layout and typography pass, not a change to what the app claims. (Raised and settled directly with the owner before any mockup work started.)

---

## 2. Scope

### In scope
- `ui/theme.py` — two new helper factories: `stat_card()` (the sector-forecast card pattern: eyebrow, serif number, confidence bar, agreement/interval caption, exploratory tag) and `page_header()` (title + right-aligned stat, replacing the boxed alert pattern). A confidence-bar QWidget/factory for the small horizontal indicator under each sector number.
- `ui/stage4_report.py` — migrate the sector-forecast section, the Final Outputs table, and the Validated Accuracy panel onto `theme.stat_card`/a shared card grid (finishing what SP2d-1 started on this file). Fold the Regional Verdicts cards into a compact table inside the Validated Accuracy card, as approved in the mockup.
- `ui/causal_chain_widget.py` — `BSPAlertBanner` loses its colored box (background tint + colored border) and becomes a plain header stat using `theme.page_header()`. The "critically exceeded" wording moves from a callout headline to a small caption under the eyebrow label; the number itself gets no more visual weight than any other stat on the page.

### Out of scope
- Any other screen (`landing.py`, `stage3_swarm_canvas.py`, `stage3_canvas.py`, `accuracy_view.py`, etc. — later SP2d slices, per the original program).
- The "Interact" tab (`stage5_interact.py`) — not shown in the approved mockups, different content type (Q&A, not a stat display).
- Any change to what's *computed*: `honesty.py`'s caveat/marker logic, `DebateEngine.consensus()`, conformal interval math, the CPI projection formula — none of it changes.
- Re-litigating SP2d-1's approved tokens (palette, `SERIF`/`MONO` fonts) — reused as-is throughout.

### One small, deliberate scope addition (not pure styling)
The approved mockup shows all three sector cards (gas/food/electricity) with an agreement percentage. Checked against the real code: gas already gets one, via the separate Swarm Consensus panel's `master_verdict.confidence_pct`. Food and electricity's confidence is *also* already computed -- `main_window.py::_on_food_complete`/`_on_elec_complete` already call `self._food_engine.consensus()`/`self._elec_engine.consensus()` and store the result in `self._food_agreement`/`self._elec_agreement` -- but neither value is currently passed into `Stage4ReportPanel.set_sector_forecasts()`, which only ever receives the three point estimates. This plan wires the three already-computed agreement values through so all three top-row cards show one, matching the mockup. This is a genuine (if small) display addition, not styling -- flagged explicitly per the project's own convention of never letting a scope decision hide inside an unrelated commit.

### Non-negotiable
- No regressions: the Report still builds, all `stage4`/`causal_chain_widget`/`main_window` tests stay green (updated where they assert on structure that's genuinely changing, e.g. `BSPAlertBanner`'s box styling — never by deleting a coverage-losing assertion without replacing it).
- Every honesty caveat, agreement percentage, interval, and "exploratory/not validated" tag that renders today still renders after the restyle, with the same text — verified explicitly, not assumed, per the file's own testing note below.

---

## 3. Components

### 3.1 `ui/theme.py` additions

```python
def stat_card(eyebrow_text, value, unit='', color=INK, meta='', tag_kind='exploratory',
              confidence_frac=None) -> tuple[QFrame, QVBoxLayout]:
    """One sector-forecast card: eyebrow, serif value (+ smaller unit), an
    optional confidence-bar row, a muted meta caption, and the exploratory/
    validated tag. Returns (frame, layout) like `card()` so callers can still
    append/adjust content."""

def page_header(eyebrow_text, title, right_eyebrow=None, right_value=None,
                 right_caption=None) -> QFrame:
    """Plain page-header row: title on the left, an optional stat on the
    right (eyebrow + serif number + muted caption). No border, no fill --
    replaces the boxed-alert pattern."""

def confidence_bar(low_frac: float, width_frac: float) -> QFrame:
    """A 5px horizontal track with a dark-ink filled segment from
    `low_frac` to `low_frac + width_frac` (both 0-1). Purely decorative
    indicator of where within its own range a value sits -- not a new
    statistic, just a visual echo of numbers already shown in the caption
    beside it."""
```

All three follow SP2d-1's existing convention: pure-ish factories returning styled `QWidget`s, no external state, importable under offscreen Qt for tests.

### 3.2 `stage4_report.py` migration

- Sector-forecast section (`set_sector_forecasts`, currently a compact row-list built from `sector_forecast_rows()` in `ui/sector_forecast.py`, which carries no confidence field): replace the row-list with `theme.stat_card(...)` per sector, now also given each sector's agreement percentage via a new `gas_agreement`/`food_agreement`/`elec_agreement` parameter (see 3.3 below for where the values come from).
- Final Outputs / Validated Accuracy: migrate onto `theme.card()` (already used elsewhere) with `theme.eyebrow`/`theme.muted`/`theme.hairline` throughout, replacing the remaining 44 stray `setStyleSheet` calls this file carries.
- Regional Verdicts: currently one bordered sub-card per region (`rv_card`/`rvfl` loop around line 680-730, each region its own `QFrame` with a name/estimate header row plus an agreement caption). Replaced with a compact table (one row per region: name, signed estimate, agreement%) inside the *same* `rv_card` it already lives in today. **Correction after checking the plan against `_build_right`'s actual parameters:** the mockup showed this table inside the Validated Accuracy card (the right column), but that card is built by `_build_right`, which never receives `master_verdict`/`regional_verdicts` -- only `_build_swarm_left` does. Moving the table across that boundary would mean threading a new parameter through, which is a larger change than "styling only." The table stays in its current card, in its current column, next to Swarm Consensus -- compact instead of moved.

### 3.3 `main_window.py` (the scope addition)

- `_push_sector_forecasts()` currently calls `self._stage4.set_sector_forecasts(self._gas_estimate, self._food_estimate, self._elec_estimate)`. Extended to also pass `self._gas_agreement, self._food_agreement, self._elec_agreement` -- all three already exist and are already populated by the existing completion handlers; nothing new is computed.

### 3.4 `causal_chain_widget.py`

- `BSPAlertBanner` keeps its class name and public interface (whatever `main_window.py`/other callers rely on today) but its internals switch from a bordered/tinted `QFrame` to `theme.page_header()`. `Projected CPI: {value:.2f}%` becomes the header's right-hand stat; "BSP TARGET" + exceeded/within status becomes the eyebrow label; the baseline/sector-impact breakdown becomes the caption underneath, same as today's text content.

---

## 4. Testing

- Extend `test_theme.py` (per SP2d-1's existing pattern): `stat_card(...)` returns a frame containing the given eyebrow/value/meta text; `page_header(...)` returns a frame with title and, when given, the right-hand stat text; `confidence_bar(...)` returns a `QFrame`.
- Before touching `stage4_report.py`/`causal_chain_widget.py`: grep every existing test asserting on their internals (`test_stage4_swarm.py`, `test_stage4_classic_honesty.py`, `test_main_window.py`, and the other 6 files found referencing `BSPAlertBanner`/`_build_swarm_left`/`_build_left`/`regional_verdicts`/`_consensus` this session) and catalog which assertions are text-content (must still pass unchanged) vs. structure/styling (may need updating).
- One explicit regression test per migrated section confirming the actual caveat/interval text set in this session's earlier work (agreement percentages, `narrow room` markers, `90% calibrated interval` line, the "exploratory — not validated" tag) still renders post-restyle — mirrors SP2d-1's own "still contains the SP2a note" guard.
- Full suite green. Launch the real app afterward (per this project's UI-change convention) and visually compare against the approved mockup before calling this done.

---

## 5. Deliverables (definition of done)

1. `ui/theme.py` gains `stat_card`, `page_header`, `confidence_bar`; tested.
2. `stage4_report.py` fully migrated (sector cards, outputs table, validated accuracy, regional verdicts) — SP2d-1's unfinished work on this file completed.
3. `causal_chain_widget.py`'s `BSPAlertBanner` restyled to the plain header pattern; no boxed/tinted alert remains.
4. Report screen matches the approved mockup (`full-mockup-v3.html` from this session's brainstorming companion, `.superpowers/brainstorm/1304-1786231728/content/`); every honesty caveat and confidence number present and unchanged in substance; full suite green.

## 6. Why it matters

Closes out SP2d-1's stated intent ("prove the tokens on the Report, then roll out") on the one screen it started but didn't finish, and removes a specific, user-identified credibility problem: an alert-box pattern that reads as generated rather than authored, on the app's most-seen screen, competing with the actual goal of looking like a rigorous analytical tool. Confidence communication itself doesn't change — only whether the page looks like someone designed it on purpose.
