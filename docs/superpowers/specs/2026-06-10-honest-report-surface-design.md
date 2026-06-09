# ph_economic_ai — Honest Surfacing on the Report Screen (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** the benchmark/validation work on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The app's **Report screen** (`ui/stage4_report.py`, "Final Outputs") presents confident gas-price forecasts — `next week = avg/4`, `next month = avg` (swarm "AI est."), `3/6-month = ml.forecast()` (ML) — with a chart band of `± cv_rmse` (uncalibrated) and the swarm's self-rated "confidence %". These are exactly the forecasts the benchmark **proved are informationally efficient** (no method beats naive). So the app's surface looks confident about the one thing we showed is unpredictable, while the validated, calibrated findings live only in the separate "Methodology & Accuracy" view.

**Goal:** make the Report screen **honest at the surface** — without removing the swarm demo and without touching the slow LLM path. Read the frozen `accuracy_report.json` (instant) and: label the forecasts as exploratory, show the **calibrated** interval, and add a compact **validated-accuracy** strip.

**Success criteria:** a viewer of the Report screen can see (a) that the forecasts are exploratory/unvalidated, (b) the real calibrated uncertainty, and (c) the validated verdicts (efficiency + the one positive, MoM nowcast) — all sourced from committed artifacts, with graceful fallback if the benchmark hasn't been run.

---

## 2. Scope

### In scope
- A small pure-Python module `ui/honest_surface.py` (no PyQt) with testable formatters that digest `accuracy_report.json`.
- Three edits to `ui/stage4_report.py::_build_right`: exploratory caption on Final Outputs; conformal band on the forecast chart; a "Validated accuracy" strip.
- Graceful degradation when the report is absent.

### Out of scope
- The swarm/LLM pipeline, model choices, cold-start (separate concern).
- Recomputing anything live — the strip only *reads* the frozen report.
- The "Methodology & Accuracy" view (already exists; we add a pointer to it, not duplicate it).
- Other screens (landing, interact, economy overview).

---

## 3. Architecture

```
ui/
├── honest_surface.py   # NEW — pure formatters (no PyQt):
│                       #   load_validated() -> dict | None
│                       #   conformal_halfwidth(report, level='0.9') -> float | None
│                       #   validated_summary_lines(report) -> list[str]
└── stage4_report.py    # _build_right: + exploratory caption, conformal band, accuracy strip
```

### 3.1 `honest_surface.py` (pure, unit-tested)
- `load_validated() -> dict | None`: wraps `benchmark.report.load_report()`; returns `None` on `FileNotFoundError` (benchmark not run).
- `conformal_halfwidth(report, level='0.9') -> float | None`: returns `float(report['conformal_widths'][level])` if present, else `None`.
- `validated_summary_lines(report) -> list[str]`: builds 3–4 plain-text lines from the report:
  - Forecast verdict: `f"1-month RON95 forecast: efficient — no method beats random walk (skill {headline_skill_vs_random_walk:+.2f})."`
  - Honest interval (if `conformal_widths['0.9']`): `f"Best estimate ≈ last price; 90% interval ±₱{qhat90:.2f}."`
  - Predictability map (from `audit` + `nowcast_mom`/`mom_longsample` if present): `"Fuel/FX/YoY-inflation: efficient · MoM inflation: predictable (ARIMA, DM-significant)."`
  - Pointer: `"Full detail: Methodology & Accuracy tab."`
  - Each line is omitted gracefully if its source key is missing; if `report is None`, returns a single line: `"Validated accuracy unavailable — run `python -m ph_economic_ai.benchmark.run`."`

### 3.2 `stage4_report.py::_build_right` edits
1. **Exploratory caption** above the Final Outputs metric grid: a muted QLabel — *"Exploratory forecasts — not validated. Backtest shows no method beats naive persistence for these (see Methodology & Accuracy)."* The existing metric cards are unchanged in value but now sit under this caption.
2. **Conformal band:** compute `half = conformal_halfwidth(report) or cv_rmse`. Use `± half` in `ax.fill_between`. Add a caption under the chart: *"90% calibrated interval (conformal)"* when the conformal width was used, else *"±cross-val RMSE (uncalibrated)"*.
3. **Validated-accuracy strip:** a `_card('Validated accuracy')` rendering `validated_summary_lines(report)` as wrapped QLabels, inserted into `self._right` directly after the Final Outputs card (before the causal-chain panel).

The report is loaded once at the top of `_build_right` via `honest_surface.load_validated()`.

---

## 4. Data Flow

```
benchmark/artifacts/accuracy_report.json  (frozen, committed)
        │  honest_surface.load_validated()
        ▼
{headline_skill_vs_random_walk, conformal_widths, audit, nowcast_mom, mom_longsample, ...}
        │  conformal_halfwidth() / validated_summary_lines()
        ▼
stage4_report._build_right ─► exploratory caption + conformal band + Validated-accuracy strip
```
No network, no recompute, no ollama — a single JSON read on Report render.

---

## 5. Error Handling
- `accuracy_report.json` missing → `load_validated()` returns `None`; the strip shows the "run benchmark" line; the chart band falls back to `± cv_rmse` with the "(uncalibrated)" caption. No crash.
- Report present but a key missing (e.g., older report without `mom_longsample`) → `validated_summary_lines` omits that line; `conformal_halfwidth` returns `None` → band falls back. Each access guarded with `.get(...)`.
- `_build_right` wraps the strip/band additions so a formatting error never blocks the rest of the Report (consistent with the existing `try/except` around the chart).

## 6. Testing
- `test_honest_surface.py` (pure, no PyQt):
  - `validated_summary_lines(full_report)` includes the efficiency verdict, the ±₱ interval, and the MoM-predictable line.
  - `validated_summary_lines(None)` returns the single "run benchmark" line.
  - `validated_summary_lines(report_missing_keys)` omits absent lines without raising.
  - `conformal_halfwidth({'conformal_widths': {'0.9': 10.42}})` == 10.42; missing → `None`.
- Existing `test_main_window.py` smoke test still constructs the window (the Report panel imports `honest_surface`).

## 7. Deliverables (definition of done)
1. `ui/honest_surface.py` with the three pure helpers + tests.
2. `stage4_report._build_right`: exploratory caption, conformal band (+ caption), validated-accuracy strip.
3. Graceful fallback when the benchmark report is absent.
4. Tests green; window smoke test unaffected.
5. No change to the swarm/LLM path or runtime.

## 8. Why it matters
Closes the gap between what the app **shows** and what was **proven**. A viewer (or thesis panel) sees, on the main result screen: the forecasts are exploratory, the *real* calibrated uncertainty, and the honest verdict (efficient; one validated positive). It makes the app itself answer the founding question — "how do people know the one thing it does is accurate?" — at the surface, not buried in a tab.

## 9. Sources / references
- Frozen artifact: `ph_economic_ai/benchmark/artifacts/accuracy_report.json` (keys: `headline_skill_vs_random_walk`, `conformal_widths`, `calibration`, `audit`, `nowcast_mom`, `mom_longsample`).
- Existing read-only renderer pattern: `ui/accuracy_view.py`.
