# Honest Surfacing on the Report Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the app's Report screen honest at the surface — label the gas-price forecasts as exploratory, draw the calibrated 90% conformal interval, and add a "Validated accuracy" strip — all by reading the frozen `accuracy_report.json` (instant, no ollama).

**Architecture:** A new pure module `ui/honest_surface.py` (no PyQt) holds the testable formatters that digest the frozen report; `ui/stage4_report.py::_build_right` calls them to add a caption, swap the chart band, and render a small accuracy card. Graceful fallback when the report is absent.

**Tech Stack:** Python 3.10, PyQt6, matplotlib, pytest. Reads `benchmark.report.load_report()`.

**Spec:** `docs/superpowers/specs/2026-06-10-honest-report-surface-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `benchmark/report.py::load_report()` → frozen report dict with keys `headline_skill_vs_random_walk`, `conformal_widths` (e.g. `{'0.9': 10.42, ...}`), `audit` (list of `{target, verdict, ...}`), `nowcast_mom`, `mom_longsample` (`{n_long, mom:{verdict,...}, driver_ablation:{...}}`). Raises `FileNotFoundError` if absent.
- `ui/stage4_report.py::Stage4ReportPanel._build_right(self, regressor, df, cv_rmse, scenario, consensus)` — builds the "Final Outputs" card. Current relevant lines:
  - `card, cl = self._card('Final Outputs')` (creates card + layout)
  - metric grid built into `cl`
  - chart: `ax.fill_between(xs, forecast_prices - cv_rmse, forecast_prices + cv_rmse, alpha=0.15, color='#1C1E26')` then `cl.addWidget(canvas)`
  - tail: `self._right.addWidget(card)` then `self._right.addWidget(self._chain_widget)` then `self._right.addStretch()`
  - helpers available: `self._card(title) -> (frame, layout)`, `self._muted(text) -> QLabel`. `QLabel` is imported.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Create:** `ph_economic_ai/ui/honest_surface.py`; test `ph_economic_ai/tests/test_honest_surface.py`.
**Modify:** `ph_economic_ai/ui/stage4_report.py` (`_build_right`).

---

## Task 1: Pure honest-surface helpers

**Files:**
- Create: `ph_economic_ai/ui/honest_surface.py`
- Test: `ph_economic_ai/tests/test_honest_surface.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_honest_surface.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest

from ph_economic_ai.ui.honest_surface import (
    conformal_halfwidth, validated_summary_lines,
)

_FULL = {
    'headline_skill_vs_random_walk': -0.18,
    'conformal_widths': {'0.5': 2.4, '0.9': 10.42, '0.95': 16.0},
    'audit': [
        {'target': 'fuel', 'verdict': 'efficient'},
        {'target': 'fx', 'verdict': 'efficient'},
        {'target': 'inflation', 'verdict': 'efficient'},
    ],
    'mom_longsample': {'n_long': 143, 'mom': {'verdict': 'beats_best_naive',
                                              'best_method': 'arima'}},
}


def test_conformal_halfwidth_reads_level():
    assert conformal_halfwidth(_FULL, '0.9') == pytest.approx(10.42)
    assert conformal_halfwidth(_FULL, '0.5') == pytest.approx(2.4)


def test_conformal_halfwidth_missing_returns_none():
    assert conformal_halfwidth({'conformal_widths': {}}, '0.9') is None
    assert conformal_halfwidth(None) is None
    assert conformal_halfwidth({}, '0.9') is None


def test_summary_lines_full_report():
    lines = validated_summary_lines(_FULL)
    text = ' || '.join(lines)
    assert 'efficient' in text and 'random walk' in text
    assert '10.42' in text                      # calibrated interval surfaced
    assert 'predictable' in text                # MoM positive surfaced
    assert any('Methodology' in l for l in lines)


def test_summary_lines_none_report():
    lines = validated_summary_lines(None)
    assert len(lines) == 1
    assert 'benchmark.run' in lines[0]


def test_summary_lines_missing_keys_no_crash():
    lines = validated_summary_lines({'something': 1})   # present but no known keys
    assert isinstance(lines, list)
    assert any('Methodology' in l for l in lines)       # pointer still present
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py -v`
Expected: FAIL — `ModuleNotFoundError: ph_economic_ai.ui.honest_surface`

- [ ] **Step 3: Implement**

Create `ph_economic_ai/ui/honest_surface.py`:

```python
"""Honest, read-only digest of the frozen benchmark report for the Report screen.

Pure functions (no PyQt) so they are unit-testable. They surface the *validated*
results — calibrated interval, efficiency verdict, the one predictable target —
so the app's main screen tells the truth, not just the swarm's confident guesses.
"""
from typing import Optional


def load_validated() -> Optional[dict]:
    """Load the frozen accuracy report; return None if it has not been generated."""
    try:
        from ph_economic_ai.benchmark.report import load_report
        return load_report()
    except Exception:
        return None


def conformal_halfwidth(report: Optional[dict], level: str = '0.9') -> Optional[float]:
    """Calibrated conformal half-width for the given level, or None if unavailable."""
    if not report:
        return None
    val = (report.get('conformal_widths') or {}).get(level)
    return float(val) if val is not None else None


def validated_summary_lines(report: Optional[dict]) -> list:
    """Plain-text lines digesting the validated findings for the Report strip.

    Each line is omitted gracefully if its source key is missing. A None/empty
    report yields a single 'run the benchmark' line."""
    if not report:
        return ['Validated accuracy unavailable — run `python -m ph_economic_ai.benchmark.run`.']

    lines: list = []
    skill = report.get('headline_skill_vs_random_walk')
    if skill is not None:
        lines.append(f'1-month RON95 forecast: efficient — no method beats random walk '
                     f'(skill {skill:+.2f}).')

    qhat = conformal_halfwidth(report, '0.9')
    if qhat is not None:
        lines.append(f'Best estimate ≈ last price; 90% interval ±₱{qhat:.2f}.')

    audit = report.get('audit') or []
    eff = [a.get('target') for a in audit if a.get('verdict') == 'efficient' and a.get('target')]
    mom_long = report.get('mom_longsample') or {}
    mom_inner = mom_long.get('mom') if isinstance(mom_long, dict) else None
    mom_verdict = (mom_inner or {}).get('verdict') or (report.get('nowcast_mom') or {}).get('verdict')
    if eff or mom_verdict:
        eff_str = '/'.join(eff) if eff else 'fuel/FX/inflation'
        mom_str = ('MoM inflation: predictable' if mom_verdict == 'beats_best_naive'
                   else 'MoM inflation: not better than naive')
        lines.append(f'{eff_str}: efficient · {mom_str}.')

    lines.append('Full detail: Methodology & Accuracy tab.')
    return lines
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/honest_surface.py ph_economic_ai/tests/test_honest_surface.py
git commit -m "feat(ui): pure honest-surface helpers digesting the frozen report"
```

---

## Task 2: Wire honest surfacing into the Report screen

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py` (`_build_right`)
- Test: import smoke + window smoke

- [ ] **Step 1: Load the report + add the exploratory caption**

In `ph_economic_ai/ui/stage4_report.py::_build_right`, immediately after the line:
```python
        card, cl = self._card('Final Outputs')
```
insert:
```python
        from ph_economic_ai.ui import honest_surface as _hs
        _report = _hs.load_validated()
        _exp = self._muted('Exploratory forecasts — not validated. Backtest shows no '
                           'method beats naive persistence for these (see Methodology & Accuracy).')
        _exp.setWordWrap(True)
        cl.addWidget(_exp)
```

- [ ] **Step 2: Swap the chart band to the calibrated conformal interval**

In the same method, find the chart band line:
```python
                ax.fill_between(xs, forecast_prices - cv_rmse, forecast_prices + cv_rmse,
                                alpha=0.15, color='#1C1E26')
```
Replace it with:
```python
                _band = _hs.conformal_halfwidth(_report) or cv_rmse
                ax.fill_between(xs, forecast_prices - _band, forecast_prices + _band,
                                alpha=0.15, color='#1C1E26')
```
Then find the line `cl.addWidget(canvas)` (inside the chart `try`) and immediately after it insert:
```python
                _bcap = self._muted('90% calibrated interval (conformal)'
                                    if _hs.conformal_halfwidth(_report) is not None
                                    else '±cross-val RMSE (uncalibrated)')
                _bcap.setWordWrap(True)
                cl.addWidget(_bcap)
```

- [ ] **Step 3: Add the "Validated accuracy" strip**

Find the tail of `_build_right`:
```python
        self._right.addWidget(card)
        # Causal chain panel below the ML outputs
        self._right.addWidget(self._chain_widget)
        self._right.addStretch()
```
Insert a strip between the Final Outputs card and the chain widget so it reads:
```python
        self._right.addWidget(card)
        try:
            acc_card, acc_l = self._card('Validated accuracy')
            for _line in _hs.validated_summary_lines(_report):
                _ql = QLabel(_line)
                _ql.setWordWrap(True)
                _ql.setStyleSheet('font-size:12px;color:#475467;')
                acc_l.addWidget(_ql)
            self._right.addWidget(acc_card)
        except Exception:
            pass
        # Causal chain panel below the ML outputs
        self._right.addWidget(self._chain_widget)
        self._right.addStretch()
```
(`QLabel`, `self._card`, `self._muted` are already available in this module.)

- [ ] **Step 4: Import smoke + verify the helpers are reachable**

Run: `python -c "import ph_economic_ai.ui.stage4_report; from ph_economic_ai.ui import honest_surface; print('import OK', honest_surface.conformal_halfwidth({'conformal_widths':{'0.9':10.42}}))"`
Expected: prints `import OK 10.42` (the module imports cleanly and the helper resolves).

- [ ] **Step 5: Window smoke test (no new breakage)**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: only the pre-existing `test_on_run_requested_accepts_4_args` may fail; nothing new. (Construction doesn't call `_build_right` — that runs on `populate` after a simulation — so this confirms no import/wiring regression.)

- [ ] **Step 6: Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass except the one documented pre-existing failure. Report counts.

- [ ] **Step 7: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py
git commit -m "feat(ui): honest surfacing on Report (exploratory label, conformal band, validated-accuracy strip)"
```

---

## Final verification

- [ ] **Helpers + suite green**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py ph_economic_ai/tests/test_main_window.py -q`
Expected: honest-surface tests pass; main_window only the pre-existing failure.

- [ ] **Manual visual check (optional, needs a GUI session)**

Launch the app, run a simulation, open Report: the Final Outputs panel shows the "Exploratory — not validated" caption, the chart band is captioned "90% calibrated interval (conformal)" (since `accuracy_report.json` is committed), and a "Validated accuracy" card appears with the efficiency verdict, the ±₱ interval, and the MoM-predictable line.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1 pure helpers (`load_validated`, `conformal_halfwidth`, `validated_summary_lines`) → Task 1. §3.2 edit 1 exploratory caption → Task 2 Step 1; edit 2 conformal band + caption → Task 2 Step 2; edit 3 validated-accuracy strip → Task 2 Step 3. §5 error handling (None report → "run benchmark" line; missing keys omitted; strip wrapped in try/except; band falls back to cv_rmse) → Tasks 1, 2. §6 testing (full/None/missing-keys/halfwidth + window smoke) → Tasks 1, 2.

**Placeholder scan:** none — all code steps contain complete code; no TBD/vague items.

**Type consistency:** `load_validated() -> dict|None`, `conformal_halfwidth(report, level='0.9') -> float|None`, `validated_summary_lines(report) -> list[str]` defined in Task 1, called identically in Task 2 (`_hs.load_validated()`, `_hs.conformal_halfwidth(_report)`, `_hs.validated_summary_lines(_report)`). `self._card(title) -> (frame, layout)` and `self._muted(text) -> QLabel` used per their existing signatures. `_report`/`_hs` are defined at the top of `_build_right` (Step 1) before their uses in Steps 2–3. The chart band fallback (`_hs.conformal_halfwidth(_report) or cv_rmse`) preserves prior behavior when the report is absent.
```
