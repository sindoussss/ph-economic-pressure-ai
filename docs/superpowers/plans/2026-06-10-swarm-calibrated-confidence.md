# Calibrated Confidence on the Swarm Consensus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the swarm's agent-agreement number masquerading as "confidence" — relabel it "agent agreement" everywhere on the Report screen and add the validated calibrated 90% conformal interval to the Swarm Consensus box.

**Architecture:** Add one pure helper (`calibrated_interval_line`) to the existing tested `ui/honest_surface.py`, then edit `ui/stage4_report.py::_build_swarm_left` to relabel three "confidence" strings and render the calibrated interval line (read from the frozen `accuracy_report.json`). Graceful fallback if the report is absent.

**Tech Stack:** Python 3.10, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-10-swarm-calibrated-confidence-design.md`.

**Prereqs (on branch `feature/accuracy-evaluation-phase1`):**
- `ui/honest_surface.py` exists with `load_validated()`, `conformal_halfwidth(report, level='0.9')`, `validated_summary_lines(report)` (all tested in `tests/test_honest_surface.py`).
- `ui/stage4_report.py::_build_swarm_left(self, master_verdict, consensus)` — current exact lines to edit:
  - `card, cl = self._card('Swarm Consensus')`
  - `conf = consensus.get('confidence_pct', 0)`
  - `sub_lbl = QLabel(f'Master judge estimate · {conf}% confidence')`
  - `for label, value in [('Low', low), ('High', high), ('Confidence', f'{conf}%')]:`
  - `cf_layout.addLayout(range_row)` then `cl.addWidget(consensus_frame)`
  - regional loop: `conf_lbl = QLabel(f'Confidence: {rv.confidence:.0%}')`
  - `QLabel` imported; `self._muted` available.

**Conventions:**
- Tests in `ph_economic_ai/tests/`, path shim at top. Single test: `python -m pytest ph_economic_ai/tests/test_FILE.py -v`.
- **Git hygiene:** staging clean; commit ONLY each task's files via explicit paths. NEVER `git add -A`/`.`. `git status --short` before committing.
- Stay on branch `feature/accuracy-evaluation-phase1`.

---

## File Structure
**Modify:** `ph_economic_ai/ui/honest_surface.py` (+`calibrated_interval_line`), `ph_economic_ai/ui/stage4_report.py` (`_build_swarm_left`); test `ph_economic_ai/tests/test_honest_surface.py` (append).

---

## Task 1: `calibrated_interval_line` helper

**Files:**
- Modify: `ph_economic_ai/ui/honest_surface.py` (append)
- Modify: `ph_economic_ai/tests/test_honest_surface.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `ph_economic_ai/tests/test_honest_surface.py`:

```python
from ph_economic_ai.ui.honest_surface import calibrated_interval_line


def test_calibrated_interval_line_full():
    line = calibrated_interval_line({'conformal_widths': {'0.9': 10.42}})
    assert line is not None
    assert '10.42' in line and 'calibrated' in line and '90%' in line


def test_calibrated_interval_line_custom_level():
    line = calibrated_interval_line({'conformal_widths': {'0.8': 5.0}}, level='0.8')
    assert '80%' in line and '5.00' in line


def test_calibrated_interval_line_missing_returns_none():
    assert calibrated_interval_line(None) is None
    assert calibrated_interval_line({'conformal_widths': {}}) is None
    assert calibrated_interval_line({}) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py -k calibrated_interval_line -v`
Expected: FAIL — `ImportError: cannot import name 'calibrated_interval_line'`

- [ ] **Step 3: Implement — append to honest_surface.py**

Append to `ph_economic_ai/ui/honest_surface.py`:

```python
def calibrated_interval_line(report: Optional[dict], level: str = '0.9') -> Optional[str]:
    """One-line calibrated interval for the given level, or None if unavailable.

    e.g. '90% calibrated interval: ±₱10.42/L (conformal, validated)'."""
    qhat = conformal_halfwidth(report, level)
    if qhat is None:
        return None
    pct = int(round(float(level) * 100))
    return f'{pct}% calibrated interval: ±₱{qhat:.2f}/L (conformal, validated)'
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py -v`
Expected: PASS (existing 5 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/honest_surface.py ph_economic_ai/tests/test_honest_surface.py
git commit -m "feat(ui): calibrated_interval_line helper for the swarm consensus box"
```

---

## Task 2: Relabel agreement + add calibrated interval in the swarm box

**Files:**
- Modify: `ph_economic_ai/ui/stage4_report.py` (`_build_swarm_left`)
- Test: import smoke + window smoke

- [ ] **Step 1: Load the report at the top of `_build_swarm_left`**

Immediately after `card, cl = self._card('Swarm Consensus')`, insert:
```python
        from ph_economic_ai.ui import honest_surface as _hs
        _report = _hs.load_validated()
```

- [ ] **Step 2: Relabel the subtitle**

Replace:
```python
        sub_lbl = QLabel(f'Master judge estimate · {conf}% confidence')
```
with:
```python
        sub_lbl = QLabel(f'Master judge estimate · {conf}% agent agreement')
```

- [ ] **Step 3: Relabel the range-row column**

Replace:
```python
        for label, value in [('Low', low), ('High', high), ('Confidence', f'{conf}%')]:
```
with:
```python
        for label, value in [('Low', low), ('High', high), ('Agent agreement', f'{conf}%')]:
```

- [ ] **Step 4: Add the calibrated interval line**

Find:
```python
        cf_layout.addWidget(val_lbl)
        cf_layout.addWidget(sub_lbl)
        cf_layout.addLayout(range_row)
        cl.addWidget(consensus_frame)
```
Insert the calibrated line between `cf_layout.addLayout(range_row)` and `cl.addWidget(consensus_frame)` so it reads:
```python
        cf_layout.addWidget(val_lbl)
        cf_layout.addWidget(sub_lbl)
        cf_layout.addLayout(range_row)
        _cal = _hs.calibrated_interval_line(_report)
        if _cal:
            _cal_lbl = QLabel(_cal)
            _cal_lbl.setWordWrap(True)
            _cal_lbl.setStyleSheet('font-size:9px;font-weight:600;color:#1C7C54;')
            cf_layout.addWidget(_cal_lbl)
        cl.addWidget(consensus_frame)
```

- [ ] **Step 5: Relabel the regional verdict line**

Replace:
```python
            conf_lbl = QLabel(f'Confidence: {rv.confidence:.0%}')
```
with:
```python
            conf_lbl = QLabel(f'Agent agreement: {rv.confidence:.0%}')
```

- [ ] **Step 6: Import smoke**

Run: `python -c "import ph_economic_ai.ui.stage4_report; from ph_economic_ai.ui import honest_surface as h; print('import OK', h.calibrated_interval_line({'conformal_widths':{'0.9':10.42}}))"`
Expected: prints `import OK 90% calibrated interval: ±₱10.42/L (conformal, validated)`

- [ ] **Step 7: Window + full suite (no new breakage)**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py ph_economic_ai/tests/test_honest_surface.py -q`
Expected: honest-surface tests pass; main_window only the pre-existing `test_on_run_requested_accepts_4_args` failure, nothing new.

- [ ] **Step 8: Commit**

```bash
git add ph_economic_ai/ui/stage4_report.py
git commit -m "feat(ui): relabel swarm 'confidence' as agent agreement + show calibrated interval"
```

---

## Final verification

- [ ] **No stray 'confidence' label remains on the swarm screen**

Run: `python -c "import re,io; s=open('ph_economic_ai/ui/stage4_report.py',encoding='utf-8').read(); import sys; lines=[ (i+1,l) for i,l in enumerate(s.splitlines()) if 'confidence' in l.lower() and ('QLabel' in l or \"'Confidence'\" in l or 'Confidence:' in l)]; print(lines)"`
Expected: prints `[]` for the swarm box labels (the `consensus.get('confidence_pct')` data access may still appear — that's the data key, not a UI label, and is fine). Manually confirm any remaining hit is a data-key access, not a user-facing "Confidence" label in `_build_swarm_left`.

- [ ] **Helper test green**

Run: `python -m pytest ph_economic_ai/tests/test_honest_surface.py -q`
Expected: all pass.

- [ ] **Manual visual check (optional, GUI session)**

Run a swarm simulation, open Report: the Swarm Consensus box shows "{X}% agent agreement", an "Agent agreement" column, and a green "90% calibrated interval: ±₱{qhat}/L (conformal, validated)" line; regional verdicts show "Agent agreement: {X}%".

---

## Self-Review (completed by plan author)

**Spec coverage:** §4.1 `calibrated_interval_line` → Task 1. §4.2 relabel subtitle (Step 2), range row (Step 3), regional (Step 5); add calibrated line (Step 4); report load (Step 1) → Task 2. §6 error handling (None → line omitted; wrapped logic; relabels independent of report) → Tasks 1, 2. §7 testing (helper full/custom/missing + import + window smoke) → Tasks 1, 2.

**Placeholder scan:** none — all code steps contain complete code. The final-verification grep is a guard, not a code change.

**Type consistency:** `calibrated_interval_line(report, level='0.9') -> str|None` defined in Task 1, called as `_hs.calibrated_interval_line(_report)` in Task 2 Step 4 (default level). `conformal_halfwidth` reused unchanged. `load_validated()` used per its existing signature. `_report`/`_hs` defined at the top of `_build_swarm_left` (Step 1) before use (Step 4). The three relabels are pure string changes; data key `confidence_pct` / `rv.confidence` is unchanged (only the displayed label text changes).
```
