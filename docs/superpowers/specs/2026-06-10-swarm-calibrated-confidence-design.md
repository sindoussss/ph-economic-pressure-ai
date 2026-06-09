# ph_economic_ai — Calibrated Confidence on the Swarm Consensus (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** the honest-surface work (`2026-06-10-honest-report-surface-design.md`) on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The Report screen's **Swarm Consensus** box and **Regional Verdicts** display an "X% confidence" (`stage4_report.py:184,188,233`). These percentages are **agent-agreement** metrics computed by the swarm — they are *not* statistical confidence and *not* calibrated. Presenting them as "confidence" overstates certainty for the gas-price estimate (which the benchmark proved is efficient).

**Goal:** stop the agreement number masquerading as confidence, and surface the *validated* calibrated uncertainty next to the estimate:
1. **Relabel** every "confidence" on this screen to **"agent agreement"** (honest, keeps the signal).
2. **Add** the calibrated 90% conformal interval (±₱qhat90, from the frozen report) to the Consensus box as the real uncertainty.

No fabricated numbers, nothing useful discarded.

---

## 2. Scope

### In scope
- One pure helper added to `ui/honest_surface.py`: `calibrated_interval_line(report, level='0.9')`.
- Edits to `ui/stage4_report.py::_build_swarm_left`: relabel agreement (3 spots) + add the calibrated interval line.

### Out of scope
- The swarm's agreement computation itself (unchanged; just relabeled).
- The Final Outputs panel / Validated-accuracy strip (already done in the prior honest-surface change).
- Per-region calibrated intervals — the conformal width is a single global value; per-region verdicts are only relabeled.
- The non-swarm (single-debate) consensus box (`_build_left`) — out of scope unless trivially shared; this spec targets the swarm view shown in the user's run.

---

## 3. Definitions
- **Agent agreement (`confidence_pct` / `rv.confidence`):** the swarm's internal agreement metric (clustering of agent estimates). Renamed in the UI to "agent agreement"; value unchanged.
- **Calibrated interval:** `±conformal_widths['0.9']` from the frozen `accuracy_report.json` — the 1-month forecast half-width whose ~90% empirical coverage was verified in the benchmark calibration table.

---

## 4. Architecture

```
ui/
├── honest_surface.py   # + calibrated_interval_line(report, level='0.9') -> str | None
└── stage4_report.py    # _build_swarm_left: relabel agreement x3 + add calibrated line
```

### 4.1 `honest_surface.calibrated_interval_line(report, level='0.9') -> str | None`
- Returns `None` if `conformal_halfwidth(report, level)` is `None`.
- Else returns: `f'{int(float(level)*100)}% calibrated interval: ±₱{qhat:.2f}/L (conformal, validated)'`.
- Pure; reuses the existing `conformal_halfwidth`.

### 4.2 `stage4_report.py::_build_swarm_left` edits
- At the top, load the report once: `from ph_economic_ai.ui import honest_surface as _hs; _report = _hs.load_validated()`.
- **Relabel 1 (subtitle, line 184):** `f'Master judge estimate · {conf}% agent agreement'`.
- **Relabel 2 (range row, line 188):** change the `('Confidence', f'{conf}%')` tuple's label to `'Agent agreement'` (value unchanged).
- **Add calibrated line:** after the `range_row` is added to `cf_layout`, if `calibrated_interval_line(_report)` is not `None`, add a `QLabel` with it (small, accent style) into `cf_layout`.
- **Relabel 3 (regional, line 233):** `f'Agent agreement: {rv.confidence:.0%}'`.

The calibrated line is wrapped so a formatting error can't break the consensus box.

---

## 5. Data Flow
```
accuracy_report.json ─► honest_surface.load_validated() / calibrated_interval_line()
        │
        ▼
_build_swarm_left ─► relabeled agreement + "90% calibrated interval: ±₱{qhat}/L"
```
One JSON read on Report render; no network, no recompute.

## 6. Error Handling
- Report absent → `calibrated_interval_line` returns `None` → the line is omitted; relabels still apply. No crash.
- Missing `conformal_widths['0.9']` → same `None` path.
- The added line is wrapped in `try/except` so any formatting issue leaves the rest of the consensus box intact.

## 7. Testing
- `test_honest_surface.py`:
  - `calibrated_interval_line({'conformal_widths': {'0.9': 10.42}})` → contains `'10.42'` and `'calibrated'` and `'90%'`.
  - `calibrated_interval_line(None)` and `calibrated_interval_line({'conformal_widths': {}})` → `None`.
- Import smoke: `stage4_report` imports cleanly.
- Window smoke (`test_main_window.py`): no new failures (the swarm box renders on `populate_swarm` after a run; logic covered by the helper test).

## 8. Deliverables (definition of done)
1. `calibrated_interval_line` in `ui/honest_surface.py` + tests.
2. `_build_swarm_left`: agreement relabeled (subtitle, range row, regional) + calibrated interval line added.
3. Graceful fallback when the report is absent.
4. Tests green; window smoke unaffected.

## 9. Why it matters
Removes the last place the app's surface implies calibrated certainty it doesn't have. After this, every "how sure are we?" number on the Report screen is either honestly labeled as *agent agreement* or is the *validated, calibrated* conformal interval — finishing the "honest at the surface" pass.

## 10. Sources / references
- Frozen artifact: `accuracy_report.json` (`conformal_widths`, `calibration`).
- Prior work: `ui/honest_surface.py`, `ui/accuracy_view.py`, `2026-06-10-honest-report-surface-design.md`.
