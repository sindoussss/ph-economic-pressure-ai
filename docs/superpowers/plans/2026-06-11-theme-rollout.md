# Editorial Theme Roll-out (SP2d-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Roll the merged `ui/theme.py` tokens out to the remaining card/text screens that still carry legacy or panel hexes — retiring the strays so the whole app shares one palette. Mechanical, styling-only.

**Architecture:** Per screen: import `theme`, convert the stylesheet strings that contain a legacy/panel hex into f-strings referencing the matching token, change nothing else. Viz/semantic-colour files and the already-editorial landing are intentionally **not** touched.

**Tech Stack:** Python 3.10, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-11-theme-and-report-restyle-design.md` (this is its "SP2d-2…n roll-out").

**The token mapping (apply consistently):**
| legacy/panel hex | token |
|---|---|
| `#F7F8FA` (panel bg) | `_theme.SURFACE` |
| `#EAECF0` (border) | `_theme.HAIRLINE` |
| `#EAEAEA` (legacy border/gray) | `_theme.HAIRLINE` |
| `#4A90E2` (legacy blue) | `_theme.NEUTRAL` |

Conversion pattern: a plain `'... #F7F8FA ...'` stylesheet becomes `f'... {_theme.SURFACE} ...'` (make it an f-string, replace the literal with `{token}`). Leave `#FFFFFF`/`#1C1E26` literals alone (their token values are identical — no churn needed). **Do not** change any label text, layout, or behaviour.

**Explicitly NOT in scope (leave as-is):**
- `landing.py` — already the editorial reference (no legacy/panel hexes); don't disturb.
- Viz/semantic-colour files: `kg_canvas.py`, `regional_map.py`, `causal_chain_widget.py`, `stage3_swarm_canvas.py`, `stage3_canvas.py`, `agent_graph.py` — their palettes are meaningful.
- `charts.py` / `dashboard.py` — dead path (`PriceChart` only used by the unused `dashboard.py`).
- `agent_performance.py`, `accuracy_view.py` — no legacy/panel hexes to retire.

**Conventions:** offscreen Qt. **Git hygiene:** commit ONLY listed paths per task; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`. Never add `self.show()` or visibility hacks; if a test needs visibility use `not widget.isHidden()`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/theme-rollout`

---

## Task 1: `economy_overview.py` (retire legacy `#4A90E2` + `#EAEAEA`)

**Files:** Modify `ph_economic_ai/ui/economy_overview.py`

- [ ] **Step 1:** `grep -nE "#4A90E2|#EAEAEA|#F7F8FA|#EAECF0" ph_economic_ai/ui/economy_overview.py` — list every occurrence.
- [ ] **Step 2:** Add `from ph_economic_ai.ui import theme as _theme` near the top imports.
- [ ] **Step 3:** For each stylesheet string containing one of those hexes, make it an f-string and replace the literal with the mapped token (`#4A90E2`→`{_theme.NEUTRAL}`, `#EAEAEA`→`{_theme.HAIRLINE}`, `#F7F8FA`→`{_theme.SURFACE}`, `#EAECF0`→`{_theme.HAIRLINE}`). Watch for braces: if a stylesheet already contains literal `{`/`}` (Qt `QWidget{...}`), double them (`{{`/`}}`) when converting to an f-string. No other change.
- [ ] **Step 4: Verify** — `python -c "import os;os.environ.setdefault('QT_QPA_PLATFORM','offscreen');import sys;from PyQt6.QtWidgets import QApplication;a=QApplication.instance() or QApplication(sys.argv);import pandas as pd;from ph_economic_ai.ui.economy_overview import EconomyOverviewWidget;EconomyOverviewWidget(pd.DataFrame({'date':pd.date_range('2024-01',periods=3,freq='M'),'gas_price':[58.,59.,60.]}));print('overview builds OK')"` → `overview builds OK`. Then `grep -cE "#4A90E2|#EAEAEA" ph_economic_ai/ui/economy_overview.py` → `0`.
- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/economy_overview.py && git commit -m "style(ui): economy overview uses theme tokens (retire legacy blue/gray)"`

---

## Task 2: `stage5_interact.py` (panel hexes → tokens)

**Files:** Modify `ph_economic_ai/ui/stage5_interact.py`

- [ ] **Step 1:** `grep -nE "#F7F8FA|#EAECF0|#EAEAEA|#4A90E2" ph_economic_ai/ui/stage5_interact.py` — list occurrences (~6 panel hexes expected).
- [ ] **Step 2:** Add `from ph_economic_ai.ui import theme as _theme` near the top imports.
- [ ] **Step 3:** Convert each stylesheet string containing those hexes to an f-string with the mapped token (double any literal `{`/`}` braces). Leave the SP2a honesty caption text + the Ask/Adjust/Sources behaviour untouched (style only).
- [ ] **Step 4: Verify** — `python -m pytest ph_economic_ai/tests/test_honesty_interact.py -q` → pass; then `grep -cE "#F7F8FA|#EAECF0" ph_economic_ai/ui/stage5_interact.py` → `0`.
- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/stage5_interact.py && git commit -m "style(ui): interact panel uses theme tokens"`

---

## Task 3: `stage2_setup.py` (panel hexes → tokens)

**Files:** Modify `ph_economic_ai/ui/stage2_setup.py`

- [ ] **Step 1:** `grep -nE "#F7F8FA|#EAECF0|#EAEAEA|#4A90E2" ph_economic_ai/ui/stage2_setup.py` — list (~4 panel hexes).
- [ ] **Step 2:** Add `from ph_economic_ai.ui import theme as _theme`.
- [ ] **Step 3:** Convert each matching stylesheet string to an f-string with the mapped token (double literal braces). Style only.
- [ ] **Step 4: Verify** — `python -m pytest ph_economic_ai/tests/test_stage2.py -q` → pass; then `grep -cE "#F7F8FA|#EAECF0" ph_economic_ai/ui/stage2_setup.py` → `0`.
- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/stage2_setup.py && git commit -m "style(ui): setup screen uses theme tokens"`

---

## Task 4: `policy_reco.py` (panel hexes → tokens)

**Files:** Modify `ph_economic_ai/ui/policy_reco.py`

- [ ] **Step 1:** `grep -nE "#F7F8FA|#EAECF0|#EAEAEA|#4A90E2" ph_economic_ai/ui/policy_reco.py` — list (~2 panel hexes).
- [ ] **Step 2:** Add `from ph_economic_ai.ui import theme as _theme`.
- [ ] **Step 3:** Convert each matching stylesheet string to an f-string with the mapped token (double literal braces). Style only — keep the PolicyRecoWidget structure/text.
- [ ] **Step 4: Verify** — no dedicated test; run an import + construct smoke: `python -c "import os;os.environ.setdefault('QT_QPA_PLATFORM','offscreen');import sys;from PyQt6.QtWidgets import QApplication;a=QApplication.instance() or QApplication(sys.argv);from ph_economic_ai.ui.policy_reco import PolicyRecoWidget;PolicyRecoWidget();print('policy_reco builds OK')"` → `policy_reco builds OK`. Then `grep -cE "#F7F8FA|#EAECF0" ph_economic_ai/ui/policy_reco.py` → `0`.
- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/policy_reco.py && git commit -m "style(ui): policy recommendations use theme tokens"`

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass (no behaviour changed).
- [ ] Sanity: `grep -rcE "#4A90E2" ph_economic_ai/ui/economy_overview.py` → 0 (the legacy blue is gone from the live overview).
- [ ] Manual (GUI, optional): the overview / setup / interact / policy cards read in the same editorial palette as the Report.

---

## Self-Review (completed by plan author)
**Spec coverage:** the spec's "SP2d-2…n roll-out" → Tasks 1–4 (the live screens that actually carry legacy/panel hexes). Viz/semantic files, landing, dead `charts`/`dashboard`, and hex-clean screens are explicitly excluded with reasons. Token mapping is the one defined in SP2d-1.
**Placeholder scan:** none — each task is a concrete grep-then-convert recipe against an enumerated hex set, with a build/test verify and a `grep -c … → 0` proof the strays are gone. (Per-line literals aren't pre-written because the work is a deterministic find-and-convert; the recipe + targets + verification fully specify it.)
**Type consistency:** purely string/stylesheet edits; `_theme.SURFACE/HAIRLINE/NEUTRAL` are str constants substituted into f-strings. The brace-doubling note prevents the one real footgun (Qt `QWidget{...}` selectors inside an f-string). Each screen verified by its existing test (overview/policy via construct smoke; interact via test_honesty_interact; setup via test_stage2) plus the full suite.
````
