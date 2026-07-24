    # ph_economic_ai — Unified Report/Interact Workbench (Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Inspiration:** MiroFish refs (`.mirofish_refs/mf4.png`, `mf5.png`) — one persistent two-pane workbench (report document left; right pane swaps between outputs and deep interaction).

---

## 1. Problem & Goal

Today the post-run experience is split across two stack pages — **Report** (`stage4_report.py`, the consensus/forecasts/charts) and **Interact** (`stage5_interact.py`, the Ask-an-Agent chat + Adjust/Toggle tools). The user must hop between them, and the report isn't visible while chatting.

**Goal:** merge them into one **Workbench** screen: the report document is always on the left; the right pane toggles **Outputs ⇄ Interact**. Reorganisation of existing, working functionality — no behaviour removed, no new prediction logic.

**Validated layout (companion-approved):**
- **Left (persistent report document):** Swarm Consensus · Next-month sector forecast (exploratory) · Validated-accuracy strip · Macro causal chain · Policy recommendations · Regional impact.
- **Right (toggle `Outputs` ⇄ `Interact`):**
  - **Outputs:** Final Outputs (next-week/month, 3/6-month) · forecast chart with the 90% calibrated band · feature importances.
  - **Interact:** the report-agent chat (existing "Ask an Agent") + tool chips for "Adjust inputs" and "What-if toggle".

---

## 2. Scope

### In scope
- A `WorkbenchPanel` widget: left report column + right pane with an `Outputs`/`Interact` toggle (QStackedWidget).
- **Reuse, don't rewrite:** compose the existing builders/widgets — the report-document blocks and chart blocks from `stage4_report`, and the whole `Stage5InteractPanel` as the Interact tab.
- `main_window` rewiring: one panel receives `populate_swarm`, `set_regional_estimates`, `set_chain`, `set_policy_recos`, `set_swarm_context`, `update_*_verdict`, `set_sector_forecasts`.
- Navigation: collapse the two stack pages + the two sidebar/nav entries ("Report", "Interact") into one ("Report").

### Out of scope
- Any change to the swarm/debate computation, the store, the benchmark, or the validated-accuracy content.
- The MiroFish *graph-visualisation* screens (mf2/mf3/mf6) — not part of this merge.
- Restyling the landing/overview/agent-performance screens.

---

## 3. Architecture

```
ui/
├── workbench.py            # NEW WorkbenchPanel: QHBoxLayout [ report-left | right-pane ]
│                           #   right-pane = top toggle (Outputs|Interact) + QStackedWidget
│                           #   - Outputs page  (charts/Final Outputs/feature importances)
│                           #   - Interact page (embeds Stage5InteractPanel)
├── stage4_report.py        # refactor: split builders into (a) report-document column,
│                           #   (b) outputs column — both reusable by WorkbenchPanel.
│                           #   Keep populate_swarm() public API; it delegates to the two halves.
├── stage5_interact.py      # reused as-is, embedded as the Interact page (no functional change)
├── main_window.py          # build WorkbenchPanel instead of separate stage4/stage5 pages;
│                           #   route the existing populate/context/verdict calls to it;
│                           #   single stack index for the workbench
└── sidebar.py              # merge "Report" + "Interact" nav entries into one "Report"
```

### 3.1 `WorkbenchPanel` (new)
- `__init__(self, rag, agents, regressor, ...)` — constructs the left report column (from the refactored stage4 report-document builders) and the right pane.
- Right pane: a header toggle (two pill buttons `Outputs` / `Interact`) controlling a `QStackedWidget`:
  - index 0 **Outputs** = the stage4 "Final Outputs" card + forecast chart + feature importances.
  - index 1 **Interact** = an embedded `Stage5InteractPanel`.
- Delegating public methods (so `main_window` calls don't change shape):
  - `populate_swarm(master_verdict, regressor, df, cv_rmse, scenario)` → fills the report-left consensus + the Outputs charts.
  - `set_regional_estimates`, `set_chain`, `set_bsp_alert`, `set_sector_forecasts` → report-left.
  - `set_swarm_context`, `update_gas_verdict`, `update_food_verdict`, `update_elec_verdict` → forwarded to the embedded `Stage5InteractPanel`.
- Default right-pane tab on a fresh run: **Outputs** (so the run's numbers show first); the user toggles to Interact to chat.

### 3.2 `stage4_report` refactor
- Extract the existing report-document blocks (swarm-consensus / regional verdicts / sector-forecast card / validated-accuracy strip / causal-chain / BSP banner) into methods that build a **report-document widget**, and the quantitative blocks (Final Outputs / forecast chart / feature importances) into an **outputs widget**. `WorkbenchPanel` consumes both. `populate_swarm` keeps working (delegates).
- This is the only non-trivial code change; everything else is composition + wiring.

### 3.3 `main_window` + `sidebar`
- Replace the two pages in the stack with the single `WorkbenchPanel`; update the stack index constants and `unlock_stages`/`set_active` calls (Home=0, Overview=1, Simulation=2, **Workbench/Report=3**, Agent performance, Methodology shift up by one).
- `sidebar._NAV`: remove the separate "Interact" row; the "Report" row points to the workbench. The top nav (Home/Simulation/Report/Interact) drops "Interact".
- All `self._stage5.*` calls become `self._workbench.*` (forwarded); `self._stage4.*` calls become `self._workbench.*`.

---

## 4. Data Flow
```
run completes → main_window:
   workbench.populate_swarm(...)        → report-left consensus + Outputs charts
   workbench.set_sector_forecasts(...)  → report-left 3-sector card
   workbench.set_chain / set_policy / set_regional → report-left sections
   workbench.set_swarm_context / update_*_verdict  → embedded Interact (chat context + verdicts)
user toggles right pane Outputs ⇄ Interact (no recompute; both already populated)
```

## 5. Error Handling / Migration safety
- Pure reorganisation: every existing public method (`populate_swarm`, `set_*`, `update_*`) is preserved on `WorkbenchPanel` (delegating), so `main_window`'s call sites change only the receiver object, not the calls.
- The embedded `Stage5InteractPanel` is used unchanged → the chat/Adjust/Toggle behaviour is identical.
- If the Outputs/report builders raise, they are wrapped (consistent with the existing try/except around the chart blocks) so one section can't blank the workbench.
- Sidebar index shift is the main regression risk → covered by the window smoke test asserting the run navigates to the workbench page and the panel exposes the expected methods.

## 6. Testing
- `test_workbench.py` (offscreen Qt): `WorkbenchPanel` constructs; has `populate_swarm`, `set_swarm_context`, `update_gas_verdict`, `set_sector_forecasts`; the right-pane toggle switches the `QStackedWidget` between Outputs (index 0) and Interact (index 1); the embedded panel is a `Stage5InteractPanel`.
- `test_main_window.py`: still constructs the window; running a sim navigates to the workbench stack index; `self._workbench` receives the swarm result (smoke).
- Existing `stage4`/`stage5` unit tests adapted to the refactored builders (sector card, honest-surface, interact chat) — kept green.

## 7. Deliverables (definition of done)
1. `WorkbenchPanel` (left report column + right Outputs/Interact toggle) reusing existing widgets.
2. `stage4_report` split into report-document + outputs builders (public `populate_swarm` preserved).
3. `main_window` builds + routes to the workbench; one stack page; indices fixed.
4. `sidebar`/nav merged (one "Report" entry; "Interact" removed).
5. Tests per §6; full suite green; honest-surface labels (exploratory / validated accuracy / agent agreement) intact.

## 8. Why it matters
One screen instead of two: the report stays in view while you read the charts or interrogate the agents — fewer clicks, a clearer mental model, and a polished MiroFish-style workbench. It's a reorganisation of proven functionality, so the validated, honest content carries over unchanged.
