# ph_economic_ai — Real 3-Sector Trajectory Charts (SP2c Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** Third SP2 slice: SP2a (honesty cues) + SP2b (chart polish) merged → **SP2c (real 3-sector charts — this)** → SP2d (full editorial restyle). The "heavier" one; turned out to need **no main_window plumbing** (the report self-serves the data).

---

## 1. Problem & Goal

The report charts only gas. The user wants real food + electricity lines too. The data exists and the report can reach it without new plumbing: PSA food/electricity history via committed CSVs, and the next-month point estimates already arrive via `set_sector_forecasts`.

**Goal:** add a **"Sector trajectories"** small-multiples panel — three editorial mini-charts of each sector's *recent real history* in its native unit, with a forecast marker where the unit aligns. Honest (real PSA/market data, no unit fudging), no `main_window` changes.

---

## 2. Scope

### In scope (`ui/stage4_report.py`)
- A persistent `_trajectory_holder` populated by `set_sector_forecasts`, showing 3 stacked mini-charts.
- The report loads PSA food/electricity history itself via `benchmark.psa_cpi` loaders (read-only, graceful).

### Out of scope
- Any `main_window`/swarm change; the full editorial restyle (SP2d); changing the electricity representation to % (rejected as too broad in brainstorming); the gas Outputs chart (already done in SP2b).

### Non-negotiable (honesty)
- Every plotted line is **real** data (PSA gold / `df`); no synthesized series.
- The **forecast marker** appears only where the forecast unit matches the history unit: **gas** (₱/L) and **food** (MoM %). **Electricity** is history-only (CPI MoM %) with an explicit note that its forecast is ₱/kWh (shown in the bar card) — no ₱/kWh-vs-% conflation.
- Graceful: a missing gold CSV or absent `self._df` drops that sector's chart, never crashes the report (wrapped like the existing matplotlib blocks).

---

## 3. Components / changes (`ui/stage4_report.py`)

### 3.1 Store `df`
`populate` and `populate_swarm` already receive `df`; add `self._df = df` in both so `set_sector_forecasts` can reach the gas series.

### 3.2 `_trajectory_holder` (persistent)
In `_build`, create `self._trajectory_holder` (a `QWidget` + `QVBoxLayout`, hidden initially) and add it to the scroll body (`body_layout`) under the top row, so it scrolls with the report. (Mirrors the existing persistent `_sector_holder` pattern.)

### 3.3 Build trajectories in `set_sector_forecasts`
After rendering the magnitude-bar card (SP2b), build/refresh the 3 mini-charts into `_trajectory_holder` (clear-then-rebuild, like the bar card):
- Title row: `RECENT SECTOR TRAJECTORIES`.
- **Gas:** `self._df['gas_price'].dropna().tail(12)` (₱/L) as a line; forecast marker at `last_price + gas` (the gas next-month change) at x = next month.
- **Food:** `load_food_mom().tail(12)` (MoM %) as a line; forecast marker at `food` (% — aligns) at x = next month.
- **Electricity:** `load_electricity_mom().tail(12)` (CPI MoM %) as a line; **no** marker + a muted note `next-month forecast in ₱/kWh — see card above`.
- Each mini-chart: small `Figure` (e.g. figsize (4.6, 1.3)), editorial axes (face `#FBFBFA`, top/right spines hidden, left/bottom `#E5E7EB`, `grid(axis='y', #EEEEEE, 0.6)`, muted small ticks), a short title `GAS · ₱/L` / `FOOD · MoM %` / `ELECTRICITY · CPI MoM %`. Each in its own `FigureCanvasQTAgg`, fixed height ~120 px.
- Imports: `from ph_economic_ai.benchmark.psa_cpi import load_food_mom, load_electricity_mom` inside a `try/except` (so a missing CSV just skips that sector). The whole builder is inside `try/except` (no crash).
- Show `_trajectory_holder` once at least one chart built.

### 3.4 Optional pure helper (testable seam)
A small pure function `recent_series(series, n=12) -> (xs, ys)` and/or `forecast_marker(last_value, change) -> (x, y)` could factor the prep for unit-testing. Minimal — the strings/markers are the seam; the matplotlib drawing is covered by build-without-crash.

## 4. Data flow
```
populate/populate_swarm(df,...)   -> self._df = df
food/elec debates finish -> main_window.set_sector_forecasts(gas, food, elec)
   -> bar card (SP2b)  +  _trajectory_holder:
        gas:  df['gas_price'].tail(12) + marker(last+gas)
        food: load_food_mom().tail(12) + marker(food)
        elec: load_electricity_mom().tail(12)  (history only + ₱/kWh note)
```
No network (committed CSVs); no `main_window` change.

## 5. Testing
- `test_stage4_trajectories.py` (offscreen): construct `Stage4ReportPanel`, set `panel._df` to a small DataFrame with `gas_price`, call `set_sector_forecasts(-1.8, -2.6, 0.18)`; assert `_trajectory_holder` is visible and contains ≥1 `FigureCanvasQTAgg` (ideally 3 when gold present); assert a label/note containing "₱/kWh" (the electricity note) exists.
- Graceful test: with `panel._df = None` (or gold monkeypatched to raise), `set_sector_forecasts(...)` does not crash and the bar card still renders.
- Existing `test_stage4_*` (populate paths) stay green.
- Full suite green.

## 6. Deliverables (definition of done)
1. `self._df` stored in `populate`/`populate_swarm`.
2. Persistent `_trajectory_holder` with 3 editorial mini-charts built in `set_sector_forecasts` from real data.
3. Forecast markers on gas + food; electricity history-only + ₱/kWh note.
4. Graceful on missing data; tests per §5; full suite green.

## 7. Why it matters
It delivers the food + electricity *lines* the user has wanted since the chart discussion — from genuine PSA data, with the next-month call marked exactly where it's unit-honest and explicitly caveated where it isn't. Real, defensible, and self-served by the report (no new coupling).
