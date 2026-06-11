# ph_economic_ai — Chart Polish (SP2b Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** Second slice of SP2 (editorial polish): SP2a (honesty cues, merged) → **SP2b (chart polish — this)** → SP2c (real 3-sector charts) → SP2d (full editorial restyle). The "feasible C" — no new data plumbing.

---

## 1. Problem & Goal

Two visuals in the Report look unfinished:
- **The gas forecast chart** (`stage4_report._build_right`) plots only the forecast line + a barely-visible conformal band (alpha 0.15) on a flat panel — no historical context, no grid, no spine styling.
- **The 3-sector forecast card** (`set_sector_forecasts`) is plain colored **text rows**.

**Goal:** give the gas chart the editorial "C" treatment (historical context + the 90% calibrated band as the hero + a clean, gridded, spine-trimmed look) and turn the 3-sector card into a clean **magnitude-bar** visual — using only data already in the report. No new plumbing, no overclaiming.

---

## 2. Scope

### In scope
- `ui/sector_forecast.py` — add an honest `bar` magnitude (0–1) to each row (pure, tested).
- `ui/stage4_report.py` — render the 3-sector card as bars; restyle the gas forecast chart + feature-importances chart (editorial axes).

### Out of scope
- Food/electricity historical *series* in the report (that's SP2c), the broad editorial restyle (SP2d), any change to the forecast numbers/benchmark.

### Non-negotiable (honesty)
- The sector bar is **display-only**, scaled to *its own sector's* "notable move" (gas ±5 ₱/L, food ±5 %, elec ±2 ₱/kWh) — it never implies a cross-unit comparison; the value text stays exact.
- The chart restyle changes **styling only**, not the plotted numbers; the "90% calibrated interval (conformal)" caption and the "exploratory — not validated" labels remain.

---

## 3. Components

### 3.1 `ui/sector_forecast.py` — `bar` magnitude (pure)
`sector_forecast_rows(gas, food, elec)` already returns rows `{key, label, value, value_str, direction}`. Add a `bar` field:
```python
_BAR_SCALE = {'gas': 5.0, 'food': 5.0, 'elec': 2.0}   # per-sector "full bar" move
# per row: bar = 0.0 if value is None else min(abs(value) / _BAR_SCALE[key], 1.0)
```
- `None` value → `bar = 0.0`, `direction = 'na'` (unchanged).
- Tested: gas −1.80 → 0.36; food −2.60 → 0.52; elec +0.18 → 0.09; a huge value clamps to 1.0; None → 0.0.

### 3.2 `ui/stage4_report.set_sector_forecasts` — bar rendering
Replace each plain text row with a row widget: `LABEL` · arrow · a colored bar (`QFrame`, fixed track width e.g. 120 px, filled width = `int(120 * bar)`, min 2 px when `bar>0`) · exact `value_str`. Colour by `direction` using the existing `colors` map (`up #EF4444`, `down #16A34A`, `flat`/`na` muted). Keep the `NEXT-MONTH SECTOR FORECAST` title + `exploratory — not validated` sub-caption. Whole thing stays inside the existing `try/except` so a bad value can't blank the card.

### 3.3 `ui/stage4_report._build_right` — gas chart + feature-importances restyle
Gas forecast chart (the `if forecast_prices ...` block):
- Prepend recent actuals: take `df['gas_price'].dropna().tail(K)` (K ≈ 6), plot in gray; then the forecast (dark) continues from the last actual; a dashed vertical `→ forecast` divider at the join.
- **Band as hero:** keep `_band = _hs.conformal_halfwidth(_report) or cv_rmse`; raise the fill alpha (~0.18→0.30 with a faint edge line) over the forecast segment only.
- **Editorial axes:** soft horizontal gridlines (`ax.grid(axis='y', color='#EEEEEE', linewidth=0.6)`), hide top/right spines, mute tick colours, light forecast-zone shade (`axvspan` over the forecast x-range, very low alpha).
- Apply the same axes styling (grid off or subtle, spines trimmed, muted ticks) to the feature-importances chart for consistency.
- All inside the existing `try/except` (a styling/data error must not break the report).

## 4. Data flow
`_build_right(regressor, df, cv_rmse, scenario, consensus)` already receives `df` (has `gas_price`) + the conformal report via `honest_surface`. `set_sector_forecasts(gas, food, elec)` already receives the point estimates. No new inputs.

## 5. Testing
- `test_sector_forecast.py`: `bar` values for representative inputs (gas/food/elec), clamping at 1.0, `None → 0.0`, and that `direction`/`value_str` are unchanged (no regression).
- `test_stage4` (offscreen): after `set_sector_forecasts(-1.8, -2.6, 0.18)`, the sector card renders (no crash) and contains a bar widget per sector (e.g. assert ≥3 `QFrame` children under the sector holder, or the rows exist).
- The gas-chart restyle is covered by the existing `populate_swarm`/`populate` build-without-crash tests (it stays inside the matplotlib `try/except`).
- Full suite green.

## 6. Deliverables (definition of done)
1. `sector_forecast_rows` returns an honest per-sector `bar` (0–1); tested.
2. 3-sector card renders magnitude bars + arrow + exact value, colour-coded; "exploratory" caption kept.
3. Gas forecast chart shows recent actuals + forecast + hero 90% band + forecast-zone shade + soft grid + trimmed spines; feature-importances chart matches.
4. Tests per §5; full suite green; numbers/labels unchanged in substance.

## 7. Why it matters
The two least-finished visuals become the polished, data-rich "C" the user asked for — recent-context forecast with a prominent calibrated band, and an at-a-glance sector card — without inventing data or overclaiming (bars are per-sector display scaling; values stay exact; everything still tagged exploratory/validated correctly).
