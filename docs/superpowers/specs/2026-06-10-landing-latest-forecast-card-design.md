# ph_economic_ai — Landing "Latest Forecast" Combined Card (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** the three-sector forecast card (`ui/sector_forecast.py`, PR #6) on `master`.

---

## 1. Problem & Goal

The landing page advertises "fuel, food, and electricity" and "3 sectors", but its only forecast element — the **"RECENT FUEL FORECASTS"** strip — shows **fuel only**, because the store (`AgentTrustStore`) persists just the fuel `final_estimate`. Food and electricity are computed each run and shown on the Report/Overview, but never reach the landing.

**Goal:** on the landing, show the **most recent run's full three-sector forecast** (gas/food/electricity, each with direction + unit), **combined into one card** with the existing fuel-history strip — so the landing reflects all three sectors the app actually forecasts.

**Non-goal:** full multi-sector *history* (older tiles stay fuel-only), accuracy claims (these are exploratory forecasts), or any change to how forecasts are computed.

---

## 2. Scope

### In scope
- Persist `food_estimate` + `electricity_estimate` per run in the store (idempotent migration + an update method).
- `main_window` updating the saved run with food/electricity once those debates complete.
- A **single landing card** containing: a top **"LATEST FORECAST"** 3-sector summary (reusing `sector_forecast_rows`) + the existing **"RECENT FUEL FORECASTS"** fuel-history strip below it.

### Out of scope
- Multi-sector history tiles (the history row stays fuel-only).
- The swarm-memory feature (declined earlier).
- New computation; the swarm/debate path is unchanged.

---

## 3. Data

A run produces (already, in `main_window`):
- **gas/fuel:** `master_verdict.final_estimate` (₱/L) — saved on swarm completion via `store.save_run`, which sets `self._current_run_id`.
- **food:** `_on_food_complete` → `weighted_avg` (%), captured as `self._food_estimate`.
- **electricity:** `_on_elec_complete` → `weighted_avg` (₱/kWh), captured as `self._elec_estimate`.

The fuel run is saved first (swarm completes), then food/electricity finish a moment later (parallel) → the saved row is **updated** with food/electricity.

---

## 4. Architecture

```
engine/store.py    # + food_estimate, electricity_estimate columns (idempotent migration)
                   # + update_run_sectors(run_id, food, elec)
ui/main_window.py  # call update_run_sectors(_current_run_id, _food_estimate, _elec_estimate)
                   #   from _on_food_complete / _on_elec_complete
ui/landing.py      # _build_recent_strip: ONE card = LATEST FORECAST (3 sectors) + fuel history
                   # _refresh_recent_runs: populate the latest-forecast row from runs[0]
ui/sector_forecast.py  # reused unchanged (sector_forecast_rows)
```

### 4.1 Store (`engine/store.py`)
- **Migration (idempotent):** in `_migrate`, after the `CREATE TABLE`, read `PRAGMA table_info(runs)`; for each of `food_estimate` and `electricity_estimate` not present, run `ALTER TABLE runs ADD COLUMN <name> REAL`. Safe on existing DBs and on repeat calls.
- **`update_run_sectors(run_id, food_estimate, electricity_estimate)`** — mirrors `update_run_quality`: `UPDATE runs SET food_estimate=?, electricity_estimate=? WHERE run_id=?` under the existing lock.
- `get_recent_runs` is unchanged (`SELECT *` already returns the new columns; rows from before the migration return `None` for them).

### 4.2 `main_window`
- In `_on_food_complete`, after `self._food_estimate = avg` (and the existing push), if `self._store is not None and self._current_run_id is not None`: `self._store.update_run_sectors(self._current_run_id, self._food_estimate, self._elec_estimate)`.
- Same in `_on_elec_complete` (with the current pair). Each handler writes both columns with the latest known values; whichever completes second fills the remaining one. Wrapped so a store error never breaks the run.

### 4.3 Landing — one combined card (`ui/landing.py`)
- In `_build_recent_strip` (the existing `wrap` card), **above** the "RECENT FUEL FORECASTS" heading, add:
  - a sub-heading `LATEST FORECAST` (+ a muted "exploratory" note),
  - a row `self._latest_row` (QHBoxLayout) that will hold three sector mini-tiles.
  - Remove/raise the card's `setFixedHeight(124)` so both blocks fit.
- In `_refresh_recent_runs` (already called on `showEvent` and after a run): after fetching `runs = get_recent_runs(4)`, also rebuild `self._latest_row` from `runs[0]` (the latest) via
  `sector_forecast_rows(gas=runs[0].get('final_estimate'), food=runs[0].get('food_estimate'), elec=runs[0].get('electricity_estimate'))`,
  rendering one mini-tile per sector (label, direction arrow, value string, per the existing UP/DOWN palette). If there are no runs, the latest row shows the existing "No simulations on record yet." state and the fuel strip stays as-is.

---

## 5. Data Flow
```
run → swarm saves fuel + run_id ─┐
      food/elec debates finish ──┤→ update_run_sectors(run_id, food, elec)
                                  ▼
landing _refresh_recent_runs → get_recent_runs(4)
        runs[0] → sector_forecast_rows(gas, food, elec) → LATEST FORECAST tiles
        runs[:4] fuel final_estimate → RECENT FUEL FORECASTS tiles  (same card)
```

## 6. Error Handling
- Pre-migration rows / sectors not yet saved → `None` → sector tile shows "—" (formatter already handles this).
- Store unavailable or `_current_run_id is None` → skip the update (guarded); landing falls back to fuel-only/empty.
- Migration runs every startup but only ALTERs when a column is missing (idempotent); wrapped so a migration error doesn't crash startup.

## 7. Testing
- `test_store_sectors.py`: on a temp DB — `save_run` → `update_run_sectors` → `get_recent_runs()[0]` has the food/electricity values; calling `_migrate` again does not error (idempotent); a row saved without sector update returns `None` for those columns.
- `test_landing_latest.py` (offscreen Qt): a `LandingPanel` given a fake store whose `get_recent_runs` returns one run with `final_estimate`/`food_estimate`/`electricity_estimate` → after `_refresh_recent_runs`, the latest row's labels contain all three sectors + units; with an empty store it shows the placeholder without error.
- Window smoke (`test_main_window.py`) unaffected.

## 8. Deliverables (definition of done)
1. Store: `food_estimate`/`electricity_estimate` columns (idempotent migration) + `update_run_sectors`.
2. `main_window` updates the saved run with food/electricity on completion.
3. Landing: one card with the latest 3-sector forecast above the fuel-history strip.
4. Tests per §7; full suite green; window smoke unaffected.

## 9. Why it matters
The first screen a user sees will reflect what the app actually forecasts — all three sectors for the latest run, each with direction and unit, honestly labeled — instead of fuel alone. It closes the "the landing only shows gas" gap at the place users notice it first, with a small, safe, reversible store addition.
