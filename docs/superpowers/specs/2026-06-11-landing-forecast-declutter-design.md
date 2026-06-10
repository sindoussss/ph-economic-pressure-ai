# ph_economic_ai — Landing Forecast Declutter + Confidence (Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** the landing latest-forecast card (`ui/landing.py`, PR #7) and the sector-refresh fix on `master`.

---

## 1. Problem & Goal

The landing's forecast area now shows two stacked blocks — "LATEST FORECAST" (3 sectors) and "RECENT FUEL FORECASTS" (4 fuel tiles). Two issues:
1. **No confidence on the latest block** — the 3-sector latest forecast shows direction + value but no agreement/confidence (the history tiles show it, the latest does not).
2. **Cramped + duplicated** — the most recent run appears twice: as the LATEST gas number *and* as `#5` in the history, making the area redundant and crowded.

**Goal:** add date + agreement to the latest block, and de-duplicate the history so it shows only *prior* runs as a compact track record. Landing-only, no data change.

---

## 2. Scope

### In scope (`ui/landing.py` only)
- Dynamic "LATEST FORECAST" heading: `LATEST FORECAST · <date> · <N>% agreement · exploratory`, from the latest run.
- Rename "RECENT FUEL FORECASTS" → "FUEL TRACK RECORD"; populate it from the *prior* runs (`runs[1:4]`) so the latest no longer duplicates.
- Relabel the per-tile "confidence" → "agreement" for consistency with the Report.

### Out of scope
- Store/schema, the sector estimates, the swarm/LLM path, the Report/Overview screens.
- Per-sector confidence (only the run-level `confidence_pct` exists; one agreement value is shown for the latest run).

---

## 3. Data

`get_recent_runs(limit=4)` already returns each run dict with `run_id`, `timestamp`, `final_estimate`, `confidence_pct`, `food_estimate`, `electricity_estimate`, `actual_price_change`. The latest run is `runs[0]`; prior runs are `runs[1:]`. "Agreement" = `confidence_pct`; date = `timestamp` formatted `%b %d` (as `_build_run_tile` already does). All values may be `None` → handled by existing formatters/guards.

---

## 4. Architecture (changes within `ui/landing.py`)

### 4.1 `_build_recent_strip`
- Replace the static `head = QLabel('LATEST FORECAST  ·  exploratory')` with a member `self._latest_head = QLabel('LATEST FORECAST  ·  exploratory')` (text set in `_refresh_recent_runs`).
- Rename the fuel heading text `'RECENT FUEL FORECASTS'` → `'FUEL TRACK RECORD'`.
- `self._latest_row` and `self._runs_row` layouts are unchanged.

### 4.2 `_refresh_recent_runs`
- After fetching `runs`:
  - **Latest heading:** if `runs`, set `self._latest_head` text to
    `f'LATEST FORECAST  ·  {date}  ·  {conf}% agreement  ·  exploratory'`
    where `date` = `_fmt_date(runs[0]['timestamp'])` and `conf` = `runs[0].get('confidence_pct')` (omit the agreement clause if `conf` is None). If no runs, `'LATEST FORECAST  ·  exploratory'`.
  - **Latest row:** populate `self._latest_row` from `runs[0]` (unchanged sector logic).
  - **Track record:** populate `self._runs_row` from **`runs[1:4]`** (prior runs only). If there are no prior runs (≤1 run total), show the existing "No simulations on record yet." placeholder in `self._runs_row`.
- A small helper `_fmt_date(ts)` factors the existing `datetime.fromisoformat(...).strftime('%b %d')` logic (with the same fallback) so both the heading and `_build_run_tile` use it.

### 4.3 `_build_run_tile`
- Change the displayed `'{conf}% confidence'` substring to `'{conf}% agreement'`. No other change.

---

## 5. Data Flow
```
get_recent_runs(4) → runs
  runs[0]  → latest heading (date + agreement) + _latest_row (3 sectors)
  runs[1:4]→ FUEL TRACK RECORD tiles (prior runs only; no duplicate of latest)
```
No network, no store change.

## 6. Error Handling
- No runs → latest heading `'LATEST FORECAST · exploratory'`, latest row empty, track record placeholder (existing behaviour).
- Exactly one run → latest block shows it; track record shows the placeholder.
- `confidence_pct` / `timestamp` None → omit the agreement clause / fall back to `ts[:10]` (existing `_build_run_tile` fallback).
- Existing try/except around the store fetch is preserved.

## 7. Testing (`tests/test_landing_latest.py`, offscreen Qt)
- With 3 runs (latest `run_id=6` + two priors), after `refresh_recent()`:
  - the latest heading text contains the latest run's date and `"agreement"` (and the `confidence_pct` value);
  - the FUEL TRACK RECORD tiles contain the *prior* run ids (e.g. `#5`, `#4`) and **not** the latest `#6`;
  - "FUEL TRACK RECORD" label present; the latest 3 sectors still render.
- With a single run: latest block renders; the track record shows "No simulations on record yet." (no crash, no duplicate tile).
- Existing latest-forecast and empty-store tests still pass.

## 8. Deliverables (definition of done)
1. Dynamic latest heading (date + agreement + exploratory).
2. "FUEL TRACK RECORD" showing prior runs only (no duplication).
3. Per-tile "confidence" → "agreement".
4. Tests per §7; full suite green; window smoke unaffected.

## 9. Why it matters
Removes the visible redundancy (the latest run shown twice) and the crowding, and gives the latest forecast the same agreement context the history already had — making the landing read as one clean "here's the latest call (with all three sectors and how much the agents agreed), and here's the track record behind it."
