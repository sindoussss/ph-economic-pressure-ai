# ph_economic_ai — Three-Sector Forecast Card (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** the honest-surface work (`ui/honest_surface.py`, `stage4_report.py`) on `master`.

---

## 1. Problem & Goal

A simulation run already produces a next-month forecast for **all three sectors** the app advertises — gas/fuel (the 20-agent swarm), food, and electricity (parallel `DebateEngine`s) — but the **Report** screen and the landing "RECENT WORK" strip surface **only the fuel number** (`final_estimate`, ₱/L). Users (including the owner) cannot tell what the headline number refers to, or see the food/electricity calls the app actually computed.

**Goal:** surface the three sector verdicts a run already produces, clearly and honestly, in one compact card on the Report — each with its direction, value, correct unit, and an "exploratory — not validated" label — and relabel the fuel-only history strip so it is unambiguous.

**Non-goal:** improving accuracy (all three are exploratory swarm forecasts), persisting food/electricity to the store (history stays fuel-only), or re-running anything. This is a clarity/surfacing change only.

---

## 2. Scope

### In scope
- A pure formatter that turns the three numeric estimates into display rows (per-sector unit + direction).
- A "Next-month sector forecast" card on the Report screen (`stage4_report`).
- `main_window` capturing the three numeric estimates and pushing them to the card when ready.
- Relabel the landing "RECENT WORK" strip to "RECENT FUEL FORECASTS".

### Out of scope
- Store/schema changes; food/electricity history; the swarm-memory feature (declined).
- Any change to how the three forecasts are computed.

---

## 3. Definitions / data

A run produces, in `main_window`:
- **Gas/fuel:** swarm master verdict `final_estimate` — next-month pump-price change, **₱/L** (already received when the swarm completes).
- **Food:** `self._food_engine.consensus()['weighted_avg']` — food price index monthly change, **%** (computed in `_on_food_complete`).
- **Electricity:** `self._elec_engine.consensus()['weighted_avg']` — electricity rate monthly change, **₱/kWh** (computed in `_on_elec_complete`).

Any of the three may be `None` (sector debate unavailable) → rendered as "—".

---

## 4. Architecture

```
ui/
├── sector_forecast.py   # NEW pure: sector_forecast_rows(gas, food, elec) -> list[dict]
├── stage4_report.py     # + set_sector_forecasts(gas, food, elec): render the card
└── ../ui/landing.py     # relabel "RECENT WORK" -> "RECENT FUEL FORECASTS"
main_window.py           # capture _gas_estimate/_food_estimate/_elec_estimate; push when ready
```

### 4.1 `ui/sector_forecast.py` (pure, no PyQt)
```python
SECTORS = [
    ('gas',  'Gas / fuel',   '₱/L',   '{:+.2f} ₱/L'),
    ('food', 'Food',         '%',     '{:+.2f} %'),
    ('elec', 'Electricity',  '₱/kWh', '{:+.4f} ₱/kWh'),
]

def sector_forecast_rows(gas=None, food=None, elec=None) -> list[dict]:
    """Return one display row per sector: {key, label, value, value_str, direction}.
    direction is 'up' (>0), 'down' (<0), 'flat' (==0), or 'na' (None)."""
```
- `value_str` uses the per-sector format; `None` → `'—'`, direction `'na'`.
- Pure and fully unit-testable (signs, units, None handling).

### 4.2 `stage4_report.set_sector_forecasts(gas=None, food=None, elec=None)`
- Builds (or refreshes) a card titled **"Next-month sector forecast"** with a muted subtitle **"exploratory — not validated"**.
- One row per sector from `sector_forecast_rows`: label, an up/down indicator (▲ red-ish for up / ▼ green-ish for down per the app's existing UP/DOWN palette — note "up" = price rising = bad, matching the existing economy-overview colour convention), and the value string.
- Idempotent: calling again replaces the card contents.
- Wrapped so a formatting error cannot break the rest of the Report.

### 4.3 `main_window` wiring
- Retain the three numeric estimates as they arrive: `self._gas_estimate` (from the swarm result/master verdict), `self._food_estimate` and `self._elec_estimate` (the `avg` already computed in `_on_food_complete`/`_on_elec_complete`).
- After each sector verdict arrives, call `self._stage4.set_sector_forecasts(self._gas_estimate, self._food_estimate, self._elec_estimate)` (the card shows "—" for any not-yet-ready sector and fills in as they complete). Reuse the existing readiness flow that already gates `_run_synthesizer_if_ready`.

### 4.4 Landing relabel
- In `landing.py`, change the "RECENT WORK" heading to **"RECENT FUEL FORECASTS"** (and, if a tile sub-label helps, note "gas · ₱/L"). Pure label change; no data change.

---

## 5. Data Flow
```
run → swarm (gas ₱/L) ─┐
      food debate (%) ─┤→ main_window captures _gas/_food/_elec_estimate
      elec debate (₱/kWh)┘            │ set_sector_forecasts(...)
                                       ▼
              sector_forecast_rows() → stage4 "Next-month sector forecast" card
```
No store, no network, no recompute.

## 6. Error Handling
- Missing/unavailable sector → estimate `None` → row shows "—", direction "na". No crash.
- `set_sector_forecasts` card build wrapped in try/except (consistent with the existing Report blocks).
- Called repeatedly as sectors complete; idempotent replace.

## 7. Testing
- `test_sector_forecast.py` (pure): `sector_forecast_rows(-2.40, 0.50, 0.05)` → gas down "−2.40 ₱/L", food up "+0.50 %", electricity up "+0.0500 ₱/kWh"; `None` → "—"/"na"; `0.0` → "flat".
- `stage4_report`: constructing the panel and calling `set_sector_forecasts(...)` renders without error and the card text contains all three units (focused widget test).
- Window smoke (`test_main_window.py`) unaffected.

## 8. Deliverables (definition of done)
1. `ui/sector_forecast.py` + tests.
2. `stage4_report.set_sector_forecasts` + the card.
3. `main_window` captures the three estimates and pushes them.
4. Landing strip relabeled "RECENT FUEL FORECASTS".
5. Tests green; window smoke unaffected.

## 9. Why it matters
The app claims "3 sectors" but shows one. This makes the surface match reality: a user sees the gas, food, and electricity calls the run actually made, each with direction and unit — honestly labeled exploratory. It removes the "what is this −2.40 number?" confusion at its source.
