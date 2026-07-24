# PH Economy Expansion — Design Spec
**Date:** 2026-05-21
**Status:** Approved

## Overview

Expand `ph_economic_ai` from a single gas-price predictor into a three-sector Philippine economic cascade simulator. Gas, Food, and Electricity each get independent ML predictions and dedicated LLM debates. A weather API feeds food price modeling. All three sectors are surfaced in a new bento-style "Economy Overview" dashboard tab.

---

## 1. Data Layer

### New data sources

| Source | Data | Fetch method |
|---|---|---|
| Open-Meteo | Monthly rainfall (mm) + avg temp (°C) aggregated across 3 PH agricultural zones (see below) | Free REST API, no key required; 3 lat/lon requests per fetch cycle |
| FAO Food Price Index | Philippines cereal/food sub-index, monthly | FAO public CSV endpoint; falls back to gas+weather derivation on failure |
| World Bank (existing infra) | Electricity cost indicator (annual, forward-filled) | Same `_fetch_world_bank()` pattern as CPI/BSP today |
| Derived electricity rate | `base_rate + (gas_price_change × 0.18)` | Calibrated from Meralco's ~18% oil-linked generation cost pass-through |

### Agricultural weather zones

Three Open-Meteo points are fetched and weighted by estimated food production share:

| Zone | Coordinates | Weight | Rationale |
|---|---|---|---|
| Central Luzon / Nueva Ecija | lat 15.58, lon 121.10 | 0.45 | Largest rice-producing region |
| Bicol Region | lat 13.42, lon 123.41 | 0.25 | Coconut, crops, storm exposure |
| Davao / Mindanao | lat 7.07, lon 125.61 | 0.30 | Banana, corn, coconut exports |

`rainfall_mm` and `temp_c` in the dataset are the production-weighted averages of the three zones. Weights are hardcoded constants (not learned) — they reflect PSA regional food output shares and do not change at runtime.

### New columns added to dataset

| Column | Type | Source |
|---|---|---|
| `rainfall_mm` | float | Weighted avg of 3 agricultural zones via Open-Meteo |
| `temp_c` | float | Weighted avg of 3 agricultural zones via Open-Meteo |
| `food_price_idx` | float | FAO → fallback derivation |
| `electricity_rate` | float | World Bank blend + gas formula |

### Fallback strategy

- Open-Meteo down → use per-zone seasonal rainfall averages (hardcoded monthly norms for each of the 3 coordinates); weighted average applied same as live data.
- FAO fetch fails → derive: `food_price_idx = last_known_idx + (gas_price_delta × 0.22) + (rainfall_deficit_pct × 0.15)` where `last_known_idx` is the most recent cached FAO value, or 100.0 (index baseline) if no cache exists.
- All fallbacks are logged in the existing cache metadata so the UI can show a "Derived" badge instead of "Live Data".

### Caching

Weather and food data share the existing `cache/data.json` structure with the same 24h TTL. The single cache file grows to include all new columns — no schema migration needed, existing cache is invalidated and rebuilt on next launch.

---

## 2. Feature Engineering & ML Models

### Architecture: Parallel models (Approach A)

Three independent `HistGradientBoostingRegressor` models. Gas runs first; its prediction (`gas_pred`) is passed as an input feature to Food and Electricity. This captures causal pass-through without hard-coupling training loops.

> **Data separation invariant (Phase 1+):** `gas_pred`, `food_pred`, and `electricity_pred` are transient inference-time values only. They are computed in memory during a single run and passed directly between models. They must never be written to `cache/data.json`, appended to the training DataFrame, or used as targets in any retraining step. All model training uses only observed historical/API data. See Section 8.

### Feature sets

**Gas model** (existing, unchanged)
- Features: `oil_price`, `usd_php`, `demand_index`, `psei`, `cpi`, `bsp_rate`, `remittances`, `prev_gas_price`
- Target: `gas_price`

**Food model** (new)
- Features: `oil_price`, `usd_php`, `cpi`, `rainfall_mm`, `temp_c`, `food_price_idx_lag1`, `gas_pred`
- Target: `food_price_idx`
- Weather is food-only — rainfall shortfall signals crop stress → price spike

**Electricity model** (new)
- Features: `oil_price`, `usd_php`, `gas_pred`, `bsp_rate`, `electricity_rate_lag1`
- Target: `electricity_rate`
- No weather — PH grid (coal + hydro + gas) means oil/gas pass-through dominates

### Code changes

**`utils/preprocessing.py`**
- Add `build_gas_features(df)`, `build_food_features(df, gas_pred)`, `build_electricity_features(df, gas_pred)` — each returns `(X, y, feature_cols, df)`
- Add `build_all_features(df, gas_pred)` orchestrator that calls all three and returns a dict keyed by sector name
- Keep existing `build_features()` as a thin alias for `build_gas_features()` to avoid breaking current tests

**`model.py`**
- Rename `train()` → `train_sector(X, y)` internally; keep `train()` as alias
- `main.py` calls `train_sector` three times, once per sector, storing results in `{gas: regressor, food: regressor, electricity: regressor}`

---

## 3. LLM Debate Layer

### Execution order

Gas debate → Food debate → Electricity debate → Economy Synthesizer. Sequential so each debate can receive the previous sector's verdict as context. Each runs in its own `QThread` and emits signals to the UI as tokens stream in.

### Sector agent sets

**Gas** — existing agents unchanged.

**Food** (new, 4 agents using `_MAIN_MODEL`):
- Agri Analyst — crop supply, harvest cycles, import dependency
- Supply Chain Expert — transport cost pass-through from fuel prices
- Weather Interpreter — reads rainfall/temp signal, assesses crop stress
- Trade Policy Critic — tariff, NFA buffer stock, import quota impact

**Electricity** (new, 4 agents using `_MAIN_MODEL`):
- Energy Economist — generation mix, fuel cost pass-through
- Grid Analyst — Meralco capacity, demand-supply balance
- Regulatory Expert — ERC rate review cycles, stranded cost recovery
- Demand Forecaster — industrial + residential load outlook

### Context injection

Food and electricity agents receive this prefix in their system prompt:
```
[GAS CONTEXT]
Predicted price: ₱XX.XX/L (Δ +₱X.XX vs prior month)
LLM verdict: "<gas debate summary>"
Weather: rainfall Xmm (<pct>% vs seasonal norm, weighted 3-zone avg), avg temp X°C
```

### Economy Synthesizer (new)

- Model: `qwen2.5:7b`
- Runs after all three debates complete
- Input: gas verdict + food verdict + electricity verdict (full text)
- Output: 3–5 sentence macro summary covering cascade effect and household impact
- Displayed as the top banner in the bento dashboard

### Debate configuration

No new debate config UI needed — Food and Electricity debates use the same round count and mode (Standard/Swarm) chosen in Stage 2 setup. The sector selector in Stage 2 gets three checkboxes: Gas ☑ Food ☑ Electricity ☑ (all on by default).

---

## 4. UI — Bento Economy Overview

### New file: `ph_economic_ai/ui/economy_overview.py`

Self-contained `QWidget`. Added as the first tab in `main_window.py` (before existing Stage tabs).

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  MACRO SUMMARY  (Economy Synthesizer verdict — full width)      │
│  Styled label, italic, amber left-border accent                 │
└─────────────────────────────────────────────────────────────────┘
┌──────────────────┐  ┌──────────────────┐  ┌───────────────────┐
│   GAS            │  │   FOOD           │  │  ELECTRICITY      │
│   ₱72.40/L       │  │  Index: 118.4    │  │  ₱11.20/kWh      │
│   +₱2.10 ↑       │  │  +3.2% ↑        │  │  +₱0.40 ↑        │
│  [sparkline]     │  │  [sparkline]     │  │  [sparkline]      │
│  Pressure: HIGH  │  │  🌧 85mm rain   │  │  Fuel share 18%   │
└──────────────────┘  └──────────────────┘  └───────────────────┘
┌──────────────────────────────────┐  ┌────────────────────────────┐
│  GAS → FOOD influence            │  │  WEATHER SIGNAL            │
│  Bar chart: cost pass-through    │  │  Rainfall vs seasonal avg  │
│  breakdown (transport, VAT etc)  │  │  Temp strip (12-month)     │
└──────────────────────────────────┘  └────────────────────────────┘
```

### Sector cards

Each of the three sector cards contains:
- Sector name header
- Current predicted value (large, bold)
- Month-on-month delta with directional arrow
- 6-month sparkline (matplotlib, tight layout)
- One key signal line (gas: pressure band; food: rainfall reading; electricity: fuel cost share)
- Color coding: green (stable), amber (rising), red (high/critical) — matches existing pressure gauge palette

### Animation

Cards populate sequentially as each debate thread completes — Gas first, Food second, Electricity third. Macro summary banner fades in last. While a sector is pending, its card shows a pulsing "Analyzing…" placeholder.

### Integration points

- `EconomyOverviewWidget.update_gas(result)` — called by existing gas debate signal
- `EconomyOverviewWidget.update_food(result)` — called by food debate signal  
- `EconomyOverviewWidget.update_electricity(result)` — called by electricity debate signal
- `EconomyOverviewWidget.update_summary(text)` — called by Economy Synthesizer signal

---

## 5. Error Handling

| Failure | Behavior |
|---|---|
| Open-Meteo unreachable | Fall back to seasonal rainfall norms; card shows "Weather: Seasonal avg" badge |
| FAO fetch fails | Derive food index from gas+weather formula; card shows "Derived" badge |
| Food/Electricity model has insufficient data | Sector card shows "Insufficient data" and debate for that sector is skipped |
| LLM debate for a sector fails | Sector card shows ML prediction only, no verdict text; Synthesizer skips failed sectors |
| All three sectors fail | Economy Overview tab shows error state; existing Stage tabs still work |

---

## 6. Files Changed / Created

| File | Change |
|---|---|
| `ph_economic_ai/fetcher.py` | Add `_fetch_open_meteo(zones, weights)`, `_fetch_fao_food()`, `_derive_food_from_gas()`, `_fetch_electricity()` |
| `ph_economic_ai/utils/preprocessing.py` | Add `build_food_features()`, `build_electricity_features()`, `build_all_features()` |
| `ph_economic_ai/model.py` | Rename internal `train()` → `train_sector()`; keep alias |
| `ph_economic_ai/main.py` | Train 3 models; pass all to `SimMainWindow` |
| `ph_economic_ai/ui/main_window.py` | Add Economy Overview tab; wire 3 debate threads |
| `ph_economic_ai/engine/debate.py` | Add `FOOD_AGENTS`, `ELECTRICITY_AGENTS` lists |
| `ph_economic_ai/ui/economy_overview.py` | **New** — bento layout widget |
| `ph_economic_ai/tests/test_fetcher.py` | Add weather + food + electricity fetch tests |
| `ph_economic_ai/tests/test_preprocessing.py` | Add food/electricity feature builder tests |

---

## 8. Architectural Invariant: Data Separation

**Rule:** Observed data and simulated predictions must never mix.

- **Observed data** (`oil_price`, `usd_php`, `gas_price`, `rainfall_mm`, `food_price_idx`, `electricity_rate`, etc.) comes from external APIs or the cache. It is the ground truth. It is immutable once fetched.
- **Simulated predictions** (`gas_pred`, `food_pred`, `electricity_pred`) are computed in memory at inference time. They flow forward through the sector pipeline in the current session only.

**Enforcement:**
- `_save_cache()` must never receive a DataFrame that contains `gas_pred`, `food_pred`, or `electricity_pred` columns.
- `build_food_features()` and `build_electricity_features()` receive `gas_pred` as a separate scalar/array argument — not as a column already merged into `df`.
- No retraining loop may source its training labels from a prior run's predictions.

**Why this matters:** If predictions are written back as historical data, future model training will learn from its own outputs, creating recursive simulation drift that is impossible to detect or reverse. This invariant is cheap to maintain now and prevents a class of bugs that become catastrophic at scale.

---

## 7. Out of Scope

- Transport fares, sembako basket, housing utilities (can be a future phase)
- Macro feedback loop (CPI → BSP rate → FX re-simulation)
- PSA or Meralco API integration (no free real-time API exists)
- Map visualization of regional price differences
