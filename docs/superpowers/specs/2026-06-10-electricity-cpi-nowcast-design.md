# ph_economic_ai — MoM Electricity-CPI Nowcast (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data-science thesis)
**Builds on:** the Food-CPI nowcast (`psa_cpi.py`, `food_nowcast.py`) and the MoM nowcast pipeline on `master`.

---

## 1. Problem & Goal

Electricity is the one sector the app forecasts but the benchmark never validated. This applies the same predictability audit used for fuel, transport, and food to **electricity** — closing the map.

**Goal:** nowcast month-on-month **Electricity** CPI inflation before its PSA release, using free energy-commodity prices + FX as within-month predictors, and test honestly (DM vs the strongest naive baseline + a driver-only ablation + the preliminary-data robustness re-test) whether it is nowcastable.

**Honest expectation (stated, not assumed):** PH retail electricity (Meralco) is heavily regulated and smoothed — generation-charge pass-through is deferred and ERC-approved over months — and Henry Hub `NG=F` is a rough proxy for PH (Malampaya/LNG) gas. So the within-month *driver* edge is expected to be **null** (regulatory lag breaks the within-month link), with MoM possibly forecastable via its own dynamics. The verdict is reported either way; this completes the map and is not expected to be a bold positive.

---

## 2. Scope

### In scope
- A free PSA gold loader for the `04.5.1 - Electricity` CPI sub-index (committed CSV), via the existing generalized PX-Web fetch.
- A free Yahoo energy predictor panel (Brent + natural gas + FX), committed CSV.
- An Electricity-MoM nowcast reusing `run_mom_nowcast` + `run_driver_only_ablation` + the robustness re-test.
- Report key + `run.py` wiring + accuracy-view note.

### Out of scope
- Changes to `nowcast.py` (runners reused unchanged: all non-`target` cols are features; `run_driver_only_ablation` drops `prev_mom`).
- WESM/Meralco spot data (not freely/reliably available); the swarm-memory feature.

---

## 3. Data (all free)

### 3.1 Gold target — PSA OpenSTAT
- Same PX-Web tables as transport/food: backcast `0012M4ACP28.px` (1994–2017) + current `0012M4ACP22.px` (2018–present), merged.
- Commodity group **`04.5.1 - Electricity`** (value confirmed present in the current table; the generalized `_resolve_commodity_id` matches the COICOP prefix `04.5.1` → "04.5.1 - Electricity (ND)", not the more granular "04.5.1.0"). Geolocation = Philippines.
- Output: committed `benchmark/data/psa_electricity_cpi_monthly.csv` (`date` `YYYY-MM`, `electricity_cpi` index, 2018 = 100). MoM transform via `targets.cpi_to_mom` is the target.
- If the 1994–2017 backcast table lacks `04.5.1`, the series begins 2018 (~89 months) — usable; the predictor overlap (Brent from 2007) is the binding constraint regardless.

### 3.2 Predictors (new, free)
Monthly Yahoo Finance series, `interval='max'`: Brent oil `BZ=F`, natural gas `NG=F`, and USD/PHP `PHP=X`. Inner-join dropna → committed `benchmark/data/electricity_features_monthly.csv` with columns `date, oil_price, natgas, usd_php`. Brent begins ~2007-08, so the electricity-CPI × features overlap is ~2007–2026 (~170+ months).

### 3.3 Provenance & reproducibility
PSA (official, citable) for the target; Yahoo for predictors. Committed CSVs freeze both; live fetches are one-off refresh steps (network), not run during the backtest.

---

## 4. Architecture

```
benchmark/
├── psa_cpi.py             # + fetch_electricity_cpi() (COICOP '04.5.1'),
│                          #   load_electricity_cpi(), load_electricity_mom()
├── refresh_data.py        # + build_electricity_features() -> data/electricity_features_monthly.csv
├── electricity_nowcast.py # NEW: build frame [oil,natgas,fx,prev_mom,target] -> run_mom_nowcast +
│                          #   run_driver_only_ablation + robustness -> dict
├── nowcast.py             # reused UNCHANGED
├── report.py              # + 'electricity_nowcast' key
├── run.py                 # run electricity_nowcast; print; record; write artifact
└── ui/accuracy_view.py    # electricity_nowcast_summary() note
```

### 4.1 `psa_cpi.py`
- Add `ELECTRICITY_CSV = HERE / 'data' / 'psa_electricity_cpi_monthly.csv'`.
- `fetch_electricity_cpi(out_csv=ELECTRICITY_CSV)`: merge `_fetch_px_table(BACKCAST, 1994, '04.5.1')` + `_fetch_px_table(CURRENT, 2018, '04.5.1')` (current wins), write `electricity_cpi` column. Mirrors `fetch_food_cpi`. `< 50` rows → raise (electricity backcast may be shorter than food/transport; guard set lower).
- `load_electricity_cpi(csv) -> Series`, `load_electricity_mom(csv) -> Series` (reuse `cpi_to_mom`).

### 4.2 `refresh_data.build_electricity_features(rng='max')`
Fetch `{'BZ=F':'oil_price', 'NG=F':'natgas', 'PHP=X':'usd_php'}` via `_yahoo_monthly`, inner-join dropna, write `electricity_features_monthly.csv` (cols `date, oil_price, natgas, usd_php`). Network one-off; CSV committed.

### 4.3 `electricity_nowcast.py`
- `load_electricity_features(csv=ELECTRICITY_FEATURES_CSV) -> DataFrame`.
- `run_electricity_nowcast(min_train=24, features=None, prelim_months=6) -> dict`:
  - Build the frame directly: drivers `[oil (from oil_price), natgas, fx (from usd_php)]` + `target = load_electricity_mom()` + `prev_mom = target.shift(1)`; `dropna()`.
  - `run_mom_nowcast(min_train, frame=frame)` + `run_driver_only_ablation(min_train, frame=frame)`.
  - Robustness: re-run the driver ablation on `frame.iloc[:-prelim_months]`; record `robust` + `driver_edge_robust`.
  - Return `{n, mom, driver_ablation, driver_edge, robust, driver_edge_robust}` (heavy internals dropped), identical shape to `food_nowcast`.

### 4.4 Integration
- `report.build_report(..., electricity_nowcast=None)` + `REQUIRED_KEYS`.
- `run.py`: call `run_electricity_nowcast`, print `mom` verdict + `driver_edge_robust`, record, write `electricity_nowcast_table.json`. Guard: missing gold/features → `{'verdict':'not_run', 'reason':...}`.
- `accuracy_view.electricity_nowcast_summary()`: one-line note reporting the robust verdict, flagging any full-sample `True` as a preliminary-data artifact (same shape as `food_nowcast_summary`).

---

## 5. Data Flow
```
PSA PX-Web (04.5.1 Electricity) ─► fetch_electricity_cpi ─► data/psa_electricity_cpi_monthly.csv ─┐
Yahoo Brent+natgas+FX ─► build_electricity_features ─► data/electricity_features_monthly.csv ─────┤
                                                                                                  ▼
            electricity_nowcast: build frame [oil,natgas,fx,prev_mom,target]
                                                                                                  │
              run_mom_nowcast + run_driver_only_ablation (+ robust re-test)
                                                                                                  │
              electricity_nowcast_table.json + report 'electricity_nowcast' ─► view note
```

## 6. Error Handling
- PSA `04.5.1` label not found → `fetch_electricity_cpi` raises listing available labels; committed CSV (if present) remains usable.
- Missing gold/features at run time → `run.py` records `electricity_nowcast: not_run` and continues.
- Short overlap → existing `insufficient_data` guard.
- Reserved frame names `prev_mom`/`target` respected; driver columns are free-form.

## 7. Testing
- `psa_cpi`: `load_electricity_mom` yields MoM % from a synthetic `psa_electricity_cpi_monthly.csv`.
- `electricity_nowcast.run_electricity_nowcast(features=synthetic, prelim_months=6)` with monkeypatched `load_electricity_mom` → returns `{n, mom, driver_ablation, driver_edge, robust, driver_edge_robust}`; `robust['n'] < n`.
- `report`/`accuracy_view`: `electricity_nowcast` key present and rendered.
- After adding `fetch_electricity_cpi`, re-running `fetch_transport_cpi`/`fetch_food_cpi` still reproduces their committed gold identically (the generalized fetch is unchanged; only a new wrapper is added — verified once during the data fetch task).
- Live PSA + Yahoo fetches are one-off refreshes (network), not unit tests.

## 8. Deliverables (definition of done)
1. `psa_cpi` electricity fetch + loaders + committed `psa_electricity_cpi_monthly.csv`.
2. `build_electricity_features` + committed `electricity_features_monthly.csv`.
3. `electricity_nowcast.py` reusing the runners + robustness re-test.
4. `electricity_nowcast` block in `accuracy_report.json` + `electricity_nowcast_table.json`.
5. Accuracy-view note.
6. Tests per §7; full suite green.
7. Reproducible via refresh + `python -m ph_economic_ai.benchmark.run`.

## 9. The contribution — measured result (a robust positive)

Run on the committed PSA gold + energy panel (`artifacts/electricity_nowcast_table.json`), **n = 151** backtest months (2007–2026).

| Test | Verdict | best | skill vs best naive | DM p |
|---|---|---|---|---|
| Full nowcast (drivers + own-lag) | beats_best_naive | Ridge | +26.6% | 0.0005 |
| Driver-only ablation, full (n = 151) | **beats_best_naive** | Ridge | **+28.3%** | **0.0011** |
| Driver-only ablation, robust (drop 6 preliminary, n = 145) | **beats_best_naive** | Ridge | **+28.4%** | **0.0012** |

**Sub-sample stability (additional robustness, beyond the trailing-window check):**

| Window | driver_edge | skill | DM p |
|---|---|---|---|
| ≤ 2023-12 (n = 129) | True | +26.3% | 0.006 |
| First half (~2007–2016, n = 63) | True | +29.9% | 0.020 |
| Second half (~2016–2026, n = 64) | True | +28.7% | 0.035 |

- **Result: a genuine, robust positive.** Within-month energy prices (Brent, natural gas, FX) **significantly nowcast PH electricity inflation** — Ridge beats the strongest naive baseline by ~28% (DM p ≈ 0.001). The edge is the **first driver edge in the audit that survives every stress test**: it holds after dropping the preliminary tail (`driver_edge_robust = True`), in *both* halves of the sample, and at earlier cutoffs — it is not period-specific and not a preliminary-data artifact.
- **Why (the prior was wrong, honestly):** the design expected regulation to *smooth away* the within-month signal. In fact the Meralco generation charge is a **formulaic, near-deterministic pass-through of fuel costs**, observable within the month before the PSA print — so regulation makes electricity *more* nowcastable, not less. This is a legitimate information-timing nowcast, not market-beating.
- **Place in the map:** electricity is the **second genuinely useful nowcast** (alongside headline MoM inflation) and the **only sector with a robustly-significant within-month *driver* edge** — contrasting cleanly with transport (a spurious edge, caught) and food (a clean null). The driver question now has a confirmed positive, a rejected false positive, and a confirmed null.
- **Honesty notes:** a *nowcast* (information timing), not market-beating; Henry Hub `NG=F` is an imperfect proxy for PH (Malampaya/LNG) gas, yet the edge is strong and stable; gold is official PSA data (2018 mean = 100.0); recent PSA prints are preliminary (the trailing robustness window); the 2007-based overlap spans the GFC, 2014 oil crash, and COVID.

## 10. Sources / references
- PSA OpenSTAT PX-Web: `DB/2M/PI/CPI/2018NEW/{0012M4ACP28,0012M4ACP22}.px`, commodity group `04.5.1 - Electricity`.
- Predictors: Yahoo Finance `BZ=F`, `NG=F`, `PHP=X`.
- Nowcasting: Giannone, Reichlin & Small (2008); Bańbura et al. (2013). Significance: Diebold–Mariano (1995); HLN (1997).
