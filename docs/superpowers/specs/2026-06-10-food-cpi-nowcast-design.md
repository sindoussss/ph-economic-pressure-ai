# ph_economic_ai — MoM Food-CPI Nowcast (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data-science thesis)
**Builds on:** the Transport-CPI nowcast (`psa_cpi.py`, `transport_nowcast.py`) and the MoM nowcast pipeline on `master`.

---

## 1. Problem & Goal

The Transport-CPI experiment tested a *fuel-driven* component of inflation; this one tests a *food-commodity-driven* component. **Food & non-alcoholic beverages** is the largest single contributor to Philippine headline inflation, and global food-commodity prices (grains, oilseeds) are observable within the month, before the PSA release.

**Goal:** nowcast month-on-month **Food** CPI inflation before its official release, using free global agri-commodity prices + FX as within-month predictors, and test honestly (DM vs the strongest naive baseline + a driver-only ablation, with the same preliminary-data robustness re-test as transport) whether food inflation is nowcastable.

**Honest stance:** Philippine food prices are heavily local (fish, vegetables, import-controlled rice), so global grains are an imperfect driver; an *efficient* verdict is a likely and legitimate outcome. The contribution is a fair, mechanistically-motivated test and another point on the predictability map — not a guaranteed positive. The verdict is reported either way.

---

## 2. Scope

### In scope
- Generalize the existing PSA PX-Web fetch to resolve any commodity group by COICOP code prefix; add a Food gold loader (committed CSV).
- A free Yahoo agri-futures + oil + FX predictor panel (committed CSV).
- A Food-MoM nowcast that builds its frame directly and reuses `run_mom_nowcast` + `run_driver_only_ablation`, with the trailing-preliminary robustness re-test.
- Report key + `run.py` wiring + accuracy-view note.

### Out of scope
- Changes to `nowcast.py` (the runners are reused unchanged: `run_mom_nowcast` uses all non-`target` columns as features; `run_driver_only_ablation` drops `prev_mom` by name).
- Rice-only or other sub-components (possible later targets).
- The swarm/LLM path; live forward updates beyond the committed backtest.

---

## 3. Data (all free)

### 3.1 Gold target — PSA OpenSTAT
- Same PX-Web tables as transport: backcast `0012M4ACP28.px` (1994–2017) + current `0012M4ACP22.px` (2018–present), merged.
- Commodity group **`01 - FOOD AND NON-ALCOHOLIC BEVERAGES`** (division level), resolved per table by matching the valueText whose COICOP code prefix is `01 -` (transport uses `07 -`). Geolocation = Philippines.
- Output: committed `benchmark/data/psa_food_cpi_monthly.csv` (`date` `YYYY-MM`, `food_cpi` index, 2018 = 100). MoM transform via `targets.cpi_to_mom` is the target.

### 3.2 Predictors (new, free)
Monthly Yahoo Finance series, `interval='max'`: rough rice `ZR=F`, wheat `ZW=F`, corn `ZC=F`, soybean `ZS=F`, Brent oil `BZ=F`, and USD/PHP `PHP=X`. Inner-join dropna → committed `benchmark/data/food_features_monthly.csv` with columns `date, rice, wheat, corn, soybean, oil_price, usd_php`. Agri futures begin ~2000, so the food-CPI×features overlap is ~2000–2026 (~300 months).

### 3.3 Provenance & reproducibility
PSA (official, citable) for the target; Yahoo for predictors. Committed CSVs freeze both; live fetches are one-off refresh steps (network), not run during the backtest.

---

## 4. Architecture

```
benchmark/
├── psa_cpi.py        # generalize _fetch_px_table to resolve commodity by COICOP prefix;
│                     #   + fetch_food_cpi(), load_food_cpi(), load_food_mom()
├── refresh_data.py   # + build_food_features(rng='max') -> data/food_features_monthly.csv
├── food_nowcast.py   # NEW: build food frame directly; run_mom_nowcast + driver ablation
│                     #   + robustness re-test -> {n, mom, driver_ablation, driver_edge,
│                     #     robust, driver_edge_robust}
├── nowcast.py        # reused UNCHANGED
├── report.py         # + 'food_nowcast' key
├── run.py            # run food_nowcast; print; record; write artifact
└── ui/accuracy_view.py  # food_nowcast_summary() note (robust verdict + artifact flag)
```

### 4.1 `psa_cpi.py` generalization
- Refactor `_fetch_px_table(url, first_year)` → `_fetch_px_table(url, first_year, coicop_prefix)`: resolve the commodity value id whose label (trimmed) starts with `f'{coicop_prefix} -'` (e.g. `'01 -'`, `'07 -'`), instead of a hardcoded id. A helper `_resolve_commodity_id(meta, coicop_prefix)` lists available labels in its error if no match.
- `fetch_transport_cpi` calls it with `coicop_prefix='07'` (behaviour unchanged — must reproduce the committed transport gold).
- Add `fetch_food_cpi(out_csv=FOOD_CSV)` with `coicop_prefix='01'`, writing `food_cpi` column to `psa_food_cpi_monthly.csv`.
- Add `load_food_cpi(csv=FOOD_CSV) -> Series` and `load_food_mom(csv=FOOD_CSV) -> Series` (reuse `cpi_to_mom`), mirroring the transport loaders.

### 4.2 `refresh_data.build_food_features(rng='max')`
Fetch the six Yahoo series via `_yahoo_monthly`, inner-join dropna, write `food_features_monthly.csv`. Network one-off; CSV committed.

### 4.3 `food_nowcast.py`
- `FOOD_FEATURES_CSV` loader → DataFrame indexed `YYYY-MM`.
- `run_food_nowcast(min_train=24, features=None, prelim_months=6) -> dict`:
  - `feats` defaults to the committed food features.
  - Build the frame directly: drivers = `[rice, wheat, corn, soybean, oil_price→oil, usd_php→fx]` joined with `target = load_food_mom()` and `prev_mom = target.shift(1)`; `dropna()`. (Driver column names are free-form; only `prev_mom` and `target` are reserved.)
  - `mom = run_mom_nowcast(min_train, frame=frame)`; `abl = run_driver_only_ablation(min_train, frame=frame)`.
  - Robustness: re-run `run_driver_only_ablation` on `frame.iloc[:-prelim_months]`; record `robust` block + `driver_edge_robust` (canonical verdict), exactly as `transport_nowcast`.
  - Return `{n, mom (slim), driver_ablation (slim), driver_edge, robust, driver_edge_robust}`.

### 4.4 Integration
- `report.build_report(..., food_nowcast=None)` + `REQUIRED_KEYS`.
- `run.py`: call `run_food_nowcast`, print `mom` verdict + `driver_edge_robust`, record, write `food_nowcast_table.json`. Guard: missing gold/features → `{'verdict':'not_run','reason':...}`.
- `accuracy_view.food_nowcast_summary()`: one-line note reporting the **robust** verdict, flagging any full-sample `True` as a preliminary-data artifact (same shape as `transport_nowcast_summary`).

---

## 5. Data Flow
```
PSA PX-Web (01 - FOOD) ─► fetch_food_cpi ─► data/psa_food_cpi_monthly.csv ─┐
Yahoo agri+oil+FX ─► build_food_features ─► data/food_features_monthly.csv ─┤
                                                                            ▼
                 food_nowcast: build frame [grains,oil,fx,prev_mom,target]
                                                                            │
              run_mom_nowcast + run_driver_only_ablation (+ robust re-test)
                                                                            │
              food_nowcast_table.json + report 'food_nowcast' ─► view note
```

## 6. Error Handling
- PSA fetch failure / `01 -` label not found → `fetch_food_cpi` raises listing available commodity labels. Committed CSV (if present) remains usable.
- Missing gold or features CSV at run time → `run.py` records `food_nowcast: not_run` and continues.
- Short overlap (agri futures start ~2000) → handled by the existing `insufficient_data` guard.
- Reserved-name collision: driver columns must avoid `prev_mom`/`target`; the builder names the prev column `prev_mom` and the target `target` explicitly.

## 7. Testing
- `_resolve_commodity_id` / generalized fetch: a synthetic metadata dict with several commodity labels → resolves `01 -` to the food id and `07 -` to transport; raises with the label list when absent.
- `load_food_mom`: MoM % from a synthetic `psa_food_cpi_monthly.csv`.
- `run_food_nowcast(features=synthetic, prelim_months=6)` with monkeypatched `load_food_mom` → returns `{n, mom, driver_ablation, driver_edge, robust, driver_edge_robust}`; `robust['n'] < n`; verdict keys present.
- `report`/`accuracy_view`: `food_nowcast` key present and rendered.
- Live PSA + Yahoo fetches are one-off refreshes (network), not unit tests. `fetch_transport_cpi` must still reproduce the committed transport gold after the refactor (verified by re-running it and diffing row count/2018 mean).

## 8. Deliverables (definition of done)
1. Generalized `psa_cpi` fetch + `fetch_food_cpi`/`load_food_cpi`/`load_food_mom` + committed `psa_food_cpi_monthly.csv`.
2. `build_food_features` + committed `food_features_monthly.csv`.
3. `food_nowcast.py` reusing the runners + robustness re-test.
4. `food_nowcast` block in `accuracy_report.json` + `food_nowcast_table.json`.
5. Accuracy-view note.
6. Tests per §7; transport gold unchanged after the refactor.
7. Reproducible via refresh + `python -m ph_economic_ai.benchmark.run`.

## 9. The contribution — to be filled from the real run
- **Result:** on n = [n] months, Food MoM nowcast is [verdict] (best [m], skill [x], DM p [p]); driver-only edge full-sample = [bool] ([skill]/p [p]); **robust** (drop [k] preliminary months, n = [nr]) `driver_edge_robust` = [bool].
- **Interpretation:** [if robust True] global food-commodity prices significantly nowcast PH food inflation ahead of the official figure — a genuine within-month edge. [if efficient/not robust] PH food inflation is efficient at this horizon and/or the apparent edge does not survive the preliminary-data check — consistent with the predictability map and the strongly-local nature of PH food prices.
- **Honesty notes:** a *nowcast* (information timing); global grains are an imperfect proxy for local PH food prices; gold is official PSA data faithfully loaded; recent PSA prints are preliminary (the robustness window); 1994/2000-based history spans multiple regimes.

## 10. Sources / references
- PSA OpenSTAT PX-Web: `DB/2M/PI/CPI/2018NEW/{0012M4ACP28,0012M4ACP22}.px`, commodity group `01 - FOOD AND NON-ALCOHOLIC BEVERAGES`.
- Predictors: Yahoo Finance `ZR=F`, `ZW=F`, `ZC=F`, `ZS=F`, `BZ=F`, `PHP=X`.
- Nowcasting: Giannone, Reichlin & Small (2008); Bańbura et al. (2013). Significance: Diebold–Mariano (1995); HLN (1997).
