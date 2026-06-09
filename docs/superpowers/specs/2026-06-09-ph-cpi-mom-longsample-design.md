# ph_economic_ai — MoM Nowcast Longer-Sample Confirmation (Design)

**Date:** 2026-06-09
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data science thesis)
**Builds on:** MoM CPI nowcast + driver-only ablation on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The MoM nowcast beat the best simple baseline (ARIMA, +16.2%, DM p = 0.032) but at **n = 61** months, and the driver-only edge was **suggestive but not significant** (Ridge −12% RMSE, no DM significance). Both results are sample-limited — not by CPI history (the IMF/DBnomics CPI series is long), but by the **feature history**, which was fetched with a `range='10y'` Yahoo window.

**Goal:** confirm (or revise) the MoM result on a **longer feature history** (~3× the sample) by extending the Yahoo window to `max`, then re-running the MoM nowcast and the driver-only ablation — **isolated** so existing committed results are untouched.

**Questions answered:**
1. Does the MoM nowcast still beat the best simple baseline on ~190 months?
2. Does the driver-only edge cross DM significance with ~3× the data, or stay suggestive?

**Honest framing:** a `max` window spans ~2007–2026 (GFC, 2014 oil crash, COVID), so this is a **longer, more heterogeneous** regime test — a stronger robustness check, but a changed verdict could reflect regime shifts as well as added power. Reported either way.

---

## 2. Scope

### In scope
- A longer-history feature builder (`build_long_features`, Yahoo `range='max'`) → committed `data/features_monthly_long.csv`.
- An optional `features=` parameter on `build_nowcast_frame` (default unchanged).
- A `longsample` module that runs `run_mom_nowcast` + `run_driver_only_ablation` on the long MoM frame → `mom_longsample_table.json`.
- run.py print of the short-vs-long comparison; one-line Accuracy-view note; spec record.

### Out of scope
- Re-running the full benchmark (fuel/FX/audit/YoY) on long data — only MoM + driver ablation, by decision. The main pipeline stays at its current committed state.
- Changing the headline MoM result already recorded; this is an additive confirmation.
- New modeling/methods (same panel, same verdict logic).

---

## 3. Definitions

- **Long features:** monthly Brent (`BZ=F`), USD/PHP (`PHP=X`), RBOB (`RB=F`) fetched at `range='max'`, with the RBOB→PHP gas proxy and seasonal demand computed exactly as in `build_features_csv`. Inner-join dropna determines the start (~2007).
- **Long MoM frame:** `build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom', features=long_features)` — contemporaneous oil/fx/fuel + `prev_mom` + `target` (MoM inflation), over the long overlap.
- **Confirmation verdicts:** the existing `run_mom_nowcast` (DM vs best simple baseline) and `run_driver_only_ablation` (`driver_edge` flag), applied to the long frame.

---

## 4. Architecture

```
benchmark/
├── refresh_data.py   # + build_long_features(rng='max') -> data/features_monthly_long.csv
├── nowcast.py        # build_nowcast_frame(..., features=None)  (backward compatible)
├── longsample.py     # NEW: load_long_features(); run_mom_longsample() -> dict + artifact
├── run.py            # call run_mom_longsample(); print short-vs-long comparison; report key
└── ui/accuracy_view.py  # one-line longsample note
```

### 4.1 `refresh_data.build_long_features(rng='max') -> None`
Mirror `build_features_csv` but with the configurable range, writing `data/features_monthly_long.csv` (columns: `date, oil_price, usd_php, gas_price, demand_index`). Reuses `_yahoo_monthly(ticker, rng)` (already accepts a range) and `fetcher._compute_demand`. Network one-off; the CSV is committed for offline reproducibility.

### 4.2 `nowcast.build_nowcast_frame(target_loader=None, prev_col='prev_inflation', features=None)`
Add `features=None`; when provided, use it instead of calling `_features()`. Default path unchanged (resolves `_features()` at call time, preserving existing behavior/tests).

### 4.3 `longsample.py`
- `load_long_features(csv_path=LONG_FEATURES_CSV) -> pd.DataFrame` (indexed by `date`).
- `run_mom_longsample(min_train=24, features=None) -> dict`:
  - `features` defaults to `load_long_features()`.
  - `frame = build_nowcast_frame(load_inflation_mom, 'prev_mom', features=features)`.
  - `mom = run_mom_nowcast(min_train, frame=frame)`; `abl = run_driver_only_ablation(min_train, frame=frame)`.
  - Returns `{n_long, mom: {...minus panel/calibration...}, driver_ablation: {...minus calibration...}}`.

### 4.4 Integration
- `report.py`: add `mom_longsample` key.
- `run.py`: after the driver-ablation block, call `run_mom_longsample()`, print short (from `mom_res`/`mom_abl`) vs long comparison, add `mom_longsample=` to the report, write `mom_longsample_table.json`. Guard: if `features_monthly_long.csv` is absent, record `{'verdict': 'not_run', 'reason': 'long features missing'}` and continue.
- `accuracy_view.py`: `mom_longsample_summary()` one-line note (long n, MoM verdict, driver_edge).

---

## 5. Data Flow

```
Yahoo (range=max) ─► build_long_features ─► data/features_monthly_long.csv
                                                  │
load_long_features + load_inflation_mom ─► build_nowcast_frame(features=long) ─► long MoM frame
                                                  │
                         run_mom_nowcast + run_driver_only_ablation (existing, frame=long)
                                                  │
                 mom_longsample_table.json + report 'mom_longsample' ─► accuracy_view note
```

---

## 6. Error Handling
- `features_monthly_long.csv` missing (refresh not run / network failed) → `run_mom_longsample` raises `FileNotFoundError`; `run.py` catches and records `mom_longsample: {'verdict': 'not_run', 'reason': ...}` so the main run still completes.
- Long frame shorter than `min_train + 5` (shouldn't happen with max history) → the existing `insufficient_data` guard in `run_mom_nowcast` applies.
- Yahoo `max` partial failures during refresh → `_yahoo_monthly` raises; the refresh step reports it; the committed CSV (if present from a prior run) remains usable.

---

## 7. Testing
- `build_nowcast_frame(features=...)`: with a synthetic long features DataFrame + monkeypatched `load_inflation`/loader, the frame uses the supplied features (contemporaneous), not `_features()`; columns and lagging correct.
- `run_mom_longsample(features=synthetic_long)`: returns `{n_long, mom, driver_ablation}` with the expected sub-keys; `n_long` reflects the longer frame; on a synthetic long frame with a constructed signal the MoM verdict is `beats_best_naive` (sanity that the wired path works).
- `report`/`accuracy_view`: `mom_longsample` key present and rendered.
- The real Yahoo `max` fetch is a one-off refresh step (network), not a unit test.

---

## 8. Deliverables (definition of done)
1. `build_long_features` + committed `features_monthly_long.csv`.
2. `build_nowcast_frame(features=)` (backward compatible) + `longsample.run_mom_longsample`.
3. `mom_longsample` block in `accuracy_report.json` + `mom_longsample_table.json`.
4. Accuracy-view one-line note.
5. Tests for the `features=` path and the longsample wiring.
6. Reproducible via refresh (build long features) + `python -m ph_economic_ai.benchmark.run`.

---

## 9. The contribution — measured result

Run on the committed long features (source: `artifacts/mom_longsample_table.json`),
**n = 143** months (~2007–2026, incl. GFC, 2014 oil crash, COVID).

| Metric | n = 61 (10y) | n = 143 (long) |
|---|---|---|
| MoM verdict | beats_best_naive | **beats_best_naive** |
| best method | ARIMA | ARIMA |
| skill vs best naive (random-walk) | +16.2% | **+16.3%** |
| DM p | 0.032 | **0.001** |
| driver_edge | False | **False** (Ridge 0.374 vs RW 0.413, −9.5%, not significant) |

- **Result: the MoM nowcast win HOLDS and strengthens.** Across ~2.3× the sample *and*
  a more heterogeneous regime mix, ARIMA still beats the best simple baseline (random-walk)
  by ~16%, and the DM p-value **tightens from 0.032 → 0.001** — the expected direction with
  more data. This is a robust, not fragile, positive result.
- **Driver edge: still not significant.** Even at n = 143, driver-only Ridge is directionally
  better (RMSE 0.374 vs random-walk 0.413, −9.5%) but does not clear DM significance. So the
  MoM predictability is firmly attributable to inflation's **own short-run dynamics** (ARIMA),
  not to a statistically-significant within-month driver information edge. The earlier
  underpowered driver signal did **not** resolve into significance with more data — a clean,
  honest negative on the driver-attribution question.
- **Regime-robustness note:** the verdict is unchanged despite the longer window spanning the
  GFC/2014/COVID — so the result is not an artifact of the calm 2016–2026 sub-period; it is
  stable across varied macro regimes.
- **Why it matters:** demonstrates sample-size/robustness discipline — re-testing a positive
  result on more (and more varied) data before leaning on it. The headline MoM nowcast claim
  is now durable (p = 0.001, n = 143, across regimes), with its mechanism honestly bounded
  (own-dynamics, not a significant driver edge).

---

## 10. Sources / references
- Giannone, Reichlin & Small (2008); Bańbura et al. (2013) — nowcasting.
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
- Data: DBnomics IMF IFS PH CPI; Yahoo Finance `BZ=F`/`PHP=X`/`RB=F` (max history).
