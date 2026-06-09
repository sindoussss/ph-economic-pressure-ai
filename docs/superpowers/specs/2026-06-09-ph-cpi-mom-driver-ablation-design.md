# ph_economic_ai — MoM Nowcast Driver-Only Ablation (Design)

**Date:** 2026-06-09
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data science thesis)
**Builds on:** the MoM CPI nowcast (`2026-06-08-ph-cpi-mom-nowcast-design.md`) on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The MoM nowcast beat the best simple baseline (ARIMA, +16.2% skill, DM p = 0.032). But the winner is **ARIMA, a univariate method** — so the edge may come from MoM inflation's own short-run dynamics, not from the contemporaneous within-month drivers (oil/FX/fuel) that justify calling it a *nowcast*. Driver-based Ridge also beat naive, but that is mixed with the `prev_mom` own-lag feature it also receives.

**Goal:** isolate the **pure driver information edge** with an ablation — drop the own-lag (`prev_mom`), restrict candidates to driver-based regressors (Ridge, HGB), and test whether the contemporaneous drivers *alone* beat the strongest simple baseline.

**Claim under test:**
> Using only month-_t_ oil/FX/fuel (no own-lag, no univariate dynamics), a regressor beats the best simple baseline (best of random-walk / seasonal-naive / drift), Diebold-Mariano significant.

- **If confirmed:** the nowcast has a genuine within-month *information* edge, not just time-series dynamics — strengthens the headline result.
- **If not:** the MoM win is attributable to ARIMA's univariate dynamics; the drivers alone add nothing — an honest qualification of the headline result.

Either outcome is a sharp, defensible refinement, not an inflation of the claim.

---

## 2. Scope

### In scope
- An optional `methods=` parameter on `run_mom_nowcast` (default unchanged → backward compatible).
- A `run_driver_only_ablation(min_train, frame=None)` that drops `prev_mom` and restricts candidates to `{ridge, hgb}` against the baseline pool.
- Report key `mom_driver_ablation`; run + Accuracy-view + spec integration.

### Out of scope
- New data (reuses committed `ph_cpi_monthly.csv` + `features_monthly.csv`).
- Changing the headline MoM nowcast logic (this is an additive ablation).
- The written manuscript.

---

## 3. Definitions

- **Ablation feature set:** `{oil, fx, fuel}` (contemporaneous month-_t_), with `prev_mom` **removed**.
- **Candidate methods:** `{ridge, hgb}` (driver-based regressors). ARIMA/ETS are excluded because they use the target's own history and would re-introduce the dynamics edge being isolated.
- **Baseline pool:** `{random_walk, seasonal_naive, drift}` (unchanged; the bar to beat).
- **Verdict (reuses `mom_verdict`):** `beats_best_naive` if a candidate has lower RMSE than the best baseline **and** is DM-significant (p < 0.05) against it; else `no_better_than_naive`.
- **Interpretation label:** the result is recorded as confirming (`driver_edge_confirmed`) or qualifying (`driver_edge_absent`) the headline MoM win — derived directly from the verdict.

---

## 4. Architecture

Minimal, additive; reuses the entire MoM machinery.

```
benchmark/
├── nowcast.py    # + methods= param on run_mom_nowcast; + run_driver_only_ablation()
├── report.py     # + 'mom_driver_ablation' key
├── run.py        # run the ablation; print + record
└── ui/accuracy_view.py  # + one-line ablation note in the MoM summary area
```

### 4.1 `run_mom_nowcast(min_train, baseline_pool, frame=None, methods=PANEL_METHODS)`
Add `methods=PANEL_METHODS` (default preserves current behavior). The loop backtests exactly `methods`; `mom_verdict` then sees only those methods, so restricting `methods` to `{random_walk, seasonal_naive, drift, ridge, hgb}` makes `{ridge, hgb}` the only non-baseline candidates. No other change.

### 4.2 `run_driver_only_ablation(min_train, frame=None) -> dict`
```
if frame is None: frame = build_nowcast_frame(load_inflation_mom, 'prev_mom')
driver_frame = frame.drop(columns=['prev_mom'])           # features -> oil, fx, fuel
res = run_mom_nowcast(min_train, frame=driver_frame,
                      methods=['random_walk', 'seasonal_naive', 'drift', 'ridge', 'hgb'])
res['driver_edge'] = (res['verdict'] == 'beats_best_naive')
return res
```
Returns the standard MoM result dict plus a boolean `driver_edge`. `best_method` will be `ridge`/`hgb` if confirmed, else the best baseline.

### 4.3 Integration
- `report.py`: add `mom_driver_ablation` key (the ablation dict, calibration dropped to keep it compact).
- `run.py`: after the MoM nowcast block, call `run_driver_only_ablation(MIN_TRAIN)`, print `driver_edge` + best_method + skill + DM p, add `mom_driver_ablation=` to the report, write `mom_driver_ablation_table.json`.
- `accuracy_view.py`: extend the MoM area with a one-line ablation note (`mom_driver_ablation_summary()`): "driver-only (no own-lag): [confirmed/absent] — ...".

---

## 5. Data Flow

```
ph_cpi_monthly.csv + features_monthly.csv
        │
        ▼
build_nowcast_frame(load_inflation_mom, 'prev_mom') ─► drop prev_mom ─► {oil,fx,fuel,target}
        │
        ▼
run_mom_nowcast(methods={rw,seasonal,drift,ridge,hgb})  ─► mom_verdict (DM vs best baseline)
        │
        ▼
report 'mom_driver_ablation' + mom_driver_ablation_table.json ─► accuracy_view note
```

---

## 6. Error Handling
- Driver frame shorter than `min_train + 5` → `run_mom_nowcast` returns `{verdict: 'insufficient_data', n}`; `run_driver_only_ablation` sets `driver_edge=False` and run.py records it.
- `prev_mom` absent from the supplied frame (e.g. a test frame already without it) → guard with `frame.drop(columns=['prev_mom'], errors='ignore')`.

---

## 7. Testing
- `run_mom_nowcast` with a restricted `methods` list → backtests only those; `mom_verdict` candidate set = methods minus baseline pool (e.g. `{ridge, hgb}`). Verify a row exists per requested method and none for omitted ones (no `arima`/`ets` in `rmse_by_method`).
- `run_driver_only_ablation` on a synthetic frame where MoM = f(contemporaneous `fuel`) with **no own-lag signal** → `driver_edge is True`, best_method in `{ridge, hgb}`.
- Control: synthetic frame where MoM is pure AR(1) + noise drivers → `driver_edge is False` (drivers alone don't beat naive).
- `errors='ignore'` drop: passing a frame without `prev_mom` does not raise.
- `report`/`accuracy_view`: `mom_driver_ablation` key present and rendered.

---

## 8. Deliverables (definition of done)
1. `methods=` param on `run_mom_nowcast` (backward compatible) + `run_driver_only_ablation` with `driver_edge` flag.
2. `mom_driver_ablation` block in `accuracy_report.json` + `mom_driver_ablation_table.json`.
3. Accuracy-view one-line ablation note.
4. Tests incl. the driver-edge-present and driver-edge-absent synthetic cases.
5. Reproducible via `python -m ph_economic_ai.benchmark.run`; no new data.

---

## 9. The contribution — measured result

Run on committed data (source: `artifacts/mom_driver_ablation_table.json`), n = 61 months.

- **Result: the driver-only ablation does NOT confirm a significant within-month driver edge.**
  `driver_edge = False`, verdict `no_better_than_naive`. With `prev_mom` removed and only
  driver regressors competing, the best baseline (random-walk, RMSE 0.453) is not beaten at
  DM significance. **However**, driver-only **Ridge posts the lowest RMSE of all methods
  (0.399, ~12% below random-walk)** — directionally the contemporaneous drivers help, but
  the edge is **not statistically significant at n = 61** (DM gate not cleared).
- **Interpretation (the honest, narrower reading):** the headline MoM nowcast win
  (ARIMA, +16.2%, DM p = 0.032) is attributable mainly to MoM inflation's **own short-run
  dynamics**, not to a *significant* contemporaneous-driver information edge. The drivers
  carry a **suggestive but underpowered** signal (Ridge 0.399 < 0.453) — consistent with a
  real-but-small information edge that a larger sample might confirm.
- **Why it matters:** demonstrates ablation discipline — separating a *time-series-dynamics*
  edge from an *information* edge. The ablation **prevented overclaiming** "the drivers
  nowcast inflation": the defensible statement is "MoM inflation is predictable via its own
  dynamics; the within-month driver edge is suggestive (−12% RMSE) but not significant at
  this sample size." That precision is exactly what distinguishes a credible nowcasting
  claim from a hand-wave.

---

## 10. Sources / references
- Giannone, Reichlin & Small (2008); Bańbura et al. (2013) — nowcasting and the value of timely information.
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
- Data: DBnomics IMF IFS PH CPI; Yahoo Finance (oil, USD/PHP); DOE/RBOB fuel proxy.
