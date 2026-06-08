# ph_economic_ai — Month-over-Month CPI Nowcasting (Design)

**Date:** 2026-06-08
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data science thesis)
**Builds on:** the YoY CPI nowcasting work (`2026-06-08-ph-cpi-nowcasting-design.md`) on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The YoY CPI nowcaster did **not** beat naive — because year-on-year inflation overlaps 11 of 12 months, making "last month's YoY" mechanically unbeatable. The documented next experiment is **month-over-month (MoM) inflation**, where the within-month signal (fuel/FX shocks) actually drives the print and naive persistence is weak.

**Goal / claim:**
> A MoM-CPI nowcaster estimates Philippine month-over-month inflation *before PSA's release* from intra-month-observable drivers, and beats the **strongest simple baseline** (the best of random-walk / seasonal-naive / drift), Diebold-Mariano significant. This is the honest numeric "yes": a meaningful win against a *strong* benchmark, not a hollow win against a weak one.

**Why the strong-baseline bar matters:** MoM inflation is noisy and strongly seasonal (food/holiday calendar effects), so plain random-walk ("last MoM") is a weak benchmark that almost anything beats. A defensible result must beat the **best** simple baseline — for CPI MoM that is typically seasonal-naive. The verdict is gated on DM significance against *that* baseline.

---

## 2. Scope

### In scope
- MoM inflation target (`targets.cpi_to_mom`, `load_inflation_mom`).
- A generalized nowcast frame (`build_nowcast_frame(target_loader=...)`) reused for MoM.
- A MoM nowcast runner (`run_mom_nowcast`) that backtests the full panel and assigns a verdict via **DM against the best simple baseline** (not random walk).
- A pure, unit-testable verdict helper (`mom_verdict`).
- Report + run + figure + Accuracy-view integration (`nowcast_mom`).

### Out of scope
- New data (reuses committed `ph_cpi_monthly.csv` + `features_monthly.csv`).
- Other targets / horizons.
- The written thesis manuscript.

---

## 3. Definitions

- **Target:** `mom_t = (CPI_t / CPI_{t-1} - 1) * 100` (month-over-month inflation %).
- **Eligible features (same integrity rule as YoY nowcast):** contemporaneous month-_t_ `oil`, `fx`, `fuel` (all published before CPI_t's release) + `prev_mom` (= `mom_{t-1}`, already published). No same-month CPI-derived feature except the lagged `prev_mom`.
- **Baseline pool:** `{random_walk, seasonal_naive, drift}` — all already in the forecaster panel. `random_walk` = last MoM; `seasonal_naive` = MoM 12 months ago; `drift` = last MoM + mean step.
- **Best simple baseline:** the pool member with the lowest backtest RMSE.
- **Model methods:** `{arima, ets, ridge, hgb}` (panel minus the baseline pool).
- **Verdict:**
  - `beats_best_naive` if any model method has RMSE < best-baseline RMSE **and** Diebold-Mariano of (model loss vs best-baseline loss) is significant (p < 0.05, model better).
  - else `no_better_than_naive`.

---

## 4. Architecture

Reuses the existing benchmark machinery; the only genuinely new logic is the DM-vs-best-baseline verdict.

```
benchmark/
├── targets.py    # + cpi_to_mom(), load_inflation_mom()
├── nowcast.py    # generalize build_nowcast_frame(target_loader, target_name);
│                 # + run_mom_nowcast(min_train, baseline_pool); + mom_verdict(...)
├── report.py     # + 'nowcast_mom' key
├── figures.py    # reuse plot_nowcast (actual vs nowcast vs best-baseline)
├── run.py        # run MoM nowcast; write nowcast_mom block + figure
└── ui/accuracy_view.py  # + nowcast_mom panel
   (reuses)       # walk_forward, forecasters, significance.diebold_mariano, conformal
```

### 4.1 `targets.cpi_to_mom(cpi_index) -> pd.Series`
`(cpi / cpi.shift(1) - 1) * 100`, dropna. `load_inflation_mom(csv_path=CPI_CSV)` reads the committed CPI index and returns the MoM series.

### 4.2 `nowcast.build_nowcast_frame(target_loader=load_inflation, target_name='inflation')`
Generalize the existing builder: `prev_<target_name>` column from the chosen target; contemporaneous `oil/fx/fuel`; `target` column. The YoY caller keeps the default; the MoM caller passes `target_loader=load_inflation_mom, target_name='mom'`. Existing YoY behavior unchanged (default args).

### 4.3 `nowcast.mom_verdict(rmse_by_method, loss_by_method, baseline_pool) -> dict`
Pure function (no I/O), unit-testable:
- `best_naive` = `min(baseline_pool, key=rmse_by_method.get)`.
- candidates = model methods with `rmse < rmse[best_naive]`.
- for each candidate, `diebold_mariano(loss[cand], loss[best_naive])`; significant if `p < 0.05` and `dm_stat < 0` (candidate lower loss).
- returns `{verdict, best_method, best_naive, best_skill_vs_naive, dm_p}` where `best_skill_vs_naive = 1 - rmse[best_method]/rmse[best_naive]` and, if no candidate qualifies, `best_method = best_naive`, verdict `no_better_than_naive`.

### 4.4 `nowcast.run_mom_nowcast(min_train, baseline_pool=('random_walk','seasonal_naive','drift'), frame=None) -> dict`
- Build the MoM frame (`build_nowcast_frame(load_inflation_mom, 'mom')`) unless `frame` supplied.
- If `len(frame) < min_train + 5` → `{verdict: 'insufficient_data', n}`.
- For each of the 7 methods: `walk_forward` → store `rmse` and the per-step squared-error `loss` array (aligned across methods on the same backtest indices).
- `mom_verdict(...)`; conformal calibration on the chosen method's residuals.
- Return `{verdict, best_method, best_naive, best_skill_vs_naive, dm_p, n, calibration, rmse_by_method}`.

### 4.5 Integration
- `report.py`: add `nowcast_mom` key (the run dict, heavy arrays dropped).
- `run.py`: call `run_mom_nowcast(MIN_TRAIN)`, print the verdict + which baseline was the bar, add `nowcast_mom=` to the report, write `nowcast_mom_table.json` + a figure (actual vs nowcast vs best-baseline).
- `accuracy_view.py`: `nowcast_mom_summary()` one-line panel.

---

## 5. Data Flow

```
ph_cpi_monthly.csv ─► cpi_to_mom() ─► mom_t ─┐
features_monthly.csv ─► oil_t, fx_t, fuel_t ─┤
                                             ▼
        build_nowcast_frame(load_inflation_mom, 'mom')  (contemporaneous + prev_mom)
                                             ▼
        run_mom_nowcast: per-method walk_forward (rmse + loss) ─► mom_verdict (DM vs best simple baseline)
                                             ▼
        report 'nowcast_mom' + nowcast_mom_table.json + figure ─► accuracy_view panel
```

---

## 6. Error Handling
- Frame shorter than `min_train + 5` → `{verdict: 'insufficient_data', n}` (no raise).
- Missing CPI CSV → `load_inflation_mom` raises `FileNotFoundError`; `run.py` catches and records `nowcast_mom: {verdict: 'insufficient_data'}`.
- ARIMA/ETS per-fold failures already fall back to random walk (existing forecasters).
- Empty baseline-pool intersection (shouldn't happen; pool ⊂ panel) → guard: if no pool method present, fall back to `random_walk` as best_naive.

---

## 7. Testing
- `cpi_to_mom`: on a CPI index growing 1%/month, MoM ≈ 1.0.
- `build_nowcast_frame(target_loader=...)`: with a MoM loader (monkeypatched), `prev_mom` is the lagged target, drivers contemporaneous, leakage guard holds (only `prev_mom` is CPI-derived).
- `mom_verdict` (pure, the key tests):
  - a model with RMSE below the best baseline and DM-significant → `beats_best_naive`.
  - a model that beats `random_walk` but **not** the lower-RMSE `seasonal_naive` → `no_better_than_naive` (the hollow-win guard).
  - no candidate below best baseline → `no_better_than_naive`, `best_method == best_naive`.
- `run_mom_nowcast`: synthetic MoM driven by contemporaneous fuel → `beats_best_naive`; pure seasonal MoM + noise drivers → `no_better_than_naive`; short series → `insufficient_data`.
- `report`/`accuracy_view`: `nowcast_mom` key present and rendered.

---

## 8. Deliverables (definition of done)
1. `cpi_to_mom` + `load_inflation_mom`; generalized `build_nowcast_frame`.
2. `mom_verdict` (pure, DM-vs-best-baseline) + `run_mom_nowcast`.
3. `nowcast_mom` block in `accuracy_report.json` + `nowcast_mom_table.json` + a figure.
4. Accuracy-view MoM nowcast panel.
5. Tests incl. the hollow-win guard and synthetic beats/ties/insufficient cases.
6. Reproducible via `python -m ph_economic_ai.benchmark.run`; no new data.

---

## 9. The contribution — measured result

Run on committed data (source: `artifacts/nowcast_mom_table.json`), n = 61 months.

- **Finding: the MoM-CPI nowcaster BEATS the best simple baseline.** Best method
  **ARIMA**, RMSE **0.380** vs the strongest baseline **random-walk** 0.453 (seasonal-naive
  0.534, drift 0.458) — **skill +16.2%**, **Diebold-Mariano p = 0.032** (significant).
  Verdict: `beats_best_naive`. This is the **first and only genuine numeric "yes"** in
  the whole project, and it cleared the hard bar (best-of-baselines + DM significance),
  so it is not a hollow win.
- **Honest mechanism nuance:** the winner is **ARIMA, a univariate method** — so the edge
  comes primarily from modeling MoM inflation's own short-run autocorrelation/mean-reversion
  dynamics, which the random-walk benchmark ignores; it is **not** purely a within-month
  driver (oil/FX/fuel) information edge. That said, the **driver-based Ridge also beats
  random-walk** (RMSE 0.398 < 0.453), so the contemporaneous drivers do carry some signal —
  ARIMA simply edged it. The honest headline is "MoM inflation has exploitable short-run
  structure beyond naive," with the driver edge a secondary, weaker contributor.
- **Why it matters (data science):** completes the predictability map —
  forecasting (efficient), YoY nowcast (efficient), **MoM nowcast (predictable, +16% skill,
  p=0.032)** — and demonstrates rigorous baseline selection (beating the *strongest* naive,
  DM-tested), the methodological point that separates a real result from a hollow one. The
  contrast YoY-efficient vs MoM-predictable is itself the interesting finding: persistence
  hides predictability at the annual frame but reveals it at the monthly frame.
- **Caveat:** n = 61 months is modest; the result is significant at 5% but not overwhelming,
  and ARIMA's edge is dynamics-driven. A longer sample and a driver-only ablation (exclude
  `prev_mom`/own-lags to isolate the pure nowcast information edge) are natural confirmations.
- **Driver-only ablation result (see `2026-06-09-ph-cpi-mom-driver-ablation-design.md` §9):**
  performed — the within-month driver edge is **not significant** (`driver_edge = False`).
  Driver-only Ridge has the lowest RMSE (0.399 vs random-walk 0.453, −12%) but does not clear
  DM significance at n = 61. So the headline MoM win is attributable to **own short-run
  dynamics** (ARIMA), with the contemporaneous-driver edge **suggestive but underpowered**.

---

## 10. Sources / references
- Giannone, Reichlin & Small (2008); Bańbura et al. (2013) — nowcasting.
- Atkeson & Ohanian (2001) — naive inflation benchmark.
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
- Data: DBnomics IMF IFS PH CPI; Yahoo Finance (oil, USD/PHP); DOE/RBOB fuel proxy.
