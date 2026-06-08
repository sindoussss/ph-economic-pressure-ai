# ph_economic_ai — CPI Inflation Nowcasting (Design)

**Date:** 2026-06-08
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data science thesis)
**Builds on:** the benchmark package + predictability audit on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The predictability audit showed PH fuel, FX, and inflation are **efficient** at the 1-month-ahead *forecast* horizon — no method beats naive persistence. That closes the door on forecasting the unknowable future. **Nowcasting opens a different, legitimate door:** estimate an official number **before it is published**, using within-period information that is already observable.

PSA releases monthly CPI around the **5th–7th business day of the following month**. By the end of month _t_, the month-_t_ values of fuel prices, oil, and FX are already known. A nowcast uses those to estimate **inflation_t before its official release**.

**Goal / claim:**
> A CPI nowcaster estimates Philippine monthly headline inflation *before PSA's official release*, using only intra-month-observable drivers, and beats the naive "last published inflation" benchmark (Diebold-Mariano significant). The edge is **information timing**, not market-beating — so it is consistent with the efficiency finding.

This is the honest numeric "yes": predicting the **present before it is announced**, not the future.

---

## 2. Scope

### In scope
- A **contemporaneous (lag-0) feature frame** for inflation_t (`nowcast.py::build_nowcast_frame`).
- A **nowcast runner** (`run_nowcast`) reusing the forecaster panel + Diebold-Mariano + conformal, with the naive baseline = last published inflation.
- A documented, enforced **feature-eligibility rule** (only intra-month-observable drivers).
- Report + run + figure + Accuracy-view integration of the nowcast result.

### Out of scope
- Nowcasting fuel or other targets (CPI only; YAGNI).
- New data collection — reuses committed `features_monthly.csv` + the CPI series (`ph_cpi_monthly.csv`).
- Sub-monthly / partial-month vintages (real-time data vintages) — a future refinement, not this spec.
- The written thesis manuscript.

---

## 3. Definitions & integrity rule

- **Target:** headline YoY CPI inflation for month _t_, `inflation_t = (CPI_t / CPI_{t-12} - 1) * 100` (existing `targets.cpi_to_yoy`).
- **Naive baseline:** `inflation_{t-1}` (last *published* inflation) — i.e. random walk on the inflation series. This is the benchmark a nowcast must beat.
- **Eligible features (intra-month-observable, known before CPI_t release):**
  - `oil_t` — month-_t_ average Brent (daily series complete by end of _t_).
  - `fx_t` — month-_t_ average USD/PHP (daily, complete by end of _t_).
  - `fuel_t` — month-_t_ fuel pump proxy (DOE weekly within _t_).
  - `prev_inflation` = `inflation_{t-1}` (already published).
- **Integrity rule (enforced):** a feature for month _t_ may enter the frame only if its month-_t_ value is observable **before** PSA's CPI_t release. The four above qualify (all complete by end of _t_; CPI_t releases ~7 days into _t+1_). No same-month CPI-derived feature is allowed except the *lagged* `prev_inflation`. This rule is stated in the module docstring and enforced by construction (the frame builder only ever pulls from the eligible set).

---

## 4. Architecture

Reuses the existing benchmark machinery; the only new concept is a contemporaneous frame.

```
benchmark/
├── nowcast.py        # build_nowcast_frame(); run_nowcast(min_train)
├── run.py            # (extend) call run_nowcast; write nowcast block + figure
└── figures.py        # (extend) plot_nowcast(dates, actual, nowcast, naive)
   (reuses)           # efficiency.run_panel, forecasters, significance.diebold_mariano,
                      # conformal, targets.load_inflation, targets._features
```

### 4.1 `build_nowcast_frame() -> pd.DataFrame`
Assembles, on the common monthly index:
- `target` = `inflation_t` (from `targets.load_inflation()`).
- `oil` = `features['oil_price']` (month _t_), `fx` = `features['usd_php']` (month _t_), `fuel` = `features['gas_price']` (month _t_) — **contemporaneous (lag-0)**.
- `prev_inflation` = `inflation` shifted 1 month.
Returns columns `['oil', 'fx', 'fuel', 'prev_inflation', 'target']`, dropna'd. Feature cols = all but `target`.

**Why lag-0 is causal here:** the walk-forward trains only on rows with index `< i` (complete, already-published `(drivers_τ, inflation_τ)` pairs for τ < t), then applies the learned mapping to month _t_'s drivers to estimate the not-yet-released `inflation_t`. No future information is used; the "contemporaneity" is between drivers and target *within* month _t_, which is exactly what a nowcast exploits and is legitimate because the drivers are published before the target.

### 4.2 `run_nowcast(min_train) -> dict`
- Build the frame; `feature_cols = ['oil', 'fx', 'fuel', 'prev_inflation']`.
- Call `efficiency.run_panel(frame, PANEL_METHODS, min_train, feature_cols, target_col='target')` (PANEL_METHODS = the 7 standard forecasters). `random_walk` here = last published inflation = the naive nowcast benchmark.
- Best non-naive method by skill; conformal calibration on its residuals.
- Verdict: `'beats_naive'` if any method has `dm_p < 0.05 and skill_vs_rw > 0`, else `'no_better_than_naive'`.
- Return `{verdict, best_method, best_skill, best_dm_p, panel, calibration, n}`.

### 4.3 Integration
- `report.py`: add a `nowcast` key (the `run_nowcast` dict minus heavy internals).
- `run.py`: after the audit, call `run_nowcast(MIN_TRAIN)`, print the verdict, add `nowcast=` to the report, write `nowcast_table.json` + the figure.
- `figures.py::plot_nowcast`: line chart of actual inflation vs nowcast vs naive over the backtest.
- `ui/accuracy_view.py::nowcast_summary()`: one-line panel (verdict, best method, skill vs naive, DM p).

---

## 5. Data Flow

```
ph_cpi_monthly.csv ─► targets.load_inflation() ─► inflation_t ─┐
features_monthly.csv ─► oil_t, fx_t, fuel_t ───────────────────┤
                                                               ▼
                              nowcast.build_nowcast_frame (contemporaneous + prev_inflation)
                                                               ▼
                              efficiency.run_panel (naive = last published inflation) + DM + conformal
                                                               ▼
                              run_nowcast verdict ─► report 'nowcast' + nowcast_table.json + figure
                                                               ▼
                                                  ui/accuracy_view nowcast panel
```

---

## 6. Error Handling
- Frame shorter than `min_train + 5` after joins/dropna → `run_nowcast` returns `{verdict: 'insufficient_data', n}` instead of raising.
- Missing `ph_cpi_monthly.csv` → `load_inflation` raises `FileNotFoundError`; `run.py` catches and records `nowcast: {verdict: 'insufficient_data'}` so the rest of the run completes.
- ARIMA/ETS per-fold failures already fall back to random walk (existing forecasters).

---

## 7. Testing
- `build_nowcast_frame`: returns the expected columns; `prev_inflation` at row _t_ equals inflation at _t-1_ (lagged); `oil`/`fx`/`fuel` at row _t_ equal the month-_t_ driver values (contemporaneous, **not** shifted); no NaNs.
- **Leakage guard:** assert the frame contains **no** same-month CPI-derived column other than `target` (only `prev_inflation` is CPI-derived and it is lagged).
- `run_nowcast` on a **synthetic** frame where `target = 0.8*fuel_t + noise` → `verdict == 'beats_naive'` with a method beating naive; on a random-walk inflation series with noise drivers → `'no_better_than_naive'`.
- `report`/`accuracy_view`: `nowcast` key present and rendered.

---

## 8. Deliverables (definition of done)
1. `build_nowcast_frame` (contemporaneous, eligibility-respecting) + `run_nowcast` (panel + DM vs naive + conformal + verdict).
2. `nowcast` block in `accuracy_report.json` + `nowcast_table.json` + a nowcast-vs-actual-vs-naive figure.
3. Accuracy-view nowcast panel.
4. Tests incl. the leakage guard and synthetic beats-naive / ties-naive cases.
5. Reproducible via `python -m ph_economic_ai.benchmark.run`; no new data.

---

## 9. The contribution, stated for the write-up
- **Finding (filled from the real run):** the CPI nowcaster [beats / does not beat] the last-published-inflation benchmark — best method [method], skill vs naive [x], DM p [p], over [n] months.
- **Interpretation:** nowcasting succeeds (where forecasting failed) because within-month fuel/oil/FX movements are informative about the *current* month's inflation that the previous print does not yet reflect — an information-timing edge, fully consistent with the efficiency result. If it does **not** beat naive, that itself is informative: PH headline inflation is so persistent that even contemporaneous drivers add little before release.
- **Why it matters (data science):** demonstrates the honest, defensible form of numeric economic prediction — **estimating the present before it is announced** — with rigorous validation (causal backtest + DM + calibrated intervals), and a clear methodological line between what *is* (nowcasting) and *isn't* (forecasting) predictable for an emerging-market CPI.

---

## 10. Sources / references
- Giannone, Reichlin & Small (2008) — "Nowcasting: The real-time informational content of macroeconomic data."
- Bańbura, Giannone, Modugno & Reichlin (2013) — nowcasting survey.
- Atkeson & Ohanian (2001) — naive inflation-forecast benchmark (why naive is the bar to beat).
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
- Data: DBnomics IMF IFS PH CPI; Yahoo Finance (oil, USD/PHP); DOE/RBOB fuel proxy.
