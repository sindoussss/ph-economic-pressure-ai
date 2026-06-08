# ph_economic_ai — Market-Efficiency + Pass-Through Mechanism (Contribution-Strengthening) Design

**Date:** 2026-06-08
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** Phase 1 (honest accuracy benchmark) + Phase 2 (gated feature ablation), both on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

Phases 1–2 produced a rigorous, reproducible result: a gradient-boosting model does **not** beat naive random walk at the 1-month-ahead Philippine RON95 pump-price forecast (best variant skill −0.007). On its own this reads as a *failed experiment*. The goal here is to **strengthen it into a defensible research contribution** suitable for a thesis — with **no new data** (uses the committed monthly World Bank RON95 gold + RBOB/FX features).

**The contribution claim:**

> Philippine retail gasoline is **informationally efficient at the monthly horizon** — no standard forecasting method beats naive persistence — and this is a **direct consequence of the DOE automatic pricing formula's near-mechanical pass-through** of an input (landed cost) that is itself a random walk.

Two pillars:
- **A. Efficiency (empirical):** across a panel of standard forecasters, none is *statistically* better than random walk (Diebold-Mariano test).
- **B. Mechanism (explanatory):** the measured DOE pass-through (elasticity + lag) plus a random-walk driver mathematically implies the efficiency result — turning "ML failed" into "here is why no method can win."

This reframes a negative ML result into a positive economic finding with a causal mechanism.

---

## 2. Scope

### In scope (all on existing monthly data)
- A **forecaster panel** (`forecasters.py`) exposing standard methods through the existing causal `walk_forward`.
- A **Diebold-Mariano significance test** (`significance.py`, hand-coded on scipy) vs random walk.
- A **pass-through regression** (`passthrough.py`): elasticity, lag structure, R², plus a random-walk check on the driver.
- Report + Accuracy-view integration of the panel table, DM p-values, and pass-through coefficients.
- New dependency: **statsmodels** (for ARIMA/ETS). DM is hand-coded (scipy only).

### Out of scope
- New data collection (weekly, MOPS) — that was the deferred Phase 3.
- The written thesis manuscript (lit review, prose) — a separate, later effort; this produces the Results/Discussion *evidence* it will cite.
- Any change to the live track record or UI beyond a read-only results panel.

---

## 3. Architecture

New modules under `ph_economic_ai/benchmark/` (headless; import nothing from `ui/`):

```
benchmark/
├── forecasters.py    # predict_fn wrappers: random_walk, seasonal_naive, drift,
│                     #   arima, ets, ridge, hgb, structural_passthrough
├── significance.py   # diebold_mariano(loss_a, loss_b) -> {dm_stat, p_value}
├── passthrough.py    # estimate_passthrough(df) -> elasticity, lags, R²; driver RW check
└── efficiency.py     # run_panel(frame, ...) -> rows[{method, skill_vs_rw, dm_p, ...}]
```

Each forecaster is a `predict_fn(X_train, y_train, x_next) -> float`, reusing `backtest.walk_forward` (strictly causal, already tested). The panel runner scores every method in RON95 space against the same random-walk baseline and attaches a DM p-value.

### 3.1 Forecaster panel (`forecasters.py`)
Factory returning `predict_fn`s. Univariate methods use `y_train`; feature methods use `X`:
- `random_walk` → `y_train[-1]`; `drift` → `y_train[-1] + mean(diff(y_train))`; `seasonal_naive` → `y_train[-12]`.
- `arima` → statsmodels `ARIMA(y_train, order=(1,1,1))`, forecast 1 step (fit per fold; fall back to random_walk on convergence failure).
- `ets` → statsmodels `ExponentialSmoothing(y_train, trend='add')`, forecast 1 step (fallback on failure).
- `ridge` → sklearn `Ridge(alpha=1.0)` on `X_train`/`y_train`, predict `x_next`.
- `hgb` → the existing `_hgb_predict_fn`.
- `structural_passthrough` → reuse the Phase-2 `structural_hybrid` decomposition (predict residual over lagged proxy).

### 3.2 Diebold-Mariano (`significance.py`)
`diebold_mariano(loss_a, loss_b, h=1) -> {'dm_stat', 'p_value'}` on per-step squared-error loss series, using the standard small-sample (Harvey-Leybourne-Newbold) correction and a Student-t reference. Pure numpy/scipy. Interpretation stored alongside each panel row: positive DM + p>0.05 ⇒ "not significantly different from random walk."

### 3.3 Pass-through regression (`passthrough.py`)
- Build `cost_t = (RBOB_t / 3.785 * USDPHP_t)` (landed estimate; same constants as `fetcher._fetch_doe_prices`, excise/VAT terms absorbed by the intercept).
- OLS (statsmodels) `Δpump_t = α + β0·Δcost_t + β1·Δcost_{t-1} + ε`; report `α, β0, β1`, total pass-through `β0+β1`, per-lag split, R², and HAC (Newey-West) standard errors.
- **Driver RW check:** first-difference autocorrelation of `cost` (lag-1) and its own random-walk skill, to show the input is ≈ a random walk.
- `estimate_passthrough(df) -> dict` returning all of the above for the report.

### 3.4 Panel runner (`efficiency.py`)
`run_panel(frame, methods, min_train) -> list[dict]` where each row =
`{method, rmse, mae, skill_vs_rw, dm_stat, dm_p, n}`. Random walk is the reference; its own row has `dm_p=None`.

---

## 4. Data Flow

```
features_monthly.csv + world_bank_ron95.csv
        │
        ├─ build_feature_frame (Phase 2) ─┐
        │                                  ▼
        │                       efficiency.run_panel ──► panel rows (skill + DM p) ─┐
        │                                  │ (per method: walk_forward + DM vs RW)  │
        └─ passthrough.estimate_passthrough ──► elasticity, lags, R², driver-RW ────┤
                                                                                    ▼
                                              report.build_report(... efficiency=, passthrough=)
                                                                                    │
                                                          accuracy_report.json + figures
                                                                                    ▼
                                                     ui/accuracy_view (read-only panels)
```

---

## 5. Error Handling
- ARIMA/ETS per-fold fit failures (convergence, short series) → caught, fall back to random-walk prediction for that step, logged; the method still produces a full series.
- statsmodels absent → `forecasters.py` raises a clear ImportError naming the package; the rest of the benchmark (Phase 1/2) is unaffected since these are new modules.
- Pass-through regression with insufficient rows → returns a dict with `n` and `None` coefficients rather than raising.
- DM test with zero loss-differential variance → returns `p_value=1.0` (no detectable difference).

---

## 6. Testing
- `forecasters`: each `predict_fn` returns a finite float on a synthetic series; `drift`/`seasonal_naive` exact values on a known series; ARIMA/ETS fall back to random walk when handed a degenerate series.
- `significance`: DM on identical loss series → stat 0, p=1; DM on a clearly-worse series → correct sign and small p; symmetry `dm(a,b) == -dm(b,a)` in stat.
- `passthrough`: on synthetic data with a known constructed pass-through (`pump = prev + 0.5·Δcost`), recovered `β` ≈ 0.5 and high R².
- `efficiency`: `run_panel` returns one row per method with required keys; random-walk row has `dm_p None`; structural method scored in RON95 space.
- `report`/`accuracy_view`: new keys present and rendered.

---

## 7. Deliverables (definition of done)
1. Forecaster panel run headlessly; **panel table (method, skill vs RW, DM p)** in `accuracy_report.json` + a skill bar figure.
2. **Pass-through coefficients** (elasticity, lag split, R², driver-RW check) in the report + a regression figure.
3. Honest written conclusion in spec §9-equivalent: whether the efficiency claim holds (all DM p > 0.05) and the measured pass-through that explains it.
4. Accuracy view shows both as read-only panels.
5. Tests covering forecasters, DM, pass-through recovery, and the panel.
6. `statsmodels` added; reproducible via `python -m ph_economic_ai.benchmark.run`.

---

## 8. The contribution, stated for the eventual write-up
- **Finding:** Across {random walk, drift, seasonal-naive, ARIMA, ETS, Ridge, HGB, structural pass-through}, none achieves a statistically significant accuracy gain over random walk at the 1-month horizon (DM p > 0.05) for NCR RON95, 2017–2025.
- **Explanation:** DOE's automatic pricing passes through landed cost with total elasticity β≈<measured> at lag 0–1 month (R²≈<measured>); landed cost is itself a random walk (Δ-autocorrelation ≈ 0). A near-unit pass-through of a random-walk input yields a random-walk output — so naive persistence is the efficient forecast, and added model complexity cannot help.
- **Why it matters:** quantifies fuel-price predictability and the pass-through elasticity for PH policy (OPSF/excise debates), and demonstrates an honest, reproducible evaluation protocol for a data-poor regulated market.

---

## 9. Sources / methods references (for the write-up)
- Diebold, F.X. & Mariano, R.S. (1995), "Comparing Predictive Accuracy."
- Harvey, Leybourne & Newbold (1997), small-sample DM correction.
- Vovk, Gammerman & Shafer — conformal prediction (Phase 1 intervals).
- World Bank Global Fuel Prices Database (gold series), DOE automatic oil pricing (pass-through formula).
