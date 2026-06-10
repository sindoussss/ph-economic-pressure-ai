# Forecastable or Efficient? A Reproducible Predictability Audit and Nowcast of Philippine Fuel, FX, and Inflation

**Author:** Sindous
**Draft:** 2026-06-10 · grounded in the frozen `ph_economic_ai/benchmark/artifacts/accuracy_report.json` and companion tables.
**Status:** Working draft for review. `[expand]` marks places needing the author's institutional/contextual detail; all empirical values are taken verbatim from committed artifacts.

---

## Abstract

Claims that machine learning or multi-agent AI systems can "predict the economy" are common and rarely tested against a hard baseline. This thesis asks a narrower, answerable question for the Philippine case: can standard methods forecast monthly fuel prices, the peso–dollar exchange rate, and inflation better than naive persistence — and if a series resists forecasting, can the official figure at least be *nowcast* before its release? I build a small, fully reproducible benchmark that evaluates every claim with a strictly causal walk-forward backtest, a seven-method forecaster panel, Diebold–Mariano significance tests against the *strongest* simple baseline, and split-conformal prediction intervals. The result is a **predictability map**. One-month-ahead forecasts of premium gasoline (RON95), USD/PHP, and year-on-year inflation are **informationally efficient**: no method significantly beats a random walk, reproducing Meese–Rogoff (FX) and Atkeson–Ohanian (inflation) for an emerging market. The single genuine positive is a **nowcast of month-on-month inflation**: using within-month observable oil, FX, and fuel data plus the previous print, an ARIMA model beats the strongest naive baseline by 16.2% (Diebold–Mariano p = 0.032, n = 61), a result that strengthens to 16.3% (p = 0.001, n = 143) on a longer 2007–2026 sample spanning the global financial crisis and COVID. A driver-only ablation shows this edge comes from inflation's own short-run dynamics, not a statistically significant contemporaneous-driver signal. A separate test of transport inflation illustrates the protocol working in reverse: an apparently significant fuel-driven nowcast (+14.8%, p = 0.021) proves to be an artifact of three preliminary data points and is rejected by the robustness check — the same discipline that confirms the one true positive also discards a false one. The contribution is methodological as much as empirical: an honest, reproducible protocol that separates what is forecastable from what is efficient in a data-poor economy, and that bounds its one positive finding rather than overstating it.

---

## 1. Introduction

### 1.1 Motivation
Fuel and food prices and the inflation they feed dominate Philippine household budgets and monetary-policy debate. Weekly fuel price changes are front-page news; the monthly inflation print from the Philippine Statistics Authority (PSA) moves expectations and frames the central bank's target. A credible ability to anticipate next month's fuel price or inflation rate would be valuable to households, firms, and policymakers. `[expand: local policy context — excise schedule, OPSF history, BSP target band.]`

### 1.2 The gap
A wave of applications — including multi-agent "swarm" and large-language-model systems — claim to forecast prices or "the economy," but almost none report a like-for-like comparison against the simplest defensible benchmark: assuming next month looks like this month. Without that comparison, an impressive-looking forecast says nothing. The prior question is therefore not "how good is the model?" but "is this series forecastable at all, relative to naive persistence — and how would anyone know?"

### 1.3 Research questions
- **RQ1.** Can standard methods forecast one-month-ahead PH fuel, FX, and year-on-year inflation better than a random walk?
- **RQ2.** If forecasting fails, what mechanism explains the efficiency?
- **RQ3.** Can the official inflation figure be *nowcast* before its release, and is any edge an information-timing effect or a time-series-dynamics effect?
- **RQ4.** Is any positive result robust to a larger, more varied sample?

### 1.4 Contributions
1. A small, fully reproducible benchmark (`ph_economic_ai/benchmark/`) that turns "is it accurate?" from an assertion into a re-runnable measurement.
2. A **predictability map** of Philippine macro series: a clean boundary between the efficient (unforecastable beyond naive) and the forecastable.
3. A genuine positive: a month-on-month inflation nowcast that beats the strongest naive baseline, significance-tested and robustness-checked.
4. A discipline of *honest bounding*: removing fabricated confidence, requiring a beat over the strongest baseline, and using ablation to attribute the win to a mechanism rather than overclaiming.

### 1.5 Roadmap
Section 2 reviews efficiency, nowcasting, and forecast-evaluation literature. Section 3 documents data and the fuel-proxy validation. Section 4 details the methodology. Section 5 reports results. Section 6 discusses the efficient-vs-forecastable boundary and the role of honesty as method. Section 7 concludes.

---

## 2. Background and Literature Review

### 2.1 Market efficiency and the random-walk benchmark
The reference result for exchange-rate forecasting is Meese and Rogoff (1983): structural models fail to beat a random walk out of sample at short horizons. The random walk is not a strawman but a stringent benchmark; beating it requires genuine exploitable structure. This thesis treats the random walk (and its siblings, drift and seasonal-naive) as the bar every method must clear.

### 2.2 Inflation forecasting and the naive benchmark
Atkeson and Ohanian (2001) show that sophisticated Phillips-curve forecasts struggle to beat a naive forecast of inflation. The lesson generalizes: for highly persistent series, the previous value is hard to improve upon. A central theme below is that *persistence itself* hides predictability at the annual frame and reveals it monthly.

### 2.3 Nowcasting
Nowcasting (Giannone, Reichlin and Small, 2008; Bańbura et al., 2013) estimates a target *before its official release* using information already observable within the reference period. The edge is one of *information timing*, not of beating an efficient market: oil, FX, and retail fuel for a given month are observable before the PSA publishes that month's CPI. This distinction (RQ3) is the hinge of the thesis.

### 2.4 Comparing forecasts
Point-error differences can be noise. Diebold and Mariano (1995) provide a test for equal predictive accuracy; Harvey, Leybourne and Newbold (1997) give a small-sample correction. Every "beats"/"efficient" verdict here is a DM test, not a raw RMSE comparison.

### 2.5 Distribution-free uncertainty
Split-conformal prediction (Vovk, Gammerman and Shafer, 2005; Angelopoulos and Bates, 2023) yields intervals with finite-sample coverage guarantees under exchangeability, without distributional assumptions. This replaces ad hoc or fabricated "confidence" numbers with calibrated intervals whose coverage is measured.

### 2.6 Forecast accuracy measures
Scale-free comparison uses MASE (Hyndman and Koehler, 2006) alongside RMSE/MAE/MAPE and a skill score, skill = 1 − RMSE_model/RMSE_baseline.

### 2.7 Philippine context
Domestic fuel prices track world product prices (MOPS/Brent) with a lag under the deregulated automatic-pricing regime; the PSA releases CPI on a fixed monthly calendar. `[expand: citations to DOE circulars and PSA release schedule.]` The gap this thesis fills is a reproducible, honestly-bounded predictability audit for PH macro — most swarm/LLM "prediction" work in this space is unvalidated.

---

## 3. Data

All series are committed as CSVs and regenerable via `refresh_data.py`; the benchmark reads only frozen files, so every number is reproducible offline.

### 3.1 Fuel (ground truth)
Premium gasoline (RON95), monthly PHP/litre, from the World Bank **Global Fuel Prices Database** (Open Database License). The committed gold series underpins the one-month forecast backtest over **2019-11 to 2025-03 (n = 79 months)**.

### 3.2 Exchange rate
USD/PHP monthly from Yahoo Finance (`PHP=X`).

### 3.3 Inflation
Philippine CPI from the IMF **International Financial Statistics** via DBnomics, transformed to year-on-year (YoY) and month-on-month (MoM) inflation.

### 3.4 Predictors
Brent crude (`BZ=F`), USD/PHP (`PHP=X`), an RBOB-gasoline→PHP landed-cost proxy (`RB=F`), and a seasonal demand index. Two feature panels are built: a standard ~10-year window and a **long window (Yahoo `max`, 2007–2026, 177 months)** used for the robustness re-test.

### 3.5 Fuel-proxy validation
Because high-frequency retail RON95 is not freely available as a long gold series, an RBOB-derived proxy is validated against the World Bank gold: Pearson **r = 0.91** with a mean bias of **−₱5.88/L** (disclosed, not corrected away). The proxy is fit for *directional/relative* validation, a stated limitation.

### 3.6 Reproducibility
`python -m ph_economic_ai.benchmark.run` regenerates every artifact (report, ablation, efficiency panel, pass-through, audit, YoY and MoM nowcasts, driver ablation, longer-sample confirmation, and figures) from the committed CSVs.

---

## 4. Methodology

### 4.1 Causal walk-forward backtest
The validity foundation. At each step t, models train only on data through t and predict t+1 (expanding window, minimum train = 24 months). A leakage-guard test asserts no future information enters any feature. This is what makes the accuracy claim defensible rather than in-sample.

### 4.2 Forecaster panel
Seven methods: random-walk, drift, seasonal-naive, ARIMA(1,1,1), ETS (additive trend), Ridge regression, and HistGradientBoosting. The first three are baselines; the last four are candidates.

### 4.3 Metrics
MAE, RMSE, MAPE, MASE, and skill = 1 − RMSE_model/RMSE_baseline.

### 4.4 Significance
Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction. A method "beats" a baseline only if it has lower RMSE *and* DM p < 0.05; otherwise the verdict is "efficient"/"no better than naive."

### 4.5 Calibrated uncertainty
Split-conformal intervals at nominal 50/80/90/95%, with an empirical-coverage calibration table reported alongside the nominal levels.

### 4.6 Forecasting vs nowcasting
Forecasting uses only information available *before* the reference month (lagged features + previous value). Nowcasting adds *within-month observable* drivers (contemporaneous oil/FX/fuel) plus the previous print — modelling the real situation in which those inputs are known before the PSA release. An eligibility rule keeps only genuinely pre-release information.

### 4.7 Baseline discipline (the hollow-win guard)
A candidate must beat the *strongest* of {random-walk, seasonal-naive, drift}, not the weakest. This prevents a "win" that merely clears a baseline no one would use; it is enforced in code and unit-tested.

### 4.8 Ablation and robustness
A **driver-only ablation** drops the own-lag (previous MoM) and restricts candidates to the driver regressors, isolating any within-month information edge from time-series dynamics. A **longer-sample re-run** rebuilds features on the 2007–2026 window and repeats the MoM nowcast and ablation.

### 4.9 Integrity infrastructure
The fabricated "90% confidence" from the original application was removed and replaced with the conformal interval. A frozen `accuracy_report.json` plus a hash-chained, two-phase track record make results tamper-evident and quotable.

---

## 5. Results

### 5.1 One-month forecasting is efficient (RQ1)

**Fuel.** Over n = 79, the best ML model (HistGradientBoosting) has RMSE 4.099 vs the random walk's 4.069 — a skill of **−0.0075**: no improvement. Against the seasonal-naive baseline the model's skill is +0.64, i.e. the gain is over a *bad* baseline, not the strong one.

On the efficiency panel (n = 52), no method significantly beats the random walk:

| Method | RMSE | Skill vs RW | DM p vs RW |
|---|---|---|---|
| random_walk | 4.069 | 0.000 | — |
| drift | 4.110 | −0.010 | 0.504 |
| ETS | 4.183 | −0.028 | 0.275 |
| Ridge | 4.105 | −0.009 | 0.881 |
| **HGB** | **4.099** | **−0.008** | **0.921** |
| ARIMA | 4.383 | −0.077 | 0.037 (worse) |
| seasonal_naive | 11.497 | −1.826 | 0.0001 (worse) |

The ML methods are statistically *indistinguishable* from the random walk (DM p ≈ 0.88–0.92); ARIMA and seasonal-naive are significantly worse. Fuel is efficient at the one-month horizon.

**The predictability audit** extends this verdict across targets:

| Target | n | Verdict |
|---|---|---|
| Fuel (RON95) | 52 | efficient (no method beats RW) |
| USD/PHP | 38 | efficient |
| Inflation (YoY) | 59 | efficient |

All three are efficient — reproducing Meese–Rogoff (FX) and Atkeson–Ohanian (inflation) for the Philippines.

**Calibration (fuel forecast).** The conformal intervals (half-widths ₱2.56/5.88/10.42/11.86 at 50/80/90/95%) over-cover at the upper levels (measured coverage 0.58, 0.88, 1.00, 1.00 at n = 79) — i.e. conservative, honestly reported rather than tuned.

### 5.2 Mechanism (RQ2)
A pass-through regression of the RON95 change on contemporaneous and lagged driver changes (n = 77) gives β₀ = 0.31, β₁ = 0.24, **total pass-through β = 0.56**, R² = 0.33, with a near-zero driver autocorrelation (ACF₁ = 0.16). Interpretation: domestic fuel reflects a *partial, lagged* pass-through of a driver (world product price) that is itself close to a random walk. A predictable level built on an unpredictable, near-random-walk driver is exactly what produces an efficient series — the mechanism behind RQ1.

The Phase-2 gated feature ablation selected the `passthrough_lags` variant as the best-justified specification; it still did not beat the random walk, but it closed the model-vs-RW gap and tightened intervals.

### 5.3 Nowcasting (RQ3)

**YoY nowcast.** Adding within-month oil/FX/fuel plus the previous print, before the PSA release, still yields **no_better_than_naive** (n = 61). Year-on-year inflation overlaps 11 of 12 months with its prior value, so persistence is mechanically near-unbeatable.

**MoM nowcast — the numeric "yes."** Targeting month-on-month inflation, the honest bar is to beat the strongest of {random-walk, seasonal-naive, drift} by a DM test. The result is **beats_best_naive**:

| Method | RMSE (n = 61) |
|---|---|
| ARIMA | **0.380** |
| Ridge | 0.398 |
| ETS | 0.414 |
| random_walk (best naive) | 0.453 |
| drift | 0.458 |
| HGB | 0.457 |
| seasonal_naive | 0.534 |

ARIMA beats the random walk by **+16.2% skill, DM p = 0.032**. Month-on-month inflation carries genuine within-month signal that the annual frame hides.

**Driver-only ablation.** Dropping the own-lag and restricting to driver regressors gives **driver_edge = False** (n = 61): driver-only Ridge (RMSE 0.399) is directionally better than the random walk (0.453, −12%) but does not clear DM significance. The MoM win is therefore attributable to inflation's **own short-run dynamics** (captured by ARIMA), with the contemporaneous-driver edge suggestive but not significant.

### 5.4 Robustness (RQ4)
Rebuilding features on the 2007–2026 window (n = 143, spanning the GFC, the 2014 oil crash, and COVID) and re-running:

| Metric | n = 61 | n = 143 |
|---|---|---|
| MoM verdict | beats_best_naive | beats_best_naive |
| best method | ARIMA | ARIMA |
| skill vs best naive | +16.2% | **+16.3%** |
| DM p | 0.032 | **0.001** |
| driver_edge | False | False |

The MoM win **holds and strengthens** (p tightens from 0.032 to 0.001) across ~2.3× the data and a far more heterogeneous regime mix — robust, not fragile. The driver edge remains directional (driver-only Ridge 0.374 vs RW 0.413, −9.5%) but never significant: the earlier underpowered signal resolves into an honest negative rather than a confirmation.

### 5.5 A spurious positive caught: the Transport-CPI nowcast

If any series should yield a significant within-month *driver* edge, it is **Transport** CPI — mechanically driven by fuel, which is observable before the official release. Using the official PSA Transport-CPI series (OpenSTAT, by commodity group, 2018 = 100, 1994–present) as a fresh gold target and the same free fuel/FX predictors, the nowcast was re-run (n = 151 backtest months).

On the **full sample**, the driver-only model looked like the sought-after positive: fuel-only Ridge beat the best naive baseline by **+14.8%** (DM p = 0.021). Taken at face value, this would license the headline "the fuel-driven component of inflation is nowcastable ahead of the official figure."

A robustness re-test dissolved it. PSA's three most recent prints are **preliminary** and anomalous — Transport CPI 130 → 142 → 156 → 148 for early 2026 (i.e. +9.5%, +10.0%, −5.0% MoM), values the agency revises in later vintages. Dropping the trailing six preliminary months collapses the skill from +14.8% to **zero**:

| Test | Verdict | skill vs best naive | DM p |
|---|---|---|---|
| Driver-only, full sample (n = 151) | beats_best_naive | +14.8% | 0.021 |
| Driver-only, robust (drop 6 preliminary months, n = 145) | no_better_than_naive | 0.0 | — |

The entire "edge" rested on roughly three unreliable observations. The **canonical verdict is therefore that Transport MoM inflation is also efficient** — no robust within-month fuel edge — consistent with the rest of the map. More importantly, this is a worked example of the audit doing its job: it caught a positive that a naive analysis would have published, traced it to preliminary real-time data, and reported the robust null. The robustness re-test (`driver_edge_robust`) is baked into the pipeline, so the check is permanent and reproducible.

### 5.6 The predictability map (synthesis)

| Target | Setup | Verdict |
|---|---|---|
| Fuel / FX / YoY inflation | 1-month forecast | efficient (no method beats RW) |
| YoY inflation | nowcast (pre-release) | no better than naive |
| **MoM inflation** | **nowcast (pre-release)** | **predictable — ARIMA +16%, robust (p = 0.001, n = 143)** |
| MoM inflation | nowcast, driver-only | edge suggestive (−9 to −12%) but not significant |
| Transport-CPI MoM | nowcast, driver-only | apparent +14.8% edge **not robust** (preliminary-data artifact) → efficient |

---

## 6. Discussion

### 6.1 The efficient-vs-forecastable boundary
The same series is "efficient" at the annual frame and "forecastable" monthly. The reconciliation is persistence: YoY inflation shares eleven of twelve months with its lag, so the naive forecast is mechanically excellent and little room remains; the MoM transform removes that overlap and exposes a short-run autoregressive structure ARIMA can exploit. Predictability was not absent — it was hidden by the framing.

### 6.2 Reproducing landmark results for an emerging market
Finding FX and inflation efficient at one month reproduces Meese–Rogoff and Atkeson–Ohanian outside their original developed-economy settings, adding external validity rather than novelty for its own sake.

### 6.3 Honesty as method
Three design choices each prevented a specific overclaim: removing the fabricated 90% confidence (replaced by measured conformal coverage); requiring a beat over the *strongest* baseline (the hollow-win guard); and the driver-only ablation (which stopped "MoM is predictable" from silently becoming "the drivers predict inflation"). The longer-sample re-test then asked whether the one positive survived more and more varied data. This is the discipline that separates a credible finding from a dashboard number.

The Transport-CPI nowcast (§5.5) is the sharpest illustration. A full-sample run produced an apparently significant fuel edge (+14.8%, p = 0.021) — precisely the bold "AI nowcasts fuel-driven inflation" headline one might want to claim. A real-time-data robustness check showed it rested on three preliminary, not-yet-revised PSA observations and vanished once they were removed. The same discipline that confirmed the one true positive (headline MoM) here *rejected* a false one. A method that only ever confirms is not doing robustness; a method that rejects its own most attractive result when the data do not support it is.

### 6.4 What the MoM result is and is not
It is a robust, significant ability to nowcast month-on-month inflation slightly ahead of publication, driven by the series' own dynamics. It is *not* evidence that contemporaneous drivers significantly improve the nowcast, and *not* a claim to forecast levels months ahead. The honest interval, not a point estimate, is the deliverable.

### 6.5 Practical relevance
For fuel/FX/YoY inflation, the defensible product is a calibrated interval around the naive forecast — and the honest statement that no model beats it. For MoM inflation, a pre-release nowcast with its interval is genuinely useful. Effort is best spent where predictability exists.

### 6.6 Limitations
Monthly resolution; an RBOB fuel proxy with disclosed bias (r = 0.91, −₱5.88/L); modest samples (n = 38–143); conformal coverage that is approximate at small n (and here conservative); and a CPI series via IMF/DBnomics rather than the PSA microdata. The application's LLM/agent "swarm" is an interface and explanation layer, not a validated predictor — its agent-agreement numbers are labelled as such, distinct from the calibrated intervals.

---

## 7. Conclusion and Future Work

This thesis replaces the assertion "AI predicts the economy" with a measured map of what is and is not predictable in Philippine macro data. One-month forecasts of fuel, FX, and year-on-year inflation are informationally efficient; the lone genuine positive is a month-on-month inflation nowcast that beats the strongest naive baseline by ~16% and survives a 2.3× larger, regime-varied sample (DM p = 0.001), attributable to the series' own dynamics rather than a significant driver edge. The contribution is a reproducible, honestly-bounded audit protocol for a data-poor economy.

**Future work.** (i) Confirm or upgrade the within-month driver edge with weekly resolution and true MOPS data, where more observations could resolve the currently-underpowered signal. (ii) Accumulate a live, hash-chained track record of matured nowcasts. (iii) Extend the audit to additional CPI components (e.g. food) using the same free PSA OpenSTAT commodity-group source, re-testing each against the preliminary-data robustness check that exposed the Transport result. (iv) Re-evaluate the Transport-CPI edge once the 2026 prints are finalized, since the present null is partly a consequence of using preliminary vintages. (v) A real-time DOE bulletin parser for daily/weekly fuel ground truth.

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 16(4), 494–591.
- Atkeson, A., & Ohanian, L. E. (2001). Are Phillips curves useful for forecasting inflation? *Federal Reserve Bank of Minneapolis Quarterly Review*, 25(1), 2–11.
- Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the real-time data flow. In *Handbook of Economic Forecasting* (Vol. 2A, pp. 195–237). Elsevier.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665–676.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1–2), 3–24.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Data: World Bank Global Fuel Prices Database; IMF International Financial Statistics (via DBnomics); Yahoo Finance; Philippine Statistics Authority; Department of Energy (Philippines).

---

## Appendices

### Appendix A — Reproducibility
- Regenerate all artifacts: `python -m ph_economic_ai.benchmark.run`.
- Refresh source data (network): `refresh_data.build_features_csv`, `build_long_features`, World Bank workbook loader.
- Committed artifacts: `accuracy_report.json`, `ablation_table.json`, `audit_table.json`, `nowcast_table.json`, `nowcast_mom_table.json`, `mom_driver_ablation_table.json`, `mom_longsample_table.json`, `backtest_predictions.csv`, `figures/*.png`.

### Appendix B — Full panels
The seven-method efficiency panel (§5.1), the MoM nowcast panel (§5.3), and the audit table (§5.1) are reproduced verbatim from the artifacts above. `[expand: paste full JSON-derived tables as formatted appendix tables.]`

### Appendix C — Calibration and pass-through
- Fuel forecast conformal calibration: nominal vs measured coverage at 50/80/90/95% (§5.1).
- Pass-through regression (§5.2): n = 77, β₀ = 0.31, β₁ = 0.24, β_total = 0.56, R² = 0.33, driver ACF₁ = 0.16, with HAC standard errors. `[expand: full coefficient table with SEs.]`

### Appendix D — Software architecture
The `benchmark/` package: `backtest.py` (causal walk-forward), `forecasters.py` (panel), `significance.py` (DM/HLN), `conformal.py` (intervals + calibration), `efficiency.py` (panel + pass-through), `nowcast.py` (YoY/MoM + ablation), `longsample.py` (robustness), `audit.py` (`Target` registry + verdicts), `report.py` / `figures.py` / `run.py` (assembly). The PyQt application renders the frozen report; it does not recompute it.
