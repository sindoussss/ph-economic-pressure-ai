# Thesis Manuscript Outline — Predictability of the Philippine Economy

> **Status:** Outline scaffold, drafted from the committed specs (`docs/superpowers/specs/`) and benchmark artifacts (`ph_economic_ai/benchmark/artifacts/`). Every number below is from a committed artifact; citations marked `[ref]` are seeded in the specs' §10. Fill prose around these bullets.

**Working title:**
*Forecastable or Efficient? A Reproducible Predictability Audit and Nowcast of Philippine Fuel, FX, and Inflation*

**Type:** Data-science thesis (empirical, methods + applied economics).

**One-sentence contribution:** A rigorous, reproducible framework that maps which monthly Philippine macro series are *forecastable* vs *informationally efficient*, finding broad efficiency (no method beats random walk for fuel, FX, and year-on-year inflation) but a single robust positive — **month-over-month inflation is nowcastable** (ARIMA +16% over the best naive, DM p=0.001, n=143), with the mechanism honestly bounded to own-dynamics rather than a contemporaneous-driver edge.

---

## Abstract (~250 words, write last)
- Problem: can monthly PH macro series be forecast better than naive persistence?
- Method one-liner: causal walk-forward backtest + 7-method forecaster panel + Diebold-Mariano significance + split-conformal intervals; forecasting vs nowcasting; ablation + longer-sample robustness.
- Key results: fuel/FX/YoY-inflation efficient; MoM-inflation nowcast beats best naive (ARIMA +16.3%, DM p=0.001, n=143), driven by own-dynamics (driver edge not significant); DOE fuel pass-through β≈0.56.
- Contribution: an honest, reproducible predictability-audit protocol for a data-poor emerging-market economy; a clean efficient-vs-forecastable boundary.

---

## 1. Introduction
- 1.1 Motivation — why predicting PH fuel/inflation matters (household impact, BSP target, OPSF/excise policy debates).
- 1.2 The gap — "AI predicts the economy" claims are rarely validated against a hard baseline; what does *accurate* even mean here?
- 1.3 Research questions:
  - RQ1: Can standard methods forecast 1-month-ahead PH fuel, FX, and YoY inflation better than random walk?
  - RQ2: If not, *why* (mechanism)?
  - RQ3: Can the official inflation print be **nowcast** before release, and is that edge from information or from time-series dynamics?
  - RQ4: Is any positive result robust to more/varied data?
- 1.4 Contributions (bullet list; the reproducible framework + the efficiency map + the MoM nowcast finding + the honesty/ablation discipline).
- 1.5 Thesis roadmap.

## 2. Background & Literature Review *(the main net-new writing — specs seed the citations)*
- 2.1 Market efficiency & the random-walk benchmark — **Meese & Rogoff (1983)** (FX random walk beats structural models).
- 2.2 Inflation forecasting & the naive benchmark — **Atkeson & Ohanian (2001)** (naive hard to beat).
- 2.3 Nowcasting — **Giannone, Reichlin & Small (2008)**, **Bańbura et al. (2013)** (timely information, present-before-release).
- 2.4 Forecast comparison — **Diebold & Mariano (1995)**; **Harvey, Leybourne & Newbold (1997)** (small-sample correction).
- 2.5 Distribution-free uncertainty — conformal prediction (**Vovk, Gammerman & Shafer**; Angelopoulos & Bates).
- 2.6 Philippine fuel pricing — DOE automatic pricing / MOPS pass-through; PSA CPI release calendar.
- 2.7 Gap this thesis fills — a reproducible, honestly-bounded audit for PH macro; most swarm/LLM "prediction" work is unvalidated.

## 3. Data
*(Source: `targets.py`, `ground_truth.py`, `refresh_data.py`; specs 2026-06-05 §3, audit §3.3, nowcast §3.)*
- 3.1 Fuel (gold): World Bank Global Fuel Prices DB, PH premium RON95, monthly PHP/litre, 2017–2025 (99 mo). Open Database License.
- 3.2 FX: Yahoo Finance USD/PHP monthly.
- 3.3 Inflation: DBnomics IMF IFS PH CPI → YoY and MoM transforms (`cpi_to_yoy`, `cpi_to_mom`).
- 3.4 Predictors: Brent (BZ=F), USD/PHP, RBOB→PHP landed-cost proxy, seasonal demand index; standard (10y) and **long (max, 2007–2026, 177 mo)** feature sets.
- 3.5 Proxy validation — RBOB proxy vs World Bank gold: **Pearson r = 0.91, bias −₱5.88/L** (`proxy_validation.py`; figure `proxy_scatter.png`). State as a disclosed limitation.
- 3.6 Reproducibility — committed CSVs + one-command `refresh_data` / `benchmark.run`.

## 4. Methodology
*(Source: `backtest.py`, `forecasters.py`, `significance.py`, `conformal.py`, `efficiency.py`, `nowcast.py`, `audit.py`.)*
- 4.1 Causal walk-forward backtest (expanding window, strictly ≤ t; leakage-guard test) — the validity foundation.
- 4.2 Forecaster panel: random-walk, drift, seasonal-naive, ARIMA(1,1,1), ETS, Ridge, HGB.
- 4.3 Metrics: MAE, RMSE, MAPE, **MASE**, **skill score** vs baseline.
- 4.4 Significance: **Diebold-Mariano** (HLN-corrected); "beats" = lower RMSE *and* DM p<0.05.
- 4.5 Calibrated uncertainty: split-conformal intervals + empirical coverage table.
- 4.6 Forecasting vs **nowcasting** — the information-timing distinction; intra-month-observable eligibility rule (integrity).
- 4.7 Baseline discipline — beating the *best* simple baseline; the hollow-win guard.
- 4.8 Ablation & robustness — driver-only ablation; longer-sample re-run.
- 4.9 Reproducibility & integrity infra — frozen `accuracy_report.json`, hash-chained two-phase track record.

## 5. Results
*(Source: `accuracy_report.json` + the `*_table.json` artifacts + figures.)*
- 5.1 **Forecasting is efficient (RQ1).**
  - Fuel: best HGB/ablation skill −0.007 vs random walk; no method significantly beats RW. (`figures/baseline_bars.png`, `method_skill_bar.png`)
  - Efficiency panel (fuel): all methods DM p>0.05 vs RW; ARIMA/seasonal-naive significantly *worse*.
  - Predictability audit: **fuel, FX, YoY-inflation all `efficient`** (`audit_table.json`, `figures/audit_verdicts.png`).
- 5.2 **Mechanism (RQ2).** DOE pass-through regression: total β = 0.56 (β0=0.31, β1=0.24), R²=0.33, driver Δ-autocorrelation=0.16 → partial, lagged pass-through of a near-random-walk driver ⇒ efficiency. (`figures/passthrough.png`)
- 5.3 **Nowcasting (RQ3).**
  - YoY nowcast: `no_better_than_naive` (persistence dominates).
  - **MoM nowcast: `beats_best_naive` — ARIMA RMSE 0.380 vs random-walk 0.453, +16.2% skill, DM p=0.032, n=61.** (`nowcast_mom_table.json`, `figures/nowcast_mom.png`)
  - Driver-only ablation: **`driver_edge=False`** — Ridge directionally best (−12%) but not significant ⇒ win is own-dynamics, not a within-month information edge.
- 5.4 **Robustness (RQ4).** Longer sample (n=143, 2007–2026): MoM **holds and strengthens** — ARIMA +16.3%, **DM p 0.032 → 0.001** across GFC/2014/COVID; driver edge still not significant. (`mom_longsample_table.json`)
- 5.5 Calibration — conformal coverage table (nominal vs measured) showing honest intervals.

## 6. Discussion
- 6.1 The efficient-vs-forecastable boundary — *why YoY is efficient but MoM is forecastable* (persistence/overlap argument).
- 6.2 Reproducing landmark results for an emerging market (Meese-Rogoff FX; Atkeson-Ohanian inflation).
- 6.3 Honesty as method — fake-confidence removal, hollow-win guard, ablation, longer-sample: each prevented a specific overclaim.
- 6.4 What the MoM result is and isn't — predictable via own dynamics; driver edge suggestive but unproven; not a crystal ball.
- 6.5 Policy/practical relevance — where forecasting effort is/ isn't worthwhile; honest interval + present-before-release as the only defensible "product".
- 6.6 Limitations — monthly resolution; RBOB-proxy bias; n still modest; conformal exchangeability caveat; LLM/swarm layer is an interface, not validated prediction.

## 7. Conclusion & Future Work
- 7.1 Summary of the predictability map (the 4-row table).
- 7.2 Contribution restated (framework + finding + discipline).
- 7.3 Future work — weekly resolution + true MOPS data; MoM driver edge with longer/weekly data; live track-record accumulation; bulletin-PDF DOE parser.

## References
- Pull the seeded list from spec §9/§10 across all six specs (Meese-Rogoff 1983; Atkeson-Ohanian 2001; Giannone-Reichlin-Small 2008; Bańbura et al. 2013; Diebold-Mariano 1995; Harvey-Leybourne-Newbold 1997; Vovk et al.; World Bank GFP DB; DOE; PSA; IMF IFS). Expand to full bibliographic entries.

## Appendices
- A. Reproducibility: commands (`refresh_data`, `benchmark.run`), committed artifacts inventory.
- B. Full forecaster-panel and audit tables (from the `*_table.json`).
- C. Conformal calibration tables; pass-through regression full output (HAC SE).
- D. Software architecture: the `benchmark/` package modules and the `Target` abstraction.

---

## Source map (section → committed evidence)
| Section | Spec | Artifact(s) |
|---|---|---|
| 3 Data | 2026-06-05; audit; nowcast | `data/*.csv`, `proxy_validation` in report |
| 4 Methodology | 2026-06-05; efficiency-mechanism | `backtest/forecasters/significance/conformal/efficiency/nowcast/audit` |
| 5.1–5.2 | efficiency-mechanism; audit | `accuracy_report.json`, `audit_table.json`, figures |
| 5.3 | nowcast; mom-nowcast; driver-ablation | `nowcast_*`, `mom_driver_ablation_table.json` |
| 5.4 | mom-longsample | `mom_longsample_table.json` |
| 6 Discussion | all specs' §9 interpretations | — |

**Writing order suggestion:** 4 → 5 → 3 (you know these cold from the code) → 6 → 2 (literature, the real new effort) → 1 → Abstract.
