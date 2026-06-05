# ph_economic_ai — Thesis-Level Accuracy & Evaluation Design

**Date:** 2026-06-05
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous

---

## 1. Problem & Goal

`ph_economic_ai` is a PyQt6 desktop app whose impressive surface area — a
multi-agent LLM debate/swarm/evolution layer (ollama), RAG over a NEDA corpus,
live data briefs, policy recommendations, causal chains — sits on top of one
small scientific core: a `HistGradientBoostingRegressor` that forecasts
Philippine retail pump prices.

**Goal:** raise the project to a *thesis-level quality bar* — the methodology and
the accuracy claim must withstand expert scrutiny, even without a formal panel.

**The single defended claim:** *the 1-month-ahead Philippine RON95 retail
gasoline pump-price forecast is accurate*, where "accurate" is defined
quantitatively and proven reproducibly.

The credibility problem is an **evidence** problem, not a modeling problem. We do
not need a fancier model; we need a reproducible, honest evaluation around the
existing model, plus an interface that lets people see the proof. The work splits
into a **scientific core** (proves accuracy, runs headless) and a **presentation
layer** (renders the proof, computes nothing).

### Honest accuracy gaps in the current code (the things being fixed)

1. `model.predict()` returns a hardcoded `confidence=90.0` and `pred_std=0.0`
   (`ph_economic_ai/model.py:49`). The displayed confidence is fictional.
2. The only legitimate accuracy signal today is `cross_val_rmse`
   (`model.py:66`); it is not surfaced as a defensible headline claim and has no
   baseline to beat.
3. The gas "ground truth" is a **proxy**: DOE/CKAN was decommissioned, so gas =
   `RBOB futures ÷ 3.785 × FX × 1.35 + 12` (`fetcher.py:346`). Defensible as a
   live *feature*, but must never be presented as truth without validation.
4. Food and electricity outputs are **deterministic transforms of gas**
   (×0.22, ×0.18 pass-through in `data.py` / `fetcher.py`). "Predicting" them is
   partly circular; the app currently presents them as forecasts.
5. A synthetic data generator (`data.py`) uses those same coefficients —
   evaluating on synthetic data would be circular and meaningless.

---

## 2. Scope

The work is **two phases**. Phase 1 makes the confidence *truthful*; Phase 2
makes it *higher* — but only by reducing real error, with every gain gated behind
a measured backtest improvement. Phase 1 must land first because it is the
measuring instrument Phase 2 is judged by.

### In scope — Phase 1 (honest measurement)
- A headless, reproducible **benchmark package** that proves the 1-month gas
  forecast claim against an open authoritative dataset and a naive baseline.
- **Calibrated uncertainty** (split conformal) replacing the fake confidence.
- A **hash-chained, prediction-locked live track record** fed by real DOE prices.
- A read-only in-app **"Methodology & Accuracy"** view that renders the artifacts.
- Honest relabeling of derived outputs and of the agent layer's role.

### In scope — Phase 2 (earning higher confidence)
- Re-grounding the model in the finished-product pass-through that actually sets
  PH pump prices, plus more efficient interval construction. See §9 for the
  ranked, individually-gated levers.

### Out of scope (YAGNI / future work)
- Experiment-tracking infra (DVC/MLflow), competing model zoo, LaTeX paper.
- A claim that the LLM swarm beats the HGB baseline (left as stated future work,
  with the harness in place to test it later).
- Independent forecasting of food/electricity (they remain derived, now labeled).

---

## 3. Ground Truth Strategy (two-tier)

| Tier | Source | Coverage | Use | License |
|---|---|---|---|---|
| **Gold (frozen backtest)** | World Bank Global Fuel Prices DB — PH premium gasoline RON95, monthly | Dec 2015 – Apr 2025 (~113 mo) | Headline RMSE/MAE/MAPE/skill + proxy validation | Open Database License |
| **Live (ongoing record)** | DOE weekly Oil Monitor advisory, aggregated to monthly | Ongoing | Truth side of the live track record | Public (gov) |

- The **World Bank series** carries the headline claim — frozen, citable,
  reproducible, enough months for a real walk-forward backtest.
- The **DOE scraper** supplies the truth for live grading; the RBOB proxy is
  demoted to a live input *feature* only, never the truth.
- The **RBOB proxy is validated** against World Bank (correlation + bias) so the
  "your data is fake" objection is answered with a chart.

---

## 4. Architecture

### 4.1 Benchmark package (scientific core)

New package `ph_economic_ai/benchmark/`, importing nothing from `ui/`, requiring
neither PyQt nor ollama, runnable as `python -m ph_economic_ai.benchmark.run`.

```
ph_economic_ai/benchmark/
├── ground_truth.py     # load + cache World Bank RON95 monthly series
├── doe_scraper.py      # scrape DOE weekly advisory → monthly (live truth)
├── proxy_validation.py # correlation/bias of RBOB proxy vs World Bank
├── baselines.py        # random-walk (no-change) + seasonal-naive
├── backtest.py         # walk-forward expanding-window harness (causal)
├── metrics.py          # MAE, RMSE, MAPE, MASE + skill score vs baseline
├── conformal.py        # split-conformal q̂ per level + empirical coverage
├── report.py           # emit frozen accuracy_report.json + figures
├── run.py              # one-command entrypoint
└── artifacts/          # committed proof
    ├── accuracy_report.json
    ├── backtest_predictions.csv
    └── figures/*.png
```

**Key decisions:**
- **Walk-forward, expanding window, strictly causal.** At month *t*, train only
  on data ≤ *t*, predict *t+1*, step forward. Prevents lookahead leakage.
- **Headline metric = skill score:** `skill = 1 − RMSE_model / RMSE_naive`.
  Positive means it beats random walk. The honest outcome "model ≈ random walk"
  is fully on the table and is reported either way.
- **MASE** included as the standard scale-free forecasting metric.
- **Frozen artifacts committed to git** with a data hash + date range; re-running
  regenerates them and `git diff` exposes any change (tamper-evidence for the
  backtest half).

### 4.2 Calibrated uncertainty (split conformal)

- During the walk-forward backtest, collect out-of-sample absolute residuals
  `|actual − predicted|`.
- For nominal level *p*, the interval half-width `q̂_p` = the *p*-th percentile of
  residuals. Interval = `prediction ± q̂_p`.
- **Empirical coverage** is then measured: fraction of backtest months where the
  true price fell inside the band. Reported as a **calibration table** (nominal
  50/80/90/95% vs. measured). This table is the proof the uncertainty is honest.
- `conformal.py` writes `q̂` per level into `accuracy_report.json`.
- `model.predict()` stops returning constants; it returns the point prediction
  plus interval bounds from the stored `q̂`. The literals `90.0` / `0.0` are
  deleted. UI consumers receive the real interval.
- **Caveat (disclosed):** conformal assumes exchangeable residuals, mildly
  violated by time-series drift. Mitigation: compute `q̂` on a rolling recent
  residual window; state this in limitations.

### 4.3 Live track record (credibility mechanism)

Hardens the existing `engine/ground_truth.py` + `AgentTrustStore`.

- **Append-only, hash-chained log.** Each prediction row:
  `{timestamp, horizon=1mo, target_month, predicted_price, interval_low,
  interval_high, model_version, data_hash, prev_row_hash}`. Each row's hash
  includes the previous row's hash → past predictions cannot be edited
  undetectably.
- **Two-phase, locked at prediction time.** The prediction + interval are written
  and frozen when made. When the real price for `target_month` is known (DOE
  scraper), a *separate* grading row links to it and records realized error.
  Prediction and outcome are never written together → no hindsight.
- **DOE scraper feeds the truth side**; RBOB proxy never used as truth here.
- **Live scorecard** uses the *same* metrics as the backtest (rolling MAE/RMSE/
  skill-vs-naive/coverage over matured predictions), so live and frozen numbers
  are directly comparable; divergence is a drift signal.
- **Constraint (disclosed):** the live record only matures over months. The
  frozen backtest carries the headline claim; the live record is the honesty
  mechanism that grows in the open.

### 4.4 In-app "Methodology & Accuracy" view (presentation layer)

Read-only PyQt view alongside `ui/stage4_report.py`; renders artifacts, computes
no science. Four panels:

1. **Headline card** — one honest sentence from `accuracy_report.json`:
   *"1-month-ahead RON95 forecast: MAE ₱X.XX, skill score +Y% vs. random walk,
   over N months (World Bank, 2015–2025)."* If skill ≤ 0, it says so plainly.
2. **Backtest panel** — predicted-vs-actual chart with conformal bands; baseline
   comparison bar (model vs. random-walk vs. seasonal-naive); calibration table.
3. **Live track record panel** — hash-chained log table (predicted → actual →
   error, with maturity status); rolling live scorecard; a "verify chain" button
   that recomputes hashes and shows ✓/✗.
4. **Methodology & limitations panel** — plain-language statement of what the
   claim is and isn't; proxy-vs-World-Bank validation chart; explicit disclosure
   that **food and electricity are deterministic transforms of gas, not
   independent forecasts**; conformal exchangeability caveat; cited data sources.

### 4.5 Honest framing changes (cross-cutting)
- Food/electricity outputs get a visible **"derived, not independently
  forecast"** label wherever they appear.
- The agent swarm/debate is framed as a **reasoning/explanation interface**, not
  the source of the accuracy number, unless/until an experiment shows it beats
  the HGB baseline (stated future work).

---

## 5. Data Flow

```
World Bank RON95 (monthly) ─┐
                            ├─→ backtest.py (walk-forward) ─→ residuals ─→ conformal.py ─┐
RBOB proxy (existing)  ─────┘         │                                                  │
                                      ├─→ metrics.py ──────────────────────────────────→ report.py
                       baselines.py ──┘                                                  │
                                                                                         ▼
                                                                    artifacts/ (accuracy_report.json, CSV, figures)
                                                                                         │
DOE scraper (weekly→monthly) ─→ live grading ─→ hash-chained log (AgentTrustStore) ──────┤
                                                                                         ▼
                                                          ui/ Methodology & Accuracy view (read-only)
```

---

## 6. Error Handling

- **Network failures** (World Bank / DOE): benchmark uses a committed local cache
  of the gold series so the backtest is reproducible offline; the live scraper
  fails soft (logs, skips grading this cycle) exactly like existing fetchers.
- **DOE page/format change**: scraper raises a typed error, grading cycle is
  skipped, log records a gap rather than guessing.
- **Sklearn internals change** (gain importances already guard this in
  `model.py`): conformal/backtest depend only on public `predict`, so they are
  insulated.
- **Empty/short series**: backtest requires a minimum window length; below it,
  `run.py` exits with a clear message rather than emitting a meaningless report.

---

## 7. Testing

Extends the existing `tests/` suite:
- `baselines`: random-walk on a known series returns the previous value;
  seasonal-naive returns the value 12 months prior.
- `backtest`: assert strict causality — no training row dated after the predicted
  month (leakage guard).
- `conformal`: synthetic data with known noise → empirical coverage ≈ nominal.
- `hash-chain`: mutating any past row makes verification fail.
- `metrics`: MAE/RMSE/MAPE/MASE/skill against hand-computed values.
- `report`: `accuracy_report.json` schema is stable and contains the headline
  fields the UI reads.

---

## 8. Deliverables (definition of done)

1. `ph_economic_ai/benchmark/` package runnable headless in one command.
2. Committed `accuracy_report.json` + figures = the frozen, reproducible proof.
3. `model.predict()` returns real conformal intervals; the `90.0`/`0.0` literals
   are gone.
4. Hash-chained, prediction-locked live track record with a verifiable chain.
5. Read-only in-app "Methodology & Accuracy" view rendering all of the above.
6. Honest labels on derived outputs and the agent layer.
7. Tests covering causality, coverage, tamper detection, and metrics.

---

## 9. Phase 2 — Accuracy Improvement (earning higher confidence)

**Principle:** "higher confidence" is *not* a setting — it is the consequence of
smaller real error. Conformal band width is a direct function of out-of-sample
residuals, so the only honest way to tighten the displayed interval is to make
the model wrong by less. Every lever below is therefore **gated**: it is merged
only if the Phase 1 backtest shows it *improves the skill score and/or narrows
mean band width at equal coverage*. A lever that doesn't measurably help is
reported as a negative result and dropped — not shipped.

**Why there is genuine room:** DOE's weekly pricing is effectively a formula —
landed cost of *finished* gasoline (MOPS Singapore / RBOB) × FX + fixed taxes and
margins, passed through with a ~1-week lag. Much of next month's pump price is
therefore already determined by the last few weeks of refined-product prices and
FX. The current model leaves this on the table: it feeds on *Brent crude*
(`oil_price`, `BZ=F`) plus a *synthetic cosine* `demand_index`.

### Ranked, individually-gated levers

| # | Lever | Mechanism | Effort |
|---|---|---|---|
| 1 | **Finished-gasoline features** — feed RBOB (`RB=F`, already fetched) / MOPS gasoline with explicit lags, replacing reliance on Brent crude | Pump price tracks refined product, not crude. Largest single expected gain. | Med |
| 2 | **Structural + ML hybrid** — compute analytical landed cost (`RBOB × FX × conv + fixed taxes`) and have ML predict only the *residual margin* | Removes the deterministic part from the learning problem; grounded in the actual DOE mechanism, so it is both more accurate and more defensible. | Med-High |
| 3 | **Pass-through lag features** — trailing 2–4 week MOPS/FX aggregates | At a 1-month horizon much of the answer is already locked in by recent weeks. | Low |
| 4 | **Drop synthetic `demand_index`** | A cosine carries no information; removing it cuts variance. | Low |
| 5 | **Weekly resolution + longer history** | More samples → lower-variance model and tighter conformal quantiles. | Med |
| 6 | **Normalized / Mondrian conformal** — scale band half-width by local volatility | Same nominal coverage, narrower bands in calm regimes, honestly wider around shocks. Makes confidence as high as it truthfully can be. | Med |
| 7 | **Tune HGB via nested CV** (currently `min_samples_leaf=5, max_leaf_nodes=15`, barely tuned) | Marginal vs. the above but cheap; nested CV avoids selection leakage. | Low |

**Recommended core:** levers **1 + 2 + 3** (re-ground the model in the
finished-product pass-through) typically move a Brent-based model from "≈ random
walk" to "clearly beats it," which is what tightens the honest bands. Lever **6**
then presents that confidence as favorably as the truth allows.

### Methodology guards (so Phase 2 stays honest)
- Each lever evaluated through the **same walk-forward, causal backtest** as
  Phase 1 — no separate, friendlier evaluation.
- Feature/lag selection and hyperparameter tuning happen **inside** the training
  fold (nested CV); never on the test window. The leakage guard test from §7
  covers this.
- Results recorded as a **lever-by-lever ablation table** in
  `accuracy_report.json` (skill score and mean band width before/after each
  lever), so the improvement is auditable, not asserted.

### Honest limit (stated, not hidden)
No 1-month forecast can stay confident through a genuine oil shock. The target is
**tight bands in normal months, honestly wide around shocks** — exactly what
normalized conformal (#6) produces. Uniform high confidence is the fake 90%
returning in disguise and is explicitly rejected.

---

## 10. Sources

- World Bank Global Fuel Prices Database —
  https://datacatalog.worldbank.org/search/dataset/0066829/global-fuel-prices-database
  (monthly retail premium gasoline RON95, Dec 2015 – Apr 2025, Open Database License)
- DOE Oil Monitor — https://doe.gov.ph/oil-monitor (weekly retail advisory)
- CEIC PH Retail Petroleum NCR RON95 (monthly, paywalled — reference only)
- GlobalPetrolPrices.com — PH Octane-95 (2016–present, reference/secondary)
```
