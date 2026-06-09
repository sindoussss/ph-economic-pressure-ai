# ph_economic_ai — Predictability Audit of the Philippine Economy (Design)

**Date:** 2026-06-08
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data science thesis)
**Builds on:** Phase 1 (accuracy benchmark), Phase 2 (ablation), Efficiency+Mechanism — all on branch `feature/accuracy-evaluation-phase1` (PR #1).

---

## 1. Problem & Goal

The thesis goal is "predicting the economy." The honest, defensible form of that — for a data-science thesis — is a **predictability audit**: apply one rigorous, reproducible framework to several Philippine economic series and report **which are predictable and which are informationally efficient (random walk), with mechanisms where they exist.**

This reframes the existing fuel result (efficient, not predictable) from a dead-end into one data point in a systematic study, and gives the thesis both rigorous **negative** results (efficiency) and a likely **positive** one (inflation).

**Contribution claim:**
> A reproducible framework audits the 1-month-ahead predictability of three Philippine economic series. Fuel and FX are informationally efficient (no standard forecaster beats random walk; DM p > 0.05); inflation is [predictable/efficient per measurement]. Mechanisms are quantified where they exist (DOE fuel pass-through).

**Targets (decided):** fuel (RON95, done), FX (USD/PHP), inflation (CPI YoY).

---

## 2. Scope

### In scope
- A **`Target` abstraction + registry** (`targets.py`) generalizing the benchmark beyond RON95.
- An **audit runner** (`audit.py`) that scores each target through the existing efficiency panel + DM test + conformal calibration and assigns an **efficient/predictable verdict**.
- **Data wiring** for two new gold series: FX monthly (Yahoo) and CPI monthly (FRED), as committed CSVs via `refresh_data.py`.
- **Report + run + Accuracy-view + figure** integration: a cross-target audit table.

### Out of scope
- New targets beyond the three (GDP, employment, trade) — YAGNI.
- The written thesis manuscript + literature review (separate, later; this produces the Results/Discussion evidence).
- Weekly resolution / true MOPS (deferred Phase 3).
- Re-deriving food/electricity (partly derived; excluded as audit targets).

---

## 3. Architecture

New/changed modules under `ph_economic_ai/benchmark/` (headless; no `ui/` imports):

```
benchmark/
├── targets.py        # Target dataclass + TARGETS registry (fuel, fx, inflation)
├── audit.py          # run_audit(target_names) -> per-target verdict + panel + calibration
├── refresh_data.py   # (extend) write data/usd_php_monthly.csv, data/ph_cpi_monthly.csv
├── run.py            # (extend) run_audit across targets; write audit table + figure
└── data/
    ├── world_bank_ron95.csv   # existing (fuel gold)
    ├── usd_php_monthly.csv     # new (fx gold)
    └── ph_cpi_monthly.csv      # new (cpi index -> inflation gold)
```

### 3.1 Target abstraction (`targets.py`)
```
@dataclass
class Target:
    name: str                       # 'fuel' | 'fx' | 'inflation'
    load_gold() -> pd.Series        # monthly 'YYYY-MM' indexed target series
    build_frame() -> pd.DataFrame   # lagged feature frame incl. 'target' col + prev_<name>
    has_mechanism: bool             # fuel True (pass-through), others False
```
`TARGETS` is a dict `name -> Target`. A generic helper `build_target_frame(target_series, driver_df, drivers)` builds `prev_<target>` + 1-month lags + 3-month MAs of each driver, dropna'd to common support. The existing `build_feature_frame` is left intact; `fuel`'s `build_frame` reuses it for continuity, `fx`/`inflation` use `build_target_frame`.

- **fuel:** gold = `ground_truth.load_world_bank_ron95`; drivers = oil, usd, gas proxy, demand (from `features_monthly.csv`); mechanism = pass-through.
- **fx:** gold = `usd_php_monthly.csv`; drivers = lagged FX, oil, CPI.
- **inflation:** gold = CPI index → YoY % change; drivers = lagged inflation, fuel, FX.

### 3.2 Audit runner (`audit.py`)
```
run_audit(target_names, min_train) -> list[dict]
```
For each target: build frame, call the existing `efficiency.run_panel` (7 forecasters, DM vs random walk) on its feature columns, compute split-conformal calibration on the best non-RW method, and assign a verdict:
- **'predictable'** if any method has `dm_p < 0.05 and skill_vs_rw > 0`,
- **'efficient'** otherwise.

Each row: `{target, verdict, best_method, best_skill, best_dm_p, n, panel: [...], calibration: [...]}`. `best_method` = highest skill with dm_p<0.05 if predictable, else 'random_walk'.

### 3.3 Data wiring (`refresh_data.py` extension)
- `build_fx_csv()`: Yahoo `PHP=X` monthly (10y) → `data/usd_php_monthly.csv` (`date,usd_php`).
- `build_cpi_csv()`: FRED `fredgraph.csv?id=PHLCPIALLMINMEI` (monthly CPI index) → `data/ph_cpi_monthly.csv` (`date,cpi_index`). Fallback source documented (DBnomics IMF IFS) if FRED id is retired. Loader converts index → YoY inflation %.
All committed so `run.py` is offline-reproducible (same pattern as the WB workbook).

### 3.4 Integration
- `report.py`: add an `audit` key (list of per-target verdict rows). The existing fuel-centric keys remain (fuel stays the detailed headline; audit is the cross-target summary).
- `run.py`: after the existing fuel run, call `run_audit(['fuel','fx','inflation'])`, write `artifacts/audit_table.json`, add `audit=` to the report, and emit a cross-target verdict figure.
- `figures.py`: `plot_audit_verdicts(rows)` — per-target best-skill bar colored by verdict.
- `accuracy_view.py`: `audit_summary()` panel — `target | verdict | best method | skill | DM p`.

---

## 4. Data Flow

```
data/world_bank_ron95.csv ─┐
data/usd_php_monthly.csv  ─┼─ targets.TARGETS[name].build_frame() ─► frame ─┐
data/ph_cpi_monthly.csv   ─┘                                                 ▼
features_monthly.csv ──────────────────────────────► efficiency.run_panel (per target)
                                                                            │
                                              audit.run_audit ─► per-target verdict rows
                                                                            ▼
                                   report.build_report(... audit=) + audit_table.json + figure
                                                                            ▼
                                                  ui/accuracy_view audit panel
```

---

## 5. Error Handling
- FRED/Yahoo fetch failure in `refresh_data` → keep the committed CSV; log and continue (don't crash the audit).
- A target's gold series too short after lags (`< min_train + buffer`) → that target's row reports `verdict='insufficient_data'` with `n`, rather than raising; the audit continues for other targets.
- CPI series id retired at FRED → `build_cpi_csv` prints the documented DBnomics fallback URL; the loader still reads whatever committed CSV exists.
- ARIMA/ETS per-fold failures already fall back to random walk (existing `forecasters.py`).

---

## 6. Testing
- `targets`: each `Target.load_gold()` returns a sorted monthly Series from a tiny committed/tmp CSV; `build_frame()` has a `target` column and lagged features with no NaNs; the inflation loader converts a known CPI index to correct YoY %.
- `build_target_frame`: lag columns equal the raw driver shifted by one (causality); MA columns correct on a known series.
- `audit`: a synthetic *predictable* target (target = 0.9·driver_lag1 + noise) → verdict 'predictable' with a method beating RW; a synthetic random-walk target → verdict 'efficient'; insufficient-length target → 'insufficient_data'.
- `report`/`accuracy_view`: `audit` key present and rendered.

---

## 7. Deliverables (definition of done)
1. `Target` registry covering fuel, fx, inflation; generic `build_target_frame`.
2. `run_audit` producing per-target verdicts via the existing panel + DM + conformal.
3. Committed FX + CPI monthly gold CSVs (real data via `refresh_data`).
4. Cross-target **audit table** in `accuracy_report.json` + `audit_table.json` + a verdict figure.
5. Accuracy view shows the audit panel.
6. Tests for loaders, frame causality, and verdict logic.
7. Reproducible: `python -m ph_economic_ai.benchmark.run` regenerates everything; `refresh_data` rebuilds the gold CSVs.

---

## 8. The contribution — measured results

Run on committed data (1-month-ahead, World Bank fuel + Yahoo FX + DBnomics/IMF CPI;
n = 79 backtest months for fuel). Source: `artifacts/audit_table.json`.

| Target | Verdict | Best method | Skill vs RW | DM p |
|---|---|---|---|---|
| fuel (RON95) | **efficient** | random_walk | 0.00 | — |
| FX (USD/PHP) | **efficient** | random_walk | 0.00 | — |
| inflation (CPI YoY) | **efficient** | random_walk | 0.00 | — |

- **Finding:** all three Philippine macro series are **informationally efficient at the
  1-month horizon** — across the full forecaster panel (random walk, drift,
  seasonal-naive, ARIMA, ETS, Ridge, HGB), *no method significantly beats naive
  persistence* (no method clears DM p < 0.05 with positive skill) for any target.
- **Interpretation, grounded in the literature:** the FX result reproduces the
  classic **Meese-Rogoff (1983)** random-walk-beats-structural-models finding; the
  inflation result reproduces **Atkeson-Ohanian (2001)**, who showed naive forecasts
  are hard to beat (here, next-month YoY inflation is so persistent that random walk
  already captures it); the fuel result follows from the partial, lagged DOE
  pass-through (β ≈ 0.56) of a near-random-walk cost driver (Efficiency+Mechanism
  spec). Three independent series, one consistent conclusion.
- **Why it matters (data science):** a rigorous, reproducible **predictability-audit
  protocol** (causal walk-forward + 7-method panel + Diebold-Mariano + calibrated
  conformal intervals) applied to an emerging-market economy, reproducing two
  landmark efficiency results for the Philippines and showing *where forecasting
  effort is and isn't worthwhile* — the honest, defensible form of "predicting the
  economy."
- **Honest caveat:** "efficient" here means *unbeatable by these methods at monthly
  resolution on this data* — not a proof of strict efficiency. Higher frequency
  (weekly), longer history, or richer feature sets could change the FX/inflation
  verdicts; that is stated future work, not a claim closed here.

---

## 9. Sources / references (for the write-up)
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997).
- Meese & Rogoff (1983) — exchange-rate random-walk benchmark (FX efficiency context).
- Atkeson & Ohanian (2001) — inflation forecasts vs naive (inflation predictability context).
- Vovk, Gammerman & Shafer — conformal prediction.
- Data: World Bank Global Fuel Prices DB (fuel); Yahoo Finance `PHP=X` (FX); FRED `PHLCPIALLMINMEI` / DBnomics IMF IFS (CPI).
