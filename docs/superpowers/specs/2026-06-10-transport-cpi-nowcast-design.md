# ph_economic_ai — MoM Transport-CPI Nowcast (Design)

**Date:** 2026-06-10
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous (data-science thesis)
**Builds on:** the MoM CPI nowcast pipeline (`benchmark/nowcast.py`) and the predictability audit on `master`.

---

## 1. Problem & Goal

The thesis has one validated positive (headline MoM inflation is nowcastable) but its *driver* edge is not significant — the win is own-dynamics. **Transport** inflation, by contrast, is mechanically driven by fuel, and fuel (Brent + USD/PHP) is **observable within the month, before the PSA release**. This makes Transport CPI the strongest free candidate for a *significant within-month driver edge* — a fuel→inflation pass-through nowcast.

**Goal:** nowcast month-on-month **Transport** CPI inflation before its official release, using free within-month-observable fuel/FX data, and test honestly (Diebold–Mariano vs the strongest naive baseline) whether (a) it beats persistence and (b) the **driver-only** model beats persistence — the latter being the bold, honest claim "Strata nowcasts the fuel-driven component of inflation ahead of the official figure."

**Honest stance:** the verdict is reported either way. If Transport MoM is efficient, that is a legitimate result; the design does not assume a win.

---

## 2. Scope

### In scope
- A free, automated PSA gold loader for monthly **Transport** CPI (PX-Web), committed as a frozen CSV.
- A Transport-MoM nowcast that reuses the existing `nowcast.py` runners (panel, DM-vs-best-naive, hollow-win guard, driver-only ablation, conformal).
- Report key + `run.py` wiring + a one-line accuracy-view note.

### Out of scope
- Other CPI components (food, housing) — a later, separate target if this succeeds.
- Weekly resolution (blocked by paid fuel gold; not pursued).
- The swarm/LLM path; live forward updates beyond the committed backtest.

---

## 3. Data (all free)

### 3.1 Gold target — PSA OpenSTAT (confirmed reachable, JSON PX-Web API)
- **Source:** PSA OpenSTAT PX-Web, database path `DB/2M/PI/CPI/2018NEW/`.
- **Table:** `0012M4ACP28.px` — "Consumer Price Index for All Income Households **by Commodity Group** (2018=100): **January 1994 – present**" (long history → high power). The shorter `0012M4ACP22.px` (2018-based start) is a fallback if `ACP28`'s dimensions prove harder to parse.
- **Selection:** Geolocation = **Philippines** (national), Commodity Group = **Transport**, all months. Dimension codes/value IDs are discovered at runtime via the table's PX-Web **metadata GET** (the `.px` endpoint returns its `variables`), then a **POST query** (`response.format = json-stat2`) returns the series.
- **Decision:** use the long `ACP28` (1994–present) for maximum power, with a **regime-heterogeneity caveat** (1994–present spans multiple oil and FX regimes), mirroring the MoM longer-sample treatment.
- **Output:** a frozen, committed `benchmark/data/psa_transport_cpi_monthly.csv` with columns `date` (`YYYY-MM`), `transport_cpi` (index, 2018=100). The MoM transform (`cpi_to_mom`) yields the nowcast target.

### 3.2 Predictors (existing, free)
Brent (`BZ=F`), USD/PHP (`PHP=X`), RBOB→PHP fuel proxy — from the committed `features_monthly_long.csv` (2007–present). Used **contemporaneously** (within-month) plus the previous Transport MoM, matching the nowcast convention.

### 3.3 Provenance & reproducibility
PSA is the authoritative, citable source. The committed CSV freezes the series; the live PX-Web fetch is a one-off refresh step (network), not run during the backtest.

---

## 4. Architecture

```
benchmark/
├── psa_cpi.py            # NEW: PX-Web fetch + parse -> Transport CPI monthly;
│                         #   fetch_transport_cpi() (network, writes gold CSV);
│                         #   load_transport_cpi(csv) -> index Series;
│                         #   load_transport_mom(csv) -> MoM inflation % (reuses cpi_to_mom)
├── transport_nowcast.py  # NEW: build frame (features=long, target=transport MoM) ->
│                         #   run_mom_nowcast + run_driver_only_ablation -> dict + artifact
├── nowcast.py            # reused unchanged (build_nowcast_frame(features=, target_loader=),
│                         #   run_mom_nowcast(frame=), run_driver_only_ablation(frame=))
├── report.py             # + 'transport_nowcast' key
├── run.py                # run transport_nowcast; print; record; write artifact
└── ui/accuracy_view.py   # one-line transport-nowcast note
```

### 4.1 `psa_cpi.py`
- `PSA_PXWEB_URL` = `https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/2018NEW/0012M4ACP28.px`.
- `fetch_transport_cpi(out_csv=TRANSPORT_CSV) -> None`: GET metadata → resolve the Commodity-Group dimension code and the value id whose label is "Transport", plus Geolocation = Philippines; POST the json-stat2 query; parse to a monthly `date,transport_cpi` frame; write the committed CSV. Network one-off; raises on failure (no fabrication).
- `load_transport_cpi(csv=TRANSPORT_CSV) -> pd.Series`: read the committed CSV → index Series (`YYYY-MM`).
- `load_transport_mom(csv=TRANSPORT_CSV) -> pd.Series`: `cpi_to_mom(load_transport_cpi(...))` (reuse `targets.cpi_to_mom`).

### 4.2 `transport_nowcast.py`
- `run_transport_nowcast(min_train=24, features=None) -> dict`:
  - `features` defaults to the long feature frame (`longsample.load_long_features()`).
  - `frame = nowcast.build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom', features=features)`.
  - `mom = run_mom_nowcast(min_train, frame=frame)`; `abl = run_driver_only_ablation(min_train, frame=frame)`.
  - Returns `{n, mom: {...minus panel/calibration...}, driver_ablation: {...}, driver_edge: bool}`.

### 4.3 Integration
- `report.build_report(..., transport_nowcast=None)` + `REQUIRED_KEYS`.
- `run.py`: call `run_transport_nowcast`, print `mom`/`driver_edge`, record, write `transport_nowcast_table.json`. Guard: if the gold CSV is absent, record `{'verdict':'not_run','reason':'transport gold missing'}` and continue.
- `accuracy_view`: `transport_nowcast_summary()` one-line note (verdict, best method, skill, DM p, driver_edge).

---

## 5. Data Flow
```
PSA PX-Web (ACP28) ─► fetch_transport_cpi ─► data/psa_transport_cpi_monthly.csv (committed)
                                                  │ load_transport_mom (cpi_to_mom)
features_monthly_long.csv ──┐                     ▼
                            └─► build_nowcast_frame(features=long, target=transport MoM)
                                                  │
                 run_mom_nowcast + run_driver_only_ablation (existing)
                                                  │
                 transport_nowcast_table.json + report 'transport_nowcast' ─► view note
```

## 6. Error Handling
- PSA fetch failure (network / dimension labels changed) → `fetch_transport_cpi` raises with a clear message; the committed CSV (if present) remains usable. `run.py` records `not_run` if the CSV is absent.
- "Transport" label/whitespace variance → match case-insensitively and trimmed; if no commodity-group value matches, raise listing the available labels (so the fix is obvious).
- Short overlap with predictors (long features start 2007) → the Transport×features join begins ~2007; the existing `insufficient_data` guard applies.

## 7. Testing
- `psa_cpi` parser: a synthetic json-stat2 payload (two commodity groups incl. Transport, a few months) → `_parse` returns the Transport monthly series correctly; `load_transport_mom` yields MoM % from a synthetic CSV.
- `transport_nowcast.run_transport_nowcast(features=synthetic_long)` with a monkeypatched `load_transport_mom` → returns `{n, mom, driver_ablation, driver_edge}` with expected keys; `n` reflects the frame.
- `report`/`accuracy_view`: `transport_nowcast` key present and rendered.
- The live PSA fetch is a one-off refresh (network), not a unit test.

## 8. Deliverables (definition of done)
1. `psa_cpi.py` + committed `psa_transport_cpi_monthly.csv` (Transport, monthly).
2. `transport_nowcast.py` reusing the existing runners.
3. `transport_nowcast` block in `accuracy_report.json` + `transport_nowcast_table.json`.
4. Accuracy-view one-line note.
5. Tests for the parser, MoM transform, and nowcast wiring.
6. Reproducible via refresh (fetch gold) + `python -m ph_economic_ai.benchmark.run`.

## 9. The contribution — to be filled from the real run
- **Result:** on n = [n] months, Transport MoM nowcast is [beats_best_naive / no better than naive] (best method [m], skill [x], DM p [p]); the **driver-only** model is [significant / not] (`driver_edge` = [bool], best driver RMSE [r] vs naive [rb]).
- **Interpretation:** [if driver_edge True] Strata nowcasts the **fuel-driven component of inflation** ahead of the official figure — a genuine, significant within-month information edge, contrasting with the headline result where the driver edge was underpowered. [if efficient] Transport MoM is also efficient at this horizon — an honest negative that still sharpens the predictability map.
- **Honesty notes:** this is a *nowcast* (information-timing), not a claim to beat a market; the gold is official PSA data; the 1994-based window spans multiple regimes (caveat).

## 10. Sources / references
- PSA OpenSTAT PX-Web: `DB/2M/PI/CPI/2018NEW/0012M4ACP28.px` (CPI by commodity group, 2018=100, 1994–present).
- Nowcasting: Giannone, Reichlin & Small (2008); Bańbura et al. (2013). Significance: Diebold–Mariano (1995); HLN (1997).
- Predictors: Yahoo Finance `BZ=F`/`PHP=X`/`RB=F`.
