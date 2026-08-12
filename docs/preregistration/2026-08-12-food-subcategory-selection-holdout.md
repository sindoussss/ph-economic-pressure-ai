# Pre-registration: selection-holdout test of six PSA food sub-categories

Written 2026-08-12, before any of the six series below have even been
fetched. Extends `RSK-004`'s protocol (`benchmark/selection.py`) to six new
targets that did not exist when the original eleven headline verdicts were
tested.

## Disclosure, first

**I have not seen any result for any of these six targets.** Unlike the
2026-08-08 pre-registration (which pre-registered the *method* for targets
whose full-sample verdict was already published), these six series have
never been fetched, backtested, or looked at in any form before this
document. This is a genuinely blind pre-registration.

## What is being tested

Six targets, each run through both a full nowcast and a driver-only
ablation (`benchmark/food_subcategory_nowcast.py::run_subcategory_nowcast`,
Task 3). The frame both the nowcast entry point and
`selection.run_selection_holdout` actually consume is built by the shared
helper, `food_subcategory_nowcast._build_subcategory_frame` --
`run_subcategory_nowcast` is a separate full-sample entry point built on top
of that same helper, not itself on the selection-holdout path:

| Target | Setup | Frame | Candidates | Baseline pool |
|---|---|---|---|---|
| Rice MoM | full nowcast | `food_subcategory_nowcast._build_subcategory_frame('rice', ...)` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Rice MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Meat MoM | full nowcast | same pattern, category='meat' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Meat MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Fish MoM | full nowcast | same pattern, category='fish' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Fish MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Dairy & eggs MoM | full nowcast | same pattern, category='dairy_eggs' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Dairy & eggs MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Vegetables MoM | full nowcast | same pattern, category='vegetables' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Vegetables MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Sugar MoM | full nowcast | same pattern, category='sugar' | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Sugar MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |

Same candidate/baseline-pool split full-nowcast vs. driver-only rows already
use for food/electricity/transport. The baseline pool is identical between
the two setups: `mean` is never a candidate in either one -- it isn't in
`PANEL_METHODS` at all -- it is always part of the baseline pool.

## What is unchanged

`run_selection_holdout` itself: not modified. `min_train=24`,
`holdout_frac=0.30` (`DEFAULT_HOLDOUT_FRAC`), `MIN_HOLDOUT_PREDICTIONS=12`,
confirmation criterion `holdout_skill > 0 and holdout_p < 0.05 and
dm_stat < 0` — all identical to every other target this protocol has ever
been run against.

## Feasibility, confirmed

All six series were fetched live from PSA OpenSTAT on 2026-08-12 (Mechanical
step 1). Each came back with the same depth as the existing electricity
series: 391 rows, 1994-01..2026-07 — no truncated backcast for any of the
six.

Each category's frame was then built with
`food_subcategory_nowcast._build_subcategory_frame` (Mechanical step 2),
*before* any backtest ran:

| category | frame rows | date range |
|---|---|---|
| rice | 228 | 2007-08 .. 2026-07 |
| meat | 228 | 2007-08 .. 2026-07 |
| fish | 228 | 2007-08 .. 2026-07 |
| dairy_eggs | 228 | 2007-08 .. 2026-07 |
| vegetables | 228 | 2007-08 .. 2026-07 |
| sugar | 228 | 2007-08 .. 2026-07 |

All six categories join to the identical range (2007-08..2026-07, 228 rows)
because the frame is bounded by the food predictor panel's own coverage
(`food_nowcast.load_food_features()`), not by the PSA CPI series, which are
all much longer (391 rows back to 1994) than the predictor panel.

None of the six falls under the ~60-70 row full-frame floor this document
flagged as worth watching (228 rows is well clear of it, matching the
existing `food_mom_full`/`food_mom_driver_only` frame's own size). Whether
any individual row's *holdout* segment clears the ~24-row power floor
(2x `MIN_HOLDOUT_PREDICTIONS`) is determined by `run_selection_holdout`'s own
`split_point` once the backtest actually runs (Mechanical step 4) and is
reported per-row in the Result section below, not here.

## Decision rule, stated before running

For each of the twelve rows (six categories × two setups) independently,
using the returned dict's `verdict` field (`run_selection_holdout` returns
`verdict: 'confirmed_on_holdout'` or `verdict: 'not_confirmed_on_holdout'`;
`confirmed` is only a local variable inside `selection.py`, not part of the
returned dict):

| outcome | conclusion | action |
|---|---|---|
| `not_confirmed_on_holdout` | no selection-honest edge found for this category/setup | report as null in the benchmark artifacts; app-facing labels for this category stay "exploratory, not validated" |
| `confirmed_on_holdout` | a selection-honest edge survives | do **not** promote directly to a validated claim. Check against the audit family's Bonferroni threshold. Note: `multiple_testing.build_family()` currently reads only `accuracy_report.json`'s hardcoded candidate list and does not include selection-holdout results — these twelve rows are NOT automatically part of that family. Wiring a confirmed result into the family (if any row confirms) is a separate decision requiring its own review, because growing the family from its current size retroactively tightens `bonferroni_threshold = alpha/m` for every existing member, which could flip an already-published result. Do not add these rows to the family without that review. Flag to the owner before any manuscript or app wording changes. |

Twelve independent tests; no result on one changes the pre-registered
expectation for another.

## What would make this run uninformative

If any category's frame, once built in Task 5, has a holdout row count
under roughly 2x `MIN_HOLDOUT_PREDICTIONS` (i.e. under ~24), the honest
reading for that row is reduced power, not a verdict — recorded as such
rather than folded silently into the results table.

## Reported whatever the outcome

For every one of the twelve rows: selection-stage skill, holdout-stage
skill, shrinkage, holdout DM p-value, n and cut, and the verdict string
verbatim — the same fields every other `selection_holdout.json` entry
already carries.

## Mechanical steps, to be done only after this document is committed

1. **[Separate, owner-gated action — not part of this implementation plan.]**
   Fetch the six series live (`psa_cpi.fetch_rice_cpi()` etc.), commit the
   CSVs and provenance sidecars. The fetch should assert/log the first
   available month for each series before committing the CSV, so a
   truncated backcast is caught immediately rather than discovered later.
2. Record actual frame sizes here, in a "Feasibility, confirmed" section,
   before running any backtest.
3. Extend `selection.run()` with the twelve rows above. If any row
   confirms, wiring it into the Bonferroni family (per the corrected
   decision-rule text above) is an explicit additional step requiring its
   own review — not implied by this step alone.
4. Run `python -m ph_economic_ai.benchmark.selection`, writing the extended
   `selection_holdout.json`.
5. Report the verdict table verbatim in this document's own "Result"
   section (added after running, never edited into the sections above).

## Result

*(Not run. This section is added after Task 5's live fetch and the actual
backtest run — both deliberately outside this implementation plan's scope,
per `DEC-010`.)*
