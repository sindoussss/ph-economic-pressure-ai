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
Task 3), then through `selection.run_selection_holdout` on the same frame:

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
use for food/electricity/transport (`mean` moves from candidate to baseline
either way).

## What is unchanged

`run_selection_holdout` itself: not modified. `min_train=24`,
`holdout_frac=0.30` (`DEFAULT_HOLDOUT_FRAC`), `MIN_HOLDOUT_PREDICTIONS=12`,
confirmation criterion `holdout_skill > 0 and holdout_p < 0.05 and
dm_stat < 0` — all identical to every other target this protocol has ever
been run against.

## Feasibility

Not yet checkable — none of the six series have been fetched. This document
is committed *before* Task 5 (fetching) runs, per this project's `DEC-010`
discipline: the frame sizes will be recorded here, in this same document,
once Task 5's fetch completes and before any backtest actually runs.

## Decision rule, stated before running

For each of the twelve rows (six categories × two setups) independently,
using `run_selection_holdout`'s own `confirmed` field:

| outcome | conclusion | action |
|---|---|---|
| `not_confirmed_on_holdout` | no selection-honest edge found for this category/setup | report as null in the benchmark artifacts; app-facing labels for this category stay "exploratory, not validated" |
| `confirmed_on_holdout` | a selection-honest edge survives | do **not** promote directly to a validated claim. Check against the audit family's Bonferroni threshold (twelve new tests added to the family), same treatment `fuel_audit`'s confirmation received. Flag to the owner before any manuscript or app wording changes. |

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
   CSVs and provenance sidecars.
2. Record actual frame sizes here, in a "Feasibility, confirmed" section,
   before running any backtest.
3. Extend `selection.run()` with the twelve rows above.
4. Run `python -m ph_economic_ai.benchmark.selection`, writing the extended
   `selection_holdout.json`.
5. Report the verdict table verbatim in this document's own "Result"
   section (added after running, never edited into the sections above).

## Result

*(Not run. This section is added after Task 5's live fetch and the actual
backtest run — both deliberately outside this implementation plan's scope,
per `DEC-010`.)*
