# Pre-registration: selection-holdout re-test of the remaining headline verdicts

**`RSK-004`. Written 2026-08-08, before any of the nine runs below.** The
original mitigation (`benchmark/selection.py`, committed 2026-07-28) ran this
protocol against two targets only -- headline MoM (short) and the fuel
exploratory edge -- and predates this vault's pre-registration discipline
(`DEC-010`), so no prior pre-registration document exists to amend. This one
is written fresh, covering every remaining target `RSK-004`'s own closure
condition names: "every headline verdict is reported through it."

## Disclosure, first

**I have already seen the full-sample verdict for all nine targets below** --
they are the manuscript's own published predictability map, all null except
electricity's driver-only row, which is withdrawn (the +28.3% artifact
`docs/defense/mean-baseline-finding.md` derives). That was never hidden and
could not be: it is the thesis's headline result.

**What I have not seen is the selection-holdout-specific numbers** for any of
these nine: the two-stage split, the selection-segment skill, the
holdout-segment skill, the shrinkage between them, and the holdout DM p-value.
Those come from a genuinely different measurement than the full-sample verdict
-- `run_selection_holdout` fits the model choice on an early segment only and
scores it on a later segment that choice never touched, which the full-sample
audit does not do. This is the same relationship the original two targets had
to their own full-sample verdicts, and the same reason `RSK-004` exists: a
single-stage p-value on a chosen winner is optimistic, even when the winner
turns out to be a naive-losing null.

## What is being tested, exactly

Nine targets, using the frame-building calls `benchmark/corrected_audit.py`
already established and reviewed -- nothing new invented, just the same
frames run through `run_selection_holdout` instead of (or in addition to)
`run_mom_nowcast`/`run_panel`.

| Target | Setup | Frame | Candidates | Baseline pool |
|---|---|---|---|---|
| USD/PHP | 1-month forecast | `TARGETS['fx'].build_frame()` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| YoY inflation | 1-month forecast | `TARGETS['inflation'].build_frame()` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Headline MoM (long, 2007-2026) | full nowcast | `build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom', features=load_long_features())` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Food MoM | full nowcast | `_build_food_frame(load_food_features())` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Food MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Electricity MoM | full nowcast | `_build_electricity_frame(load_electricity_features())` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Electricity MoM | driver-only (the withdrawn flagship) | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Transport MoM | full nowcast | `build_nowcast_frame(target_loader=load_transport_mom, prev_col='prev_mom', features=load_long_features())` | `arima, ets, ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |
| Transport MoM | driver-only | same, `prev_mom` dropped | `ridge, hgb` | `random_walk, seasonal_naive, drift, mean` |

The candidate/baseline-pool split for full-nowcast vs. driver-only rows is
`corrected_audit.py`'s own (`FULL_NEW` minus `mean` vs. `DRV_NEW` minus
`mean` -- `mean` moves from candidate to baseline either way, matching how
`fuel_audit` already treats it in `selection.run()`).

## What is unchanged

`run_selection_holdout` itself: not modified, not reimplemented. Same
defaults the two existing targets already use -- `min_train=24`,
`holdout_frac=0.30` (`DEFAULT_HOLDOUT_FRAC`), `MIN_HOLDOUT_PREDICTIONS=12`.
Same confirmation criterion, unchanged: `holdout_skill > 0 and holdout_p <
0.05 and dm_stat < 0`. Same output shape, appended to the same
`selection_holdout.json` artifact `headline_mom` and `fuel_audit` already
live in, under nine new keys.

## Feasibility, checked in advance

`corrected_predictability_map.json`'s own `n` fields are stale -- they predate
the 2026-07-28 calendar-completeness fix and still show pre-correction counts
(fuel n=52, headline n=61, food/electricity/transport n=151, none matching
the manuscript's current, corrected sample sizes). Read directly from each
target's own frame builder instead of trusting that artifact:

| Target | Frame rows (`n`) | Selection cut | Holdout rows |
|---|---|---|---|
| fx | 105 | 74 | 31 |
| inflation (YoY) | 104 | 73 | 31 |
| headline MoM (long) | 214 | 150 | 64 |
| food / electricity / transport MoM (all four rows each) | 227 | 159 | 68 |

Every target clears `MIN_HOLDOUT_PREDICTIONS` (12) by a wide margin -- the
smallest holdout is 31 rows, over 2.5x the minimum. Nothing here is at risk
of `insufficient_data`, unlike the concern that would have applied had the
stale artifact's much smaller counts (e.g. fx at 38) been trusted instead.

## Decision rule, stated before running

For each of the nine targets independently, using `run_selection_holdout`'s
own `confirmed` field:

| outcome | conclusion | action |
|---|---|---|
| `not_confirmed_on_holdout` (expected, for all nine) | selection-honest re-test agrees with the published null / withdrawn verdict | report the two-stage numbers in the manuscript's methodology section alongside `headline_mom`/`fuel_audit`; no verdict changes; `RSK-004` closes once all nine are reported |
| `confirmed_on_holdout` (not expected for any) | a selection-honest edge survives where the full-sample audit reported none -- a genuinely new finding, not a confirmation | do **not** promote to a positive claim. Treat exactly as `fuel_audit`'s confirmation was treated: report as exploratory, check against the audit family's Bonferroni threshold, register a `CLM-*-EXPLORATORY` claim if it survives that check, and flag to the owner before any manuscript wording changes |

The nine are independent tests of independent targets; no result on one
changes the pre-registered expectation for another, and none is expected to
flip the map's headline claim ("uniformly negative within the confirmatory
family") regardless of outcome, since even a confirmed holdout result would
land in the same exploratory-not-confirmatory category fuel's already does.

## What would make this run uninformative

Checked above and not triggered: every target clears the holdout-size
minimum by a wide margin. The one condition that would still apply if the
actual run's frame sizes differ from what was checked here (e.g. a data
refresh landing between this document and the run) is the same one `RSK-034`
named: if any target's holdout row count falls under roughly 2x
`MIN_HOLDOUT_PREDICTIONS` at run time, the honest reading is reduced power,
not a changed verdict, and that target's row should say so rather than be
folded into the table silently.

## Reported whatever the outcome

For every one of the nine: selection-stage skill, holdout-stage skill,
shrinkage, holdout DM p-value, n and cut, and the verdict string verbatim --
the same fields `headline_mom` and `fuel_audit` already report, so the
extended `selection_holdout.json` is uniform across all eleven targets rather
than treating the new nine differently.

## Mechanical steps, to be done only after this document is committed

1. Extend `selection.run()` to add the nine targets above, reusing
   `corrected_audit.py`'s frame-building calls rather than re-deriving them.
2. Run `python -m ph_economic_ai.benchmark.selection`, writing the extended
   `selection_holdout.json`.
3. Report the verdict table verbatim in [[06 Work/Risk Register]] and
   [[06 Work/Current Handoff]], including if every one comes back
   `not_confirmed_on_holdout` as expected -- that closes `RSK-004`, and the
   closure is the finding, not an anticlimax to skip past.
4. If the manuscript's methodology section (currently reporting only
   `headline_mom`/`fuel_audit`'s two-stage numbers) should be extended to
   show all eleven, that is a wording/scope decision for the owner, not
   assumed here.

## Result, run 2026-08-08

**All nine came back `not_confirmed_on_holdout`, exactly the expected outcome.**
No verdict changes; the manuscript's published null and withdrawn claims all
hold under the selection-honest re-test.

| Target | n | Selection skill | Holdout skill | Shrinkage | Holdout DM p | n holdout |
|---|---|---|---|---|---|---|
| USD/PHP forecast | 105 | +0.0023 | -0.0448 | +0.0471 | 0.4632 | 31 |
| YoY inflation forecast | 104 | +0.2049 | +0.0433 | +0.1616 | 0.7122 | 31 |
| Headline MoM (long) | 214 | +0.0651 | +0.0253 | +0.0398 | 0.6679 | 64 |
| Food MoM, full nowcast | 227 | +0.0964 | +0.0004 | +0.0960 | 0.9949 | 68 |
| Food MoM, driver-only | 227 | -0.1038 | -0.0239 | -0.0799 | 0.5090 | 68 |
| Electricity MoM, full nowcast | 227 | -0.0492 | -0.0065 | -0.0427 | 0.7325 | 68 |
| Electricity MoM, driver-only | 227 | -0.0300 | -0.0003 | -0.0296 | 0.9905 | 68 |
| Transport MoM, full nowcast | 227 | +0.0559 | -0.0360 | +0.0919 | 0.8465 | 68 |
| Transport MoM, driver-only | 227 | -0.0435 | +0.0083 | -0.0517 | 0.9284 | 68 |

Frame sizes matched the feasibility check above exactly (105/104/214/227),
confirming nothing shifted between this document being written and the run.
Every holdout comfortably cleared `MIN_HOLDOUT_PREDICTIONS` (31-68 rows
against a floor of 12); no target returned `insufficient_data`.

**Worth noting rather than skipping past: YoY inflation's selection-stage
skill (+20.5%) was the largest of any target in either the original two or
these nine, and it fully evaporates on the holdout (+4.3%, DM p=0.71) --
the single cleanest illustration in this dataset of exactly the selection
bias `RSK-004` exists to catch. Electricity's driver-only row, the withdrawn
flagship, shows the smallest shrinkage of the nine (-0.0296, i.e. the holdout
is marginally less negative than selection) precisely because there was
never an edge to shrink away from -- both stages sit at essentially zero,
consistent with `mean-baseline-finding.md`'s account of what the +28.3%
artifact actually was.

`selection_holdout.json` now carries all eleven targets. `RSK-004` closes on
this evidence: every headline verdict the manuscript reports has now been
run through the selection-holdout protocol, and none of them changes.
