# Pre-registration: does the regional derivation beat doing nothing?

**Written 2026-08-01, before the estimator and the decision rule were fixed.**
This is the confirmatory form of `CLM-REGIONAL-EXPLORATORY-001`.

## Disclosure, first

**I have already seen a pooled result over 2024 to 2026**: zero-change scored
0.645 against the multiplier model's 0.520. That exploratory run is recorded in
the Claim Registry and is what motivated this document.

So this is **not blind**. What was NOT fixed when that number was computed, and
is fixed here, is everything that decides what the number MEANS: the naive pool,
the paired test, the significance level, the invalidation conditions, and above
all **what happens to the app if the derivation loses**. That last one is the
point of writing this. A result with no pre-committed action is an invitation to
argue about the action once the result is in.

A genuinely blind test is not available: the panel is what it is and no new
weeks exist. Recording that plainly is better than manufacturing a hold-out by
pretending I have not seen the pooled figure.

## The question

The app displays 17 regional price-change figures. Four region groups are
debated; the other 13 are those scaled by a freight premium. **Is that
derivation more accurate than a trivial alternative?**

This is not `Q-ENG-009`, which asks whether a level premium SHOULD scale a
change and closed as infeasible. This asks whether the thing being shipped
performs, which is answerable and has never been asked.

## The comparison

Per region R and pricing week t, the forecast error against DOE's published
change. Four arms, all seeing identical actuals:

| arm | definition |
|---|---|
| **DERIVATION** | the app's own: `delta_R = delta_ref * mult_R / mult_anchor` |
| **DELTA-EQUAL** | `delta_R = delta_ref`, the premium removed |
| **ZERO** | `delta_R = 0`, no regional move at all |
| **PERSISTENCE** | `delta_R` = that region's change in the previous week |

`ZERO` is in the pool because for a mean-reverting weekly change it is the
analogue of the historical mean, and `DEC-008` exists because omitting the mean
from a naive pool turned four nulls into four false positives.

`PERSISTENCE` is in the pool because a random walk is the standard comparator
and its absence would leave the test open to the same objection.

**The reference change is the ACTUAL national change, not a forecast.** This
isolates the regional step. A test driven by the app's own national forecast
would confound two questions, and the national one is already graded.

## Primary metric and test

**Mean absolute error in PHP/L**, pooled across regions and weeks.

**Paired bootstrap over WHOLE WEEKS**, 10,000 resamples, seed 20260801.
Resampling weeks rather than observations because regions within a week share
the national move and are not independent. The statistic is the DIFFERENCE in
MAE between the derivation and each naive arm, and the interval is on that
difference.

**Alpha 0.05, Bonferroni-corrected across the three comparisons**, so 0.0167.

## Decision rule, fixed here

Let `d = MAE(derivation) - MAE(best naive arm)`.

| outcome | conclusion | action |
|---|---|---|
| corrected CI for `d` lies entirely **below 0** | the derivation beats the best naive | keep the regional figures as they are |
| corrected CI **contains 0** | no measurable difference | **stop presenting per-region figures as forecasts.** Keep the map, label every derived figure as "no better than assuming no regional change", and say so in `regional_basis` |
| corrected CI lies entirely **above 0** | the derivation is WORSE than doing nothing | **the same labelling, plus the four unmeasurable regions stop showing a number at all** |

The second and third rows do **not** delete the regional map. `DEC-021`'s spirit
holds: the number stays, the claim around it stops overreaching, and that has
been this project's answer every previous time. What changes is what the
interface asserts.

**The four unmeasurable regions are treated differently in row three and only in
row three.** Ilocos, Cagayan Valley, Central Luzon and CAR have no DOE series,
so they cannot be in the test at all. If the derivation is measurably worse than
nothing WHERE it can be checked, then showing a precise number where it cannot
be checked is asserting an accuracy the evidence contradicts.

## Reported whatever the outcome

* Per-region MAE and n, not only the pool. A pooled win hiding eleven regional
  losses is not a win.
* The count of ungradable regions, by name.
* Whether each region's own premium was measured or assumed.

## What would make this uninformative

* Fewer than 60 paired weeks.
* A reference series whose changes are mostly zero, which would make every arm
  identical.
* **Actuals too noisy to separate the arms.** Weekly regional changes have
  split-half reliability 0.685 to 0.948, and a noisy target systematically
  flatters `ZERO`, which can never be wrong by more than the true move. If the
  three intervals all contain 0, the honest reading is that this data cannot
  rank the arms, NOT that they are equivalent.

That last one is the real risk here and it is stated in advance because it is
the finding most likely to be reported as "the derivation is no worse".

## What this cannot say

It says nothing about whether a level premium should scale a change, nothing
about the four regions with no series, and nothing about the app end to end:
it grades one step, driven by a known national change.
