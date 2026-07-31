# Pre-registration: are the freight multipliers right as LEVEL premiums?

**`Q-ENG-009` Phase 2b. Written 2026-08-01, before the intervals were computed.**

## Disclosure, first, because it limits what this document can claim

**Two of the region ratios below were already computed and seen**, as an
exploratory follow-up recorded in Amendment 2 of the Phase 2 pre-registration:
Western Visayas at 0.997 and Davao at 0.962 against a claimed 1.05 for both.

So this is **not a blind pre-registration for those two**. What was fixed before
anything was computed is the estimator, the interval procedure, the decision rule,
the correction policy, and the list of regions — and the other seven regions in
that list have not been looked at in any form. The honest description is: two
confirmatory-looking rows carry prior exposure and are flagged in the output;
seven do not.

Recording this rather than quietly re-registering a known result is the point.
`DEC-010` is about not assembling a family after seeing results, and the family
here is fixed by DOE's coverage rather than by choice.

## What is being tested, and what is NOT

Phase 2 asked whether a freight premium should scale a price CHANGE and **closed
as infeasible**: the two hypotheses differ by 5 percent of a slope, and resolving
that from weekly retail changes needs 527 to 779 years of data.

This asks a different and answerable question. `engine/swarm.py` defines each
multiplier as a LEVEL premium -- "freight/logistics premium over NCR" -- so
whatever it is used for, the constant itself makes a checkable claim:

    regional pump price level  ~  NCR level * multiplier

A level is about 70 PHP/L and a 5 percent premium is about 3.5 PHP/L, against
weekly changes of under 1. **The signal is roughly seventy times larger than the
one Phase 2 failed on**, which is why this is worth running when that was not.

**This does not settle whether the multiplication should be applied to changes at
all.** That question stays closed and unanswered. This settles only whether the
constants are the right numbers for the thing they are documented to be.

## The estimator

For each region R and each pricing week t where both R and NCR have a level:

    ratio_R(t) = level_R(t) / level_NCR(t)

Reported as the **median ratio** across weeks, with a **95 percent bootstrap
interval, 10,000 resamples of whole weeks**, seed fixed at 20260801. Weeks are
resampled rather than observations, because prices within a week are not
independent. Median rather than mean, for the same reason the panel uses a median
within a region: one outlying city-week should not move the estimate.

## Which regions, fixed before running

Every region whose provinces appear in the DOE panel, which is a coverage fact
rather than a selection:

| region | multiplier | source |
|---|---|---|
| Western Visayas | 1.05 | Visayas *(seen)* |
| Central Visayas | 1.04 | Visayas |
| Eastern Visayas | 1.07 | Visayas |
| Zamboanga | 1.08 | Mindanao |
| Northern Mindanao | 1.06 | Mindanao |
| Davao Region | 1.05 | Mindanao *(seen)* |
| SOCCSKSARGEN | 1.07 | Mindanao |
| Caraga | 1.07 | Mindanao |
| BARMM | 1.10 | Mindanao |
| CALABARZON | 1.03 | South Luzon, if fetched in time |
| MIMAROPA | 1.08 | South Luzon, if fetched in time |
| Bicol Region | 1.06 | South Luzon, if fetched in time |

NCR is the base at 1.00 by construction and is not tested. Central Luzon, Ilocos,
Cagayan Valley and CAR have no DOE series at all (`DEC-044`) and cannot be tested
at any point; they are reported as untestable rather than omitted.

**A region needs at least 52 paired weeks**, else it is reported as insufficient
rather than estimated.

Family size is however many of the twelve have data. With a family this size the
Bonferroni-corrected level is 0.05 divided by that count, and it is applied to the
interval rather than to a p-value: intervals are widened accordingly and both the
uncorrected and corrected intervals are printed.

## Decision rule, fixed in advance

Per region:

| outcome | conclusion | action |
|---|---|---|
| corrected interval contains the multiplier | consistent | leave that multiplier alone |
| corrected interval excludes it, and contains 1.00 | no premium detectable | see the correction policy |
| corrected interval excludes both | premium exists but is the wrong size | see the correction policy |
| fewer than 52 paired weeks | insufficient | report, change nothing |

**Correction policy, fixed here so it is not chosen once the numbers are in.** A
multiplier whose interval excludes it is replaced by the **median ratio, rounded
to two decimals**, which is the precision the existing table is written to. No
region is corrected on a point estimate whose corrected interval covers the
current value, and no region without data is touched by inference from a
neighbour.

**A region that is corrected keeps its ORIGINAL value recorded beside it**, in
the code, with the date and this document named.

## Why this does not require refitting the pass-through

`DEC-021` bars moving one half of a fitted pair. The 0.79 pass-through in
`engine/anchoring.py` was fitted on the NATIONAL series, as an OLS slope of the
mechanical value beating a no-change baseline on national MAE, 2.21 against 2.64.
No regional multiplier entered that fit. The two constants are applied in
sequence, crude to national and then national to regional, but they were never
estimated together, so correcting one does not invalidate the other.

This paragraph is here because the claim needed checking rather than assuming,
and `DEC-021` is exactly the rule that would have been violated by assuming it.

## What would make this test uninformative

* A region whose paired weeks all fall in one short period, so the ratio reflects
  one episode. Reported as the span of paired weeks per region.
* A ratio whose week-to-week spread is so wide that the interval spans more than
  0.10, which is twice the largest premium in the table.
* NCR itself being unrepresentative as a base. It is a single metro of 13 cities
  and its own split-half reliability on changes is 0.685; on LEVELS reliability is
  reported alongside, because a noisy base biases every ratio.

## What this test cannot do

It cannot say whether multiplying a change by these constants is right. It cannot
speak for the four regions with no source. And a corrected multiplier remains an
observed price ratio, not a fitted forecast parameter: the regional figures stay
DERIVED and stay labelled that way (`DEC-044`, Phase 0).
