# Pre-registration: do regional freight premiums scale a price CHANGE?

**`Q-ENG-009`, Phase 2. Written 2026-08-01, BEFORE the regression was run.**

`DEC-010` exists because this project once assembled a hypothesis family after
seeing which results were interesting, and `RSK-003` records what that cost. This
document is written first, committed first, and is not edited after the numbers
exist. Anything below that turns out to be wrong gets a correction appended, not
a rewrite.

## The claim under test

`engine/swarm.py` gives each of 17 regions a freight multiplier and
`derive_regional_estimates` applies it to a price CHANGE:

    region_change = reference_change * multiplier

The multipliers are LEVEL premiums. Zamboanga's pump price sits about 8 percent
above NCR's because shipping fuel there costs a roughly fixed number of pesos per
litre. Applying that ratio to a change asserts the premium itself moves 8 percent
when crude moves, and freight rates track bunker prices and shipping capacity
rather than crude in lockstep.

Two models, and they make different predictions about the same coefficient:

| model | prediction |
|---|---|
| **CURRENT**, the premium scales the change | `b` equals the region's multiplier |
| **DELTA-EQUAL**, freight sits in the level | `b` equals 1, and the multiplication should be removed |

## The regression

For each region R, over pricing weeks t:

    delta_R(t) = a + b * delta_NCR(t) + e(t)

NCR is the reference because it is the multiplier table's own base, `1.00`, and
because it is the only debated region with near-complete weekly coverage from
2019.

**OLS, with Newey-West standard errors at 4 lags.** Weekly fuel changes are
autocorrelated by construction, since one crude move propagates over several
weeks, and plain OLS standard errors would be too narrow. The lag length is set
here, in advance, and not tuned to the result.

## Which regions

Only the debated groups with a DOE series:

| region | multiplier | CURRENT predicts | DELTA-EQUAL predicts |
|---|---|---|---|
| Western Visayas | 1.05 | `b = 1.05` | `b = 1.00` |
| Davao Region | 1.05 | `b = 1.05` | `b = 1.00` |

Central Luzon is excluded because DOE publishes nothing for it (`DEC-044`). NCR
is the reference and cannot be regressed on itself.

**The family is two tests.** It is declared here at two so it cannot grow after
the fact. Both are reported whatever they show, and neither is dropped for being
uninteresting. With two tests the Bonferroni-corrected level is 0.025.

## The discriminating margin is 0.05, and that may be too small

Both hypotheses live 0.05 apart. **If the 95 percent interval for `b` contains
both 1.00 and the multiplier, the honest answer is that this data cannot tell
them apart**, and that is a declared outcome rather than a disappointment to be
written around. It is stated here because the alternative — discovering the
interval is wide and then reporting whichever endpoint is nearer — is exactly the
failure `CLM-BASELINE-001` was written about.

A rough expectation, recorded so it can be checked against reality: with about
150 weeks, a residual spread near 0.3 PHP/L and a reference spread near 1.0, the
standard error on `b` lands around 0.025 and the interval spans roughly 0.10.
**That is wider than the gap being tested.** Underpowered is the most likely
outcome and it is not a null result about freight; it is a statement about this
sample.

## Decision rule, fixed in advance

| outcome | conclusion | action |
|---|---|---|
| CI excludes the multiplier, contains 1.00 | DELTA-EQUAL supported | remove the multiplication, and refit the 0.79 pass-through in the same pass per `DEC-021` |
| CI excludes 1.00, contains the multiplier | CURRENT supported | keep the multiplication, refit the multipliers to the estimated slopes; they have never been fitted at all |
| CI contains BOTH | **cannot distinguish** | change nothing. Report the interval, and say the multipliers remain unvalidated rather than validated |
| CI excludes BOTH | neither model is right | report a national change with a per-region uncertainty band, and open a new question |
| the two regions disagree | no single answer | report both, change nothing, and do not pick the region that supports a preferred model |

`DEC-021` binds the first two rows: the pass-through coefficient was fitted
against the current multiply-the-delta behaviour, so one half of a fitted pair
cannot move alone.

## Window and construction, all fixed before running

**Window: pricing weeks from 2023-01-03 onward.** Chosen because it is the common
coverage of all three regions and for no other reason. Western Visayas has 1, 1, 2
and 7 weeks in 2019 to 2022 against 36 to 46 a year from 2023, because the earlier
weeks were published as scans with no text layer. Fitting three regions on three
different periods would confound calibration with sample.

**Regional level: the MEDIAN of city prices in that region-week.** Median rather
than mean because the set of cities reporting changes week to week and one
outlying station should not move the regional figure. Per city the price is DOE's
own `common` column, falling back to the midpoint of the published range when
DOE declines to name one.

**A region-week needs at least 3 reporting cities**, else it is dropped. A
"regional" median over one city is that city.

**Changes are computed only between CONSECUTIVE pricing weeks, exactly 7 days
apart.** The panel has gaps. Differencing across a gap would label a three-week
move as a one-week move, which is precisely the defect `ADR-003` fixed nationally
and `RSK-001` was raised for. A gap ends a run; it does not get bridged.

**Where two documents cover one city-week**, the later filename wins and the
count of such collisions is reported. 411 exist. This rule is arbitrary and is
declared as arbitrary; it is recorded so it is not chosen later to suit an
outcome.

## What would invalidate this test rather than answer it

Recorded now, so they are not rationalised later:

* Fewer than 60 usable weekly changes in either region.
* A reference series whose own changes are mostly zero, which would leave `b`
  estimated off a handful of moves.
* Residuals dominated by one week, meaning one event rather than a relationship.

Each is checked and reported alongside the coefficient whatever it says.

## What this test cannot do

It compares two regions against NCR. It says nothing about the other 14
multipliers, and **nothing at all about Central Luzon**, which is both a DEBATED
region and one with no source. A result here does not license repairing the whole
table; it licenses a statement about Western Visayas and Davao.

---

# Amendment 1, 2026-08-01, written AFTER the first run

**The test was run, and its result is WITHDRAWN as invalid.** This section is
appended rather than folded into the text above, so the original decision rule
stays readable exactly as it was committed.

## What the run returned

| region | b | 95% CI | verdict under the rule above |
|---|---|---|---|
| Western Visayas | 0.371 | [0.013, 0.729] | NEITHER |
| Davao Region | 0.391 | [0.029, 0.752] | NEITHER |

Under the pre-registered rule that is `NEITHER`, whose action is to report a
national change with a per-region band and open a new question. **That is not
being done, because the coefficient does not measure what the rule assumed it
measured.**

## Why it is withdrawn

The pre-registration predicted a residual spread near 0.3 PHP/L. The run gave
2.0, and the reference series contains week-over-week moves of 10 to 20 PHP/L.
Philippine retail adjustments are roughly 0.20 to 3.00 per week; an 18 PHP/L
weekly move is not a price change, it is a measurement change.

Tracing it: **the NCR level tracks which document TEMPLATE DOE used that week.**

| template | cities | brand columns | level range, 2026 |
|---|---|---|---|
| older | 9 | 3, `TOTAL FLYING V UNIOIL` | 56 to 61 |
| newer | 12 | 10, `PETRON SHELL CALTEX PHOENIX ... PTT INDEPENDENT` | 72 to 96 |

Within each template the series moves smoothly. Every large jump sits on a
template switch. A regression of one region's change on another's is therefore
partly a regression of one sheet layout on another, and `b` around 0.37 is
consistent with two series whose common signal is swamped by template noise:
error in the regressor attenuates the slope toward zero.

Separately, one file was mis-dated. `ncr-price-monitoring-for-june-2-8-2026`
parsed as 8 February 2026, because a numeric shape was tried before the month
NAME. A June sheet landed in a February pricing week, won the later-filename
tie-break, and moved that week's level by 18 PHP/L on its own. Fixed, with the
month-name patterns now tried first; it changed the date of 2 files in 2520.

## The rule this exposes

The pre-registration listed three conditions that would invalidate the test:
too few changes, a mostly-flat reference, and one dominant week. **All three
passed.** None of them asked whether the series was measuring a constant thing
over time, and that is the condition that failed.

A pre-registered invalidation list is only as good as the failure modes it
imagined. Adding this one after the fact is legitimate precisely because it
withdraws a result rather than producing one: nothing here converts a null into
a finding, and no branch of the decision rule was edited to fit what came back.

## What has to happen before this test is rerun

1. Establish whether the two templates report the same quantity. The newer sheet
   lists more brands, so its overall range and common price are drawn from a
   wider pool. That alone could shift a level without any price moving.
2. If they do not, the panel needs a per-template basis before any differencing,
   and a change that spans a template switch has to be dropped the way a change
   spanning a calendar gap already is.
3. Re-run unchanged in every other respect. The decision rule above stands as
   written and is not to be renegotiated on the strength of having seen 0.37.

## How fragile the coefficient was, measured

Fixing the one mis-dated file and rebuilding moved the estimate:

| region | b before | b after | R2 before | R2 after |
|---|---|---|---|---|
| Western Visayas | 0.371 | 0.649 | 0.307 | 0.547 |
| Davao Region | 0.391 | 0.611 | 0.379 | 0.599 |

**One file in 2520 moved the coefficient by 0.22 to 0.28**, more than four times the
0.05 gap the test is trying to resolve. That is the argument for withdrawal
stated as a number: an estimate that swings this far on a single date parse is
not yet measuring freight, whatever it reports. The verdict stays WITHDRAWN,
because the template artifact that caused the withdrawal is still present.
