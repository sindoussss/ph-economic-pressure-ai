# Pre-registration: the freight-multiplier level test, re-run on the basis-guarded panel

**`RSK-034`. Written 2026-08-07, before the corrected panel was scored.**
Amends `docs/preregistration/2026-08-01-regional-level-premiums.md` rather than
replacing it: the estimator, the interval procedure, the region list, and the
decision and correction rules are unchanged. Only the input panel changes.

## Disclosure, first

**I have already seen the published result**, `regional_level_premiums.json`,
committed 2026-08-01 and folded into the app the same day (`ADR-017`):

| region | old multiplier | verdict | corrected to |
|---|---|---|---|
| CALABARZON | 1.03 | refuted | 0.98 |
| Western Visayas | 1.05 | refuted | 1.00 |
| Central Visayas | 1.04 | refuted | 1.00 |
| Eastern Visayas | 1.07 | refuted | 1.00 |
| Zamboanga | 1.08 | refuted | 0.98 |
| Northern Mindanao | 1.06 | refuted | 0.99 |
| Davao Region | 1.05 | refuted | 0.96 |
| SOCCSKSARGEN | 1.07 | refuted | 0.99 |
| Caraga | 1.07 | refuted | 1.02 |
| MIMAROPA | 1.08 | consistent | 1.08 (unchanged) |
| Bicol Region | 1.06 | consistent | 1.06 (unchanged) |

Nine of eleven testable regions were refuted and corrected, always downward.
MIMAROPA and Bicol were the positive control and were left alone.

## What is different in the input, exactly

Same defect as the accuracy test, same fix, same date: `regional_grading._price`
merged DOE's `common` price with the midpoint of its range, and
`tools/regional_multiplier_backtest.py`'s own `_price` (which this script
imports) carried a second, independent copy of the same bug, fixed in the same
pass (`RSK-034` follow-up, `RSK-036`). The two implementations now delegate to
one. `regional_level_premiums.py` reads through both fixes automatically; no
separate data step is needed.

**Why this one is lower-stakes than the accuracy re-run, stated in advance so
it is not discovered convenient after the fact.** The corrections all moved
multipliers DOWN. The basis contamination inflates the midpoint-derived figure
UPWARD relative to the common price (mean difference -0.758, common minus
midpoint). A ratio built partly on the inflated midpoint is therefore biased
upward, which means the correction the original run reported is, if anything,
UNDERSTATED rather than an artifact of the contamination -- the same
observation the RSK-036 follow-up recorded when this was first found. The
expectation going in is that the nine corrections hold or sharpen slightly, not
that they reverse. This expectation is recorded before the run so a matching
result cannot later be presented as more surprising than it is, and a
non-matching result is not dismissed as noise because it disagrees with this
paragraph.

## What is unchanged

The estimator (median ratio of region level to NCR level, paired by week), the
interval procedure (95 percent bootstrap, 10,000 resamples of whole weeks, seed
20260801), the region list (every region with DOE coverage, the same twelve),
`MIN_WEEKS` (52), and the Bonferroni correction applied to the interval rather
than to a p-value, family size equal to however many of the twelve have data.

## Decision rule, reaffirmed unchanged

Per region, identical to the original:

| outcome | conclusion | action |
|---|---|---|
| corrected interval contains the multiplier | consistent | leave that multiplier alone |
| corrected interval excludes it, and contains 1.00 | no premium detectable | apply the correction policy below |
| corrected interval excludes both | premium exists but is the wrong size | apply the correction policy below |
| fewer than 52 paired weeks | insufficient | report, change nothing |

**Correction policy, unchanged.** A multiplier whose interval excludes it is
replaced by the median ratio, rounded to two decimals. A region already
consistent (MIMAROPA, Bicol) is left alone unless this re-run's interval says
otherwise. No region without data is touched by inference from a neighbour. A
corrected region keeps its immediately-prior value recorded beside it, with
this document named, exactly as the original required.

**MIMAROPA and Bicol remain the positive control here too.** If either moves
under this re-run despite being consistent before, that is reported plainly
rather than folded quietly into "the multipliers were re-measured."

## Reported whatever the outcome

The same items the original committed to: the corrected interval and verdict
per region, which regions were `*seen` before this document (none -- unlike the
original, nothing here has been computed early), and the full table above
updated with whatever the new run produces, not merely the regions that moved.

## What would make this re-run uninformative

Everything the original listed (a region whose paired weeks cluster in one
short period; a ratio whose spread exceeds 0.10; NCR's own reliability as a
base), plus the same paired-week-count caveat as the accuracy re-run: if a
region's `n_weeks` falls under `MIN_WEEKS` because of pairs the guard now
refuses, it moves from a measured verdict to INSUFFICIENT, and that is reported
as a coverage loss, not folded into "consistent."

## Mechanical steps, fixed here

1. `tools/regional_level_premiums.py` gains a `basis_guard: true` field in its
   written JSON, mirroring the accuracy artifact
   (`honesty.regional_basis` already reads this field via
   `_regional_level_premiums()` and was built for exactly this).
2. Run `python -m ph_economic_ai.tools.regional_level_premiums`.
3. Report the printed table verbatim in [[06 Work/Risk Register]] and
   [[06 Work/Current Handoff]], including if every region stays unchanged.
4. Update `swarm.ALL_REGIONS` / `MEASURED_MULTIPLIERS` only for regions the
   decision rule's correction policy actually triggers on -- not a blanket
   overwrite -- with each changed constant's prior value recorded beside it
   per the original document's requirement.
5. Update `honesty.regional_basis`'s caveat only insofar as `basis_guard` now
   makes it disappear on its own.
