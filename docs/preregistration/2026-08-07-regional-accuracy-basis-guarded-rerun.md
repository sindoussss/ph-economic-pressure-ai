# Pre-registration: the regional accuracy test, re-run on the basis-guarded panel

**`RSK-034`. Written 2026-08-07, before the corrected panel was scored.**
Amends `docs/preregistration/2026-08-01-regional-accuracy.md` rather than
replacing it: the question, the arms, the metric, and the decision rule are
unchanged. Only the input panel changes. Written as its own document because
`DEC-010` requires the decision to be committed before the run, and pointing at
an old document without saying what is different here would leave that decision
implicit.

## Disclosure, first

**I have already seen the result on the uncorrected panel**, because it is what
`RSK-034` is about. `regional_accuracy.json`, committed 2026-08-01:

| comparison | diff | 99.83% CI | verdict |
|---|---|---|---|
| DERIVATION vs ZERO | +0.095 | [-0.110, +0.309] | contains 0, no measurable difference |
| DERIVATION vs DELTA-EQUAL | +0.019 | [+0.014, +0.024] | excludes 0, derivation measurably worse |
| DERIVATION vs PERSISTENCE | -0.429 | [-0.660, -0.198] | excludes 0, derivation beats persistence |

294 paired weeks, best naive arm ZERO, published verdict NO MEASURABLE
DIFFERENCE (against the family's best naive arm, which the Bonferroni
correction is applied relative to).

**This is why an amendment, not a blind re-run.** A genuinely blind test is not
available on this panel for the same reason the original document gave: the
panel is what it is. What changes here is not the question but the data quality
the question is asked of.

## What is different in the input, exactly

`regional_grading._price` merged DOE's `common` price with the midpoint of its
published range with no record of which, and the two differ by a mean 0.758
PHP/L. A constant offset cancels in a week-over-week change, so the effect is
confined to region-week pairs where the basis SWITCHES between consecutive
weeks: measured at 9.2 percent of the pairs in the panel this test built.

The fix (`RSK-034`, code committed 2026-08-06) makes the basis travel with the
price and refuses a change measured across a switch. `regional_levels()` --
what `tools/regional_accuracy.py` calls to build its panel -- already reads
through the fix; no separate data step is needed. Re-running
`python -m ph_economic_ai.tools.regional_accuracy` today scores the corrected
panel automatically. The paired-week count will fall by up to 9.2 percent; the
exact number is not known until the run completes, which is the reason to fix
the rule now rather than after seeing it.

## What is unchanged

The question (does the derivation beat a trivial alternative), the four arms
(DERIVATION, DELTA-EQUAL, ZERO, PERSISTENCE), the reference (NCR, actual
national change, not a forecast), the metric (paired-week bootstrap on MAE,
10,000 resamples, seed 20260801), and alpha (0.05, Bonferroni-corrected across
three comparisons to 0.0167). Reusing the same seed is deliberate: the only
thing that should move the answer is which weeks survive the guard, not a new
source of randomness layered on top.

## Decision rule, reaffirmed unchanged

Let `d = MAE(derivation) - MAE(best naive arm)`, best naive arm chosen the same
way as before (lowest pooled MAE among DELTA-EQUAL, ZERO, PERSISTENCE).

| outcome | conclusion | action |
|---|---|---|
| corrected CI for `d` lies entirely **below 0** | the derivation beats the best naive | keep the regional figures as they are; remove the price-basis caveat from `honesty.regional_basis` |
| corrected CI **contains 0** | no measurable difference | keep presenting the labelled state already shipped: derived figures marked no better than assuming no regional change; remove the price-basis caveat (the label about NOT being graded stays -- only the "predates the guard" clause goes) |
| corrected CI lies entirely **above 0** | the derivation is WORSE than doing nothing | same labelling, plus the four unmeasurable regions stop showing a number at all, exactly as the original document's third row specified |

Identical to the original's decision rule. It is restated in full, not
referenced, so nothing about executing this re-run requires re-deriving or
re-interpreting it from memory once the new numbers are visible.

**If the verdict changes from NO MEASURABLE DIFFERENCE to either of the other
two**, that is reported as a verdict change caused by a data-quality
correction, not as a new finding dressed as one. If the verdict does **not**
change, that is reported too, with the new interval, not silently treated as
"nothing to update."

## Reported whatever the outcome

The same three items the original document committed to: per-region MAE and n,
not only the pool; the count of ungradable regions, by name; whether each
region's own premium was measured or assumed. Plus one item specific to this
re-run: **the exact change in paired-week count**, old n against new n, so the
9.2 percent estimate is checked against what actually happened rather than
assumed to have applied uniformly.

## What would make this re-run uninformative

Everything the original document listed, plus one new condition specific to
the correction: **if the paired-week count falls enough that `MIN_WEEKS` (60)
is not met for the pool as a whole**, or falls so far that a region's own
per-region count drops below what the original noise analysis assumed, the
honest reading is that the guard fix cost more power than it bought precision,
not that the derivation changed. This is not expected -- 9.2 percent of 294 is
about 27 pairs, leaving well over 60 -- but it is stated in advance because it
is exactly the kind of thing that gets rationalized away after the fact if it
is not.

## Mechanical steps, fixed here

1. `tools/regional_accuracy.py` gains a `basis_guard: true` field in its
   written JSON, so the artifact self-reports which side of `RSK-034` it was
   computed on (`honesty.regional_basis` already reads this field and was
   built for exactly this).
2. Run `python -m ph_economic_ai.tools.regional_accuracy`.
3. Report the printed verdict and the table above verbatim in
   [[06 Work/Risk Register]] and [[06 Work/Current Handoff]], including if it
   is unchanged.
4. Update `honesty.regional_basis`'s caveat only insofar as `basis_guard` now
   makes it disappear on its own -- no separate wording change unless the
   decision rule's action column requires one.
