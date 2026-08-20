# Pre-registration: confirmatory test of `CLM-FUEL-EXPLORATORY-001`

Written 2026-08-20. Registers the confirmatory test the manuscript says this
claim is a candidate for (§5.3, "it is registered as `CLM-FUEL-EXPLORATORY-001`
and is a candidate for preregistered confirmation, not a result"), and closes
the Gate 3 item that named it.

## Disclosure, first, and it is unfavourable

**I have seen this result in full.** Unlike the 2026-08-12 pre-registration of
six food sub-categories, which was written before those series had ever been
fetched, this document is written about a result that has been computed,
published in a thesis manuscript, and argued about at length:

    ridge, skill_vs_rw +0.1211, dm_stat -2.1656, dm_p 0.0337, n = 72

So **this is not a blind pre-registration and nothing here can make the existing
result confirmatory.** A protocol written after seeing an effect cannot
retroactively pre-specify it; claiming otherwise would be the exact pattern the
size study of §5.10 exists to distrust, and would be worse than leaving the claim
exploratory, because it would dress a post-hoc finding in the vocabulary of a
planned one.

What this document does is narrower and is the only honest thing available: it
freezes, in advance, the test that *future* data will be put to, on a window that
has never been used for selection or evaluation, so that if the effect is real
someone can establish it without re-deciding anything after seeing the answer.

## What is frozen, exactly

The specification is the audit's fuel panel, unchanged, as of `41c1279`:

| Item | Value |
|---|---|
| Target | `targets.TARGETS['fuel']`, World Bank RON95 monthly, PHP per litre |
| Frame builder | `Target.build_frame()`, no modification |
| Features | `prev_fuel`, `oil_price_lag1`, `oil_price_ma3`, `usd_php_lag1`, `usd_php_ma3`, `gas_price_lag1`, `gas_price_ma3`, `demand_index_lag1`, `demand_index_ma3` |
| Model | `ridge`, the single method named here; **no selection over candidates** |
| Baseline | `random_walk` |
| Evaluation | walk-forward, `min_train = MIN_TRAIN = 24` |
| Statistic | Diebold-Mariano on the squared-error differential, HLN-corrected |

The model is named in advance precisely so that no selection occurs. That
distinction is the point of the exercise and it is worth stating plainly: the
audit applies a Bonferroni threshold of 0.0125 because `verdict_from_panel`
takes the best of four candidates (`arima`, `ets`, `ridge`, `hgb`) and judges
that maximum, so it must pay for k = 4. **A pre-declared test of one frozen
specification selects nothing, so α = 0.05 applies undivided.** Pre-declaration
buys back the factor of four, and that is the whole benefit being claimed for it.

## The confirmatory window

The committed fuel frame runs **2017-04 to 2025-03**, 96 rows, because the
World Bank RON95 gold series ends there.

The confirmatory window is **every month after 2025-03**, obtained by refreshing
the World Bank workbook through `refresh_data.build_world_bank_csv()`. No month
in that window has been used to select a model, tune a hyper-parameter, or score
a verdict in this project.

The model is refit walk-forward as it always was; what may not happen is any
change to the specification above once a single month of the new window has been
looked at.

## Decision rule, fixed now

- **Direction is pre-declared**: ridge beats the random walk, skill > 0. A
  one-sided test is therefore legitimate and is what will be used.
- **α = 0.05, one-sided, single test.** No multiplicity correction, because this
  is one pre-declared hypothesis and not a family.
- **Confirmed** if skill > 0 and the one-sided DM p < 0.05.
- **Not confirmed** otherwise, and the claim is then retired rather than
  re-specified. No second window, no alternative baseline, no re-selection.
- The result is recorded whichever way it falls.

## When it may be evaluated, and the problem with that

A test run before it can detect the effect is not a confirmation, so the trigger
is stated in advance. Scaling the observed statistic, `|t| = 2.1656` at n = 72:

| Test | Power | Required n | Calendar time |
|---|---|---|---|
| two-sided α = 0.05 | 80% | ~120 | ~10.0 years |
| one-sided α = 0.05 | 80% | ~95 | ~7.9 years |
| one-sided α = 0.05 | 50% | ~42 | ~3.5 years |

**Trigger: n ≥ 95 monthly predictions in the confirmatory window.** About 17
months exist there today.

**This is not achievable within the thesis timeline, and that is the finding.**
Pre-registration is supposed to make this visible before the fact rather than
after, and here it does: the effect is small enough, and the monthly series slow
enough, that honest confirmation is roughly eight years away. An interim look at
17 months would have power far below 50% and could not distinguish a real effect
from its absence, so no interim look is authorised. If one is taken anyway it is
descriptive and must not be reported as confirmation.

The consequence for the manuscript is that `CLM-FUEL-EXPLORATORY-001` stays
exploratory, permanently as far as this thesis is concerned. Nothing in this
document changes what may be claimed in it.

## What is unchanged

- No manuscript verdict, table or figure changes because of this document.
- The audit continues to report fuel as `efficient`, which it has since
  PR #55 corrected `verdict_from_panel` to pay for its selection.
- `CLM-FUEL-EXPLORATORY-001` remains registered as exploratory and is not
  promoted.

## Why this is worth registering anyway

Two reasons, neither of which is "so the gate can be ticked".

The specification stops drifting. Frozen here, the claim can be tested years from
now against exactly what was meant in 2026, rather than against whatever the
panel has become, which is the failure that made every sample size in the
manuscript stale twice in one month.

And the infeasibility is now a recorded fact rather than a discovery waiting to
happen. Anyone who later proposes confirming this claim can read that it needs
about 95 fresh monthly observations, and decide with that in hand instead of
running an underpowered test and reporting whatever it returns.
