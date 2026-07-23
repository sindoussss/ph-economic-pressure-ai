# The Missing Mean Baseline — a self-audit that overturns the MoM positives

**Status:** Finding, verified against the committed data with the repository's own
walk-forward, Diebold–Mariano test, and `mom_verdict` function. Reproduce with
`python -m ph_economic_ai.benchmark.corrected_audit`.

**One-line bottom line.** The nowcast baseline pool `{random_walk, seasonal_naive,
drift}` omits the *historical mean*. For the mean-reverting month-on-month (MoM)
rate series, the mean — not the random walk — is the strong naive baseline. When
the mean is added, **every MoM "positive" in the manuscript (headline inflation,
food, and the flagship electricity driver edge) falls to "no better than naive."**
The efficiency findings (fuel, FX, YoY) are unaffected, and the core thesis — that
these series are not beatable by real structure — is *strengthened*.

---

## 1. Why this is a real hole, not a nitpick

The manuscript's central methodological safeguard is the *hollow-win guard*
(§4.7): a candidate must beat the **strongest** simple baseline, not a strawman.
That guard is implemented as:

```python
# nowcast.py
BASELINE_POOL = ('random_walk', 'seasonal_naive', 'drift')
best_naive = min(pool, key=lambda m: rmse_by_method[m])   # strongest = lowest RMSE
```

For a **persistent level series** (fuel PHP/L, USD/PHP, YoY inflation), the random
walk is genuinely the strongest of these, and the unconditional mean is useless
(predicting the 2007 average price in 2025). So the pool is correct there.

For a **mean-reverting rate series** — and every MoM inflation target is one — the
random walk is a *weak* baseline (it chases each oscillation), and the strongest
naive is the **mean**. The mean is not in the pool. So the guard silently fails
for exactly the series where it matters, and any model that reverts toward the
mean (ARIMA, Ridge, ETS) clears the random walk and is stamped `beats_best_naive`
— without beating the mean.

The mean forecaster is the expanding-window unconditional mean, uses only past
data (no leakage), and is *simpler* than every candidate. If a candidate cannot
beat it, the candidate has no demonstrated edge — by the manuscript's own logic.

---

## 2. Evidence chain

All numbers below are from the committed CSVs, `MIN_TRAIN = 24`, the repository's
`walk_forward`, `diebold_mariano`, and `mom_verdict`.

### 2.1 The electricity "driver edge" is reproduced by a pure-noise feature

Decomposing the flagship electricity driver-only nowcast (mechanism-faithful:
month-on-month **% change** regressors, Ridge with standardization) by feature
subset:

| driver subset (Δ%, scaled Ridge) | skill vs best naive | DM p | verdict |
|---|---|---|---|
| oil + natgas + fx (full) | +0.276 | 0.003 | edge |
| oil + fx (anchor-validation set) | +0.283 | 0.002 | edge |
| natgas only | +0.319 | 0.000 | edge |
| oil only | +0.322 | 0.000 | edge |
| fx only | +0.278 | 0.003 | edge |

Every single driver *alone* yields a ~28–32% "edge" — yet the univariate
change-on-change correlations of those drivers with the target are ≈0 (oil +0.06,
natgas −0.05, fx −0.00). A near-zero-correlation feature cannot be predicting.
The tell:

| forecaster (electricity MoM, n=151) | RMSE | skill vs RW | DM p vs RW |
|---|---|---|---|
| random_walk (baseline) | 3.340 | 0.000 | — |
| **Ridge on PURE NOISE** | **2.383** | **+0.286** | **0.0015** |
| MEAN of target (no drivers) | 2.352 | +0.296 | 0.0011 |
| AR(1) toward mean | 2.404 | +0.280 | 0.0005 |

**Ridge on a random noise column beats the random walk by 28.6%** — identical to
the "driver edge" — because regularized Ridge collapses to its intercept (the
training mean), and the mean crushes the random walk on this oscillatory series.
The electricity "within-month driver edge" is mean-reversion mislabeled as a fuel
channel. (This also vindicates `anchor_validation.json`, which found the
electricity change-on-change relationship a null all along.)

### 2.2 No winning method beats the mean, on any target

The decisive test for every "positive": does the paper's winning method beat the
**mean**, not just the random walk?

| Target | winner skill vs **RW** (paper) | mean skill vs RW | winner vs **mean** | verdict |
|---|---|---|---|---|
| Headline MoM (short, n=61) | ARIMA +16.2% (p=0.032) | +12.6% | **+4.1% (p=0.36)** | does not beat mean |
| Headline MoM (long, n=143) | ARIMA +16.3% (p=0.001) | +12.2% | **+4.6% (p=0.19)** | does not beat mean |
| Food MoM (n=151) | ARIMA +16.0% (p=0.005) | +12.8% | **+3.7% (p=0.46)** | does not beat mean |
| Electricity MoM (n=151) | Ridge/ARIMA +25–27% (p<0.001) | +29.6% | **−6.0% (p=0.32)** | mean beats the winner |

Every headline positive is, at most, ~4% better than the mean in point terms, and
in no case is that difference significant (p = 0.19–0.46). For electricity the
mean is *better* than the winner.

### 2.3 The verdicts flip — shown with the paper's own `mom_verdict`

Adding `'mean'` to the pool and re-running the whole map through the unchanged
machinery (`python -m ph_economic_ai.benchmark.corrected_audit`) flips **6 of 8**
nowcast verdicts; the other two were already null:

| Target | setup | paper pool | mean-in-pool |
|---|---|---|---|
| Headline MoM (short) | full nowcast | beats_best_naive (+0.16) | **null** |
| Headline MoM (long, n=143) | full nowcast | beats_best_naive (+0.16) | **null** |
| Food MoM | full nowcast | beats_best_naive (+0.16) | **null** |
| Food MoM | driver-only | null | null |
| Electricity MoM | full nowcast | beats_best_naive (+0.27) | **null** |
| Electricity MoM | driver-only (flagship) | beats_best_naive (+0.28) | **null** |
| Transport MoM | full nowcast | null | null |
| Transport MoM | driver-only | beats_best_naive (+0.15) | **null** |

Note the last row: the transport pre-robustness edge (§5.5) is *also* killed by
the mean baseline — so it was a mean-reversion artifact as well as a
preliminary-data artifact. Every `beats_best_naive` verdict in the entire MoM
audit collapses once the correct strong baseline is present.

The forecast-side nulls, by contrast, are untouched: with the mean added as a
*candidate*, fuel/FX/YoY remain `efficient`, and the mean's own skill vs the
random walk is strongly negative (fuel −1.75, FX −2.09, YoY −1.70) — confirming
the mean cannot manufacture a false positive on a persistent level series.

---

## 3. What falls, what survives

**Falls (requires rewriting §5.3–5.8, §6, §7, and the abstract):**
- The electricity "confirmed within-month **driver** edge" (§5.7) — not a driver
  edge (noise reproduces it) and does not beat the mean.
- The headline MoM nowcast positive (§5.3), including the robust n=143 result.
- The food own-dynamics positive (§5.6).
- The "three-way discrimination" narrative loses its *confirmed true positive*
  leg: the within-month driver channel now has no confirmed positive
  (transport rejected, food null, electricity = mean-reversion artifact).

**Survives — and is reinforced:**
- The efficiency findings (fuel, FX, YoY are unforecastable beyond naive). Adding
  a baseline can only make a null *harder* to overturn; the mean's skill vs RW is
  strongly negative on these level series (−1.75 / −2.09 / −1.70), so it never
  becomes the best naive and the verdicts stay `efficient`.
- The transport rejection (§5.5) — still an artifact (now by two independent
  routes: preliminary-data *and* the mean baseline).
- The food commodity-driver null (§5.6) — still a null.
- The reproducibility, honesty-as-method, DM/HLN, power, and multiple-testing
  infrastructure — which is *what made this finding possible in an afternoon*.

**The corrected story is stronger, not weaker.** "AI/models cannot predict these
series beyond trivial naive methods" becomes more complete: MoM rate series are
mean-reverting, so they appear beatable against a random walk, but no model
(ARIMA, Ridge, drivers, or a swarm) beats simply predicting the mean. The random
walk was the wrong yardstick for the rate series; the mean is the right one.

---

## 4. The fix

**Canonical (destructive — rewrites cited numbers and test expectations, so make
it a reviewed commit):**

```python
# nowcast.py
BASELINE_POOL = ('random_walk', 'seasonal_naive', 'drift', 'mean')
PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'mean',
                 'arima', 'ets', 'ridge', 'hgb']
```
and add `'mean'` to the driver-only ablation's method list in
`run_driver_only_ablation`. The `mean` forecaster is already provided in
`forecasters.py`. Then regenerate: `python -m ph_economic_ai.benchmark.run`, and
update the unit tests that assert the old MoM verdicts.

**Non-destructive (already provided):** `benchmark/corrected_audit.py` re-derives
the entire map with the mean in the pool and writes
`benchmark/artifacts/corrected_predictability_map.json`, leaving every frozen
artifact intact — use it for the before/after in the revision.

---

## 5. Reproduction

- `python -m ph_economic_ai.benchmark.corrected_audit` — the full corrected map
  (forecast efficiency + every MoM nowcast, old vs corrected verdict).
- The noise-Ridge / mean / AR(1) diagnostic for electricity, and the
  driver-subset decomposition, are documented in §2.1–2.2 above and are
  reconstructable from the committed frames via `walk_forward` + `mom_verdict`.

---

*Meta-note. This document is the audit turned on its own most-cited positives. A
protocol that can reject its own flagship result when the data do not support it
is doing exactly what §6.3 of the manuscript claims for it.*
