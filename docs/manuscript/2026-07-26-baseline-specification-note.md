# Mind Which Naive: Baseline Specification and Spurious Predictability on Mean-Reverting Targets

**Author:** Sindous
**Draft:** 2026-07-26 · standalone methodological note
**Reproducibility:** every number below is produced by a committed module in
`ph_economic_ai/benchmark/` and frozen in `artifacts/`. Reproduce the whole note with
`python -m ph_economic_ai.benchmark.baseline_theory`,
`… .baseline_size`, `… .vulnerability_survey`, and `… .power`.
Check it against the artifacts with `python -m ph_economic_ai.benchmark.manuscript_check`.

> ### ARTIFACT-DIVERGENCE notice (2026-07-28)
>
> The calendar correction of 2026-07-28 changed the empirical sample sizes this note's
> simulation cells were anchored to. Three references to **n = 203** no longer match any
> artifact; the corresponding empirical sample is now 202.
>
> This does **not** affect the note's result. The closed form, the size and power study, and
> the FRED-MD census are simulation and census work whose n values are design choices, not
> measurements. The correction also confirmed the result empirically: every month-on-month
> null still holds against the mean on the rebuilt panels, so the finding was not an artifact
> of the irregular calendar.
>
> The simulation cells should be re-anchored to the corrected empirical sizes when §5 of the
> companion manuscript is rewritten, so that the illustrative n matches the audit it
> illustrates.

---

## Abstract

Forecast-evaluation practice requires a candidate model to beat a naive benchmark, and the random walk is the conventional choice. This note quantifies what happens when that convention is applied to a mean-reverting target. For a covariance-stationary series with lag-1 autocorrelation ρ, a forecaster carrying **no information beyond the unconditional mean** exhibits an apparent skill over the random walk of exactly

**S(ρ) = 1 − [2(1 − ρ)]^(−1/2)**,

which is positive if and only if ρ < ½ and equals ≈ +29.2% at ρ = 0. Three results follow. First, the expression is accurate: it matches simulation through a walk-forward backtest to ≤ 0.011 across ρ ∈ [−0.2, 0.8], and predicts the observed mean-versus-random-walk gap on five real macroeconomic targets to ≤ 0.022. Second, the induced size distortion is severe rather than marginal: on simulated targets containing no signal at all, a Diebold–Mariano protocol whose naive pool omits the mean returns a significant "edge" in **99.7%** of replications at ρ = 0, n = 203, against a nominal 5%; the rate **increases with sample size**, so the reflexive robustness response of re-estimating on a longer sample amplifies the error rather than exposing it. Including the mean restores exact size (0.0% in every cell) while retaining 80.7% power to detect a genuine driver. Third, the exposure is not exotic: in FRED-MD, the standard monthly US macro panel, **80.2% of 126 series** fall below the ρ = ½ threshold once each series' own *recommended* stationarity transformation is applied — including **every** differenced series and 92% of growth rates, while **no** series left in levels is affected. Differencing removes precisely the persistence that makes a random walk a valid benchmark. A worked case is supplied in which a +28.3% "driver edge" (DM p = 0.0011) survived sub-sample stability tests, a real-time-vintage robustness re-test, a Bonferroni correction, and a mechanistically accurate story, while the model was in fact 1.8% *worse* than a constant. Inverting S(ρ) yields a one-line diagnostic applicable to any reported skill-over-random-walk.

**Keywords:** forecast evaluation · naive benchmark · Diebold–Mariano · size distortion · nowcasting · mean reversion

---

## 1. Introduction

A forecast comparison is a comparison, and it inherits whatever weakness the comparator has. Contemporary practice is careful about the *test* — Diebold–Mariano with a small-sample correction, out-of-sample walk-forward design, corrections for multiple hypotheses — and comparatively casual about the *benchmark*, which is very often "the naive forecast," meaning the random walk.

That convention has a strong justification for persistent series. If a series has a unit root, the no-change forecast is optimal, and a model appearing to beat it is producing a spurious result — a point made forcefully in the evaluation literature (§2). The concern this note addresses is the mirror image, and it is much less commonly stated: on a series that is *not* persistent, the random walk is a structurally weak comparator, and beating it is uninformative in the opposite direction.

The distinction matters because the targets of applied nowcasting are routinely *rates* — month-on-month inflation, growth rates, changes in changes — obtained by differencing a level series in order to induce stationarity. That transformation is applied for an entirely correct reason and has a side effect nobody intends: it removes the persistence that made the random walk a sensible benchmark in the first place.

Section 3 gives the magnitude of the resulting distortion in closed form and validates it. Section 4 measures the false-positive rate it produces and confirms that correcting it preserves power. Section 5 measures how much of the standard macro target space is affected. Section 6 presents a case in which the artifact defeated five independent-looking robustness checks, and Section 7 states the diagnostic.

**What is and is not claimed.** That the unconditional mean belongs in a naive pool is *not* new, and no such claim is made here; §2 attributes it. The derivation in §3 is elementary, resting on two variance calculations, and may be folklore among forecasters even where it is not written down. The contributions are the **magnitude**, the **exact condition** under which it applies, the **operating characteristic** it induces, the **measured exposure** of the standard target space, and a **usable diagnostic**. Section 5 measures targets, not published findings, and no inference should be drawn about any particular published result.

---

## 2. Related work

Two literatures bracket this problem from opposite sides.

**The naive benchmark is too hard to beat.** Hyndman and Koehler (2006) motivate scale-free accuracy measures constructed against a naive benchmark, making the benchmark choice load-bearing for the measure itself. Hewamalage, Ackermann and Bergmeir (2023) catalogue the resulting evaluation pitfalls and note that on a series which genuinely is a random walk the naive forecast is optimal by construction, so an apparent improvement is spurious. Beck, Dovern and Vogl (2025) make the empirical case at scale: across exchange rates, NASDAQ100 constituents and FRED-MD, machine-learning methods fail to beat a no-change forecast, and the closer a series is to a random walk the less useful complex models become. Their prescription — include the naive benchmark or the comparison is uninformative — is sound and is adopted here.

**The naive benchmark can be too easy to beat.** The ingredients are standard: the unconditional mean is the optimal forecast for white noise, the no-change forecast is optimal for a random walk, and the mean becomes more competitive as autocorrelation falls. Atkeson and Ohanian (2001) is, in substance, an argument about which baseline an inflation forecast must clear. Lunsford and West (2026) study the closely related question of what happens when a random-walk model is used to forecast a *stationary* process, finding it has unusually low bias — a result favourable to the random walk on a bias criterion, and complementary to the RMSE-based question asked here.

What is absent from applied practice is a statement of *how large* the distortion is, *when* it applies, and *how to test a reported result for it*. Notably, the two failure modes are distinguished by exactly the quantity this note makes explicit: Beck et al. study **levels** (exchange rates, equity prices) which are near-random-walks, where naive is hard to beat; the affected case is **differenced rates**, where it is too easy. A protocol that adopts only the first warning is fully protected against one error and entirely exposed to the other.

---

## 3. A closed form for baseline-induced spurious skill

### 3.1 Derivation

Let $\{y_t\}$ be covariance-stationary with variance $\sigma^2$, mean $\mu$, and lag-1 autocorrelation $\rho$. Consider two forecasts of $y_t$:

- the **random walk**, $\hat y_t = y_{t-1}$, with error variance
  $E[(y_t - y_{t-1})^2] = 2\sigma^2(1-\rho)$;
- the **unconditional mean**, $\hat y_t = \mu$, with error variance
  $E[(y_t - \mu)^2] = \sigma^2$.

Hence $\text{RMSE}_{RW} = \sigma\sqrt{2(1-\rho)}$ and $\text{RMSE}_{mean} = \sigma$. A forecaster carrying no information beyond $\mu$ therefore records a skill against the random walk of

$$S(\rho) \;=\; 1 - \frac{\text{RMSE}_{mean}}{\text{RMSE}_{RW}} \;=\; 1 - \frac{1}{\sqrt{2(1-\rho)}},$$

with

$$S(\rho) > 0 \iff 2(1-\rho) > 1 \iff \rho < \tfrac{1}{2}.$$

Three properties are worth stating explicitly.

1. **The crossover at ρ = ½ is exact**, not asymptotic or approximate.
2. **Only ρ enters.** No AR(1) assumption is required; the result holds for any stationary process, since the random-walk error variance is $2\sigma^2(1-\rho_1)$ for any such process.
3. **At ρ = 0, S = 1 − 1/√2 ≈ 0.2929.** On a white-noise rate, a model that has learned nothing whatsoever is credited with a ~29% improvement over the random walk.

The inverse is the practically useful form. Given a reported skill $s$ against a random walk,

$$\rho_{\text{implied}}(s) \;=\; 1 - \frac{1}{2(1-s)^2}$$

is the autocorrelation at which a wholly uninformative forecaster would post exactly that skill. Comparing $\rho_{\text{implied}}$ against the target's measured $\rho$ is a direct test (§7).

### 3.2 Validation against a real estimator

The derivation uses the true $\mu$; an implementation uses an expanding-window sample mean, and the comparison runs through a walk-forward backtest. Simulating stationary AR(1) targets and scoring them with the same `walk_forward` routine used in applied work gives:

| ρ | S(ρ) predicted | simulated | \|error\| |
|---|---|---|---|
| −0.20 | +0.354 | +0.348 | 0.006 |
| 0.00 | +0.293 | +0.286 | 0.007 |
| 0.20 | +0.209 | +0.202 | 0.008 |
| 0.40 | +0.087 | +0.078 | 0.009 |
| **0.50** | **+0.000** | **−0.011** | **0.011** |
| 0.60 | −0.118 | −0.129 | 0.011 |
| 0.80 | −0.581 | −0.587 | 0.006 |

Maximum absolute error 0.011, with the small negative bias attributable to finite-sample estimation of the mean. The sign change occurs at ρ = 0.5 as derived.

![S(rho) with simulation and real targets.](../../ph_economic_ai/benchmark/artifacts/figures/fig5_spurious_skill.png)

**Figure 1.** S(ρ) (black), simulated through a walk-forward backtest (crosses), and five real month-on-month targets at their measured ρ (circles). Shaded: where an uninformative forecaster is credited with positive skill.

### 3.3 Validation against real targets

Applied to five real month-on-month macroeconomic targets (Philippine CPI components and aggregates, 2007–2026), measuring each series' ρ and comparing the predicted spurious skill against the mean-versus-random-walk gap actually observed in a walk-forward backtest:

| Target | n | ρ | S(ρ) predicted | observed | \|error\| |
|---|---|---|---|---|---|
| Headline MoM inflation | 61 | +0.343 | +12.75% | +12.58% | 0.002 |
| Headline MoM (long sample) | 143 | +0.369 | +11.01% | +12.22% | 0.012 |
| Food MoM inflation | 151 | +0.375 | +10.56% | +12.76% | 0.022 |
| **Electricity MoM inflation** | 151 | **+0.002** | **+29.22%** | **+29.59%** | 0.004 |
| Transport MoM inflation | 151 | +0.140 | +23.76% | +24.77% | 0.010 |

The closed form accounts for the observed gap on every target to within 0.022. Electricity month-on-month inflation is essentially white noise (ρ = +0.002), and the ~29% figure at that point is not a coincidence — it is §3.1's value of S(0).

---

## 4. The induced size distortion

S(ρ) describes an expected magnitude. The operationally decisive question is the **false-positive rate**: how often does a protocol declare significance when there is nothing to find?

### 4.1 Design

The experiment is paired by construction. A target is simulated as a stationary AR(1) with three pure-noise features, so no model can legitimately win and **every rejection is a false positive**. Each method is scored once through the same walk-forward backtest used in applied work. The verdict rule — lower RMSE than the best pool member *and* a significant HLN-corrected Diebold–Mariano advantage over it at α = 0.05 — is then evaluated **twice on the identical fitted losses**, changing only the pool:

- **Pool A** (conventional): {random walk, drift, seasonal naive}
- **Pool B** (corrected): {random walk, drift, seasonal naive, **historical mean**}

Because data, models and losses are identical across arms, every difference is attributable to the pool. Under Pool A the mean is also removed from the *candidate* set, since a protocol that never evaluated it could not have crowned it the winner; retaining it would simulate a procedure nobody uses.

### 4.2 Size

300 replications per cell; nominal α = 0.05.

| ρ | n = 82, Pool A | n = 203, Pool A | Pool B (either n) |
|---|---|---|---|
| 0.00 | 43.3% | **99.7%** | 0.0% |
| 0.20 | 19.7% | 87.7% | 0.0% |
| 0.35 | 4.7% | 26.3% | 0.0% |
| 0.50 | 1.0% | 1.3% | 0.0% |
| 0.70 | 0.0% | 0.0% | 0.0% |

Three observations.

**The distortion is not marginal.** At ρ = 0 and n = 203 the protocol rejects on 99.7% of datasets containing no signal whatsoever. A nominal 5% test that rejects essentially always does not have inflated size; it has no operating characteristics at all.

**The distortion is confined to ρ < ½.** At ρ = 0.5 the rate collapses to ~1% and at ρ = 0.7 to zero. §3.1 derives that threshold analytically from two variance calculations; this simulation reaches it independently, sharing no assumption beyond stationarity.

**More data makes it worse.** The rate rises with n in every row (43.3% → 99.7% at ρ = 0; 19.7% → 87.7% at ρ = 0.2). This is the most practically damaging property. "Re-estimate on a longer sample" is the reflexive robustness response to a suspicious result, and here it *amplifies* the error: a larger sample estimates the spurious mean-versus-random-walk gap more precisely, tightening the p-value. A researcher following standard practice will read the artifact strengthening as confirmation.

![False-positive rate by rho, both pools.](../../ph_economic_ai/benchmark/artifacts/figures/fig6_size_distortion.png)

**Figure 2.** Rejection rate on data containing no signal. Pool A (red) is catastrophically oversized below ρ = ½ and worsens with n; Pool B (blue) is at zero throughout. Dotted line: nominal α = 5%.

### 4.3 Power

A correction that merely blinds the test is not a correction. Repeating the experiment with a genuine, contemporaneously observable driver (β = 0.6 on the first feature):

| ρ | n | Pool A | **Pool B** |
|---|---|---|---|
| 0.00 | 61 | 82.0% | 17.3% |
| 0.00 | 151 | 100.0% | **80.7%** |
| 0.35 | 61 | 42.0% | 13.0% |
| 0.35 | 151 | 96.3% | **73.0%** |

Pool B retains 80.7% power at ρ = 0, n = 203 and 73.0% at ρ = 0.35 — adequate detection of real signal at realistic sample sizes.

**The Pool A column must not be read as superior power.** On these same targets Pool A has a 43–100% false-positive rate (§4.2), so its "detections" confound size with power and are not comparable. At n = 82 Pool B's power falls to 13–17%, which reflects the sample size rather than a defect of the correction: at that n the test's minimum detectable effect against the mean is ~13%, so a moderate driver is genuinely beyond resolution.

---

## 5. Exposure: how much of the standard target space is affected?

A natural response is that this describes an unusual corner of the data. It does not. FRED-MD (McCracken and Ng, 2016) is the standard monthly US macro panel for benchmarking nowcasting methods. It ships a **recommended stationarity transformation per series** (its transformation code), and applying it is precisely what a nowcasting study does before modelling — so the transformed series *are* the objects the literature forecasts.

Measuring ρ for each transformed series and comparing against the ρ = ½ threshold (126 series with ≥ 120 observations):

| | count | share |
|---|---|---|
| ρ < 0.5 — affected | 101 / 126 | **80.2%** |
| ρ < 0.2 — false-positive rate ≥ 20% | 73 / 126 | 57.9% |
| ρ < 0.05 — false-positive rate ≈ 100% | 56 / 126 | 44.4% |

Median ρ across the panel is **+0.107**. The median spurious skill a wholly uninformative forecaster would post on the affected series is **+29.3%**.

The decomposition by transformation is the structural result:

| Transformation | Affected |
|---|---|
| First difference | **19 / 19 (100%)** |
| Second log difference | **33 / 33 (100%)** |
| Log difference (growth rate) | 48 / 52 (92.3%) |
| Level | **0 / 11 (0%)** |
| Log | **0 / 10 (0%)** |

**Every differenced series in FRED-MD is affected; no series left in levels is.**

![FRED-MD rho distribution by transform.](../../ph_economic_ai/benchmark/artifacts/figures/fig7_fredmd_exposure.png)

**Figure 3.** Lag-1 autocorrelation across 126 FRED-MD series after each series' recommended transform, split by whether it involved differencing. The groups barely overlap. This is the mechanism of §3.1 stated in reverse. A random walk is a good benchmark exactly when a series is persistent; differencing exists to remove persistence. The standard stationarity transformation therefore moves a target out of the regime where the random-walk benchmark is appropriate and into the regime where it is structurally weak — silently, because the transformation is applied for an unrelated and entirely correct reason.

The most affected series are ordinary quantities whose transformed form carries *negative* autocorrelation, where S(ρ) exceeds +42%:

| Series | ρ | S(ρ) |
|---|---|---|
| CES2000000008 (construction avg. hourly earnings) | −0.638 | +44.8% |
| CES0600000008 (goods-producing avg. hourly earnings) | −0.587 | +43.9% |
| CES3000000008 (manufacturing avg. hourly earnings) | −0.555 | +43.3% |
| CPIMEDSL (CPI: medical care) | −0.548 | +43.2% |
| CUSR0000SAS (CPI: services) | −0.541 | +43.0% |

**This section measures the target space, not published findings.** It does not claim that any particular published result is spurious, and no such inference is licensed by it. What it establishes is that the condition under which the distortion of §4 operates is the common case in the standard panel rather than an exception.

---

## 6. A worked case: five robustness checks, one artifact

The following is drawn from the author's own predictability audit of Philippine macroeconomic series, and is reported because it documents the failure mode end to end rather than in the abstract. That audit — including the full corrected predictability map, the affected targets in detail, and the withdrawn results in their original form — is reported separately in [`2026-06-10-thesis-manuscript.md`](2026-06-10-thesis-manuscript.md); only what bears on the baseline question is reproduced here.

The audit examined whether electricity CPI inflation could be nowcast from within-month observable energy drivers (Brent, natural gas, FX) before the statistical agency's release. The driver-only specification returned, on n = 203 walk-forward months, **Ridge +28.3% over the best naive baseline, DM p = 0.0011**. The result then passed, in sequence:

1. **Statistical significance** — HLN-corrected DM, p = 0.0011.
2. **Sub-sample stability** — ≤ 2023-12 (+26%, p = 0.006), 2007–2016 (+30%, p = 0.020), 2016–2026 (+29%, p = 0.035). Not period-specific.
3. **A real-time-vintage robustness re-test** — dropping trailing preliminary observations left the edge intact, a check that had *correctly rejected* an apparently significant transport-CPI edge in the same audit.
4. **A Bonferroni family-wise correction** over six confirmatory tests — survived at adjusted p ≤ 0.028.
5. **A mechanism** — the regulated generation charge is a formulaic, within-month-observable fuel pass-through. Institutionally accurate, and it made the result feel safe.

The baseline pool was the check the audit did not have. Adding the historical mean:

| Method | RMSE |
|---|---|
| **mean** | **2.3515** |
| Ridge | 2.3936 |
| ARIMA | 2.4925 |
| random walk | 3.3399 |

| Comparison | Skill | DM p |
|---|---|---|
| Ridge vs random walk | **+28.3%** | 0.0011 |
| **Ridge vs mean** | **−1.8%** | 0.371 |

Ridge is **worse than predicting a constant**. The entire +28.3% is the distance between the random walk and the mean. Applying §3.1's inversion: a +28.3% skill implies ρ ≈ 0.027; electricity MoM measures **ρ = +0.002**. The result is fully accounted for by the target's own autocorrelation, with no reference to any driver, model or mechanism.

The corrected verdict is a null, and it is not an underpowered one: the minimum detectable effect for that test against the mean is **5.8%** at 80% power, so the test had ample resolution to see a small genuine edge and instead found the model performing worse than a constant.

**Why the guards failed is the instructive part, and none of them malfunctioned.** Each answered a question other than "is the baseline appropriate?" Significance testing asks whether an edge over the *chosen* comparator is real. Sub-sample stability asks whether an effect is period-specific — but S(ρ) is a property of the target, so the artifact is stable everywhere by construction, and stability testing actively *corroborated* the error. Vintage robustness asks about data revisions. Multiple-comparison correction asks whether many tests inflate false positives, offering no protection when all of them share one misspecification. And a plausible mechanism supplied narrative confirmation.

Robustness checks compound only when they are **independent**. Five checks that all condition on the baseline provide the reassurance of five and the coverage of one.

---

## 7. A diagnostic

The check is cheap and requires no access to a study's models or code — only its reported skill and its target series.

> **Given a reported skill $s$ over a random-walk benchmark on a stationary target:**
> 1. Compute $\rho_{\text{implied}} = 1 - \dfrac{1}{2(1-s)^2}$.
> 2. Measure the target's actual lag-1 autocorrelation $\hat\rho$.
> 3. If $\rho_{\text{implied}} \approx \hat\rho$, the reported skill is **fully consistent with a forecaster carrying no information beyond the unconditional mean**, and the comparison does not distinguish the two.
> 4. If $\hat\rho < \tfrac12$, report skill against the **mean** as well as the random walk. This is one additional row in a results table.

The diagnostic is *not* proof of a spurious result — a genuinely informative model on a low-ρ target will also show skill in this range, and step 4 is what separates the cases. Its value is in identifying which reported edges cannot be distinguished from the artifact on the evidence presented.

The corresponding recommendation for practice is narrow: **on any target with $\hat\rho < \tfrac12$, the naive pool should include the unconditional mean.** Where $\hat\rho \geq \tfrac12$ the addition is harmless — the mean is a poor predictor of a persistent series and never binds, which is also why this correction cannot manufacture a false negative on level-series forecasts.

---

## 8. Limitations

**The effect is not new, and is claimed only in magnitude.** §2 attributes the qualitative point. The derivation is elementary and may be unwritten folklore; the contributions are the magnitude, the exact condition, the operating characteristic, the exposure measurement, and the diagnostic.

**The size study is a simulation.** It uses AR(1) targets with Gaussian innovations and a single ML candidate (Ridge; HistGradientBoosting was checked in one cell and behaves identically for ~100× the compute). Heavier tails, structural breaks, or richer candidate sets are not covered, though the mechanism depends only on ρ and should be insensitive to them.

**The FRED-MD census measures targets, not findings.** It establishes that the affected regime is common. It does not establish that any published result is spurious, and this note makes no such claim. Establishing that would require each study's series and specification.

**The worked case is a single audit** on a single country's data, reported because the author has full access to it, not as representative evidence.

**Power is not free at small n.** At n = 82 the corrected pool detects a moderate driver in only 13–17% of replications. The correction removes a false-positive problem; it does not create power that the sample size does not support. Reporting a minimum detectable effect alongside every null is the appropriate accompaniment.

---

## 9. Conclusion

The random walk is the conventional naive benchmark, and on a persistent series it is the right one. On a mean-reverting series it is a structurally weak comparator, and the magnitude of that weakness is exactly $S(\rho) = 1 - [2(1-\rho)]^{-1/2}$, positive precisely when $\rho < \tfrac12$. The resulting size distortion is not marginal: at ρ = 0 and n = 203 a nominal 5% protocol rejects on 99.7% of datasets containing nothing, and the rate grows with sample size, so the standard robustness response amplifies rather than exposes it. Four out of five series in the standard US macro panel sit in the affected regime after their own recommended stationarity transformation — every differenced series, and no series in levels — because differencing removes exactly the persistence that justified the benchmark.

The practical consequence is a single additional row in a results table. The methodological consequence is larger: a stack of robustness checks that all condition on the baseline offers the reassurance of many and the coverage of one, and no amount of significance testing detects a misspecified comparison. Baseline specification is not a preliminary to the analysis. On a mean-reverting target it *is* the analysis.

---

## References

- Atkeson, A., & Ohanian, L. E. (2001). Are Phillips curves useful for forecasting inflation? *Federal Reserve Bank of Minneapolis Quarterly Review*, 25(1), 2–11.
- Beck, N., Dovern, J., & Vogl, S. (2025). Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability. *Applied Intelligence*, 55, 395.
- Diebold, F. X. (2015). Comparing predictive accuracy, twenty years later: A personal perspective on the use and abuse of Diebold–Mariano tests. *Journal of Business & Economic Statistics*, 33(1), 1–9.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. *Data Mining and Knowledge Discovery*, 37, 788–832.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Lunsford, K. G., & West, K. D. (2026). Random walk forecasts of stationary processes have low bias. *Journal of Business & Economic Statistics* (forthcoming; NBER WP 34112).
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.

---

## Appendix — Reproduction

| Result | Module | Artifact |
|---|---|---|
| S(ρ), inversion, simulation + real-target validation (§3) | `benchmark/baseline_theory.py` | `baseline_theory.json` |
| Size and power grids (§4) | `benchmark/baseline_size.py` | `baseline_size.json` |
| FRED-MD census (§5) | `benchmark/vulnerability_survey.py` | `vulnerability_survey.json` |
| Minimum detectable effects (§4.3, §6) | `benchmark/power.py` | `power.json` |
| The worked case (§6) | `benchmark/run.py`, `corrected_audit.py` | `accuracy_report.json`, `corrected_predictability_map.json` |

The FRED-MD snapshot is frozen at `benchmark/data/fredmd_snapshot.csv`; refresh with
`python -m ph_economic_ai.tools.refresh_fredmd`. All modules are pure numpy/scipy/pandas
and require no LLM, GPU or network on the run path.
