# Mind Which Naive: How the Choice of Baseline Decides a Predictability Verdict — with a Reproducible Audit of Philippine Fuel, FX, and Inflation

**Author:** Sindous
**Draft:** 2026-06-10 · results text still reflects the artifact set frozen before the 2026-07-28 calendar correction. See the divergence notice below.
**Companion note:** the methodological result of §4.7, §5.10 and §5.11 is also written up standalone, without the Philippine application, in [`2026-07-26-baseline-specification-note.md`](2026-07-26-baseline-specification-note.md). The two are intended to coexist: this manuscript develops the result in full as its methodological spine, the note is the extractable version for a forecasting venue. Both are checked against the same artifacts by `python -m ph_economic_ai.benchmark.manuscript_check`, so divergence between them is detected rather than assumed away.
**Status:** Draft, **not consistent with the current artifacts**. The §1.1 macro figures (2022–2025 episode) are drawn from PSA and BSP releases, cited under *Primary institutional and data sources* in the References (specific release reference numbers to be confirmed in the final version).

> ### ARTIFACT-DIVERGENCE notice (2026-07-28, partially reconciled)
>
> The calendar correction of 2026-07-28 rebuilt every monthly panel on a complete month index, which recovered months that had been silently dropped and changed every sample size in this document.
>
> **Reconciled.** All 53 stale sample sizes have been corrected against the committed artifacts (n = 51/52 → 72, 61 → 82, 77 → 97, 79 → 99, 143 → 190, 145 → 197, 151 → 203, plus the sentiment and anchor windows). The companion note is fully consistent. Correcting a sample size is factual accuracy, not reinterpretation, so it did not wait on Gate 3.
>
> **Outstanding, and it needs an authorial decision.** The `fuel` audit verdict moved from `efficient` to `predictable`, and on the efficiency panel Ridge now beats the random walk by +16.1% (DM p = 0.009) where it previously scored −0.9% (p = 0.881). §5.2 and the new §5.2.1 state this accurately and report it as exploratory. **The abstract, §1, §6 and the summary tables still describe the predictability map as uniformly negative.** That framing is the thesis's central claim, and revising it is a judgement about what the paper argues, not a number to be substituted — so it has deliberately been left to the author rather than rewritten mechanically.
>
> The automated check reports the remaining divergences as `verdict` mismatches, which is the correct signal: the numbers now agree and the narrative does not yet.
>
> Reproduce with `python -m ph_economic_ai.benchmark.manuscript_check`.

---

## Abstract

Claims that machine learning or multi-agent AI systems can "predict the economy" are common and rarely tested against a hard baseline. This thesis asks a narrower, answerable question for the Philippine case: can standard methods forecast monthly fuel prices, the peso–dollar exchange rate, and inflation better than naive persistence — and if a series resists forecasting, can the official figure at least be *nowcast* before its release? I build a small, fully reproducible benchmark that evaluates every claim with a strictly causal walk-forward backtest, an eight-method forecaster panel, Diebold–Mariano significance tests against the *strongest* simple baseline, and split-conformal prediction intervals. The result is a **predictability map**, and it is uniformly negative. One-month-ahead forecasts of premium gasoline (RON95), USD/PHP, and year-on-year inflation are **informationally efficient**: no method significantly beats a random walk — a null the accompanying power analysis bounds to a ~25% minimum detectable effect, so it rules out a large exploitable edge rather than every edge — reproducing Meese–Rogoff (FX) and Atkeson–Ohanian (inflation) for an emerging market. No nowcast target is predictable either, once the baseline pool is correctly specified.

That last qualification is the thesis's principal methodological contribution, and it arrived by self-refutation. An earlier version of this audit reported four significant positives, including a flagship within-month "driver edge" for electricity (Ridge +28.3%, DM p = 0.0011) that survived sub-sample stability tests, a preliminary-data robustness re-test, a Bonferroni family-wise correction, and carried an institutionally accurate mechanism. All were measured against a baseline pool of {random-walk, seasonal-naive, drift}, which omits the **historical mean**.

That the mean belongs in such a pool follows from existing work on baseline choice; what this thesis adds is the magnitude. For a covariance-stationary target with lag-1 autocorrelation ρ, a forecaster carrying no information beyond the unconditional mean shows an apparent skill over the random walk of **S(ρ) = 1 − [2(1 − ρ)]^(−1/2)**, positive exactly when ρ < ½ and ≈ +29% at ρ = 0. The expression matches simulation through the audit's own backtest to ≤ 0.011, and predicts the observed mean-versus-random-walk gap on all five real targets to ≤ 0.022. Inverted it becomes a diagnostic: the +28.3% electricity edge implies ρ ≈ 0.027, and electricity month-on-month inflation measures ρ = +0.002 — the flagship finding is fully accounted for by the target's own autocorrelation, with no driver, model, or mechanism required. With the mean in the pool every positive becomes a null (electricity Ridge is **1.8% worse than a constant**, p = 0.37; headline inflation falls from +16.2% to +4.1%, p = 0.36). A symmetry check confirms the correction cannot manufacture false negatives: on persistent level series ρ exceeds ½, the mean scores −1.7 to −2.1, and the forecasting nulls are untouched.

A size study quantifies the consequence: on simulated targets containing no signal at all, the mean-free protocol returns a significant "edge" in **99.7%** of replications at ρ = 0 and n = 203, against a nominal 5%. The distortion vanishes above ρ = ½ — the crossover the closed form predicts — and, critically, it *grows with sample size*, so the reflexive robustness response of re-running on a longer sample amplifies the error rather than exposing it. Adding the mean restores correct size (0.0% across every cell) while retaining power to detect a genuine driver (80.7% at n = 203). The exposure is not specific to Philippine data: a census of **FRED-MD**, the standard US macro panel, finds **80.2%** of its 126 series sit below the ρ = ½ crossover once the recommended stationarity transformation is applied — including **every** differenced series and 92% of growth rates, while no series left in levels is affected. Differencing removes exactly the persistence that makes a random walk a valid benchmark. Because S(ρ) is a property of the target rather than of the estimation, the artifact neither decays out of sample nor varies across sub-samples, and is unaffected by multiple-comparison correction — so a stack of robustness checks that all condition on the baseline offers the reassurance of five checks and the coverage of one.

The contribution is therefore methodological as much as empirical: an honest, reproducible protocol that separates what is forecastable from what is efficient in a data-poor economy, and a demonstration that the choice of naive baseline — not the model, the significance test, the robustness check, or the multiple-comparison correction — silently determined the verdicts. Every guard in the audit passed; only the baseline was wrong. A companion contribution turns the map outward: it conditions a program-aided *anchoring* layer that lets a small, offline language-model system produce physically coherent estimates on commodity hardware — each series anchored to the signal its own backtest identified, and each anchor regressed against the same real data, so that the one anchor that predicts (fuel, correlation 0.60 against realized pump changes) is distinguished from the two that only guard magnitude (electricity, food) rather than presented as uniform successes.

---

## 1. Introduction

### 1.1 Motivation
Fuel and food prices, and the inflation they feed, dominate Philippine household budgets and monetary-policy debate — never more visibly than during the 2022–2025 period that motivates this study. The Russian invasion of Ukraine in February 2022 drove world crude and refined-product prices to multi-year highs, and because Philippine pump prices have been market-determined since the Downstream Oil Industry Deregulation Act of 1998 (Republic Act 8479) — adjusted on a near-weekly cycle that tracks Mean of Platts Singapore quotations — the shock passed quickly to the pump, lifting gasoline and diesel to record levels by mid-2022 and prompting targeted fuel subsidies for the transport sector (e.g. the *Pantawid Pasada* programme). Excise taxes on petroleum products, raised in tranches under the 2018 TRAIN Law (Republic Act 10963), kept retail prices structurally elevated; proposals to suspend them during the 2022 spike were debated but not enacted. *(Institutional basis: RA 8479, RA 10963, the DOE Oil Industry Management Bureau's MOPS-based weekly pump-price mechanism, and the Pantawid Pasada program under DOE Joint Memorandum Circular No. 001, s. 2018 — cited under Primary institutional and data sources in the References.)*

The pass-through to consumer prices was rapid. Headline CPI inflation climbed through 2022 (averaging 5.8% for the year) to **8.7% year-on-year in January 2023** — its highest since November 2008, a roughly fourteen-year peak — well above the Bangko Sentral ng Pilipinas (BSP) target band of 2–4%, with food and fuel-intensive components among the largest contributors. The BSP responded by raising its policy rate from **2.0% to 6.50%** in a sequence of hikes from May 2022 through October 2023 (a cumulative 400 basis points). Inflation then receded: the annual average eased from 6.0% in 2023 to **3.2% in 2024** (back inside the 2–4% target) and further to about **1.7% in 2025** — below the band, the lowest in nearly a decade. Throughout, the monthly inflation print released by the Philippine Statistics Authority (PSA) was a closely watched signal that moved expectations and framed each BSP decision.

In that environment, a credible ability to anticipate next month's fuel price or inflation rate — even by the short interval between the close of within-month data and the official PSA release — would be valuable to households, firms, and the central bank. This is the practical question the thesis makes precise and tests. *(Sourced from PSA and BSP releases — Jan-2023 peak 8.7% [PSA CPI, highest since Nov-2008]; annual averages 5.8/6.0/3.2/1.7% for 2022–2025 [BSP]; policy rate 2.0%→6.50% by Oct-2023 [BSP]; target band 2–4% — cited under Primary institutional and data sources in the References.)*

### 1.2 The gap
A wave of applications — including multi-agent "swarm" and large-language-model systems — claim to forecast prices or "the economy," but almost none report a like-for-like comparison against the simplest defensible benchmark: assuming next month looks like this month. Without that comparison, an impressive-looking forecast says nothing. The prior question is therefore not "how good is the model?" but "is this series forecastable at all, relative to naive persistence — and how would anyone know?"

There is a second gap behind that one, and this thesis found it the hard way. Asking whether a model beats "the naive benchmark" presumes that *which* naive benchmark is a settled detail. It is not. On a mean-reverting target the random walk is not a neutral yardstick but a structurally weak one, and a model can clear it while carrying no information whatsoever. An earlier version of this audit reported four significant positives on exactly that basis; all four were withdrawn once the naive pool included the historical mean. Because the failure is in the comparison rather than the estimation, no amount of significance testing, sub-sample stability, or multiple-comparison correction detects it — a point developed in §4.7 and §5.7.

### 1.3 Research questions
- **RQ1.** Can standard methods forecast one-month-ahead PH fuel, FX, and year-on-year inflation better than a random walk?
- **RQ2.** If forecasting fails, what mechanism explains the efficiency?
- **RQ3.** Can the official inflation figure be *nowcast* before its release, and is any edge an information-timing effect or a time-series-dynamics effect?
- **RQ4.** Is any positive result robust to a larger, more varied sample?
- **RQ5.** How much does the choice of naive baseline determine these verdicts — by what magnitude, under what condition, at what false-positive rate, and across how much of the standard macro target space?

RQ5 was not part of the original design. It was forced by the audit's own results and is answered in §4.7 (a closed form and a size study), §5.7 (the withdrawn flagship), and §5.10 (a FRED-MD census). It is now the thesis's principal contribution, and RQ1–RQ4 are answered *under* it.

### 1.4 Contributions
1. A small, fully reproducible benchmark (`ph_economic_ai/benchmark/`) that turns "is it accurate?" from an assertion into a re-runnable measurement.
2. A **predictability map** of Philippine macro series: forecast and nowcast targets alike show no detectable edge over a properly-specified naive baseline.
3. A **quantified mirror to the "mind the naive forecast" warning** (§2.5, §4.7). The literature warns that the naive benchmark is *too hard* to beat on random-walk-like series (Hewamalage et al., 2023; Beck et al., 2025). The complementary failure — that on mean-reverting rates it is *too easy* — is given an exact magnitude here: a forecaster carrying no information beyond the unconditional mean scores S(ρ) = 1 − [2(1 − ρ)]^(−1/2) against the random walk, positive exactly when the target's lag-1 autocorrelation ρ < ½, and ≈ +29% at ρ = 0. The derivation is elementary; the contribution is its use as a **diagnostic**. Inverted, it converts any reported skill-over-random-walk into the autocorrelation that would produce it for free, which a reader can compare against the target's measured ρ. Applied to this audit's own flagship finding — an electricity "driver edge" of +28.3% that cleared significance, sub-sample stability, a trailing-preliminary-months re-test, a Bonferroni correction, and an institutionally accurate mechanism — it returns an implied ρ of 0.027 against a measured 0.002: fully explained without any driver. The expression matches simulation to ≤ 0.011 and the observed gap on all five real targets to ≤ 0.022. A companion **size-and-power study** turns this into an operating characteristic: the mean-free protocol returns a false positive on 99.7% of pure-noise datasets at ρ = 0, n = 203 (nominal 5%), the distortion grows with n rather than washing out, and the corrected pool restores exact size while keeping 80.7% power. A FRED-MD census (§5.10) shows 80.2% of the standard US macro panel sits in the affected regime after its own recommended transform — every differenced series, and no series in levels — so the condition is the common case, not an artefact of one dataset.
4. A **case study in non-independent robustness**: the same episode shows that a stack of guards which all condition on the baseline provides the reassurance of five checks and the coverage of one. Because S(ρ) is a property of the target rather than of the estimation, the artifact is stable out of sample and across sub-samples *by construction* — so stability testing actively corroborates it.
5. A discipline of *honest bounding*: removing fabricated confidence, requiring a beat over the strongest baseline — including the mean — using ablation to attribute any win to a mechanism rather than overclaiming, and reporting the superseded map alongside the corrected one so the change is auditable rather than quietly absorbed.
6. A companion application contribution — **benchmark-conditioned anchoring** (§6.6): a program-aided method that makes a small, offline language-model system produce physically coherent estimates by anchoring each series to the signal its *own* backtest identified, with every anchor regressed against the same real data and honestly separated into a validated pass-through predictor (fuel) and magnitude guards that do not forecast (electricity, food).

### 1.5 Roadmap
Section 2 reviews efficiency, nowcasting, and forecast-evaluation literature, and §2.5 sets out the two opposing failure modes of naive-benchmark choice that frame RQ5. Section 3 documents data and the fuel-proxy validation. Section 4 details the methodology; §4.7 derives the closed form for baseline-induced spurious skill and measures the resulting false-positive rate. Section 5 reports results: the predictability map (§5.1–§5.8), the sentiment keystone (§5.9), a FRED-MD census of how much of the standard target space is exposed (§5.10), and a minimum-detectable-effect for every null (§5.11). Section 6 discusses what the corrected boundary means, why five independent-looking robustness checks all passed on an artifact (§6.3), and the offline anchoring layer (§6.6). Section 7 concludes.

---

## 2. Background and Literature Review

This thesis sits at the intersection of four literatures: market efficiency and the random-walk benchmark, the predictability of inflation, nowcasting, and the statistics of forecast evaluation and uncertainty quantification. Each supplies a piece of the protocol used here, and each frames a specific way the empirical claims could be wrong. The review proceeds from the benchmark this thesis must beat, through the one setting where beating it is plausible, to the tools that decide whether a "beat" is real.

### 2.1 Market efficiency and the random-walk benchmark
The conceptual anchor is the efficient-market hypothesis (Fama, 1970): if prices already incorporate available information, future changes are unforecastable from that information, and the best predictor of tomorrow is today. In forecasting practice this becomes the random walk — a deceptively strong benchmark. The canonical demonstration is Meese and Rogoff (1983), who found that structurally-motivated exchange-rate models fail to beat a random walk out of sample at short horizons, despite using realized values of their own regressors. Four decades of re-examination have left the result largely intact: Cheung, Chinn and Pascual (2005) reconfirmed it across model classes and currencies, and the survey by Rossi (2013) concludes that exchange-rate predictability, where it exists, is fragile and horizon- and period-dependent. The lesson is not that markets are literally efficient but that the random walk is an *empirically* high bar; beating it requires genuine, stable, exploitable structure rather than in-sample fit. Accordingly, this thesis treats the random walk and its near-relatives — drift and seasonal-naive — as the family of baselines every candidate method must clear, and reports the *strongest* of them as the bar (Section 4.7).

### 2.2 Inflation forecasting and the naive benchmark
A parallel result holds for inflation. Atkeson and Ohanian (2001) showed that elaborate Phillips-curve forecasts struggle to beat a naive forecast that simply projects recent inflation forward, prompting a large literature on *why* inflation became "hard to forecast" (Stock and Watson, 2007) and on the conditions under which any model adds value (the survey of Faust and Wright, 2013). The common thread is persistence: a highly autocorrelated series leaves little for a model to add beyond its own lag. This thesis takes that thread literally and turns it into a testable mechanism. Year-on-year inflation is an overlapping twelve-month difference, so consecutive observations share eleven months of information and the naive forecast is mechanically excellent; the month-on-month transform removes that overlap and exposes whatever short-run dynamics remain. The framing in which a series is measured, not the series itself, can therefore determine whether it appears forecastable — a point the empirical results (Section 5.3) make concrete.

### 2.3 Nowcasting
Forecasting and *nowcasting* are distinct problems often conflated in applied "AI predicts the economy" work. Nowcasting (Giannone, Reichlin and Small, 2008; Bańbura, Giannone, Modugno and Reichlin, 2013) estimates a target *before its official release* by exploiting information that is already observable within, or shortly after, the reference period — the "real-time data flow." Related contributions include Evans (2005) on real-time GDP and the mixed-frequency activity index of Aruoba, Diebold and Scotti (2009); the review by Bok et al. (2018) synthesizes the field. The crucial conceptual point is that a nowcast edge is one of *information timing*, not of beating an efficient market: for a given calendar month, world oil, the exchange rate, and retail fuel are observable before the Philippine Statistics Authority publishes that month's CPI. A nowcast that beats persistence is therefore not a violation of efficiency but an exploitation of publication lag — a weaker, and far more defensible, claim. This distinction is the hinge of RQ3: the nowcast is precisely where, if anywhere, a genuine positive result should be found. This thesis finds none that survives a correctly specified naive baseline (§5.3, §5.7), which makes the distinction matter in a different way than intended — the publication-lag argument explains why a nowcast edge *would* be defensible, and its absence here is therefore not attributable to an efficiency constraint.

### 2.4 Evaluating and comparing forecasts
A lower error in one sample can be noise. Diebold and Mariano (1995) formalized the comparison with a test of equal predictive accuracy based on the loss-differential series; Harvey, Leybourne and Newbold (1997) supplied the small-sample correction used throughout this thesis, and West (1996) developed the asymptotics for predictive inference. A known subtlety is that the standard Diebold–Mariano test is designed for *non-nested* models; when one model nests the benchmark — as a model that can collapse to a random walk does — the test is conservative, and Clark and West (2007) propose an adjusted statistic. The retrospective by Diebold (2015) clarifies the test's intended scope. This thesis compares distinct method classes (e.g. ARIMA versus random walk) and applies the HLN-corrected Diebold–Mariano test as the arbiter of every "beats"/"efficient" verdict, treating raw RMSE ordering as suggestive only; the conservativeness under near-nesting biases *against* false positives, which suits an audit whose priority is not to overclaim.

### 2.5 Which naive benchmark? Two opposite failure modes

The significance test presupposes a benchmark, and the choice is not innocuous. Two literatures warn about it from opposite directions, and this thesis sits in the gap between them.

The first warns that the naive forecast is **too hard** to beat, and that apparent wins over it are spurious. Hyndman and Koehler (2006) motivate scale-free measures built on a naive benchmark; Hewamalage, Ackermann and Bergmeir (2023) catalogue the resulting evaluation pitfalls, noting that on a series which is genuinely a random walk the naive forecast is optimal by construction, so a model that appears to beat it has produced a spurious result. Beck, Dovern and Vogl (2025) make the empirical case at scale: across exchange rates, NASDAQ100 constituents, and FRED-MD, machine-learning models fail to beat a no-change forecast, and the closer a series is to a random walk the less useful complex models become. Their conclusion — include the naive benchmark, or your comparison is uninformative — is the discipline this audit adopts.

The second, less commonly stated in applied work, is the **mirror failure**: on a series that is *not* close to a random walk, the naive forecast is a weak benchmark, and beating it is uninformative in the other direction. The forecasting literature records the ingredients — the unconditional mean is the optimal forecast for white noise, the naive forecast is optimal for a random walk, and the mean becomes competitive as autocorrelation falls — and West and Lunsford (2026) study the closely related question of what happens when a random-walk model is used to forecast a *stationary* process, finding it has unusually low bias. What is not standard is a statement of how large the resulting distortion is, or a rule for when it applies. Section 4.7 supplies both: a forecaster carrying no information beyond the unconditional mean out-scores the random walk by exactly S(ρ) = 1 − [2(1 − ρ)]^(−1/2), positive if and only if the lag-1 autocorrelation ρ < ½.

The two failure modes are complementary and are distinguished by the target's persistence. Beck et al. study *levels* (exchange rates, equity prices) which are close to random walks, where naive is hard to beat. Every nowcast target in this thesis is a *month-on-month rate* with ρ well below ½, where naive is too easy to beat. An audit that adopts only the first warning — as the earlier draft of this one did — is fully protected against one error and entirely exposed to the other.

### 2.6 Distribution-free uncertainty
Honest forecasts require honest intervals. Conformal prediction (Vovk, Gammerman and Shafer, 2005; Shafer and Vovk, 2008) constructs prediction sets with finite-sample coverage guarantees under the sole assumption of exchangeability, without distributional or model-correctness assumptions; the split-conformal variant (Lei et al., 2018) makes this computationally trivial for a fitted regressor, and the tutorial of Angelopoulos and Bates (2023) gives the modern treatment. This machinery replaces the ad hoc — or, in the application this thesis grew out of, *fabricated* — "confidence" numbers common in deployed tools with intervals whose empirical coverage is measured and reported alongside the nominal level (Section 5.1). Where exchangeability is strained by time-series dependence, the reported coverage is treated as approximate and disclosed as such, rather than asserted.

### 2.7 Forecast accuracy measures and combination
Comparing errors across series of different scales motivates scale-free measures; this thesis uses the Mean Absolute Scaled Error (Hyndman and Koehler, 2006) alongside RMSE, MAE, MAPE and an explicit skill score, skill = 1 − RMSE_model / RMSE_baseline. Because the evaluation runs a panel of methods rather than a single model, the forecast-combination literature is also relevant: Bates and Granger (1969) established that combinations often dominate their constituents, and the survey of Timmermann (2006) catalogues when and why. Here the panel is used diagnostically — to locate the best honest competitor to the baseline — rather than to manufacture a winner by combination, but the literature motivates reporting the full panel rather than a single hand-picked model.

### 2.8 Philippine context
Two institutional facts make the Philippine case tractable. First, under the Downstream Oil Industry Deregulation Act (Republic Act 8479, 1998), domestic fuel pricing is deregulated and oil companies adjust pump prices on a near-weekly cycle that tracks world product prices (Mean of Platts Singapore, and Brent as a proxy) with a short lag. The pass-through literature — "rockets and feathers" (Bacon, 1991; Borenstein, Cameron and Gilbert, 1997) — documents that such retail prices follow upstream costs partially and asymmetrically, which is exactly the partial, lagged pass-through estimated in Section 5.2 and the mechanism behind the efficiency of the fuel series. Second, the Philippine Statistics Authority releases the Consumer Price Index on a fixed monthly calendar and publishes commodity-group detail (including Transport) through OpenSTAT, with recent vintages marked preliminary and subsequently revised — a feature that proves decisive for the Transport-CPI robustness result (Section 5.5). The gap this thesis fills is the absence, for Philippine macro series, of a reproducible and honestly-bounded predictability audit: most fuel- or inflation-"prediction" work in this space, including recent multi-agent and large-language-model systems, reports no like-for-like comparison against the naive benchmark and no out-of-sample significance test, and is therefore not assessable as a forecasting claim at all.

---

## 3. Data

All series are committed as CSVs and regenerable via `benchmark/refresh_data.py`; the benchmark reads only frozen files, so every number is reproducible offline.

### 3.1 Fuel (ground truth)
Premium gasoline (RON95), monthly PHP/litre, from the World Bank **Global Fuel Prices Database** (Open Database License). The committed gold series underpins the one-month forecast backtest over **2019-11 to 2025-03 (n = 99 months)**.

### 3.2 Exchange rate
USD/PHP monthly from Yahoo Finance (`PHP=X`).

### 3.3 Inflation
Philippine CPI from the IMF **International Financial Statistics** via DBnomics, transformed to year-on-year (YoY) and month-on-month (MoM) inflation.

### 3.4 Predictors
Brent crude (`BZ=F`), USD/PHP (`PHP=X`), an RBOB-gasoline→PHP landed-cost proxy (`RB=F`), and a seasonal demand index. Two feature panels are built: a standard ~10-year window and a **long window (Yahoo `max`, 2007–2026, 177 months)** used for the robustness re-test.

### 3.5 Fuel-proxy validation
Because high-frequency retail RON95 is not freely available as a long gold series, an RBOB-derived proxy is validated against the World Bank gold: Pearson **r = 0.91** with a mean bias of **−₱5.88/L** (disclosed, not corrected away). The proxy is fit for *directional/relative* validation, a stated limitation.

### 3.6 Reproducibility
`python -m ph_economic_ai.benchmark.run` regenerates every artifact (report, ablation, efficiency panel, pass-through, audit, YoY and MoM nowcasts, driver ablation, longer-sample confirmation, and figures) from the committed CSVs.

---

## 4. Methodology

### 4.1 Causal walk-forward backtest
The validity foundation. At each step t, models train only on data through t and predict t+1 (expanding window, minimum train = 24 months). A leakage-guard test asserts no future information enters any feature. This is what makes the accuracy claim defensible rather than in-sample.

### 4.2 Forecaster panel
Eight methods: random-walk, drift, seasonal-naive, the **historical mean**, ARIMA(1,1,1), ETS (additive trend), Ridge regression, and HistGradientBoosting. The first four are baselines; the last four are candidates. The historical mean is the expanding-window unconditional mean of the training window (no leakage): it is the optimal constant predictor, and for a mean-reverting series it — not the random walk — is the baseline to beat (§4.7).

### 4.3 Metrics
MAE, RMSE, MAPE, MASE, and skill = 1 − RMSE_model/RMSE_baseline.

### 4.4 Significance
Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction. A method "beats" a baseline only if it has lower RMSE *and* DM p < 0.05; otherwise the verdict is "efficient"/"no better than naive."

### 4.5 Calibrated uncertainty
Split-conformal intervals at nominal 50/80/90/95%, with an empirical-coverage calibration table reported alongside the nominal levels.

### 4.6 Forecasting vs nowcasting
Forecasting uses only information available *before* the reference month (lagged features + previous value). Nowcasting adds *within-month observable* drivers (contemporaneous oil/FX/fuel) plus the previous print — modelling the real situation in which those inputs are known before the PSA release. An eligibility rule keeps only genuinely pre-release information.

### 4.7 Baseline discipline (the hollow-win guard)
A candidate must beat the *strongest* of {random-walk, seasonal-naive, drift, **historical mean**}, not the weakest. This prevents a "win" that merely clears a baseline no one would use; it is enforced in code and unit-tested.

**Why the mean belongs in the pool.** The choice of baseline pool is not cosmetic; on a mean-reverting target it decides the verdict. A random walk predicts *last month's value*, so on a series that reverts it is systematically wrong by roughly √2 times the series' standard deviation, while the constant mean is wrong by one standard deviation. Every month-on-month inflation target in this audit is such a rate. A pool of {random-walk, seasonal-naive, drift} therefore sets the bar at a predictor that is *structurally weak for this class of target*, and any model that merely reverts to the mean clears it — scoring as "skill" what is actually the absence of a random walk.

That the mean belongs in a naive pool is not a new observation. It is implicit in the scale-free measures of Hyndman and Koehler (2006); the Atkeson–Ohanian (2001) result is precisely an argument about which baseline an inflation forecast must clear; and the textbook comparison — the mean is optimal for white noise, the naive forecast for a random walk, and the mean grows more competitive as autocorrelation falls — is standard (§2.5). The derivation below is elementary, resting on two variance calculations, and may well be folklore among forecasters even where it is not written down. What is nevertheless absent from applied practice, and what this section supplies, is a statement of **how large the distortion is, exactly when it applies, and how to test a published result for it**.

**A closed form for the artifact.** Let the target be covariance-stationary with variance σ² and lag-1 autocorrelation ρ. The random-walk forecast error is yₜ − yₜ₋₁, with variance 2σ²(1 − ρ); the mean forecast error is yₜ − μ, with variance σ². A forecaster carrying **no information beyond the unconditional mean** therefore exhibits an apparent skill over the random walk of

**S(ρ) = 1 − [2(1 − ρ)]^(−1/2),  which is positive if and only if ρ < ½.**

The crossover at ρ = ½ is exact, and only ρ enters — so the result holds for any stationary target, not merely an assumed AR(1). At ρ = 0 it gives S = 1 − 1/√2 ≈ **+29.3%**: on a white-noise rate, a model that has learned nothing at all is credited with a ~29% improvement over the random walk.

**Validation.** `benchmark/baseline_theory.py` checks the expression two ways. Against the project's own walk-forward estimator on simulated AR(1) targets — which uses an expanding-window mean, not the true μ — predicted and simulated skill agree to ≤ 0.011 across ρ ∈ [−0.2, 0.8]. Against the five real MoM targets:

| Target | ρ (lag-1) | S(ρ) predicted | mean-vs-RW observed | abs. error |
|---|---|---|---|---|
| Headline MoM | +0.343 | +12.8% | +12.6% | 0.002 |
| Headline MoM (long) | +0.369 | +11.0% | +12.2% | 0.012 |
| Food MoM | +0.375 | +10.6% | +12.8% | 0.022 |
| **Electricity MoM** | **+0.002** | **+29.2%** | **+29.6%** | 0.004 |
| Transport MoM | +0.140 | +23.8% | +24.8% | 0.010 |

The closed form accounts for the observed gap on every target to within 0.022.

![S(rho) against the target's lag-1 autocorrelation: the closed form, the simulation check, and every real target at its measured rho.](../../ph_economic_ai/benchmark/artifacts/figures/fig5_spurious_skill.png)

**Figure 5.** The closed form S(ρ) (black), the same quantity simulated through the project's walk-forward backtest (crosses), and the five real month-on-month targets at their measured ρ (circles). The shaded region is where a forecaster carrying no information beyond the unconditional mean is nonetheless credited with positive skill over the random walk. Electricity MoM sits at ρ ≈ 0, where that credit is ≈ +29%. Inverting it is diagnostic: the audit's flagship electricity "driver edge" of **+28.3%** implies ρ ≈ 0.027, and electricity MoM in fact measures ρ = +0.002. The headline result of the earlier draft is therefore quantitatively explained by the target's own autocorrelation, with no reference to any driver, model, or mechanism. A referee can apply the same inversion to any reported edge: if the implied ρ matches the target's measured ρ, the edge is consistent with pure baseline weakness.

**Why every robustness check passed.** S(ρ) is a property of the *target*, not of the estimation. It requires no data mining, no leakage, and no overfitting — so it does not decay out of sample, does not vary across sub-samples, is untouched by dropping the trailing preliminary months, and is unaffected by multiple-comparison correction. Any guard that does not interrogate the baseline will pass it, which is exactly what §5.7 records.

**How often does this produce a false positive? A size study.** S(ρ) gives the *expected* spurious skill; the operationally decisive quantity is the rate at which the protocol declares significance when there is nothing to find. `benchmark/baseline_size.py` measures it directly. A target is simulated as a stationary AR(1) with pure-noise features, so no model can legitimately win and **every rejection is a false positive**; the project's real `walk_forward` scores each method once; and `mom_verdict` is then evaluated twice on the *identical* fitted losses, changing only the baseline pool. The comparison is paired by construction, so nothing but the pool can explain the difference.

**False-positive rate (nominal α = 0.05), 300 replications per cell:**

| ρ | n = 82, pool without mean | n = 203, pool without mean | either n, pool **with** mean |
|---|---|---|---|
| 0.00 | 43.3% | **99.7%** | 0.0% |
| 0.20 | 19.7% | 87.7% | 0.0% |
| 0.35 | 4.7% | 26.3% | 0.0% |
| 0.50 | 1.0% | 1.3% | 0.0% |
| 0.70 | 0.0% | 0.0% | 0.0% |

Three things follow, and together they are the substance of this thesis's methodological claim.

*The size distortion is not marginal; at low persistence the test is uninformative.* At ρ = 0 and n = 203 the protocol rejects the null on **99.7%** of datasets containing no signal whatsoever. A nominal 5% test that rejects essentially always is not a slightly liberal test — it has no operating characteristics at all. Electricity MoM, the target carrying this audit's withdrawn flagship finding, has measured ρ = +0.002.

*The distortion is confined to ρ < ½, exactly where the closed form says it must be.* At ρ = 0.5 the rate collapses to ~1%, and at ρ = 0.7 to zero. §4.7 derives that crossover analytically from two variance calculations; this simulation reaches the same threshold from an entirely different direction, sharing no assumption beyond stationarity.

*More data makes it worse.* The rate rises with n in every row — 43.3% → 99.7% at ρ = 0, 19.7% → 87.7% at ρ = 0.2. This is the most practically damaging property, because "re-run it on a longer sample" is the reflexive response to a suspicious result. Here that move *amplifies* the error: a larger sample estimates the spurious mean-versus-random-walk gap more precisely, so the p-value tightens. It is exactly what happened in the earlier draft, where the long-sample re-run moved the headline MoM result from p = 0.032 to p = 0.001 and was reported as the finding having "held and strengthened" (§5.4).


![False-positive rate against rho for both sample sizes and both baseline pools.](../../ph_economic_ai/benchmark/artifacts/figures/fig6_size_distortion.png)

**Figure 6.** Rejection rate on simulated data containing no signal, so every rejection is a false positive. The mean-free pool (red) is catastrophically oversized below ρ = ½ and worsens with sample size — the gap between the dashed (n = 82) and solid (n = 203) lines. With the mean in the pool (blue) the rate is zero everywhere. The dotted line is the nominal α = 5%.

**Power is preserved.** A fix that merely blinds the audit would be no fix. Repeating the experiment with a genuine, contemporaneously-observable driver (β = 0.6), the mean-inclusive pool detects it in **80.7%** of replications at ρ = 0, n = 203 and 73.0% at ρ = 0.35. The corresponding without-mean figures (100% and 96.3%) must not be read as superior power: on those same targets that pool has a 43–100% false-positive rate, so its "detections" confound size with power. Only the corrected column measures detection of real signal. At n = 82 power falls to 13–17%, an honest reflection of the sample size rather than a defect of the correction, and consistent with the 13.2% minimum detectable effect reported for that target in §5.11.

**The bar is not vacuous.** A controlled test (`test_mean_baseline_rejects_the_beat_the_random_walk_artifact`) constructs a target as `0.6 × Δfuel + noise` while offering only the fuel *level* as a feature, so the driver is unrecoverable by construction. Ridge is then measurably **worse than the constant** (RMSE 0.643 vs 0.623) yet beats the random walk (0.871) by 26% — scoring as `beats_best_naive` under a mean-free pool. The companion test supplies the true driver, and Ridge wins decisively (91% skill over the mean, DM p < 0.001). The stricter bar rejects the artifact without suppressing genuine signal.

Adding the mean cannot manufacture a false *negative* either: on the persistent level series (fuel, FX, YoY inflation) ρ sits far above ½, the mean scores −1.7 to −2.1 in skill, and it never becomes the binding baseline — so the §5.1 efficiency verdicts are untouched. The correction bites exactly where S(ρ) predicts and nowhere else. Consequences for the nowcast verdicts are in §5.3–§5.8; the derivation is documented in `docs/defense/mean-baseline-finding.md`, with the superseded (mean-free) map retained in `corrected_predictability_map.json`.

### 4.8 Ablation and robustness
A **driver-only ablation** drops the own-lag (previous MoM) and restricts candidates to the driver regressors, isolating any within-month information edge from time-series dynamics. A **longer-sample re-run** rebuilds features on the 2007–2026 window and repeats the MoM nowcast and ablation.

### 4.9 Multiple comparisons
Where an audit conducts several confirmatory Diebold–Mariano tests — each of the form "beats the strongest naive baseline" — testing them at α = 0.05 individually inflates the family-wise false-positive rate. The family is defined *dynamically*, from the artifacts, as every DM test the benchmark actually ran: the `accuracy_report.json` nodes returning a `beats_best_naive` verdict with a computable p-value, plus every scored row of `selection_holdout.json` (§4.9.1). The efficiency findings are excluded, since accepting a null raises a power question (quantified in §5.1), not a false-positive one; so are per-method panel rows, because a panel is one hypothesis tested with K candidates rather than K hypotheses, and the audit's own fuel row, because `selection_holdout.json`'s `fuel_audit` is that same hypothesis re-tested honestly and counting both would charge one claim twice. Two corrections are applied and reported (`multiple_testing.json`, regenerated by `benchmark.multiple_testing`): the **Bonferroni** procedure controlling the family-wise error rate (the strictest), and the **Benjamini–Hochberg** step-up controlling the false-discovery rate.

Both procedures are two-sided, as the underlying DM test is, so the record stores the *direction* of every p-value alongside it. This is not bookkeeping: a DM test rejects just as readily when the model is significantly **worse** than its naive baseline, and two of this family's three sub-α p-values are exactly that (§5.8). A record that reported only how many tests came in under 0.05 would present a model losing badly as a near-finding.

Under the corrected baseline pool (§4.7) the `accuracy_report.json` half of the family is **empty** — there are no panel positives left to correct. The procedure is retained, and unit-tested against synthetic families, for two reasons: a future genuine positive must still be corrected, and the earlier draft's experience is itself instructive. That draft ran six confirmatory tests and reported that four survived Bonferroni — arithmetically correct, and completely uninformative about the actual error, because a family-wise correction guards against testing *many* hypotheses and offers no protection when every hypothesis is measured against an unsuitable baseline. Multiple-comparison control is not a substitute for baseline specification (§5.8).

### 4.9.1 Selection-honest re-test (post-selection inference)

A Diebold–Mariano verdict computed on the same sample used to choose which candidate to report is optimistic even when the chosen candidate turns out to be a naive-losing null: the act of picking the best of several candidates on a fixed sample inflates the apparent skill of whichever one is picked. `benchmark/selection.py` guards against this directly rather than by correction. For each target, the candidate model is chosen on a chronological **selection segment** (`min_train = 24` months, the initial `holdout_frac = 0.30` reserved) using the same panel and naive pool as the target's headline verdict, then re-scored, unchanged, on a **holdout segment** the selection step never saw. A run is scored `confirmed_on_holdout` only if the holdout skill is positive, the holdout Diebold–Mariano test rejects at α = 0.05, and the sign is a genuine win (`dm_stat < 0`); every holdout segment is required to clear `MIN_HOLDOUT_PREDICTIONS = 12` or the run is reported as underpowered rather than folded into a verdict. This is a stricter, non-overlapping counterpart to §4.9's family-wise correction: multiple-comparison control corrects for testing many hypotheses, while this protocol corrects for choosing the best of several candidates on the sample that also scores it.

The protocol originally covered two of the manuscript's targets, the fuel efficiency edge (§5.2.1) and the headline MoM nowcast (§5.3). It was extended to the remaining nine headline verdicts the predictability map reports — USD/PHP and YoY inflation forecasts, the long-sample headline MoM nowcast, and food, electricity and transport MoM in both full-nowcast and driver-only form — pre-registered in `docs/preregistration/2026-08-08-selection-holdout-remaining-headline-verdicts.md` before being run, using the same frame-building calls the audit itself already uses. Results for all eleven are reported together in §5.8.1.

### 4.10 Integrity infrastructure
The fabricated "90% confidence" from the original application was removed and replaced with the conformal interval. The frozen `accuracy_report.json` this benchmark writes makes the reported numbers tamper-evident and quotable. A hash-chained, two-phase track-record class (`engine/track_record.py`) also exists in the repository for the application's own run history; it is implemented and tested but not wired into normal application runs, so it does not currently make any application result tamper-evident (`CLM-TRACK-RECORD-001`, `RSK-012`).

---

## 5. Results

### 5.1 One-month forecasting is efficient (RQ1)

**Fuel.** On the efficiency panel (n = 72 forecasts, drawn from the 99-month feature-aligned window of §3.1), the best ML model (HistGradientBoosting) has RMSE 4.099 vs the random walk's 4.069 — a skill of **−0.0075**: no improvement. Against the seasonal-naive baseline the model's skill is +0.64, i.e. the gain is over a *bad* baseline, not the strong one.

On the efficiency panel (n = 72), one method now beats the random walk and the rest do not. **This is a change from the frozen artifacts and is reported as exploratory, not as a positive result** (see §5.2.1):

| Method | Skill vs RW | DM p vs RW |
|---|---|---|
| **Ridge** | **+0.161** | **0.009** |
| random_walk | 0.000 | — |
| drift | −0.009 | 0.483 |
| ETS | −0.034 | 0.242 |
| ARIMA | −0.048 | 0.299 |
| HGB | −0.074 | 0.421 |
| mean | −2.263 | 0.000 (worse) |
| seasonal_naive | −2.293 | 0.000 (worse) |

Every method except Ridge is statistically *indistinguishable* from the random walk or worse. Ridge's +16.1% (DM p = 0.009) appeared only after the 2026-07-28 calendar correction, which rebuilt the feature panel on a complete month index; on the previous gapped panel the same model scored −0.9% (p = 0.881). The correction repaired feature alignment, so a driver edge becoming visible is the expected direction rather than a surprise.

### 5.2.1 The fuel edge is exploratory, not a finding

It is reported here and deliberately not promoted, for reasons that are recorded rather than asserted.

**What supports it.** It is not the baseline artifact of §4.7: fuel's lag-1 autocorrelation is ρ = +0.9475, far above the ½ crossover, and the closed form predicts the mean should score −2.087 against the random walk here against a measured −2.263. Against 400 surrogates built by circularly rotating each driver column — destroying alignment while preserving each driver's own autocorrelation, scale and trend — the observed skill has empirical p = 0.0000 (null mean −0.152, null maximum −0.031). It also survives selection: with the model and the naive both fixed on the first 70% of rows and scored only on the remaining 29 predictions, skill moves +0.1233 → +0.1187, a shrinkage of 0.005 (DM p = 0.0296), where headline MoM on the same protocol collapses +0.0329 → −0.0057. §5.8.1 reports this protocol (§4.9.1) run against all eleven of the manuscript's headline verdicts, not only these two. The mechanism is an interaction, not a single driver: own-lag alone scores −0.030 and drivers alone −0.118, and only the combination beats the random walk, which is error correction toward the RBOB/FX-implied level and is consistent with the measured pass-through (β_total = 0.596, R² = 0.490).

**Why it is not claimed.** The audit family is three targets, so Bonferroni gives α = 0.0167 against an observed 0.0337 in sample and 0.0296 on the holdout. The holdout carries only 29 predictions. Most importantly it was never predeclared: it emerged from a data correction, which is precisely the pattern the size study of §5.10 exists to distrust. It is registered as `CLM-FUEL-EXPLORATORY-001` and is a candidate for preregistered confirmation, not a result.

**What does not change.** Every month-on-month null still holds against the historical mean on the corrected panels, so the methodological contribution of §4.7, §5.10 and §5.11 is unaffected — and is in fact strengthened, since the baseline result was not an artifact of the irregular calendar.

**Power — what "efficient" can and cannot mean here.** Accepting the null on a modest sample is only informative if the test could have rejected it, so the efficiency verdict is bounded by a power analysis (`power.json`). For the fuel forecast, the minimum skill the Diebold–Mariano test could detect at 80% power (α = 0.05) is **≈ 25%** RMSE improvement over the random walk; the observed skill is −0.3%. The honest reading is therefore **"no economically-large edge (≳ 25%) is detectable at this sample size,"** not "predictability is proven absent" — the test rules out a large, exploitable edge but cannot exclude a small one. This is the same bound Meese–Rogoff-style results carry and rarely state; here it is quantified.

**The predictability audit** extends this verdict across targets:

| Target | n | Verdict |
|---|---|---|
| Fuel (RON95) | 72 | no detectable edge over RW (efficient to a ~25% MDE) |
| USD/PHP | 38 | no detectable edge over RW |
| Inflation (YoY) | 59 | no detectable edge over RW |

All three show no detectable edge — reproducing Meese–Rogoff (FX) and Atkeson–Ohanian (inflation) for the Philippines, subject to the power bound above (which is tightest for the smallest samples, USD/PHP at n = 81).

**Calibration (fuel forecast).** The conformal intervals (half-widths ₱2.56/5.88/10.42/11.86 at 50/80/90/95%) over-cover at the upper levels (measured coverage 0.58, 0.88, 1.00, 1.00 over the disjoint 26-month validation half of the 52-forecast backtest) — i.e. conservative, honestly reported rather than tuned.

### 5.2 Mechanism (RQ2)
A pass-through regression of the RON95 change on contemporaneous and lagged driver changes (n = 97) gives β₀ = 0.31, β₁ = 0.24, **total pass-through β = 0.56**, R² = 0.33, with a near-zero driver autocorrelation (ACF₁ = 0.16). Interpretation: domestic fuel reflects a *partial, lagged* pass-through of a driver (world product price) that is itself close to a random walk. A predictable level built on an unpredictable, near-random-walk driver is exactly what produces an efficient series — the mechanism behind RQ1.

The Phase-2 gated feature ablation selected the `passthrough_lags` variant as the best-justified specification; it still did not beat the random walk, but it closed the model-vs-RW gap and tightened intervals.

![Pass-through of world product-price changes to PH RON95: a partial, lagged response built on a near-random-walk driver.](../../ph_economic_ai/benchmark/artifacts/figures/passthrough.png)

**Figure 1.** Estimated pass-through of contemporaneous and lagged driver changes to the monthly RON95 change (total β = 0.56, R² = 0.33). A partial, lagged pass-through of a driver that is itself close to a random walk is the mechanism behind the fuel series' efficiency.

### 5.3 Nowcasting (RQ3)

**YoY nowcast.** Adding within-month oil/FX/fuel plus the previous print, before the PSA release, still yields **no_better_than_naive** (n = 82). Year-on-year inflation overlaps 11 of 12 months with its prior value, so persistence is mechanically near-unbeatable.

**MoM nowcast — a null once the baseline is right.** Targeting month-on-month inflation, the honest bar is to beat the strongest of {random-walk, seasonal-naive, drift, mean} by a DM test (§4.7). The result is **no_better_than_naive**, and the binding baseline is the **historical mean**:

| Method | RMSE (n = 82) |
|---|---|
| ARIMA | 0.380 |
| **mean (best naive)** | **0.396** |
| Ridge | 0.398 |
| ETS | 0.4135 |
| random_walk | 0.453 |
| HGB | 0.457 |
| drift | 0.458 |
| seasonal_naive | 0.534 |

The decisive comparison is which baseline the candidate is measured against:

| Comparison | Skill | DM p |
|---|---|---|
| ARIMA vs random-walk | +16.2% | 0.032 |
| **ARIMA vs mean** | **+4.1%** | **0.358** |
| Ridge vs mean | −0.5% | 0.947 |

Almost the whole of the apparent +16.2% is the gap between the *random walk and the mean*, not between the model and a serious baseline: on a mean-reverting rate the random walk is wrong by roughly √2 standard deviations while a constant is wrong by one. Against the mean, ARIMA retains a residual +4.1% that is nowhere near significance (p = 0.36), and Ridge is *worse than the constant*. The honest verdict is therefore that **month-on-month inflation is not nowcastable beyond a naive baseline** at this sample size. The earlier positive was an artifact of a baseline pool that omitted the mean.

![Month-on-month inflation nowcast: ARIMA and the naive baselines against the realized pre-release actual.](../../ph_economic_ai/benchmark/artifacts/figures/nowcast_mom.png)

**Figure 2.** Month-on-month inflation nowcast (n = 82): ARIMA (own-dynamics) against the realized pre-release actual. ARIMA's apparent +16.2% edge is measured against the random walk; against the historical mean — the appropriate naive for a mean-reverting rate — the edge falls to +4.1% and is not significant (p = 0.36).

**Driver-only ablation.** Dropping the own-lag and restricting to driver regressors likewise gives **driver_edge = False** (n = 82). This verdict is unchanged by the correction: the contemporaneous-driver edge was never significant.

### 5.4 Robustness (RQ4)
Rebuilding features on the 2007–2026 window (n = 190, spanning the GFC, the 2014 oil crash, and COVID) and re-running:

| Metric | n = 82 | n = 190 |
|---|---|---|
| MoM verdict | no_better_than_naive | no_better_than_naive |
| best naive | mean | mean |
| skill vs best naive | 0.0 | 0.0 |
| driver_edge | False | False |

The null **holds across ~2.3× the data** and a far more heterogeneous regime mix. This is the mirror image of the earlier reading: what previously "held and strengthened" (p tightening from 0.032 to 0.001 against the random walk) was the *stability of the baseline artifact*, not of a forecasting edge. A larger, more varied sample estimates the historical mean more precisely, so a mean-reverting target becomes harder to beat, not easier — the correct robustness intuition, and the opposite of what the mean-free pool implied.

### 5.5 A spurious positive caught: the Transport-CPI nowcast

If any series should yield a significant within-month *driver* edge, it is **Transport** CPI — mechanically driven by fuel, which is observable before the official release. Using the official PSA Transport-CPI series (OpenSTAT, by commodity group, 2018 = 100, 1994–present) as a fresh gold target and the same free fuel/FX predictors, the nowcast was re-run (n = 203 backtest months).

On the **full sample**, the driver-only model looked like the sought-after positive: fuel-only Ridge beat the best naive baseline by **+14.8%** (DM p = 0.021). Taken at face value, this would license the headline "the fuel-driven component of inflation is nowcastable ahead of the official figure."

A robustness re-test dissolved it. PSA's three most recent prints are labelled **preliminary** and anomalous — Transport CPI 130 → 142 → 156 → 148 for early 2026 (i.e. +9.5%, +10.0%, −5.0% MoM). The check drops these trailing preliminary months from the single committed extract and re-tests on what remains; it does not compare against a later, revised release, since no archived data vintage was collected for this series. Dropping them collapses the skill from +14.8% to **zero**:

| Test | Verdict | skill vs best naive | DM p |
|---|---|---|---|
| Driver-only, full sample (n = 203) | beats_best_naive | +14.8% | 0.021 |
| Driver-only, robust (drop 6 preliminary months, n = 197) | no_better_than_naive | 0.0 | — |

The entire "edge" rested on roughly three unreliable observations. The **canonical verdict is therefore that Transport MoM inflation is also efficient** — no robust within-month fuel edge — consistent with the rest of the map. More importantly, this is a worked example of the audit doing its job: it caught a positive that a naive analysis would have published, traced it to three preliminary months, and reported the robust null. The robustness re-test (`driver_edge_robust`) is baked into the pipeline, so the check is permanent and reproducible.

Under the corrected baseline pool the transport case is now **doubly rejected**: it fails the preliminary-data robustness check *and* fails against the historical mean. Two independent guards catch the same false positive — but only the second would have caught it had the data been clean, which is the point of §4.7.

### 5.6 Food inflation: the second apparent positive, and a clean null driver

The same protocol was applied to **Food & non-alcoholic beverages** — the largest contributor to Philippine headline inflation — with a food-appropriate predictor panel: free global agri-commodity futures (rice, wheat, corn, soybean) plus oil and FX, all observable within the month. The PSA Food-CPI gold (OpenSTAT, COICOP division 01, 2018 = 100) provides the target; n = 203 backtest months (2007–2026).

Two results emerge, and both are nulls under the corrected pool:

| Test | Verdict | best naive | skill vs best naive | DM p |
|---|---|---|---|---|
| Full nowcast (drivers + own-lag) | no_better_than_naive | mean | 0.0 | — |
| Driver-only ablation, full sample (n = 203) | no_better_than_naive | mean | 0.0 | — |
| Driver-only ablation, robust (drop 6 preliminary months, n = 197) | no_better_than_naive | mean | 0.0 | — |

First, the apparent own-dynamics positive **does not survive the mean**. ARIMA's RMSE is 0.663 against the mean's 0.689 — a residual **+3.7%**, DM p = 0.456. Measured against the random walk (0.790) the same model shows +16.0% at p = 0.0046, which is the number the earlier draft reported. As with headline inflation (§5.3), essentially the whole apparent edge is the random-walk-to-mean gap.

Second, the food-commodity **driver edge remains a clean null**, and is now null by a wider margin: driver-only Ridge (0.702) is *worse* than the constant (0.689), a skill of **−1.9%** (p = 0.72). The verdict is stable at both n = 203 and n = 197 (`driver_edge_robust = False`). Global commodity prices carry no within-month signal for Philippine food inflation, consistent with its strongly-local composition (fish, vegetables, import-controlled rice). This conclusion is unchanged by the correction — it was a null before and is a null now, which is exactly the kind of result that should be insensitive to the baseline pool.

### 5.7 Electricity: the flagship edge does not survive the mean

Electricity was the audit's strongest apparent positive, and it is the one the correction overturns most sharply. Using the PSA `04.5.1 - Electricity` gold and free energy predictors (Brent, natural gas, FX), the driver-only nowcast (n = 203) gave the headline result of the earlier draft: Ridge **+28.3% over the best naive, DM p = 0.0011**, stable across sub-samples (≤2023-12 +26%, p = 0.006; 2007–2016 +30%, p = 0.020; 2016–2026 +29%, p = 0.035) and surviving the trailing-preliminary robustness check. Every check the audit had *was* passed.

The baseline pool was the check it lacked. The RMSEs make the situation unambiguous:

| Method | RMSE (driver-only, n = 203) |
|---|---|
| **mean** | **2.3515** |
| Ridge | 2.3936 |
| ARIMA | 2.4925 |
| random_walk | 3.3399 |

| Comparison | Skill | DM p |
|---|---|---|
| Ridge vs random-walk | +28.3% | 0.0011 |
| **Ridge vs mean** | **−1.8%** | **0.371** |

Ridge is **worse than predicting the historical mean**. The entire +28.3% is the distance between the random walk and a constant — the signature of a mean-reverting target measured against a baseline unsuited to it, and precisely the failure mode reproduced under controlled conditions in §4.7. The corrected verdict is **no_better_than_naive**, with `driver_edge_robust = False`.

The mechanism story must be withdrawn with the result. The Meralco generation charge *is* a formulaic pass-through of observable fuel costs, and that remains true as institutional description — the charge is recovered monthly under ERC's automatic cost-adjustment rules as a pass-through of contracted and WESM generation costs, subject to regulatory verification rather than utility discretion (cited under *Primary institutional and data sources* in the References) — but what the data do not support is the claim that this makes the monthly CPI print **nowcastable beyond a naive baseline**. A plausible mechanism was doing the work of evidence — a caution worth stating plainly, because the mechanism's plausibility is exactly what made the result feel safe. Sub-sample stability did not help either: an artifact rooted in the target's statistical character is stable by construction, so stability corroborated the wrong thing.

This is the single largest change the correction produces, and it removes the thesis's only claimed positive driven by contemporaneous observables.

### 5.8 The predictability map (synthesis)

| Target | Setup | Verdict | MDE (80% power) |
|---|---|---|---|
| Fuel / FX / YoY inflation | 1-month forecast | efficient (no method beats RW) | 24.7% |
| YoY inflation | nowcast (pre-release) | no better than naive | — |
| MoM inflation (headline) | nowcast (pre-release) | no better than naive (ARIMA +4.1% vs mean, p = 0.36) | 13.2% |
| MoM inflation (headline, n = 190) | nowcast, long sample | no better than naive (+4.6%) | 10.1% |
| MoM inflation (food) | nowcast (pre-release) | no better than naive (ARIMA +3.7% vs mean, p = 0.46) | 14.9% |
| Food-CPI MoM | nowcast, driver-only | clean null — Ridge −1.9% vs mean | 14.9% |
| Transport-CPI MoM | nowcast, driver-only | rejected twice — preliminary-data artifact *and* fails vs mean | 12.5% |
| **Electricity-CPI MoM** | **nowcast, driver-only** | no better than naive — Ridge **−1.8% vs mean** (p = 0.37) | **5.8%** |

Every target in the audit now returns the same verdict: **no detectable edge over a properly-chosen naive baseline.** The map is uniform. §5.8.1 re-tests every verdict above under a selection-honest protocol that separates model choice from the score it is reported against; none flips.

**Each null is bounded (§5.11).** The final column is the minimum skill over the *binding* baseline that a Diebold–Mariano test could have detected at 80% power — the honest scope of each "no". Two readings matter. The headline and food nulls are **loose**: at 10–15% MDE against observed edges of 3–5%, they rule out an economically large edge but cannot exclude a modest one. The **electricity driver null is tight**: a 5.8% MDE against an observed −1.8% means the test had ample power and still found the model performing worse than a constant. That is worth stating plainly, because electricity is the target whose +28.3% "edge" this thesis withdraws — the replacement null is the best-powered result in the map, not a shrug.

This is a weaker set of claims than the earlier draft made, and a stronger thesis. A predictability audit whose answer is "efficient almost everywhere" reproduces Meese–Rogoff and Atkeson–Ohanian for a new market and adds a methodological result of its own: that the *choice of naive baseline* silently determines the verdict, and that the omission of the historical mean is sufficient to manufacture four significant positives — one of which (electricity, p = 0.001) survived sub-sample stability checks, a robustness re-test, and a Bonferroni correction. Every guard in the audit passed. Only the baseline was wrong.

![Predictability map: skill vs the strongest naive baseline per target — no target shows a detectable edge once the historical mean is in the baseline pool.](../../ph_economic_ai/benchmark/artifacts/figures/predictability_map.png)

**Figure 3.** The predictability map — skill over the strongest naive baseline per target, with the historical mean in the pool. Every bar sits at or below zero: no target is predictable beyond naive. (Compare the superseded mean-free map in `corrected_predictability_map.json`, where four of these bars appear significantly positive.)

![Audit verdicts by target under the corrected baseline pool.](../../ph_economic_ai/benchmark/artifacts/figures/audit_verdicts.png)

**Figure 4.** Audit verdicts by target. The same protocol that previously produced a three-way split (positive / null / rejected) returns a uniform null once the baseline pool is corrected.

**Multiple-comparison correction (§4.9).** Under the corrected pool the panel half of the family is **empty**: there are no `beats_best_naive` positives left to correct. The live family is therefore the 23 scored rows of `selection_holdout.json`, and `multiple_testing.json` records them at **m = 23, Bonferroni threshold α/m = 0.0022**.

**Nothing survives either correction.** Three rows land under an uncorrected α = 0.05 — and 23 tests at that level buy **1.15 hits by chance alone**, so three is very close to what a family of pure nulls would produce. Two of the three are the model performing significantly *worse* than its naive baseline (`dairy_eggs_mom_driver_only`, −35.0%, p = 0.0201; `sugar_mom_driver_only`, −82.1%, p = 0.0218), which leaves exactly one nominal positive in the model's favour.

That one is the fuel efficiency edge, **and it does not survive**: at p = 0.0296 against a threshold of 0.0022 its Bonferroni-adjusted p is **0.68**, and Benjamini–Hochberg — the more permissive of the two — puts its q-value at **0.23**, an order of magnitude above the 0.05 an FDR reading would need. §5.8.1 reports it as `confirmed_on_holdout`, and that verdict stands: it answers a different question (does the edge survive *selection*), and it is the reason the row is in this family at all. What multiplicity adds is that surviving selection is not sufficient. One test in twenty-three clearing 0.05 is the expected yield of twenty-three coin flips, and the correction says so numerically rather than leaving a reader to notice.

This inverts the earlier draft, which reported that four of six confirmatory tests survived the strict Bonferroni threshold — arithmetically correct at the time. A family-wise correction controls for testing *many hypotheses*; it offers no protection whatever against every hypothesis being measured against the wrong baseline. It also offers none against a family that is never assembled: until 2026-08-16 this record read `n_tests = 0` while the 23 holdout tests sat uncorrected in the artifact beside it, because `build_family` read only `accuracy_report.json`. An empty multiplicity record behind a claimed positive is the failure this section exists to prevent, and it had occurred.

### 5.8.1 Selection-honest re-test: all eleven headline verdicts

§4.9.1 describes the protocol; two targets (fuel, headline MoM short-sample) had already been run through it. The remaining nine — every other headline verdict in §5.8's map — were pre-registered on 2026-08-08 before being run, with the expectation that all nine would return `not_confirmed_on_holdout`, matching their published null or withdrawn status. `selection_holdout.json` now reports all eleven:

| Target | n | Selection cut | Holdout n | Selection skill | Holdout skill | Shrinkage | Holdout DM p | Verdict |
|---|---|---|---|---|---|---|---|---|
| Fuel (RON95) efficiency | 96 | 67 | 29 | +12.3% | +11.9% | 0.005 | 0.0296 | **confirmed_on_holdout** |
| Headline MoM, nowcast | 106 | 74 | 32 | +3.3% | −0.6% | 0.039 | 0.948 | not_confirmed_on_holdout |
| USD/PHP forecast | 105 | 74 | 31 | +0.2% | −4.5% | 0.047 | 0.463 | not_confirmed_on_holdout |
| YoY inflation forecast | 104 | 73 | 31 | +20.5% | +4.3% | 0.162 | 0.712 | not_confirmed_on_holdout |
| Headline MoM, long sample | 214 | 150 | 64 | +6.5% | +2.5% | 0.040 | 0.668 | not_confirmed_on_holdout |
| Food MoM, full nowcast | 227 | 159 | 68 | +9.6% | +0.0% | 0.096 | 0.995 | not_confirmed_on_holdout |
| Food MoM, driver-only | 227 | 159 | 68 | −10.4% | −2.4% | −0.080 | 0.509 | not_confirmed_on_holdout |
| Electricity MoM, full nowcast | 227 | 159 | 68 | −4.9% | −0.7% | −0.043 | 0.733 | not_confirmed_on_holdout |
| Electricity MoM, driver-only | 227 | 159 | 68 | −3.0% | −0.0% | −0.030 | 0.991 | not_confirmed_on_holdout |
| Transport MoM, full nowcast | 227 | 159 | 68 | +5.6% | −3.6% | 0.092 | 0.847 | not_confirmed_on_holdout |
| Transport MoM, driver-only | 227 | 159 | 68 | −4.3% | +0.8% | −0.052 | 0.928 | not_confirmed_on_holdout |

**Ten of eleven do not survive.** Only the fuel efficiency edge is `confirmed_on_holdout`; it is still reported as exploratory rather than a finding, for the reasons given in §5.2.1 (that it was never predeclared) and in §5.8 (its Bonferroni-adjusted p is 0.68 and its BH q-value 0.23 over the 23-test family, so it survives neither correction — `multiple_testing.json`). Every other target's holdout skill either collapses toward zero or reverses sign, so none of the ten changes the verdict already reported for it in §5.1–§5.8: the map remains uniformly null in the confirmatory sense.

> **ARTIFACT-DIVERGENCE (2026-08-16).** The table above covers the eleven targets pre-registered to 2026-08-08. `selection_holdout.json` has since grown to **23** rows: twelve PSA food sub-category frames (rice, meat, fish, dairy/eggs, vegetables, sugar × full and driver-only) pre-registered in [`2026-08-12-food-subcategory-selection-holdout.md`](../preregistration/2026-08-12-food-subcategory-selection-holdout.md), which reports their verdict table in full. All twelve returned `not_confirmed_on_holdout`, so the paragraph above is unchanged in substance — but the multiplicity family in §5.8 is m = 23, not m = 11, and this table has not yet been extended to match.

**The clearest single illustration of the bias this protocol exists to catch is YoY inflation.** Its selection-stage skill, +20.5%, is the largest of any target across all eleven, selected or not — and it fully evaporates on the holdout (+4.3%, DM p = 0.71). A verdict built only from the selection segment would have reported a strong edge; the same model on data selection never touched shows almost none.

**Electricity's driver-only row — the withdrawn +28.3% flagship of §5.7 — shows the smallest shrinkage of the nine newly-run targets** (−0.030, i.e. the holdout skill is marginally less negative than selection). That is consistent with, not in tension with, §4.7's account of what the +28.3% figure actually was: both stages sit at essentially zero because there was never a real edge to shrink away from, only a mean-reverting target measured against the wrong baseline.

Frame sizes at run time matched the pre-registration's feasibility check exactly (105/104/214/227 rows for the four newly-run frame shapes), and every holdout segment cleared `MIN_HOLDOUT_PREDICTIONS` by at least 2.5×, so none of the nine is reported as underpowered.

### 5.9 Search interest does not nowcast inflation

A recurring claim in applied work is that public attention — social posts, search volume — carries early signal about prices. Because the companion application (§6.6) reasons over exactly this kind of chatter, the claim is tested here with the same machinery rather than assumed.

Monthly Google Trends search interest for four price-salient terms (`presyo ng gas`, `diesel price`, `Meralco bill`, `bigas presyo`, PH geography, 2016–2026) is joined to the headline and food MoM nowcast frames and run through the identical walk-forward + Diebold–Mariano `mom_verdict` against the mean-inclusive pool. Three specifications are tested per target: search terms alone, drivers alone, and drivers plus search terms.

| Target | sentiment only | drivers only | drivers + sentiment |
|---|---|---|---|
| Headline MoM (n = 82) | no better than naive | no better than naive | no better than naive |
| Food MoM (n = 102) | no better than naive | no better than naive | no better than naive |

**Search interest does not nowcast Philippine fuel or food inflation beyond a naive baseline**, alone or as an increment to the drivers (`sentiment_nowcast.json`).

One measurement note matters for interpreting this null, because it nearly produced a vacuous one. Google Trends normalises every term within a *single query* against one shared 0–100 scale fixed by the highest observed point. Querying the four terms together let the highest-volume English term set the scale and collapsed the low-volume Tagalog terms to a near-constant zero — `presyo ng gas` was nonzero in 1 month of 127. A regression on a constant cannot fail to be uninformative, so that null would have carried no evidential weight. Re-querying each term in its own payload restores within-term variation (`presyo ng gas`: 22 nonzero months, 16 distinct values; `bigas presyo`: 94 and 27), at the cost of cross-term level comparability, which no part of this analysis uses. The reported null is therefore a null about *search interest*, not an artifact of a degenerate feature.

### 5.10 How much of the standard target space is exposed? A FRED-MD census

The natural objection to §4.7 is that it describes a quirk of Philippine data. It does not. To measure the exposure directly, the same criterion is applied to **FRED-MD** (McCracken and Ng, 2016) — the canonical monthly US macro panel, and the dataset on which Beck, Dovern and Vogl (2025) run the complementary "naive is hard to beat" analysis.

FRED-MD ships a recommended stationarity transformation per series (its `tcode`). Applying it is precisely what a nowcasting study does before modelling, so the transformed series *are* the objects the literature forecasts. For each, the lag-1 autocorrelation is measured and compared against the ρ = ½ crossover. A series below it is *vulnerable*: on that target a model can be reported as significantly beating the random walk while carrying no information, and §4.7's size study puts the false-positive rate at 26% by ρ = 0.35, 88% by ρ = 0.2, and ~100% as ρ approaches 0.

**126 series, after the recommended transform:**

| | count | share |
|---|---|---|
| ρ < 0.5 — vulnerable | 101 / 126 | **80.2%** |
| ρ < 0.2 — false-positive rate ≥ 20% | 73 / 126 | 57.9% |
| ρ < 0.05 — false-positive rate ≈ 100% | 56 / 126 | 44.4% |

Median ρ across the panel is **+0.107**; the median spurious skill a wholly uninformative forecaster would post on the vulnerable series is **+29.3%**.

The decomposition by transformation is the structural result, and it is unambiguous:

| Transformation | Vulnerable |
|---|---|
| First difference | **19 / 19 (100%)** |
| Second log difference | **33 / 33 (100%)** |
| Log difference (growth rate) | 48 / 52 (92.3%) |
| Level | 0 / 11 (0%) |
| Log | 0 / 10 (0%) |

![Distribution of lag-1 autocorrelation across FRED-MD, split by whether the series was differenced.](../../ph_economic_ai/benchmark/artifacts/figures/fig7_fredmd_exposure.png)

**Figure 7.** Lag-1 autocorrelation across the 126 FRED-MD series after each series' own recommended stationarity transformation, separated by whether that transformation involved differencing. The two groups barely overlap: differenced series (red) sit below the ρ = ½ threshold, series left in levels (blue) sit well above it.

**Every differenced series in FRED-MD is vulnerable; no series left in levels is.** That is not a coincidence but the mechanism stated in reverse. A random walk is a good baseline exactly when a series is persistent, and the purpose of differencing is to remove persistence. The standard stationarity transformation therefore moves a series out of the regime where the random-walk benchmark is appropriate and into the regime where it is structurally weak — and it does so silently, because the transformation is applied for an unrelated and entirely correct reason.

Two consequences follow. First, the exposure is not exotic: four out of five series in the standard panel sit in the affected regime, and the most severe cases are ordinary quantities — real personal income, medical-care CPI, average weekly earnings — whose transformed form carries *negative* autocorrelation (ρ ≈ −0.49 to −0.64), where an uninformative forecaster would post an apparent skill of **+42% to +45%**. Second, the check is cheap: measuring ρ is one line, and the inversion of §4.7 converts any reported skill-over-random-walk into the autocorrelation that would produce it for free.

This section makes no claim about any particular published result, and none should be read into it. It measures the *target space*, not anyone's findings. What it establishes is that the condition under which this thesis's own flagship result proved spurious is the common case rather than the exception — which is why the baseline check belongs in the standard protocol rather than in a footnote.

### 5.11 How strong is each "no"? Minimum detectable effects

A null is only informative if the test could have rejected it, so every null in the map carries a minimum-detectable-effect (MDE): the smallest skill over the binding baseline that a Diebold–Mariano test would detect at 80% power and α = 0.05 (`power.json`, reproduced by `benchmark.power`).

Two design points matter. First, each nowcast null is bounded against the **mean**, not the random walk — bounding against a baseline that does not bind would overstate the test's power, and by exactly the S(ρ) margin of §4.7. Second, the model used is the *strongest candidate* on each target, so the bound describes the most favourable case the audit actually gave a model rather than a convenient weak one.

| Target | n | Best candidate | Observed skill | MDE @ 80% power |
|---|---|---|---|---|
| Fuel one-month forecast (vs RW) | 51 | HGB | −0.3% | 24.7% |
| Headline MoM | 61 | ARIMA | +4.1% | 13.2% |
| Headline MoM (long) | 143 | ARIMA | +4.6% | 10.1% |
| Food MoM | 151 | ARIMA | +3.7% | 14.9% |
| Electricity MoM (full) | 151 | Ridge | −4.3% | 11.6% |
| **Electricity MoM (driver-only)** | 151 | Ridge | **−1.8%** | **5.8%** |
| Transport MoM (driver-only) | 151 | Ridge | −2.8% | 12.5% |

Every observed skill sits inside its own detectable band — which is what makes each verdict a null rather than an unresolved positive, and is asserted as a unit test so a future genuine positive cannot be silently reported as a null.

The spread across rows is the honest part. The fuel forecast (n = 72) is the weakest test in the thesis at ~25% MDE; the headline and food nowcast nulls sit at 10–15%; and the electricity driver-only null is the strongest at 5.8%. So the claims are not uniform in strength, and should not be stated as though they were: for fuel and headline inflation the defensible statement is "no *large* edge is detectable at this sample size", whereas for the electricity driver channel the test genuinely had the resolution to see a small edge and did not.

That last row carries the most weight. Electricity is the target whose apparent +28.3% driver edge this thesis withdraws (§5.7), and the natural objection to a withdrawal is that the replacement null is merely underpowered. It is not: the electricity driver test is the best-powered in the map.

---

## 6. Discussion

### 6.1 The efficient-vs-forecastable boundary
The boundary this audit set out to draw turns out to lie in an unexpected place: not between series, but between *baselines*. The MoM transform does expose short-run autoregressive structure that the YoY frame hides — ARIMA's RMSE genuinely improves on the random walk's at every MoM target. What the corrected pool shows is that this structure is almost entirely **mean reversion**, and mean reversion is what a constant already captures. Removing the twelve-month overlap does not reveal exploitable signal; it reveals that the random walk was never the right yardstick for the transformed series. The honest boundary is therefore: efficient at one month, and efficient in the nowcast too — with the caveat that "efficient" here means "no edge detectable at these sample sizes" (§5.1's ~25% MDE), not "no edge exists."

### 6.2 Reproducing landmark results for an emerging market
Finding FX and inflation efficient at one month reproduces Meese–Rogoff and Atkeson–Ohanian outside their original developed-economy settings, adding external validity rather than novelty for its own sake.

### 6.3 Honesty as method
Several design choices each prevented a specific overclaim: removing the fabricated 90% confidence (replaced by measured conformal coverage); requiring a beat over the *strongest* baseline (the hollow-win guard); the driver-only ablation (which stopped "MoM is predictable" from silently becoming "the drivers predict inflation"); the trailing-preliminary-months robustness re-test; and the multiple-comparison correction.

The Transport-CPI nowcast (§5.5) illustrates the discipline working as designed. A full-sample run produced an apparently significant fuel edge (+14.8%, p = 0.021) — precisely the bold "AI nowcasts fuel-driven inflation" headline one might want to claim. A trailing-preliminary-months robustness check showed it rested on three preliminary PSA observations and vanished once they were removed. A method that only ever confirms is not doing robustness.

The Electricity-CPI result (§5.7) illustrates the harder lesson, and is the more valuable of the two. Its +28.3% driver edge passed *every* guard listed above: it was significant, it survived the preliminary-data re-test, it was stable across both sample halves and earlier cutoffs, it had a compelling and institutionally accurate mechanism, and it cleared a Bonferroni family-wise correction. It was nonetheless an artifact — the model was worse than a constant, and the entire edge was the random-walk-to-mean gap on a mean-reverting target.

The instructive point is *how* the guards failed. None malfunctioned; each answered a question that was not the one that mattered. Significance testing asks whether an edge over the chosen baseline is real, not whether the baseline is appropriate. Sub-sample stability asks whether an effect is period-specific — but an artifact rooted in the target's statistical character is stable everywhere, so stability actively corroborated the error. Multiple-comparison correction asks whether many tests inflate false positives, not whether all of them share a common misspecification. And a plausible mechanism supplied narrative confirmation that made the number feel safe. Robustness checks compound only if they are *independent*; five checks that all take the baseline as given provide the reassurance of five but the coverage of one.

The strongest version of "honesty as method" is therefore not a longer checklist but a willingness to publish the refutation of one's own headline result. The superseded map is retained in `corrected_predictability_map.json` and the derivation in `docs/defense/mean-baseline-finding.md`, so a reader can audit the change rather than take it on trust.

### 6.4 What the MoM result is and is not
The corrected verdict is uniform, not partial. Once the naive pool includes the historical mean, no month-on-month target — headline, food, or electricity — shows a robust, significant own-dynamics or driver-based edge (§5.3–§5.7). The pre-correction picture, in which headline and food appeared to have a robust own-dynamics nowcast and electricity additionally appeared driver-predictable, was the baseline-omission artifact this thesis's methodological contribution identifies (§4.7, §5.10): a mean-reverting target measured against a comparator unsuited to it. None of those three apparent positives survives, electricity least of all — its 28.3 percent driver edge is 1.8 percent *worse* than predicting a constant (§5.7). The honest interval, not a point estimate, is the deliverable, and for every MoM target that interval is the naive one.

### 6.5 Practical relevance
For every series this audit covers — fuel, FX, and YoY inflation forecasts, and headline, food, and electricity MoM nowcasts alike — the defensible product is a calibrated interval around the naive forecast, and the honest statement that no model beats it. Effort spent building a driver-based nowcast for any of them is not effort spent where predictability exists, because the audit finds none. What the audit delivers instead is constructive: the closed-form diagnostic of §4.7, which explains *why* the apparent headline, food, and electricity positives arose and gives a reader a one-line test for the same artifact in other work, and the discipline of §6.3 for catching it before it is published rather than after.

### 6.6 From audit to application: benchmark-conditioned anchoring
The audit's purpose is negative — to say what cannot be forecast — but its findings are also constructive: they specify, per series, *what* signal is worth conditioning on. The companion application (the multi-agent "swarm"; Appendix D) exploits this directly, and in doing so exposes a second methodological problem that the same discipline of honest bounding resolves. The application must run offline on commodity hardware (an 8 GB consumer GPU), which restricts it to small quantized language models (Qwen2.5-3B/7B). Such models reason adequately about the *direction* of an economic shock but are unreliable about its *magnitude*: asked for the pump-price effect of a +6.8% crude move, a 7B judge returned +₱12.93/L, roughly five times the mechanical pass-through of +₱2.72/L. A plausibility filter then discards the estimate and the report shows nothing — the failure is silent. Fine-tuning does not fix this: arithmetic and quantity estimation are known, persistent weaknesses of small models that survive supervised adaptation.

The resolution follows the program-aided paradigm (Gao et al., 2023): do not ask the model for the quantity it cannot produce. The magnitude of an oil→pump pass-through is not an opinion but accounting — crude cost per litre, revalued at the exchange rate, plus VAT — and is computed deterministically (Appendix C). This *anchor* is used three ways: injected into the prompt as a prior, so the model reasons from the correct scale; applied as a leash that clamps an estimate diverging beyond a plausibility band back toward the anchor while preserving the model's direction; and used as a fallback when the model produces nothing, so the pipeline never returns a blank. The design is deliberately conditioned on the audit — each series is anchored to the signal *its own* backtest identified as informative. Fuel and electricity receive a mechanical fuel pass-through anchor; food, which the audit found a clean null on commodity drivers (§5.6) but predictable from own dynamics, is anchored to the trailing trend of food inflation itself rather than to oil, since anchoring it to commodities would be anchoring it to what the audit proved is noise.

Because an anchor is a quantitative claim, it is itself testable against the same real series the audit uses, and is regressed there rather than asserted, with significance judged by the same HLN-corrected Diebold–Mariano test the audit applies to its own claims (`anchor_validation.json`; `tools/anchor_backtest.py`). The fuel anchor is a genuine model of the *contemporaneous* pass-through: over 78 months of World Bank RON95 against monthly Brent and USD/PHP, its predicted monthly pump change correlates **0.60 with the realized change (p < 0.001, 95% CI [0.44, 0.73])** and matches direction 74% of the time. Its mean absolute error is lower than a no-change baseline (₱2.21 vs ₱2.64), but — and this is the honest limit — that improvement is **only marginal (DM p = 0.065, significant at the 10% but not the 5% level)**: the anchor demonstrably co-moves with pass-through, yet at n = 97 the evidence that it beats naive persistence on squared error is suggestive rather than conclusive. The ordinary-least-squares slope of realized on predicted change, **0.79 ± 0.12**, is significantly non-zero (p < 0.001) but **not distinguishable from a full 1:1 pass-through at the 5% level (p = 0.084)**; it is fed back as the anchor's calibration coefficient — consistent with the partial, lagged, asymmetric adjustment the rockets-and-feathers literature documents (§2.8) — while acknowledging the data cannot rule out complete pass-through. This is a *contemporaneous* relationship, and does not contradict the one-month-ahead efficiency of §5.1: the anchor scales a *known* shock, it does not forecast one.

The other two anchors are reported with the same candour that governs the audit. Regressed against 175 months of PSA electricity CPI, the fuel-price anchor does *not* predict the monthly move (correlation 0.03–0.13 across lags; the strongest is **not significant, p = 0.08**); consistent with §5.7, where the full driver panel found no genuine edge either — the apparent 28.3 percent gain was a baseline artifact, not a real relationship the anchor's cruder proxy merely fails to recover. Against 172 months of PSA food CPI, persistence and an oil driver are **each individually significant against zero (r = 0.18, p = 0.02; r = 0.21, p = 0.006) but too weak to separate from one another** (within one standard error) or to beat a simple mean on error. Neither the electricity nor the food anchor is a useful predictor at monthly resolution. What each *is* — and what the anchor is for — is a magnitude guard: the ratio of the anchor's typical size to the realized monthly move is ≈1.0 for electricity and ≈0.9 for food, so each keeps a weak model's estimate correctly scaled even where it cannot forecast. Measured against the realized series, reconciliation more than halves the error of a simulated hallucinating model (₱3.79 → ₱1.58, a 58% reduction), and a robustness sweep over 10,357 scenarios and adversarial inputs finds no case in which the anchoring returns a non-finite or unbounded value.

The contribution is of a piece with the thesis's central discipline. Just as the audit separates the forecastable from the efficient and refuses to overstate its one positive, the application separates the anchor that *predicts* (fuel) from the two that only *guard magnitude* (electricity, food), and labels each as such rather than presenting three uniform successes. The result is not a system that forecasts the economy — the audit forbids that claim — but a small, offline, weak-model system whose numbers are physically coherent, whose corrections are transparent, and whose every anchor is validated, and bounded, against the same real data as the audit itself.

### 6.7 Does swarm size matter? An agreement-not-accuracy ablation
A multi-agent system invites the obvious question its proponents rarely answer: does the size of the ensemble earn its cost? The application's swarm — twenty agents across four regions, two elimination rounds — was ablated against three cheaper configurations (halved to two regions; a single round; and shortened agent completions), each run eight times so that run-to-run variance in the master estimate could be *measured* rather than assumed (`swarm_ablation.json`; `tools/swarm_ablation.py`). A configuration counts as a defensible economy only if its estimate range *overlaps* the full swarm's *and* its spread is no wider.

The result is a clean negative for the intuition that more agents produce a better *number*, and a modest positive for the intuition that they produce a more *stable* one. All three reduced configurations reach the same verdict — every mean falls within ₱3.1–3.6/L and overlaps the full swarm's range — so the extra agents, regions, and rounds do not move the central estimate. What the full swarm delivers is the lowest run-to-run spread (standard deviation ₱0.66/L, against ₱0.72–0.81/L for the reductions): the ensemble's value is *agreement*, not accuracy. Halving to two regions returns the same estimate in 46% less wall-clock (128 s vs 236 s per run) at the cost of a wider spread — a genuine speed–stability trade rather than a free lunch — while shortening completions is strictly dominated, saving no time and biasing the spread upward as starved agents overshoot and are clamped back.

The ablation also corrected itself, which is the point. A first pass at three repeats showed the two-region configuration as *tighter* than the full swarm; at eight repeats that ordering reversed, exposing the three-run spread as sampling noise. Reporting the reversal, rather than the flattering first number, is the same discipline the audit applies to its own results (§6.3). A final incidental observation validates the anchoring layer (§6.6) at scale: across all thirty-two runs the master estimate repeatedly settles at the physical anchor (₱2.21/L) or its clamp bound (₱4.21/L), so the weak swarm is visibly and frequently rescued by reconciliation rather than producing usable magnitudes on its own.

### 6.8 Limitations
Monthly resolution; an RBOB fuel proxy with disclosed bias (r = 0.91, −₱5.88/L); modest samples (n = 72–203) — which bound the efficiency nulls to a minimum detectable skill of ~25% (§5.1), so they rule out large edges, not small ones; conformal coverage that is approximate at small n (and here conservative); and a CPI series via IMF/DBnomics rather than the PSA microdata. The application's LLM/agent "swarm" is an interface and explanation layer, not a validated predictor — its agent-agreement numbers are labelled as such, distinct from the calibrated intervals, and its ensemble size is justified by lower verdict variance (§6.7) rather than a better estimate. The anchoring layer (§6.6) is bounded in the same spirit: its fuel anchor is a contemporaneous pass-through model, not a forecast; its electricity and food anchors are magnitude guards that do not predict at monthly resolution; and its calibration coefficient is fit to a single 2017–2025 window and may drift.

---

## 7. Conclusion and Future Work

This thesis replaces the assertion "AI predicts the economy" with a measured map of what is and is not predictable in Philippine macro data, and the map is uniformly negative. One-month forecasts of fuel, FX, and year-on-year inflation are informationally efficient, reproducing Meese–Rogoff and Atkeson–Ohanian for an emerging market. No month-on-month nowcast target — headline inflation, food, electricity, or transport — shows a detectable edge over the strongest naive baseline once that pool includes the historical mean. Search-interest data adds nothing either (§5.9). Subject to the power bound of §5.1, the answer to "can this be predicted?" is no, almost everywhere.

The more durable contribution is methodological, and it is a negative result about method rather than about the Philippines. An earlier version of this audit reported four significant positives, one of which — the electricity within-month driver edge — cleared every guard the protocol had: significance, sub-sample stability, a trailing-preliminary-months robustness re-test, a Bonferroni family-wise correction, and a mechanism that was institutionally accurate. It was still an artifact of a single specification choice: a baseline pool that omitted the constant mean, on targets that are mean-reverting rates. The lesson is that robustness checks compound only when they are *independent*, and that a stack of guards which all take the baseline as given offers the reassurance of five checks and the coverage of one. Baseline specification is not a preliminary to the analysis; on a mean-reverting target it *is* the analysis.

**Future work.** (i) Re-examine whether any within-month driver channel survives at higher frequency, where the mean is a weaker competitor because there is less time to revert — the electricity generation-charge pass-through remains institutionally real even though it does not beat a constant monthly, and weekly MOPS data would test it properly. (ii) Audit whether the published nowcasting literature on mean-reverting rate targets shares this baseline omission; the artifact reproduced here is not specific to Philippine data and the check is cheap. (iii) Extend the audit to the remaining CPI components (housing, water, services) through the same PSA OpenSTAT source. (iv) Re-evaluate the Transport-CPI series once the 2026 prints are finalised. (v) *Corrected rather than deferred*: §6.4–§6.6 previously described the application's electricity anchor, and the headline and food MoM results generally, in language that predated the baseline correction and had not been brought into line with it; both are now stated consistently with §5.3–§5.7, and the electricity anchor is described only as the magnitude guard §6.6 shows it to be.

---

## Data and Code Availability

**Data.** All series are third-party and publicly available: World Bank Global Fuel Prices Database (RON95 pump prices, ODbL); IMF International Financial Statistics via DBnomics (CPI); Philippine Statistics Authority OpenSTAT (CPI by commodity group); the Department of Energy (pump-price bulletins); and Yahoo Finance market series (Brent, USD/PHP, RBOB, natural gas). Every series used is committed to the repository as a frozen CSV, so the benchmark reads only static files and reproduces offline; the raw sources are refreshable via `refresh_data.py`.

**Code.** The full analysis code and the frozen result artifacts are in the repository at `github.com/sindoussss/ph-economic-pressure-ai` (MIT license). The entire audit — every table, significance test, multiple-comparison correction, power analysis, and figure — regenerates with `python -m ph_economic_ai.benchmark.run`; the reported numbers were verified to reproduce byte-for-byte. *[final version: archive a tagged release for a citable DOI, e.g. via Zenodo.]*

**Ethics.** The study uses only aggregate, publicly published macroeconomic and market data; it involves no human subjects, no personal data, and no proprietary or licensed-restricted material beyond the terms of the public sources above.

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in Machine Learning, 16(4), 494–591.
- Aruoba, S. B., Diebold, F. X., & Scotti, C. (2009). Real-time measurement of business conditions. *Journal of Business & Economic Statistics*, 27(4), 417–427.
- Atkeson, A., & Ohanian, L. E. (2001). Are Phillips curves useful for forecasting inflation? *Federal Reserve Bank of Minneapolis Quarterly Review*, 25(1), 2–11.
- Bacon, R. W. (1991). Rockets and feathers: The asymmetric speed of adjustment of UK retail gasoline prices to cost changes. *Energy Economics*, 13(3), 211–218.
- Bańbura, M., Giannone, D., Modugno, M., & Reichlin, L. (2013). Now-casting and the real-time data flow. In *Handbook of Economic Forecasting* (Vol. 2A, pp. 195–237). Elsevier.
- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts. *Operational Research Quarterly*, 20(4), 451–468.
- Bok, B., Caratelli, D., Giannone, D., Sbordone, A. M., & Tambalotti, A. (2018). Macroeconomic nowcasting and forecasting with big data. *Annual Review of Economics*, 10, 615–643.
- Borenstein, S., Cameron, A. C., & Gilbert, R. (1997). Do gasoline prices respond asymmetrically to crude oil price changes? *Quarterly Journal of Economics*, 112(1), 305–339.
- Cheung, Y.-W., Chinn, M. D., & Pascual, A. G. (2005). Empirical exchange rate models of the nineties: Are any fit to survive? *Journal of International Money and Finance*, 24(7), 1150–1175.
- Beck, N., Dovern, J., & Vogl, S. (2025). Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability. *Applied Intelligence*, 55, 395.
- Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. *Data Mining and Knowledge Discovery*, 37, 788–832.
- McCracken, M. W., & Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business & Economic Statistics*, 34(4), 574–589.
- Lunsford, K. G., & West, K. D. (2026). Random walk forecasts of stationary processes have low bias. *Journal of Business & Economic Statistics* (forthcoming; NBER WP 34112).
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
- Diebold, F. X. (2015). Comparing predictive accuracy, twenty years later: A personal perspective on the use and abuse of Diebold–Mariano tests. *Journal of Business & Economic Statistics*, 33(1), 1–9.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Evans, M. D. D. (2005). Where are we now? Real-time estimates of the macroeconomy. *International Journal of Central Banking*, 1(2), 127–175.
- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *Journal of Finance*, 25(2), 383–417.
- Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., & Neubig, G. (2023). PAL: Program-aided Language Models. *Proceedings of the 40th International Conference on Machine Learning (ICML)*, PMLR 202, 10764–10799.
- Faust, J., & Wright, J. H. (2013). Forecasting inflation. In *Handbook of Economic Forecasting* (Vol. 2A, Ch. 1, pp. 3–56). Elsevier.
- Giannone, D., Reichlin, L., & Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. *Journal of Monetary Economics*, 55(4), 665–676.
- Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679–688.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. *Journal of the American Statistical Association*, 113(523), 1094–1111.
- Meese, R. A., & Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? *Journal of International Economics*, 14(1–2), 3–24.
- Republic of the Philippines (1998). *Downstream Oil Industry Deregulation Act of 1998* (Republic Act No. 8479).
- Rossi, B. (2013). Exchange rate predictability. *Journal of Economic Literature*, 51(4), 1063–1119.
- Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. *Journal of Machine Learning Research*, 9, 371–421.
- Stock, J. H., & Watson, M. W. (2007). Why has U.S. inflation become harder to forecast? *Journal of Money, Credit and Banking*, 39(s1), 3–33.
- Timmermann, A. (2006). Forecast combinations. In *Handbook of Economic Forecasting* (Vol. 1, pp. 135–196). Elsevier.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- West, K. D. (1996). Asymptotic inference about predictive ability. *Econometrica*, 64(5), 1067–1084.

**Primary institutional and data sources**
- Bangko Sentral ng Pilipinas. *Monetary Policy Decisions* (policy-rate history, cumulative 2.00% → 6.50%, May 2022 – October 2023) and the *Inflation Target* (2–4% ± 1 ppt band). Manila: BSP. *[final citation: confirm the specific press-release dates/URLs.]*
- Philippine Statistics Authority. *Consumer Price Index (2018 = 100), January 2023* (headline inflation 8.7% year-on-year) and monthly CPI by commodity group. Quezon City: PSA / OpenSTAT. *[final citation: confirm the specific release reference number.]*
- World Bank. *Global Fuel Prices Database* (Open Database License, ODbL). Washington, DC.
- International Monetary Fund. *International Financial Statistics*, accessed via DBnomics.
- Republic of the Philippines, Department of Energy. *Oil Price / Pump Price Bulletins.*
- Republic of the Philippines. *Republic Act No. 8479: Downstream Oil Industry Deregulation Act of 1998* (signed 10 February 1998). Manila: Official Gazette. `officialgazette.gov.ph/1998/02/10/republic-act-no-8479/`
- Republic of the Philippines. *Republic Act No. 10963: Tax Reform for Acceleration and Inclusion (TRAIN) Act* (signed 19 December 2017, effective 1 January 2018). Manila: Official Gazette, 114 O.G. No. 3, 287 (15 January 2018).
- Republic of the Philippines, Department of Energy. *Pantawid Pasada Program*, implemented under Joint Memorandum Circular No. 001, s. 2018 (fuel-subsidy card for public-utility-vehicle operators). Taguig: DOE. `doe.gov.ph/pantawid-pasada`
- Republic of the Philippines, Department of Energy, Oil Industry Management Bureau. *Weekly pump-price adjustment mechanism*, in which retailers set advisories from the trailing week's Mean of Platts Singapore (MOPS) assessments plus freight, taxes, and margin, filed with and consolidated by DOE ahead of the Tuesday industry-norm adjustment. Taguig: DOE.
- Republic of the Philippines, Energy Regulatory Commission. *Rules on the Automatic Cost Adjustment and True-Up Mechanisms for Distribution Utilities*, under which a distribution utility's generation charge is recovered monthly as a pass-through of what it actually paid generators through bilateral contracts and the Wholesale Electricity Spot Market (WESM), subject to ERC verification of the computation rather than utility discretion. Pasig: ERC. `erc.gov.ph`
- Manila Electric Company (Meralco). *Breakdown of Charges*, the utility's own published accounting of a residential bill's generation, transmission, system loss, and other components. Pasig: Meralco. `meralco.com.ph/residential/billing-payment/understanding-your-bill/breakdown-charges`
- Yahoo Finance market data: Brent (`BZ=F`), USD/PHP (`PHP=X`), RBOB (`RB=F`), Henry Hub natural gas (`NG=F`).

---

## Appendices

### Appendix A — Reproducibility
- Regenerate all artifacts: `python -m ph_economic_ai.benchmark.run`.
- Refresh source data (network): `refresh_data.build_features_csv`, `build_long_features`, World Bank workbook loader.
- Committed artifacts: `accuracy_report.json`, `ablation_table.json`, `audit_table.json`, `nowcast_table.json`, `nowcast_mom_table.json`, `mom_driver_ablation_table.json`, `mom_longsample_table.json`, `transport_nowcast_table.json`, `food_nowcast_table.json`, `electricity_nowcast_table.json` (the last including the §5.7 sub-sample stability cuts), `multiple_testing.json`, `power.json`, `backtest_predictions.csv`, `figures/*.png` (including the Fig. 3 predictability map, rendered by `run` via `render_pub_figures`).
- **Baseline-specification results (§4.7, §5.10, §5.11).** All four are regenerated by `benchmark.run` and reproducible individually:
  - `python -m ph_economic_ai.benchmark.baseline_theory` → `baseline_theory.json` — the closed form S(ρ), its inversion, the simulation validation, and the per-target check.
  - `python -m ph_economic_ai.benchmark.baseline_size` → `baseline_size.json` — the size and power grids (300 replications per cell; ~6 min).
  - `python -m ph_economic_ai.benchmark.vulnerability_survey` → `vulnerability_survey.json` — the FRED-MD census. Reads the frozen `benchmark/data/fredmd_snapshot.csv`; refresh that with `python -m ph_economic_ai.tools.refresh_fredmd` (the only network step).
  - `python -m ph_economic_ai.benchmark.power` → `power.json` — minimum detectable effects for the fuel forecast and every nowcast null.
- **Superseded results retained for audit.** `corrected_predictability_map.json` (via `benchmark.corrected_audit`) re-derives the whole map under both the mean-free and mean-inclusive pools, so the numbers this draft withdraws remain inspectable rather than overwritten.
- **Sentiment keystone (§5.9).** `sentiment_nowcast.json`, regenerated by `benchmark.run`; the Google Trends input is frozen at `benchmark/data/google_trends_monthly.csv` and refreshed with `python -m ph_economic_ai.tools.refresh_social --trends` (query one term per payload — see §5.9).
- Application anchoring validation (§6.6): regenerate with `python -m ph_economic_ai.tools.anchor_backtest`; committed as `anchor_validation.json` (fuel/electricity/food regressions, calibration, robustness sweep, weak-model benefit).
- Swarm-size ablation (§6.7): regenerate with `python -m ph_economic_ai.tools.swarm_ablation --repeats 8`; committed as `swarm_ablation.json` (per-variant estimate ranges, spreads, and overlap verdicts).

### Appendix B — Full panels
All values are reproduced verbatim from the committed `accuracy_report.json` (regenerate with `python -m ph_economic_ai.benchmark.run`). DM p-values in the per-panel tables are vs the **random-walk** baseline (HLN-corrected); skill = 1 − RMSE/RMSE_baseline.

**B.0 The historical mean across every MoM panel.** Because the per-panel tables below quote skill and DM p against the random walk, and the corrected verdict is decided against the *mean* (§4.7), the mean's RMSE is collected here for every panel. The mean ignores features, so its value is identical for a full and a driver-only specification on the same target.

| Panel | n | best candidate RMSE | **mean RMSE** | random-walk RMSE | candidate vs mean |
|---|---|---|---|---|---|
| B.3 Headline MoM | 61 | ARIMA 0.3799 | **0.3961** | 0.4532 | +4.1% (p = 0.36) |
| B.4 Headline MoM (long) | 143 | — | **0.3625** | — | not significant |
| B.5 Transport driver-only | 151 | Ridge 1.5740 | **1.5312** | 2.0355 | −2.8% (worse) |
| B.7 Food full | 151 | ARIMA 0.6633 | **0.6890** | 0.7897 | +3.7% (p = 0.46) |
| B.7 Food driver-only | 151 | Ridge 0.7020 | **0.6890** | 0.7897 | −1.9% (worse) |
| B.8 Electricity driver-only | 151 | Ridge 2.3936 | **2.3515** | 3.3399 | −1.8% (p = 0.37) |

In four of the six panels the best candidate is *worse than a constant*; in the remaining two the residual edge is small and insignificant. Every `beats_best_naive` verdict in the earlier draft therefore rested on the random-walk column, and none survives the mean column. The superseded verdicts are preserved in the per-panel tables below, marked as such, so the change is auditable.

**B.1 Fuel one-month forecast — eight-method efficiency panel** (RON95, n = 72). Skill and DM p are vs random walk.

| Method | RMSE | MAE | Skill vs RW | DM p vs RW |
|---|---|---|---|---|
| random_walk | 4.0685 | 3.0762 | 0.0000 | — |
| drift | 4.1101 | 3.1114 | −0.0102 | 0.5043 |
| seasonal_naive | 11.4971 | 8.8848 | −1.8259 | 0.0001 (worse) |
| ARIMA(1,1,1) | 4.3834 | 3.2829 | −0.0774 | 0.0368 (worse) |
| ETS | 4.1828 | 3.1603 | −0.0281 | 0.2753 |
| Ridge | 4.1046 | 3.1473 | −0.0089 | 0.8813 |
| HGB | 4.0991 | 3.0029 | −0.0075 | 0.9209 |
| **mean** | **11.1871** | **9.1192** | **−1.7497** | **0.0000 (worse)** |

*No method significantly beats the random walk; the ML methods (Ridge, HGB) are statistically indistinguishable from it (DM p ≈ 0.88–0.92), while ARIMA and seasonal-naive are significantly worse. The **mean** row is the symmetry check of §4.7: on a persistent level series the historical mean is a catastrophic predictor (skill −1.75), so it never becomes the binding baseline here and the correction cannot manufacture a false negative in the forecasting results.*

**B.2 Predictability audit — one-month forecast verdicts.**

| Target | n | Best method | Best skill | Verdict |
|---|---|---|---|---|
| Fuel (RON95) | 72 | random_walk | 0.0 | efficient |
| USD/PHP | 38 | random_walk | 0.0 | efficient |
| Inflation (YoY) | 59 | random_walk | 0.0 | efficient |

**B.3 Headline MoM inflation nowcast — panel RMSE** (best naive = random_walk).

| Method | RMSE (n = 82) | RMSE (long, n = 190) |
|---|---|---|
| random_walk | 0.4532 | 0.4130 |
| drift | 0.4578 | 0.4159 |
| seasonal_naive | 0.5343 | 0.4761 |
| **ARIMA** | **0.3799** | **0.3458** |
| ETS | 0.4135 | 0.3735 |
| Ridge | 0.3980 | 0.3604 |
| HGB | 0.4574 | 0.4297 |
| **mean** | **0.3961** | **0.3625** |
| **Verdict (corrected)** | no_better_than_naive (best naive = mean) | no_better_than_naive (best naive = mean) |
| *Superseded (vs random walk)* | *beats_best_naive +16.2%, p = 0.032* | *beats_best_naive +16.3%, p = 0.001* |

**B.4 Headline MoM driver-only ablation** (own-lag dropped; candidates = Ridge, HGB).

| Method | RMSE (n = 82) | RMSE (long, n = 190) |
|---|---|---|
| random_walk | 0.4532 | 0.4130 |
| drift | 0.4578 | 0.4159 |
| seasonal_naive | 0.5343 | 0.4761 |
| Ridge (driver-only) | 0.3993 | 0.3739 |
| HGB (driver-only) | 0.4431 | 0.4272 |
| **mean** | **0.3961** | **0.3625** |
| **driver_edge** | False (best naive = mean) | False (best naive = mean) |

**B.5 Transport-CPI MoM nowcast — full sample vs robust** (best naive = seasonal_naive full / random_walk robust). Full-sample driver-only edge vanishes after dropping the 6 preliminary PSA months.

| Method | Full nowcast RMSE (n = 203) | Driver-only RMSE (n = 203) | Driver-only, robust RMSE (n = 197) |
|---|---|---|---|
| random_walk | 2.0355 | 2.0355 | 1.4159 |
| drift | 2.0435 | 2.0435 | 1.4235 |
| seasonal_naive | 1.8463 | 1.8463 | 1.6412 |
| ARIMA | 1.6200 | — | — |
| ETS | 1.6328 | — | — |
| Ridge | 1.6116 | 1.5740 | 1.3138 |
| HGB | 1.7342 | 1.7375 | 1.4131 |
| **mean** | **1.5312** | **1.5312** | **1.4159** |
| **Verdict (corrected)** | no_better_than_naive | no_better_than_naive (best naive = mean) | no_better_than_naive (driver_edge_robust = False) |
| *Superseded (vs random walk)* | *no_better_than_naive* | *beats_best_naive +14.8%, p = 0.021* | *no_better_than_naive* |

**B.6 Phase-2 gated feature ablation** (fuel forecast, n = 72). `band90` = 90% conformal half-width (₱/L). Selected variant: `passthrough_lags`.

| Variant | RMSE | MAE | Skill vs RW | 90% band (₱/L) |
|---|---|---|---|---|
| baseline | 4.6043 | 3.4402 | −0.1317 | 17.859 |
| drop_demand | 4.4139 | 3.3901 | −0.0849 | 16.826 |
| **passthrough_lags** (selected) | **4.0991** | **3.0029** | **−0.0075** | **14.457** |
| finished_gas | 4.9940 | 3.8109 | −0.2275 | 18.569 |
| structural_hybrid | 5.5500 | 3.9363 | −0.3642 | 19.692 |

*No variant beats the random walk, but `passthrough_lags` closes the gap (−0.13 → −0.007) and tightens the 90% band by ~19%.*

**B.7 Food-CPI MoM nowcast** (n = 203; best naive = random_walk). Full nowcast (own-lag + drivers) vs driver-only; driver-only edge is null at both windows.

| Method | Full nowcast RMSE | Driver-only RMSE (n = 203) | Driver-only, robust RMSE (n = 197) |
|---|---|---|---|
| random_walk | 0.7897 | 0.7897 | 0.7517 |
| drift | 0.7936 | 0.7936 | 0.7556 |
| seasonal_naive | 0.9151 | 0.9151 | 0.9114 |
| ARIMA | 0.6633 | — | — |
| ETS | 0.7285 | — | — |
| Ridge | 0.7020 | 0.7274 | 0.7029 |
| HGB | 0.7386 | 0.7878 | 0.7554 |
| **mean** | **0.6890** | **0.6890** | — |
| **Verdict (corrected)** | no_better_than_naive (best naive = mean) | no_better_than_naive | no_better_than_naive (`driver_edge_robust` = False) |
| *Superseded (vs random walk)* | *beats_best_naive ARIMA +16.0%, p = 0.0046* | *no_better_than_naive* | *no_better_than_naive* |

**B.8 Electricity-CPI MoM nowcast** (n = 203; best naive = **mean**). The driver-only edge is significant against the random walk at both windows and disappears against the mean, which is the whole of §5.7.

| Method | Full nowcast RMSE | Driver-only RMSE (n = 203) | Driver-only, robust RMSE (n = 197) |
|---|---|---|---|
| random_walk | 3.3399 | 3.3399 | 3.3957 |
| drift | 3.3746 | 3.3746 | 3.4311 |
| seasonal_naive | 3.5010 | 3.5010 | 3.4681 |
| ARIMA | 2.4925 | — | — |
| ETS | 2.5308 | — | — |
| **Ridge** | **2.4520** | **2.3936** | **2.4317** |
| HGB | 2.9348 | 2.8522 | 2.8944 |
| **mean** | **2.3515** | **2.3515** | — |
| **Verdict (corrected)** | no_better_than_naive (best naive = mean) | no_better_than_naive (Ridge **-1.8%** vs mean, p = 0.37) | no_better_than_naive (`driver_edge_robust` = False) |
| *Superseded (vs random walk)* | *beats_best_naive Ridge +26.6%, p = 0.0005* | *beats_best_naive +28.3%, p = 0.0011* | *beats_best_naive +28.4%, p = 0.0012* |

*Sub-sample stability (driver-only): ≤2023-12 +26.3% (p = 0.006); first half +29.9% (p = 0.020); second half +28.7% (p = 0.035) — the edge is not period-specific.*

### Appendix C — Calibration and pass-through

**C.1 Fuel one-month forecast — split-conformal calibration** (52-forecast backtest, split 26 calibration / 26 validation). `qhat` = interval half-width (₱/L) taken from the calibration half; measured = empirical coverage of that interval on the disjoint validation half.

| Nominal | qhat (₱/L) | Measured coverage |
|---|---|---|
| 0.50 | 2.5569 | 0.5769 |
| 0.80 | 5.8805 | 0.8846 |
| 0.90 | 10.4194 | 1.0000 |
| 0.95 | 11.8630 | 1.0000 |

*The 90% and 95% intervals over-cover (conservative) at this sample size; the 50%/80% are close to nominal. Coverage is reported, not tuned.*

**C.2 Pass-through regression** (Δ RON95 on contemporaneous + lagged Δ driver, n = 97; HAC errors).

| Quantity | Value |
|---|---|
| α (intercept) | 0.1328 |
| β₀ (contemporaneous) | 0.3132 |
| β₁ (one-month lag) | 0.2438 |
| **β_total = β₀ + β₁** | **0.5570** |
| R² | 0.3323 |
| driver Δ autocorrelation (ACF₁) | 0.1577 |

*Partial, lagged pass-through (β ≈ 0.56) of a near-random-walk driver (ACF₁ ≈ 0.16) — the mechanism behind the fuel series' efficiency.*

### Appendix D — Software architecture
The `benchmark/` package: `backtest.py` (causal walk-forward), `forecasters.py` (panel), `significance.py` (DM/HLN), `conformal.py` (intervals + calibration), `efficiency.py` (panel + pass-through), `nowcast.py` (YoY/MoM + ablation), `longsample.py` (robustness), `audit.py` (`Target` registry + verdicts), `report.py` / `figures.py` / `run.py` (assembly). The PyQt application renders the frozen report; it does not recompute it.
