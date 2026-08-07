# Strata — Defense Talking Points

Your job at the defense is to be the **most honest person in the room** about what Strata does and doesn't do. The science is strong *because* it's blunt about limits. Lead with that, and the hard questions stop being threats.

> **Revised 2026-07-26.** An earlier version of this sheet was built around four significant positives (headline MoM +16%, food MoM +16%, electricity drivers +28%, and a rejected transport edge). **Those positives did not survive the corrected baseline pool** and have been withdrawn. Everything below reflects the current, corrected map. If you rehearsed the old version, re-read §3 and §6 carefully — the strongest material is now the *refutation*, not the positives.

---

## 1. The one-sentence thesis

> "Strata is a rigorous, reproducible audit of **what is and isn't forecastable** in Philippine fuel and inflation — and its main methodological finding is that the *choice of naive baseline*, not the model, silently decided the verdicts."

The **contribution is the benchmark and the baseline result**, not the app. Say this first, always.

---

## 2. The frame to memorize: validated vs exploratory

Draw this line before anyone else can blur it.

| | **Validated** | **Exploratory** |
|---|---|---|
| What | The benchmark (`ph_economic_ai/benchmark`) | The swarm app, knowledge-graph/evidence sim, agent-agreement %, trust/evolution loop |
| Method | Strictly-causal walk-forward backtest, Diebold–Mariano tests vs the *strongest* naive baseline, split-conformal intervals | 20 LLM agents debating, grounded in real retrieved evidence |
| Needs an LLM? | **No** — fully reproducible with `python -m ph_economic_ai.benchmark.run` | Yes (local Ollama) |
| Claim | A measured, significance-tested result | An interface / explanation layer — **not** a validated predictor |

The boundary is enforced in code, not just asserted: `tests/test_benchmark_isolation.py` fails if `benchmark/` ever imports the app, PyQt, or any LLM provider — including via a relative import.

---

## 3. The findings — the corrected map

All skill = **% RMSE improvement over the strongest naive baseline**, walk-forward, DM-tested. The naive pool is {random walk, drift, seasonal naive, **historical mean**}.

> **ARTIFACT-DIVERGENCE (2026-08-07). Be ready for this question.** The fuel row below says "efficient." The manuscript's own divergence notice (top of `2026-06-10-thesis-manuscript.md`) flags that fuel's 1-month panel now shows a +12.1% Ridge edge over the random walk (`predictable`, DM p = 0.0337, §5.2.1) — exploratory, not confirmed: it fails the audit family's Bonferroni threshold and was never predeclared (`CLM-FUEL-EXPLORATORY-001`). Whether to promote it is an authorial decision the manuscript deliberately leaves open, not resolved here. If asked "is the map really uniform, then?": yes, within the confirmatory family reported below; the one exploratory exception is disclosed, not hidden, and is exactly the kind of finding the audit's own discipline (§6.3) says to report and not promote. Checked by `python -m ph_economic_ai.benchmark.manuscript_check`, which now covers this file too.

| Target | Setup | Verdict |
|---|---|---|
| RON95 fuel | 1-month forecast | **Efficient** — the deployed model does not beat a random walk (skill −7.3% vs RW) |
| USD/PHP · YoY inflation | 1-month forecast | **Efficient** — no method beats a random walk (skill ≈ 0% vs RW) |
| MoM inflation (headline) | nowcast | **No better than naive** — ARIMA +4.1% vs the mean, p = 0.36 |
| MoM inflation (food) | nowcast | **No better than naive** — ARIMA +3.7% vs the mean, p = 0.46 |
| Electricity-CPI | nowcast, driver-only | **No better than naive** — Ridge **−1.8%** vs the mean, p = 0.37 |
| Transport-CPI | nowcast, driver-only | **Rejected twice** — preliminary-data artifact *and* fails vs the mean |
| Food-CPI | nowcast, driver-only | **Clean null** — Ridge −1.9% vs the mean |
| Google Trends search interest | nowcast | **No better than naive** — adds nothing on either target |

**The map is uniformly negative within the confirmatory family, and that is the finding.**

**The money line:** *"An earlier version of this audit reported four significant positives. The strongest — a +28.3% electricity driver edge, DM p = 0.0011 — passed every guard I had: significance, sub-sample stability, a trailing-preliminary-months robustness re-test, a Bonferroni correction, and a mechanism that is institutionally accurate. It was still an artifact. The model was worse than predicting a constant, and the entire edge was the gap between the random walk and the mean on a mean-reverting target. I found it, I published the refutation, and I derived the condition under which it happens."*

**The closed form (know this cold).** For a stationary target with lag-1 autocorrelation ρ, a forecaster carrying *no information beyond the unconditional mean* scores

> **S(ρ) = 1 − [2(1 − ρ)]^(−1/2)** over the random walk — positive exactly when **ρ < ½**, and ≈ **+29% at ρ = 0**.

Inverted, it's a diagnostic: the +28.3% electricity edge implies ρ ≈ 0.027; electricity MoM measures **ρ = +0.002**. The finding is fully explained by the target's own autocorrelation, with no driver involved. The expression matches simulation to ≤ 0.011 and all five real targets to ≤ 0.022 (`baseline_theory.json`).

**Why every guard passed** (this is the part examiners find interesting): S(ρ) is a property of the *target*, not the estimation. So it doesn't decay out of sample, doesn't vary across sub-samples, survives a vintage re-test, and is untouched by multiple-comparison correction. **Robustness checks compound only if they're independent** — five checks that all take the baseline as given give the reassurance of five and the coverage of one.

**The size result — your strongest single number** (`baseline_size.json`). On simulated data with *nothing in it*, the mean-free protocol declares a significant edge **99.7% of the time** at ρ = 0, n = 151. Nominal α is 5%. With the mean in the pool: **0.0%**, and it still detects a genuine driver 80.7% of the time. Two follow-ups worth having ready:
- **It dies above ρ = ½** — exactly the crossover the algebra predicts. Two independent derivations, same threshold.
- **More data makes it WORSE** (43.3% → 99.7% as n goes 61 → 151). This is why my long-sample "robustness check" tightened the artifact from p = 0.032 to p = 0.001 and I read it as the finding strengthening. The standard robustness move actively amplifies this error.

**"Isn't this just a quirk of your data?" — the FRED-MD census** (`vulnerability_survey.json`, §5.9a). No. Applying the same criterion to the 126 series of FRED-MD — the standard US macro panel — after its own *recommended* stationarity transform:
- **80.2% sit below ρ = ½** (vulnerable); 57.9% below ρ = 0.2; 44% below ρ = 0.05 where the false-positive rate is ~100%.
- Median ρ = **+0.107**. Median spurious skill an uninformative model would post: **+29.3%**.
- **Every differenced series (19/19) and 92% of growth rates are vulnerable. No series left in levels is.**

**The one-sentence version to have ready:** *"Differencing removes exactly the persistence that makes a random walk a valid benchmark — so the standard stationarity transform silently moves your target into the regime where the random-walk baseline is invalid. In FRED-MD that's four out of five series."* Note carefully: this measures the **target space**, not anyone's published findings, and I don't claim any specific paper is wrong.

---

## 4. "Does it learn?" — the honest three-layer answer

Memorize these three layers; answer in exactly this order.

1. **Within a run — yes.** The swarm debates over multiple rounds; each agent sees prior rounds' estimates and revises. **But it resets every run** (history starts empty).
2. **Across runs — only on real outcomes.** A background checker grades past forecasts against the **real DOE pump price** (~5 days later); trust scores update; the swarm then *evolves* (benches low-trust agents, adjusts model tier/prompt) after a cold-start threshold. **Same-day reruns change nothing** — there's no new outcome yet.
3. **The models themselves — never.** The LLMs are frozen. Fine-tuning (`train_sft.py`) is future work. It does **not** train on your runs.

**One-liner:** *"It adapts its agent selection as real pump-price outcomes arrive over days — honest in-context/selection adaptation, not model training. It does not get smarter every click, and I'd never claim it does."*

---

## 5. The app, made honest: anchoring & the size ablation

Two follow-up contributions on the exploratory side. Both are *honesty stories*. (It runs **offline on an 8GB GPU** — local models only.)

**The magnitude problem, and the fix.** Small local models reason about *direction* fine but botch *magnitude* — a 7B judge said a +6.8% oil shock moves pump prices **+₱12.93/L** when the real pass-through is **~₱2.7/L**. The fix isn't a bigger model — it's **not asking the model to do the arithmetic at all**. The oil→pump pass-through is accounting, so it's computed deterministically (a physics "anchor") and used three ways: a **prior** in the prompt, a **leash** clamping a hallucinated estimate back toward physics, and a **fallback** when the model produces nothing. Program-aided reasoning (Gao et al. 2023, PAL) applied to macro.

**Regressed against real data — and honestly bounded** (`anchor_validation.json`):
- **Fuel anchor → significantly tracks pass-through.** Correlation **0.60** with actual monthly pump moves over 78 months (**p < 0.001**, CI [0.44, 0.73]), 74% directional. Lower MAE than a no-change baseline (₱2.21 vs ₱2.64) but only *marginally* — DM **p = 0.065**, so say "co-moves significantly; beats naive only marginally," never a flat "beats naive."
- **Electricity & food anchors → magnitude guards, NOT predictors.** They get the *scale* right (~1.0×) but do not forecast the monthly move (electricity corr ~0.03; food persistence ≈ oil ≈ a plain mean).

**A detail worth volunteering:** the anchor backtest flagged that *"a plain mean is competitive"* for food **before** the benchmark's baseline pool did. Your own validation surfaced the problem; the audit was slower to hear it.

**The line to say:** *"One anchor co-moves with reality, two only guard magnitude — and I report which is which. The anchor's job is to stop a weak model saying ₱33/kWh, not to forecast."*

**"Why 20 agents?" — the ablation** (`swarm_ablation.json`, n=8). All three cheaper configs reach the **same verdict** (means within ₱3.1–3.6/L, overlapping). The full swarm buys **lower run-to-run variance (σ 0.66 vs 0.72–0.81), not a better number** — *agreement, not accuracy*. **And the ablation corrected itself**: a 3-repeat pass suggested two-regions was tighter; 8 repeats showed that was noise.

---

## 6. Q&A bank — likely examiner questions

**Q: So your AI predicts fuel prices?**
No — and that's a *result*, not a failure. Nothing in the map shows a detectable edge over a properly-specified naive baseline. The value is knowing precisely what isn't predictable, so effort doesn't go where there's no signal.

**Q: What's your actual contribution, then?**
Three things. A reproducible predictability audit for a data-poor economy. A **closed form for baseline-induced spurious skill**, S(ρ) = 1 − [2(1−ρ)]^(−1/2), validated on simulated and real data and usable as a diagnostic on anyone's published edge. And a documented case showing that five robustness checks can all pass on an artifact when none of them interrogates the baseline.

**Q: Isn't "the mean should be in the baseline pool" already known?**
Yes, and I say so in §2.4a and §4.7. It's implicit in Hyndman & Koehler (2006), it's the substance of the Atkeson–Ohanian benchmark argument, and the textbook comparison is standard. **I don't claim the effect is new — I claim the magnitude.** The literature (Hewamalage et al. 2023; Beck, Dovern & Vogl 2025) warns the naive forecast is *too hard* to beat on random-walk-like series. Mine is the mirror case: on mean-reverting *rates* it's too easy. What I add is how large, exactly when (ρ < ½), and how to test a published result for it.

**Q: Your headline finding evaporated. Isn't the thesis now empty?**
It's a *null* thesis, which is a different thing. It reproduces Meese–Rogoff and Atkeson–Ohanian for a new market, and it adds a methodological result that doesn't depend on Philippine data at all. A referee can apply the diagnostic to their own work tomorrow. I'd rather submit a correct null than an incorrect positive — and I found it myself rather than having it found for me.

**Q: With n≈50, isn't your "efficient" verdict just underpowered?**
Partly — and **every null is bounded**, so I can tell you exactly how much (`power.json`, §5.10). The MDE at 80% power: fuel forecast **24.7%** (the weakest test in the thesis), headline MoM **13.2%**, long-sample **10.1%**, food **14.9%**, transport drivers **12.5%**, and electricity drivers **5.8%**. So I claim what each test supports and no more: for fuel, "no *large* edge is detectable"; for the electricity driver channel, the test genuinely had resolution to see a small edge and found the model 1.8% *worse* than a constant. The claims are not uniform in strength and I don't state them as if they were.

**Q: You withdrew the electricity finding — isn't the replacement null just underpowered?**
No, and this is the one to have ready. The electricity driver-only null is the **best-powered result in the entire map**: MDE 5.8% against an observed −1.8%. The test could comfortably have seen a small edge. It saw a model performing worse than a constant.

**Q: Electricity +28% — real or overfit?**
Neither. It was real arithmetic against the wrong baseline. Walk-forward, no look-ahead, DM p = 0.0011 — all correct, and all measured against a random walk that is structurally weak for a target with ρ ≈ 0. Ridge was in fact 1.8% *worse* than a constant. It's the worked example in §5.7.

**Q: How do you know transport was an artifact?**
Two independent ways now: the apparent edge rested on three preliminary CPI observations and vanished on revised data, *and* it fails against the mean. Only the second would have caught it if the data had been clean.

**Q: Is the agent-agreement % a probability?**
No. A stochastic LLM consensus signal that varies run to run (temperature ~0.8, no seed), explicitly **not** calibrated. The calibrated uncertainty lives in the benchmark's split-conformal intervals.

**Q: Why use LLM agents at all if the benchmark needs no LLM?**
The benchmark is the validated science. The swarm is an explanation/interface layer, clearly separated and never claimed as the predictor. If pressed on whether it's necessary: it isn't — it's an accessibility contribution, and I label it as exploratory throughout.

**Q: Did you just tune the anchors to look good?**
The opposite — all three were regressed against real PH series with the same DM test, and the limits reported. Fuel co-moves significantly (r 0.60) but beats naive only marginally (p = 0.065). Electricity and food don't forecast at all. I flag my own strongest result as merely marginal.

**Q: What baseline are you beating?**
The *strongest* naive per target — random walk, drift, seasonal naive, **and the historical mean**. That last one is the whole point: omitting it is what produced the earlier false positives.

**Q: How do I know you didn't p-hack?**
One pre-committed protocol, DM tests, and the strongest evidence available: **I published the destruction of my own headline result**, with the superseded map retained in `corrected_predictability_map.json` so you can audit the change rather than take it on trust.

**Q: Reproducibility?**
`pip install -r requirements.txt && python -m ph_economic_ai.benchmark.run` — no LLM, no GPU, committed data + artifacts. Every number re-derives.

**Q: Limitations?**
Modest sample sizes and correspondingly limited power; PH-specific data; CPI preliminary-data revisions; proxy series (Henry Hub for PH generation fuel, RBOB for finished product); and the app layer is exploratory, not validated.

**Q: Practical value?**
Knowing these series aren't monthly-predictable saves wasted forecasting effort — and the diagnostic lets anyone check whether a reported edge is real before acting on it.

**Q: Future work?**
A survey applying the S(ρ) diagnostic to published nowcasting results on mean-reverting targets; higher-frequency tests where the mean is a weaker competitor; extending the audit to the remaining CPI components.

---

## 7. Phrases to use / avoid

**Use:**
- "what is and isn't forecastable"
- "no detectable edge over a properly-specified naive baseline"
- "the choice of baseline decided the verdicts, not the model"
- "robustness checks compound only if they're independent"
- "I published the refutation of my own headline result"
- "validated benchmark vs exploratory app"
- "not a calibrated probability"
- "program-aided — the model doesn't do the arithmetic"
- "the swarm buys agreement, not accuracy"

**Avoid (overclaim traps):**
- "the AI predicts fuel prices" → it doesn't, by your own finding
- "MoM inflation / electricity is predictable" → **withdrawn**; both are nulls
- "a confirmed true positive" → there is no longer one
- "it learns / gets smarter every run" → only across days, on real outcomes; model is frozen
- "87% confidence" / "agreement = probability" → not calibrated
- "I discovered that baselines matter" → known; claim the *magnitude*, not the effect
- "proven efficient" → say "no detectable edge at this power"

---

## 8. If you remember nothing else

1. The contribution is the **audit plus the baseline result** — reproducible, and the map is uniformly null.
2. **S(ρ) = 1 − [2(1−ρ)]^(−1/2), positive iff ρ < ½, ≈ +29% at ρ = 0.** Know it cold; it's the defensible core.
3. You **refuted your own flagship finding** and published the before/after. That's the integrity story — lead with it, don't hide it.
4. Five guards passed on an artifact because none questioned the baseline. Independence is what makes robustness real.
5. The app is **exploratory and labeled as such** — never the predictor; anchors guard magnitude, they don't forecast.

Be the most honest person in the room. It's your strongest position.
