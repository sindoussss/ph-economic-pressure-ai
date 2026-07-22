# Strata — Defense Talking Points

Your job at the defense is to be the **most honest person in the room** about what Strata does and doesn't do. The science is strong *because* it's blunt about limits. Lead with that, and the hard questions stop being threats.

---

## 1. The one-sentence thesis

> "Strata is a rigorous, reproducible benchmark of **what is and isn't forecastable** in Philippine fuel and inflation — a strictly-causal walk-forward audit with significance tests — plus an exploratory multi-agent app that makes the predictable channels legible."

The **contribution is the benchmark (the predictability map)**, not the app. Say this first, always.

---

## 2. The frame to memorize: validated vs exploratory

Draw this line before anyone else can blur it.

| | **Validated** | **Exploratory** |
|---|---|---|
| What | The benchmark (`ph_economic_ai/benchmark`) | The swarm app, knowledge-graph/evidence sim, agent-agreement %, trust/evolution loop |
| Method | Strictly-causal walk-forward backtest, Diebold–Mariano tests vs the *strongest* naive baseline, split-conformal intervals | 20 LLM agents debating, grounded in real retrieved evidence |
| Needs an LLM? | **No** — fully reproducible with `python -m ph_economic_ai.benchmark.run` | Yes (local Ollama) |
| Claim | A measured, significance-tested result | An interface / explanation layer — **not** a validated predictor |

If a question is about prediction accuracy → it's about the **validated** side. If it's about the agents/agreement/learning → it's **exploratory**, and you say so.

---

## 3. The findings (the predictability map) — exact numbers

All skill = **% RMSE improvement over the strongest naive baseline** (random walk / drift / seasonal naive), walk-forward, DM-tested.

| Target | Setup | Verdict | Numbers |
|---|---|---|---|
| RON95 fuel · USD/PHP · YoY inflation | 1-month forecast | **Efficient** — no method beats a random walk | skill ≈ **−0.01** (−0.0075) |
| MoM inflation (headline) | nowcast | **Predictable** — own dynamics | ARIMA **+16%** (0.1627), DM **p = 0.001**, n = 143 |
| MoM inflation (food) | nowcast | **Predictable** | **+16%** (0.16), DM **p = 0.005** |
| **Electricity-CPI** | nowcast, driver-only | **Robust within-month driver edge** | Ridge **+28%** (0.2833), DM **p ≈ 0.001**, n = 151 |
| Transport-CPI | nowcast, driver-only | **Rejected** — preliminary-data artifact | apparent +14.8%, fails robustness check |
| Food-CPI | nowcast, driver-only | **Clean null** | no commodity-driver edge |

**The money line:** the within-month driver question is answered from all three sides — a **rejected false positive** (transport), a **confirmed null** (food), and a **confirmed true positive** (electricity, because the regulated generation charge is a formulaic, observable fuel pass-through). *That* is what makes the method credible: it discriminates real signal from artifact.

---

## 4. "Does it learn?" — the honest three-layer answer

Memorize these three layers; answer in exactly this order.

1. **Within a run — yes.** The swarm debates over multiple rounds; each agent sees prior rounds' estimates and revises. **But it resets every run** (history starts empty).
2. **Across runs — only on real outcomes.** A background checker grades past forecasts against the **real DOE pump price** (~5 days later); trust scores update; the swarm then *evolves* (benches low-trust agents, adjusts model tier/prompt) after a cold-start threshold. **Same-day reruns change nothing** — there's no new outcome yet.
3. **The models themselves — never.** The LLMs are frozen. Fine-tuning (`train_sft.py`) is future work. It does **not** train on your runs, and does not yet read its own track record as memory.

**One-liner:** *"It adapts its agent selection as real pump-price outcomes arrive over days — honest in-context/selection adaptation, not model training. It does not get smarter every click, and I'd never claim it does."*

Why say it this way: if you claim "learns every run," the follow-up is "show me trust change between two runs five minutes apart" — and it's flat. The honest version has no such trap.

---

## 5. The app, made honest: anchoring & the size ablation

Two follow-up contributions on the exploratory side. Both are *honesty stories* — lead with that. (It runs **offline on an 8GB GPU** now — local models only.)

**The magnitude problem, and the fix.** Small local models reason about *direction* fine but botch *magnitude* — a 7B judge said a +6.8% oil shock moves pump prices **+₱12.93/L** when the real pass-through is **~₱2.7/L**. The fix isn't a bigger model — it's **not asking the model to do the arithmetic at all**. The oil→pump pass-through is accounting, so it's computed deterministically (a physics "anchor") and used three ways: a **prior** in the prompt, a **leash** that clamps a hallucinated estimate back toward physics, and a **fallback** when the model produces nothing. This is the *program-aided* pattern (Gao et al. 2023, PAL) applied to macro.

**Conditioned on the benchmark — this is the key link.** Each sector is anchored to **the signal its own backtest found real**: fuel and electricity get a fuel pass-through; **food gets its own trailing trend, NOT oil** — because your benchmark proved food is a clean null on commodities. Anchoring food to oil would be anchoring it to what you already proved is noise. This ties the exploratory app *directly* to the validated benchmark.

**Regressed against real data — and honestly bounded** (`anchor_validation.json`):
- **Fuel anchor → validated predictor.** Correlation **0.60** with actual monthly pump moves over 78 months, beats a no-change baseline, calibrated to the empirically-fitted **0.79×** pass-through.
- **Electricity & food anchors → magnitude guards, NOT predictors.** They get the *scale* right (~1.0×) but do not forecast the monthly move (electricity corr ~0.03; food persistence ≈ oil ≈ a plain mean). Raw commodity prices can't reproduce the benchmark's electricity edge — that needs the full generation-charge formula, and I say so.

**The line to say:** *"One anchor predicts, two only guard magnitude — and I report which is which. The anchor's job is to stop a weak model hallucinating ₱33/kWh, not to forecast."*

**"Why 20 agents?" — the ablation** (`swarm_ablation.json`, n=8). All three cheaper configs reach the **same verdict** (means within ₱3.1–3.6/L, all overlapping). The full swarm's value is **lower run-to-run variance (σ 0.66 vs 0.72–0.81), not a better number** — it buys *agreement, not accuracy*. Two regions gives the same answer in half the time. **And the ablation corrected itself**: a 3-repeat pass looked like two-regions was tighter; 8 repeats showed that was noise — the same reject-your-own-flattering-result discipline as the transport artifact.

---

## 6. Q&A bank — likely examiner questions

**Q: So your AI predicts fuel prices?**
No — and that's a *result*, not a failure. Fuel (RON95) is informationally efficient: no method beats a random walk. The value is knowing precisely what is and isn't predictable, so effort goes where there's signal.

**Q: What's your actual contribution, then?**
A significance-tested predictability map of PH fuel/inflation that cleanly separates predictable channels (MoM inflation +16%, electricity drivers +28%) from efficient ones (fuel, FX, YoY) — and is honest enough to reject a false positive (transport).

**Q: Is the agent-agreement % a probability?**
No. It's a stochastic LLM consensus signal that varies run to run (temperature ~0.8, no seed). It is explicitly **not** calibrated. The calibrated uncertainty lives in the benchmark's **split-conformal intervals**.

**Q: Why use LLM agents at all if the benchmark needs no LLM?**
The benchmark is the validated science. The swarm is an explanation/interface layer that makes the drivers legible — every node grounded in real retrieved evidence — and it's clearly separated, never claimed as the predictor.

**Q: Your local models are small — how are their numbers trustworthy?**
They're not asked to do the hard part. Small models get *direction* right but *magnitude* wrong, so magnitude is computed deterministically (the pass-through anchor) and the LLM only supplies direction and qualitative judgment. Program-aided reasoning: LLM for structure, code for the arithmetic. It's how the app stays coherent on an 8GB GPU.

**Q: Did you just tune the anchors to look good?**
The opposite — I regressed all three against real PH series and *reported the failures*. Fuel predicts (correlation 0.60 over 78 months, beats a no-change baseline). Electricity and food anchors do **not** forecast — they only guard magnitude. Two of three are honest negatives; I didn't hide them.

**Q: Isn't the electricity anchor a cheat if it doesn't predict?**
It's never claimed as a predictor. Its job is *scale* — stopping a weak model saying ₱33/kWh. The benchmark's electricity *prediction* edge (+28%) needs the full generation-charge formula; raw oil can't reproduce it, and the write-up says so plainly (§6.6, §6.7 of the manuscript).

**Q: Why 20 agents if a smaller swarm gives the same answer?**
It buys *stability, not accuracy* — lower run-to-run variance (σ 0.66 vs 0.72–0.81). And I measured that with an 8-repeat ablation rather than asserting it; two regions is a valid speed tradeoff that reaches the same verdict.

**Q: You rebuilt the app since the benchmark — is it still the same thesis?**
The rebuild is conditioned *on* the benchmark: each anchor uses exactly what the audit found forecastable for that series (fuel/electricity pass-through; food own-trend, not commodities). The app and the audit now tell one story instead of two.

**Q: Electricity +28% — real or overfit?**
Walk-forward (no look-ahead), DM p ≈ 0.001, n = 151, and mechanistically grounded (the ERC generation charge is a formulaic fuel pass-through). And we rejected a similar-looking transport edge as an artifact — so the method discriminates rather than rationalizes.

**Q: How do you know transport was an artifact?**
The apparent edge appeared on preliminary CPI data and failed the robustness check on revised data — so we report it as rejected, not as a finding.

**Q: What baseline are you beating?**
The *strongest* naive per target (random walk / drift / seasonal naive), not a strawman. Skill is % RMSE improvement over that.

**Q: Why conformal intervals?**
Distribution-free, finite-sample coverage — honest uncertainty without assuming Gaussian errors.

**Q: How do I know you didn't p-hack?**
One pre-committed protocol, DM significance tests, and we report the nulls and the *rejected* positive — not just the wins. The figure shows the efficient/rejected bars too.

**Q: Reproducibility?**
`pip install -r requirements.txt && python -m ph_economic_ai.benchmark.run` — no LLM, no GPU, committed data + artifacts. Anyone can re-derive every number.

**Q: Limitations?**
Modest sample sizes; PH-specific; CPI preliminary-data revisions (which is exactly why we reject fragile edges); and the app layer is exploratory, not validated.

**Q: Why does the app matter if the benchmark is the contribution?**
It turns the validated structure into something a policymaker can interrogate, with provenance — accessibility, not a new claim. Labeled exploratory throughout.

**Q: Future work?**
Fine-tuning (currently future), more targets, track-record memory *on the predictable targets only*, and faster outcome grading.

**Q: Practical value?**
Knowing fuel is unpredictable saves wasted forecasting effort; the predictable channels (MoM inflation, electricity drivers) are where nowcasting genuinely adds value.

---

## 7. Phrases to use / avoid

**Use:**
- "what is and isn't forecastable"
- "informationally efficient — no method beats a random walk"
- "validated benchmark vs exploratory app"
- "we rejected this as a data artifact"
- "skill vs the strongest naive baseline, DM-tested"
- "not a calibrated probability"
- "program-aided — the model doesn't do the arithmetic"
- "one anchor predicts, two guard magnitude — I report which"
- "the swarm buys agreement, not accuracy"
- "anchored to what the benchmark proved is real per series"

**Avoid (overclaim traps):**
- "the AI predicts fuel prices" → it doesn't, by your own finding
- "it learns / gets smarter every run" → only across days, on real outcomes; model is frozen
- "87% confidence" / "agreement = probability" → it's not calibrated
- "self-improving AI" → say "outcome-graded agent selection," scoped and honest
- "the anchoring makes it accurate" → no — it makes it *coherent*; only the fuel anchor predicts
- "electricity/food anchors are validated predictors" → they're magnitude guards; say exactly that

---

## 8. If you remember nothing else

1. The contribution is the **predictability map**, validated and reproducible.
2. You **reject false positives** — that's the integrity story (transport artifact; the anchoring's own n=3→n=8 reversal).
3. The app is **exploratory and labeled as such** — never the predictor.
4. "Learns" = within-run debate + outcome-graded evolution over days; **the model is frozen.**
5. The app is now **anchored to the benchmark** — weak local models kept physically coherent by deterministic pass-through math (program-aided); **one anchor (fuel) validated as a predictor, two as honest magnitude guards** — every anchor regressed against real data.

Be the most honest person in the room. It's your strongest position.
