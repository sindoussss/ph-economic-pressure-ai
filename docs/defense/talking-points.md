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

## 5. Q&A bank — likely examiner questions

**Q: So your AI predicts fuel prices?**
No — and that's a *result*, not a failure. Fuel (RON95) is informationally efficient: no method beats a random walk. The value is knowing precisely what is and isn't predictable, so effort goes where there's signal.

**Q: What's your actual contribution, then?**
A significance-tested predictability map of PH fuel/inflation that cleanly separates predictable channels (MoM inflation +16%, electricity drivers +28%) from efficient ones (fuel, FX, YoY) — and is honest enough to reject a false positive (transport).

**Q: Is the agent-agreement % a probability?**
No. It's a stochastic LLM consensus signal that varies run to run (temperature ~0.8, no seed). It is explicitly **not** calibrated. The calibrated uncertainty lives in the benchmark's **split-conformal intervals**.

**Q: Why use LLM agents at all if the benchmark needs no LLM?**
The benchmark is the validated science. The swarm is an explanation/interface layer that makes the drivers legible — every node grounded in real retrieved evidence — and it's clearly separated, never claimed as the predictor.

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

## 6. Phrases to use / avoid

**Use:**
- "what is and isn't forecastable"
- "informationally efficient — no method beats a random walk"
- "validated benchmark vs exploratory app"
- "we rejected this as a data artifact"
- "skill vs the strongest naive baseline, DM-tested"
- "not a calibrated probability"

**Avoid (overclaim traps):**
- "the AI predicts fuel prices" → it doesn't, by your own finding
- "it learns / gets smarter every run" → only across days, on real outcomes; model is frozen
- "87% confidence" / "agreement = probability" → it's not calibrated
- "self-improving AI" → say "outcome-graded agent selection," scoped and honest

---

## 7. If you remember nothing else

1. The contribution is the **predictability map**, validated and reproducible.
2. You **reject false positives** — that's the integrity story.
3. The app is **exploratory and labeled as such** — never the predictor.
4. "Learns" = within-run debate + outcome-graded evolution over days; **the model is frozen.**

Be the most honest person in the room. It's your strongest position.
