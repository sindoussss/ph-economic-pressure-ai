# Reviewer's-Eye Critique — Strata / PH Predictability Audit

**Purpose.** An adversarial pre-read: where a tough reviewer at an international
journal (or a sharp defense panelist) will attack, whether each attack is fatal
or fixable, and the defense or the action to take *before* submission. Ordered
by severity. Be your own harshest reviewer here so the real ones have less to say.

The paper's genuine strength is honesty-as-method — but honesty is a *posture*,
not a *shield*. Several of these need an actual analysis or edit, not just a
well-phrased limitation.

---

## Lead with these (your strongest defenses)

State them early and unprompted; they neutralize whole classes of attack.

1. **Byte-identical reproducibility.** Re-running `benchmark.run` regenerates
   every number (verified; only a timestamp differs). Almost no empirical
   economics paper can say this. It kills "did you cherry-pick / can we trust
   the numbers."
2. **Three-way discrimination.** You *reject* a false positive (transport),
   *confirm* a null (food drivers), and *confirm* a true positive (electricity)
   with one protocol. This pre-empts "your method only ever confirms."
3. **The strongest-baseline (hollow-win) guard.** You beat the *best* naive per
   target, not a strawman — the conservative choice, which biases against false
   positives.
4. **HLN-corrected Diebold–Mariano** as the arbiter, not raw RMSE ordering.

---

## MAJOR — rejection risk if unaddressed

### M1. Multiple hypothesis testing / no correction reported
**Attack.** "You run DM tests across ~8–10 targets and samples at α = 0.05.
Some 'significant' results are expected by chance. Where is your
multiple-comparison correction?" This is the single most likely rejection
reason for a quantitative paper testing many targets.

**Fatal? No — but you must add an explicit correction, and it changes the
weaker claims.** Under Bonferroni over ~10 primary tests (α = 0.005):
- Electricity (p = 0.0005) — **survives.**
- MoM long-sample (p = 0.001) — **survives.**
- Food MoM (p = 0.0046) — **borderline** (just under 0.005).
- MoM short-sample (p = 0.032) — **does NOT survive.**
- Anchoring fuel-vs-naive (p = 0.065) — already reported as marginal.

**Action (do before submission).** Add a short "Multiple comparisons" subsection
to §4 (Methodology): state the number of primary tests, apply Bonferroni **and**
a less conservative Benjamini–Hochberg FDR, and report which findings survive
each. Frame it as a strength — your two headline positives (electricity, MoM
long-sample) survive even Bonferroni; downgrade the short-sample MoM to
"suggestive, robust only in the long sample," which you already partly do.

### M2. Small samples → accepting the null is underpowered
**Attack.** "n = 38–143. Your efficiency findings (fuel, FX, YoY) *accept* the
null. With this little power, absence of evidence isn't evidence of absence —
you may simply lack power to detect a real edge."

**Fatal? No, but the framing must change.** You cannot claim a series is
"efficient"; you can claim "no predictability detectable at this power against
the strongest naive baseline." That is a weaker, correct claim.

**Action.** (a) Reword every "efficient" conclusion to "no detectable edge / not
distinguishable from the naive baseline at this sample." (b) Add a minimal
**power / minimum-detectable-effect** statement: given n and the loss variance,
what skill *could* you have detected at 80% power? If you could detect, say, a
10% skill and found ~0, the null is informative; if you could only detect 40%,
say so honestly. This single addition disarms the attack.

### M3. External validity — one country, a shock-dominated period
**Attack.** "2019–2025 is dominated by the 2022 Ukraine oil shock and COVID
recovery — a highly unusual regime. Is this a general result or an artifact of
one episode? And it's a single emerging market."

**Fatal? No.** Your long sample (2007–2026, n = 143, spanning the GFC and COVID)
is the defense — say so louder. But reframe the *contribution*: the paper is a
**reproducible method / case study**, not a universal law. The method
generalizes; the numbers are Philippine.

**Action.** In §6 and the abstract, state explicitly: "we do not claim
generality; the contribution is a transferable, reproducible protocol,
demonstrated on the Philippine case, with a robustness re-test across two
regimes." Pre-empt, don't wait for it.

### M4. The LLM/agent app is a scientific liability in an econ venue
**Attack.** "Why is a 20-agent LLM 'swarm' in a forecasting paper? This reads as
AI hype and undercuts the serious econometrics." Even labeled exploratory, an
economics reviewer may hold it against the whole paper.

**Fatal to acceptance at some venues — this is a positioning decision.** Three
options, choose per venue:
- **Econ/forecasting venue:** cut the swarm to a one-paragraph "implementation"
  mention or an appendix; keep **only** the anchoring, reframed as the
  methodological contribution (program-aided estimation conditioned on a
  predictability audit). The anchoring is defensible; the agent theatre is not.
- **Applied-ML venue:** the anchoring + swarm becomes a co-headline; safe.
- **Split into two papers:** the audit (econ venue) and the anchoring method
  (ML venue). Often the cleanest.

**Action.** Decide this with your advisor *before* formatting. It changes the
paper's spine.

### M5. The fuel efficiency result and the RBOB proxy
**Attack.** "Your fuel series uses an RBOB proxy with a −₱5.88/L bias
(r = 0.91). Is the 'fuel is efficient' null an artifact of proxy noise?"

**Fatal? No — but the text must make unmistakable which series does what.** The
*one-month forecast backtest* uses the **World Bank RON95 gold** (n = 79), not
the proxy; the proxy is only for the app's directional/relative validation
(§3.5). Reviewers *will* conflate them if you let them.

**Action.** Add one sentence at the top of §5.1: "The forecast evaluation uses
the World Bank RON95 gold series; the RBOB proxy (§3.5) is used nowhere in the
forecast tests." Remove all ambiguity.

---

## MODERATE — revise-and-resubmit territory

### R1. Conformal coverage under time-series dependence
**Attack.** "Split-conformal assumes exchangeability; your data are serially
dependent, so the coverage guarantee doesn't hold."

**Action.** You already disclose this and (claim to) measure empirical coverage
— *report the measured coverage number prominently* next to the nominal 90%, and
cite a time-series conformal variant (EnbPI, Xu & Xie 2021; or NexCP, Barber et
al. 2023) as the honest next step. Disclosed + measured = defensible.

### R2. Is the transport rejection principled or post-hoc?
**Attack.** "You found +14.8% then explained it away as a data artifact when it
was inconvenient — that's rationalizing an unwanted result."

**Action.** Show the vintage/robustness check was a **pre-specified** part of the
protocol applied to *every* target, not invented after seeing transport. State:
"the same preliminary-vs-revised check was run on all components; only transport
failed it." That turns a weakness into evidence of rigor.

### R3. The anchoring's headline is only marginal (DM p = 0.065)
**Attack.** "Your application's flagship result doesn't clear 5%."

**Action.** Already corrected in §6.6. Reframe: the anchor's *purpose* is
magnitude-guarding (validated by the ~1.0 scale ratio and the 58% error
reduction on hallucinating models), not forecasting; the *correlation* is highly
significant (p < 0.001). Never call it "beats naive" flatly.

### R4. Data-snooping across the design choices
**Attack.** "Seven-method panel, choice of features, choice of MoM vs YoY
transform — how much of the design was fixed before seeing results?"

**Action.** State the protocol was pre-committed (you have the git history to
prove ordering — use it). A dated, hash-chained commit trail is unusually strong
evidence here; cite it explicitly as a pre-registration substitute.

### R5. Nowcast information-timing airtightness
**Attack.** "Are you certain the within-month drivers are truly observable before
the CPI release, with no vintage look-ahead?"

**Action.** Add a data-availability timeline: exact publication lag of the CPI
vs. availability of oil/FX/fuel, and confirm the walk-forward uses only
information available at each nowcast date. One figure or table settles it.

---

## MINOR — polish before submission

- **Status line** still says "Working draft." Remove.
- **Bibliography** has TODOs (PSA/BSP primary releases). Complete them.
- **No figures embedded** in the manuscript body — add the predictability map
  (§5.8), the pass-through scatter (§5.2/App C), and the audit-verdicts bar.
- **Abstract** is ~350 words in one block; most venues cap 150–250. Trim.
- **Inconsistent n** across panels (52 / 79 / 143 / 151). Add one sentence
  reconciling why each analysis uses the n it does (backtest warm-up, sample
  window, component availability).
- **Data availability / ethics / licensing statement** — add a formal one
  (World Bank ODbL, IMF via DBnomics, PSA OpenSTAT, Yahoo terms).

---

## Prioritized pre-submission action list

**Must-do (analysis, not just wording):**
1. Multiple-comparison correction (M1) — Bonferroni + BH, report survivors.
2. Power / minimum-detectable-effect statement (M2).
3. Decide the swarm's fate per venue (M4).

**Must-do (wording / clarity):**
4. Reframe "efficient" → "no detectable edge at this power" throughout (M2).
5. One sentence disambiguating gold-vs-proxy in §5.1 (M5).
6. Reframe contribution as method/case-study, not universal law (M3).
7. Pre-specification statement citing the git trail (R2, R4).

**Should-do:**
8. Report measured conformal coverage + cite TS-conformal (R1).
9. Data-availability timeline (R5).
10. Bibliography, figures, abstract trim, data statement (Minor).

**Out of my hands (yours + advisor):**
- Venue choice, methodology sign-off, domain-expert economics check, the review
  cycle itself.

---

*Bottom line: none of the major objections is fatal, but three (M1 multiple
testing, M2 power, M4 swarm positioning) require real work, not just a
sentence. Do those three and the paper moves from "a strong draft" to
"defensible under a hostile review."*
