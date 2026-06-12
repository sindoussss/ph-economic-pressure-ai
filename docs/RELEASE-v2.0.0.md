# Strata v2.0.0

The "real product" release. **Same science as v1.0.0 — a far more polished, honest, and populated app.** No finding changed; the validated benchmark is byte-for-byte the same. What changed is everything around it.

## What's new since v1.0.0

**A coherent editorial design system.** A single source of design tokens (`ui/theme.py` — warm off-white surface, Georgia serif headline numbers, mono eyebrows, hairline dividers, price up = red / down = green) rolled out across the Report, setup, overview, interact, and policy screens. The Report is the reference screen.

**In-app honesty cues.** "Validated" (the benchmark) vs "exploratory" (the swarm) is now unmistakable in the UI, and the agent-agreement % carries its caveat everywhere: *agent agreement varies per run — not a calibrated probability.*

**Real multi-sector charts.** The Report shows honest per-sector forecast magnitude bars and recent-trajectory small-multiples for gas, food, and electricity, drawn from committed real data — plus a restyled 1-month gas forecast chart (recent actuals + a hero calibrated band).

**A publication-grade predictability-map figure.** One chart (`docs/img/predictability_map.png`, regenerable via `python -m ph_economic_ai.benchmark.render_pub_figures`) showing skill vs the naive baseline per target — green = predictable, gray = efficient, red = rejected-as-artifact. Embedded atop the README Findings. It shows what *isn't* forecastable too, which is the point.

**A populated, honest simulation.** The structured swarm canvas now hangs each agent's **real retrieved RAG evidence** off it as satellite nodes (~37 → ~100+ nodes) — every dot clickable to its actual source and text. Density that's *earned*, not decorative.

**An editorial completion toast.** When a run finishes, a light card slides in from the top with the **real master verdict** (estimate + confidence) and a "View report →" action — instead of silently jumping away.

**An honest learning loop (exploratory).** An accuracy-graded trust store evolves the swarm toward agents that have actually been right — but only as real DOE pump-price outcomes arrive (days/weeks later), and the underlying models are *not* trained. Kept strictly separate from the validated benchmark.

## The findings (unchanged)

- Fuel (RON95), USD/PHP, and YoY inflation are **informationally efficient** — no method beats a random walk.
- **Month-on-month inflation is nowcastable** (~+16%, own dynamics) for headline and food.
- **Electricity-CPI shows a robust within-month driver edge** (+28%, DM p ≈ 0.001) — the regulated generation charge is a formulaic fuel pass-through.
- A transport "edge" was **caught and rejected** as a preliminary-data artifact; food commodity drivers are a **clean null**.

## Validated vs exploratory

- **Validated** — the benchmark (`ph_economic_ai/benchmark`): strictly-causal walk-forward backtests, Diebold–Mariano tests against the *strongest* naive baseline, split-conformal intervals. Fully reproducible **without any LLM**.
- **Exploratory** — the swarm, the knowledge-graph/evidence simulation, the agent-agreement %, and the trust/evolution loop. An interface and explanation layer, *not* a validated predictor.

Reproduce the benchmark: `pip install -r requirements.txt && python -m ph_economic_ai.benchmark.run`.

MIT licensed.
