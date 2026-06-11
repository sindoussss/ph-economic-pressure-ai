# Strata v1.0.0

First public release — a validated predictability benchmark for Philippine fuel & inflation, plus an exploratory multi-agent simulator.

**Validated (reproducible, no LLM):** strictly-causal walk-forward backtest with Diebold–Mariano tests and split-conformal intervals.

**Findings:**
- Fuel (RON95), USD/PHP, and YoY inflation are informationally efficient (no method beats a random walk).
- Month-on-month inflation is nowcastable (~+16%, own dynamics) for headline and food.
- Electricity-CPI shows a robust within-month driver edge (+28%, DM p ≈ 0.001) — the regulated generation charge is a formulaic fuel pass-through.
- A transport "edge" was caught and rejected as a preliminary-data artifact; food commodity drivers are a clean null.

**Exploratory:** the swarm app, knowledge-graph simulation, and agent-agreement % — an explanation layer, not a validated predictor.

Reproduce: `pip install -r requirements.txt && python -m ph_economic_ai.benchmark.run`.

MIT licensed.
