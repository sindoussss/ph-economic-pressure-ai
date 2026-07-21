# STRATA

**An honest predictability benchmark for Philippine fuel & inflation — plus a multi-agent simulator built on top.**

Most "AI predicts the economy" claims are never tested against a hard baseline. Strata does the opposite: it measures, with a strictly-causal walk-forward backtest and significance tests, *what is and isn't* forecastable in Philippine macro data — and is blunt about the limits.

> **Validated vs exploratory — read this first.**
> - **Validated:** the **benchmark** (`ph_economic_ai/benchmark`) — strictly-causal walk-forward backtests, Diebold–Mariano tests against the *strongest* naive baseline, and split-conformal prediction intervals. Fully reproducible **without any LLM**.
> - **Exploratory:** the **multi-agent app** (the swarm, the knowledge-graph simulation, the "agent agreement" %). It's an interface and explanation layer — *not* a validated predictor. Agent agreement varies run-to-run and is **not** a calibrated probability.

## Findings — the predictability map

![Predictability map — skill vs the naive baseline per target: electricity inflation +28% and month-on-month inflation +16% are predictable, transport's apparent edge is rejected as a data artifact, and 1-month fuel/FX/YoY inflation are informationally efficient.](docs/img/predictability_map.png)

| Target | Setup | Verdict |
|---|---|---|
| RON95 fuel · USD/PHP · YoY inflation | 1-month forecast | **efficient** — no method beats a random walk (skill ≈ −0.01 vs RW) |
| MoM inflation (headline) | nowcast, pre-release | **predictable** — ARIMA ~+16% over the best naive (DM p = 0.001, n = 143); own short-run dynamics |
| MoM inflation (food) | nowcast, pre-release | **predictable** — ARIMA ~+16% (DM p = 0.005); own dynamics |
| **Electricity-CPI** | nowcast, driver-only | **robust within-month driver edge** — Ridge +28% over best naive (DM p ≈ 0.001, n = 151) |
| Transport-CPI | nowcast, driver-only | apparent +14.8% fuel edge **rejected** as a preliminary-data artifact |
| Food-CPI | nowcast, driver-only | clean **null** on commodity drivers |

The within-month *driver* question is answered from all three sides: a rejected false positive (transport), a confirmed null (food), and a confirmed true positive (electricity — its regulated generation charge is a formulaic, observable fuel pass-through). Full write-up: [`docs/manuscript/2026-06-10-thesis-manuscript.md`](docs/manuscript/2026-06-10-thesis-manuscript.md).

## Reproduce the benchmark (no LLM required)

```bash
pip install -r requirements.txt
python -m ph_economic_ai.benchmark.run
```

This runs the full walk-forward audit and writes artifacts to `ph_economic_ai/benchmark/artifacts/` (`accuracy_report.json`, tables, figures). No API key, no GPU, no Qt — anyone can verify the numbers above. The boundary is enforced by `tests/test_benchmark_isolation.py`, which fails if `benchmark/` ever grows a dependency on the app.

## Run the app (interactive simulator)

The app adds a 20-agent swarm + a MiroFish-style knowledge-graph simulation on top of the benchmark. It calls a **hosted LLM over plain HTTP** — no local model, no GPU, no vendor SDK. Both supported providers have a free tier:

```bash
pip install -r requirements-app.txt

# pick one (free, no credit card):
export GROQ_API_KEY=...      # https://console.groq.com    — fastest
export GEMINI_API_KEY=...    # https://aistudio.google.com — chat + embeddings

python -m ph_economic_ai.main
```

On Windows PowerShell use `$env:GROQ_API_KEY = '...'` instead of `export`.

The app pulls live data (Brent, USD/PHP, PSA/DOE feeds), runs the swarm, and renders the simulation as a knowledge graph of what the agents actually retrieved and claimed — every node traces back to a real source.

### LLM configuration

Call sites request a **tier**, not a model, so switching providers is a config change rather than a code change:

| Variable | Default | Purpose |
|---|---|---|
| `STRATA_LLM_PROVIDER` | auto-detect from keys | `groq` or `gemini` |
| `STRATA_LLM_FAST_MODEL` | `llama-3.1-8b-instant` | the 20 bulk swarm agents |
| `STRATA_LLM_DEEP_MODEL` | `llama-3.3-70b-versatile` | judges and synthesis |
| `STRATA_LLM_EMBED_MODEL` | `gemini-embedding-001` | RAG embeddings (Gemini only) |
| `STRATA_LLM_RPM` | 28 (Groq) / 9 (Gemini) | client-side rate limit |

Free-tier model IDs get retired often — override the model variables rather than editing code. Embeddings are cached to `ph_economic_ai/cache/embeddings.npz` by content hash, so the corpus is embedded once and subsequent launches cost nothing. Without `GEMINI_API_KEY`, retrieval falls back to TF-IDF and the app still runs.

## Screenshots

| Workbench (report + interact) | Knowledge-graph simulation | Landing |
|---|---|---|
| ![workbench](docs/img/workbench.png) | ![simulation](docs/img/sim-graph.png) | ![landing](docs/img/landing.png) |

*(Drop PNGs into `docs/img/` to populate these.)*

## Repo layout

```
ph_economic_ai/
  benchmark/   # the validated, reproducible audit (walk-forward, DM tests, conformal)
  engine/      # swarm, debate, RAG, knowledge graph, trust store
  ui/          # PyQt6 app — landing, workbench (report+interact), simulation canvas
docs/
  manuscript/  # the thesis write-up
```

## How to cite

> Sindous (2026). *Strata: An honest predictability benchmark for Philippine fuel and inflation.* https://github.com/sindoussss/ph-economic-pressure-ai

## License

MIT — see [LICENSE](LICENSE).
