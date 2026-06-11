# Repo Packaging for Public Release (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the already-public repo read as a real, finished, honest research artifact — MIT license, verified `requirements.txt`, a research-first README, and a prepared `v1.0.0` release.

**Architecture:** Pure content/packaging — no app or test-code changes. Three files at repo root (`LICENSE`, `requirements.txt`, `README.md`) + a `docs/img/` slot + release notes. Verification = deps import, README links resolve, suite stays green.

**Tech Stack:** Markdown, MIT license text, pip requirements. No code.

**Spec:** `docs/superpowers/specs/2026-06-11-repo-packaging-design.md`.

**Verified facts (use verbatim):**
- Repo: `https://github.com/sindoussss/ph-economic-pressure-ai` (public).
- Reproduce (no LLM): `python -m ph_economic_ai.benchmark.run`. App: `python -m ph_economic_ai.main` (needs Ollama).
- Manuscript: `docs/manuscript/2026-06-10-thesis-manuscript.md`.
- Real deps (confirmed imported): PyQt6, pandas, numpy, scipy, scikit-learn, matplotlib, **statsmodels** (forecasters/passthrough), requests, beautifulsoup4, ollama. Dev-only: pytest.
- Findings (from `benchmark/artifacts/accuracy_report.json`): fuel/FX/YoY → efficient (skill −0.01 vs RW); MoM headline nowcast +16% (DM p=0.001, n=143, own-dynamics); MoM food +16% (p=0.005); **electricity driver edge robust** (+28.3%, DM p=0.0011, n=151, `driver_edge_robust=True`); transport apparent +14.8% edge rejected as preliminary-data artifact; food commodity drivers = clean null.

**Conventions:** Git hygiene — commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/repo-packaging`

---

## Task 1: LICENSE + requirements.txt

**Files:** Create `LICENSE`, `requirements.txt`

- [ ] **Step 1: `LICENSE`** — standard MIT, verbatim:
```
MIT License

Copyright (c) 2026 Sindous

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: `requirements.txt`** (verified deps; unpinned + Python note):
```
# Strata — runtime dependencies (Python 3.10)
PyQt6
pandas
numpy
scipy
scikit-learn
matplotlib
statsmodels
requests
beautifulsoup4
ollama

# Dev / test only:
# pytest
```

- [ ] **Step 3: Verify every listed dep actually imports**

Run:
```bash
python -c "import PyQt6, pandas, numpy, scipy, sklearn, matplotlib, statsmodels, requests, bs4, ollama; print('all deps import OK')"
```
Expected: `all deps import OK`. If any fails, it's not installed in this env — note it (the user may need to `pip install` it), but keep it in requirements only if it is genuinely imported by the package (all 10 here are confirmed). Do NOT add a package the code doesn't import.

- [ ] **Step 4: Commit**
```bash
git add LICENSE requirements.txt
git commit -m "chore: MIT license + verified requirements.txt"
```

---

## Task 2: README.md + docs/img slot

**Files:** Create `README.md`, `docs/img/.gitkeep`

- [ ] **Step 1: Create `docs/img/.gitkeep`** (empty file) so the image dir exists for user screenshots.

- [ ] **Step 2: Create `README.md`** with exactly this content:
````markdown
# STRATA

**An honest predictability benchmark for Philippine fuel & inflation — plus a multi-agent simulator built on top.**

Most "AI predicts the economy" claims are never tested against a hard baseline. Strata does the opposite: it measures, with a strictly-causal walk-forward backtest and significance tests, *what is and isn't* forecastable in Philippine macro data — and is blunt about the limits.

> **Validated vs exploratory — read this first.**
> - **Validated:** the **benchmark** (`ph_economic_ai/benchmark`) — strictly-causal walk-forward backtests, Diebold–Mariano tests against the *strongest* naive baseline, and split-conformal prediction intervals. Fully reproducible **without any LLM**.
> - **Exploratory:** the **multi-agent app** (the swarm, the knowledge-graph simulation, the "agent agreement" %). It's an interface and explanation layer — *not* a validated predictor. Agent agreement varies run-to-run and is **not** a calibrated probability.

## Findings — the predictability map

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

This runs the full walk-forward audit and writes artifacts to `ph_economic_ai/benchmark/artifacts/` (`accuracy_report.json`, tables, figures). No Ollama, no GPU — anyone can verify the numbers above.

## Run the app (interactive simulator)

The app adds a 20-agent swarm + a MiroFish-style knowledge-graph simulation on top of the benchmark. It needs a local [Ollama](https://ollama.com) and the models the swarm uses:

```bash
# 1. install Ollama, then pull the swarm models (see ph_economic_ai/engine/swarm.py), e.g.:
ollama pull qwen2.5:3b
# 2. run the app
pip install -r requirements.txt
python -m ph_economic_ai.main
```

The app pulls live data (Brent, USD/PHP, PSA/DOE feeds), runs the swarm, and renders the simulation as a knowledge graph of what the agents actually retrieved and claimed — every node traces back to a real source.

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
````

- [ ] **Step 3: Verify links + images degrade gracefully**

Run:
```bash
python -c "import os; [print(p, os.path.exists(p)) for p in ['LICENSE','requirements.txt','docs/manuscript/2026-06-10-thesis-manuscript.md','ph_economic_ai/benchmark/run.py','docs/img']]"
```
Expected: `LICENSE`, `requirements.txt`, the manuscript, `benchmark/run.py`, and `docs/img` all `True`. (Image PNGs may be absent — that's fine; GitHub shows a broken-image icon only until the user adds them, and the table still renders.)

- [ ] **Step 4: Commit**
```bash
git add README.md docs/img/.gitkeep
git commit -m "docs: research-first README (honest findings, reproduce path, run guide)"
```

---

## Task 3: Release notes + final verification

**Files:** Create `docs/RELEASE-v1.0.0.md`

- [ ] **Step 1: Create `docs/RELEASE-v1.0.0.md`** (the GitHub release body):
```markdown
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
```

- [ ] **Step 2: Commit**
```bash
git add docs/RELEASE-v1.0.0.md
git commit -m "docs: v1.0.0 release notes"
```

- [ ] **Step 3: Full suite (unchanged, must stay green)**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (no code touched).

---

## Final verification + release (after merge)

- [ ] All four artifacts exist at root/docs: `LICENSE`, `requirements.txt`, `README.md`, `docs/RELEASE-v1.0.0.md`; suite green.
- [ ] **Tag + release (run after SP1 is merged to `master`):**
```bash
git tag -a v1.0.0 -m "Strata v1.0.0 — validated benchmark + exploratory simulator"
git push origin v1.0.0
gh release create v1.0.0 --title "Strata v1.0.0" --notes-file docs/RELEASE-v1.0.0.md
```
(The `gh release create` is the user's call — offer it; it publishes the release. The tag push is safe.)
- [ ] **User action:** drop `workbench.png`, `sim-graph.png`, `landing.png` into `docs/img/` (screenshots can't be captured headlessly).

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 LICENSE → Task 1. §3.2 requirements (verified, statsmodels confirmed) → Task 1. §3.3 README full structure (pitch, validated-vs-exploratory callout, findings table, reproduce, run, screenshots, layout, citation, license) → Task 2. §3.4 v1.0.0 release notes + tag → Task 3 + final. §4 honesty (findings copied from artifacts; app tagged exploratory) → Task 2 content. §5 verification (deps import, links resolve, suite green) → Tasks 1–3 steps.
**Placeholder scan:** none — LICENSE, requirements, README, and release notes are complete verbatim content. `docs/img` PNGs are explicitly user-supplied (not a plan placeholder) and the README degrades gracefully without them.
**Consistency:** all numbers match the verified artifact read (skill −0.01; electricity +28.3%/p=0.0011/n=151; transport rejected; food null; MoM +16%). Repo URL, entrypoints, and manuscript path are the confirmed ones. requirements lists exactly the 10 confirmed-imported third-party packages, pytest noted dev-only.
````
