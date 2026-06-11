# ph_economic_ai — Repo Packaging for Public Release (SP1 Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** Sub-project 1 of "make Strata a real, public app." The repo (`github.com/sindoussss/ph-economic-pressure-ai`) is **already public** but has **no README, LICENSE, or requirements.txt** — so it reads as unfinished and is legally all-rights-reserved. SP3 (knowledge-graph simulation) is merged; SP2 (editorial restyle) is deferred.

---

## 1. Problem & Goal

A visitor today sees raw code: no explanation, no license (nobody may legally reuse it), no install path. The personal-folder leak risk is already plugged (`.gitignore`, done in SP3a). 

**Goal:** make the public repo read as a **real, finished, honest research artifact** — an MIT-licensed project with a research-first README, reproducible install, and a `v1.0.0` release — without overclaiming (the swarm app stays clearly labelled exploratory; the benchmark is the validated, reproducible anchor).

**Approved decisions:** MIT license (`Copyright (c) 2026 Sindous`); README leads research-artifact-first (validated benchmark + honest findings, app as the interactive demo, validated-vs-exploratory marked throughout).

---

## 2. Scope

### In scope (all top-level, plus `docs/img/`)
- `LICENSE` (MIT).
- `requirements.txt` (real runtime deps, Python 3.10).
- `README.md` (research-first, honest, with Reproduce + Run + Findings + layout + citation).
- `docs/img/` placeholder slots referenced by the README (user supplies PNGs).
- A `v1.0.0` annotated git tag + GitHub release notes (release creation is the user's action; the plan prepares the notes + tag command).

### Out of scope
- App behaviour / UI changes (SP2), any model/benchmark logic, a packaged installer or PyPI publish, capturing screenshots (the user drops PNGs into `docs/img/`).

### Non-negotiables
- **Honesty:** every README claim links to a reproducible benchmark number or is explicitly tagged *exploratory*. The swarm/agent "agent agreement" is described as exploratory and run-to-run variable, not a calibrated probability.
- **Reproducibility without Ollama:** the benchmark path (`python -m ph_economic_ai.benchmark.run`) must be documented as runnable with no LLM — the credibility anchor.

---

## 3. Components

### 3.1 `LICENSE`
Standard MIT text, `Copyright (c) 2026 Sindous`.

### 3.2 `requirements.txt`
The third-party runtime deps used by the package, verified against actual imports: `PyQt6`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `statsmodels` (used by the benchmark's ARIMA/ETS — **verify by grep before including**), `requests`, `beautifulsoup4`, `ollama`. Plus a note that `pytest` is dev-only (either a comment or a separate `requirements-dev.txt`). Unpinned (package names only) for simplicity, with a `# Python 3.10` header comment. The plan verifies each package is genuinely imported (no phantom deps) and that nothing imported is missing.

### 3.3 `README.md` (structure)
1. **Title + pitch** — `# STRATA` · *"An honest predictability benchmark for Philippine fuel & inflation — plus a multi-agent simulator built on top."* Optional shields (license, python) — skip if they add noise.
2. **Validated vs Exploratory** — a short callout box: *Validated* = the benchmark (strictly-causal walk-forward, DM tests, conformal intervals; reproducible). *Exploratory* = the swarm/agent app & its "agent agreement" (illustrative, varies per run, not calibrated).
3. **Findings** — the predictability-map table (verbatim-consistent with the manuscript):
   - Fuel (RON95) / USD-PHP / YoY inflation → **efficient** (no method beats random walk).
   - **MoM inflation (headline & food)** → **predictable** nowcast (~+16%), own-dynamics.
   - **Electricity-CPI** → **robust within-month driver edge** (~+28%, DM p≈0.001).
   - Transport-CPI → apparent edge **rejected** as a preliminary-data artifact.
   - Food-CPI drivers → clean **null**.
   - Link to `docs/manuscript/2026-06-10-thesis-manuscript.md`.
4. **Reproduce the benchmark (no LLM)** — `pip install -r requirements.txt` → `python -m ph_economic_ai.benchmark.run`; note outputs (`benchmark/artifacts/…`).
5. **Run the app** — install Ollama, pull models (e.g. `ollama pull qwen2.5:3b`, plus the others the swarm uses), `python -m ph_economic_ai.main`; mention the MiroFish knowledge-graph simulation + that it needs live data.
6. **Screenshots** — `docs/img/` slots: `workbench.png`, `sim-graph.png`, `landing.png`, embedded with markdown image links; a one-line note that they're user-supplied. README must read fine if the images are absent (text-first).
7. **Repo layout** — short tree (`ph_economic_ai/benchmark`, `engine`, `ui`, `docs/`).
8. **Citation** — a short "How to cite" (thesis title + year + repo URL).
9. **License** — "MIT — see LICENSE."

### 3.4 `v1.0.0` release
An annotated tag `v1.0.0` with a one-paragraph summary + the findings highlights; the plan provides the `git tag -a` command and the GitHub release body text. Actually creating the GitHub release is the user's click (or `gh release create`, offered).

---

## 4. Honesty / correctness safeguards
- The Findings numbers are copied from the manuscript/benchmark artifacts (already verified in earlier work) — the plan re-checks them against `benchmark/artifacts/accuracy_report.json` so the README can't drift from the validated source.
- The swarm/app section explicitly says "exploratory, not a validated predictor" and that agent agreement varies run-to-run.
- `requirements.txt` is verified against real imports (no phantom or missing deps) so a fresh `pip install -r` + `benchmark.run` actually works.

## 5. Testing / verification
- `pip install -r requirements.txt` resolves (or: every listed package imports) — the plan includes an import-check of each dep.
- A clean `python -m ph_economic_ai.benchmark.run` is the documented reproduce path (already known-good); the plan notes verifying it still runs.
- Markdown lint sanity: links resolve to real files (`docs/manuscript/…`, `LICENSE`), image slots point to `docs/img/…`.
- The existing test suite is untouched and stays green.

## 6. Deliverables (definition of done)
1. `LICENSE` (MIT, Sindous 2026).
2. `requirements.txt` (verified deps).
3. `README.md` (research-first, honest, Reproduce + Run + Findings + layout + citation; graceful without images).
4. `docs/img/` referenced (user supplies PNGs).
5. `v1.0.0` tag + release notes prepared.
6. Repo reads as a finished, honest, MIT-licensed research artifact; suite green.

## 7. Why it matters
This is the difference between "public raw code" and "a real, citable, reproducible project." It protects the work (license), lets anyone verify the honest findings without an LLM (reproduce path), and frames Strata exactly as what it is — a validated benchmark + an exploratory simulator — so it impresses without overclaiming, which is the bar the whole project is held to.
