# Economic Pressure AI Simulation Engine — Design Spec

**Date:** 2026-05-19
**Project:** ph_economic_ai
**Status:** Approved by user · ready for implementation

---

## Overview

A full redesign of `ph_economic_ai` into a 5-stage simulation engine modeled after the OASIS/Mirofish lifecycle. Users configure a Philippine economic pressure scenario, watch up to 10 AI agents debate it using live-fetched news and documents, and receive a structured report with a gas price forecast, pressure index, and 6-month trend. Every parameter is adjustable post-run for what-if analysis.

The app launches maximized (`showMaximized()`). The existing HistGradientBoostingRegressor model, indicators, and forecast charts are preserved and promoted to Stage 3's execution engine and Stage 4's output panel.

---

## 1. Architecture & Navigation

### Navigation model

A left sidebar with 5 numbered stage items. Clicking a stage navigates to it via `QStackedWidget`. Stages 3–5 are disabled (grayed out) until a simulation has been run at least once in the session.

```
Sidebar                Main area (QStackedWidget)
────────────           ──────────────────────────
  1  Graph Building  → RAG pipeline panel
  2  Environment     → Scenario + agent config
  3  Simulation      → Agent network canvas
  4  Report          → Debate summary + outputs
  5  Interact        → Adjust / ask / toggle
```

### Window

- Launch: `window.showMaximized()`
- Minimum size: 1200 × 720
- Background: `#F7F8FA`
- Sidebar width: 180px, background `#FFFFFF`, border-right `1px #EAECF0`
- Font: system default (Inter / Segoe UI)
- Color palette: `#1C1E26` (dark), `#F7F8FA` (surface), `#EAECF0` (border), `#9EA3AE` (muted)
- No emoji anywhere in the UI

### Existing code reuse

| Existing piece | Promoted to |
|---|---|
| `HistGradientBoostingRegressor` model | Stage 3 execution engine |
| Feature importance chart | Stage 4 output panel |
| 6-month forecast chart with ±CV-RMSE band | Stage 4 output panel |
| `build_features()` | Called by Stage 3 engine |
| Scenario input panel (oil, USD/PHP, BSP, demand) | Stage 2 inputs |

The 7 indicator charts (oil, USD/PHP, demand, PSEi, CPI, BSP rate, remittances) are retired as standalone charts. Their live values are surfaced in the Stage 1 financial data strip instead.

The old `DashboardPage` tab structure is replaced entirely. All existing logic is re-wired into the new stage structure, not deleted.

---

## 2. Stage 1 — Graph Building (RAG Pipeline)

### Purpose

Build the knowledge base before the simulation runs. Fetch live sources in parallel, load uploaded PDFs, and index everything into a TF-IDF store.

### Sources fetched on startup (parallel, 9 sources)

| Source | URL target | Content |
|---|---|---|
| DOE | doe.gov.ph | Weekly pump price bulletin + news |
| BSP | bsp.gov.ph | Press releases, rate decision statements |
| BusinessWorld | businessworld.com.ph | Top 10 energy/economy articles |
| Reuters | reuters.com | Search: "Philippines fuel" + "Brent crude OPEC" |
| Inquirer Business | business.inquirer.net | Top 8 results for oil/peso |
| Manila Bulletin | mb.com.ph | Search: gasoline price 2025 |
| OPEC | opec.org | Latest production decision bulletins |
| Yahoo Finance (crude) | finance.yahoo.com | Brent crude live price + 30-day history |
| Yahoo Finance (forex) | finance.yahoo.com | USD/PHP live rate + PSEi index |

All 9 fetches fire simultaneously via `concurrent.futures.ThreadPoolExecutor`. Each fetch uses `requests` with a 10-second timeout and `BeautifulSoup4` to strip HTML to plain text.

Financial data (Brent, USD/PHP, PSEi, BSP rate) is auto-polled every 5 minutes and displayed in a live data strip at the top of Stage 1.

### Static sources (always available)

- **NEDA PH Economic Outlook 2024–2026** — bundled with the app, pre-chunked, ~91 chunks
- **User-uploaded PDFs** — loaded via PyMuPDF (`fitz`), chunked on upload, cached for the session

### Chunking

- Window: 512 tokens (approximate character split: 512 × 4 = 2048 chars)
- Overlap: 64 tokens
- Each chunk tagged with: source name, URL/filename, fetch timestamp

### Indexing

- `sklearn.feature_extraction.text.TfidfVectorizer` fitted on all chunks
- Stored in memory as a `(n_chunks, n_terms)` sparse matrix + chunk metadata list
- Re-fitted whenever new sources are added or toggled

### UI for Stage 1

- Live financial data strip (Brent, USD/PHP, PSEi, BSP rate) — auto-updates
- Status pill: "X fetching / Y done / Z chunks indexed"
- Per-source card: source name, fetch time, progress bar, chunk count
- "3 fetching" animated cards with sweeping progress bars
- Live article feed: scrolling list of fetched articles scored by TF-IDF relevance against a default query ("Philippines gasoline price fuel oil impact") until the user sets Stage 2 inputs, then re-scored against the actual scenario text
- Upload button: opens file dialog for PDF/DOCX
- Pre-bundled NEDA card (always shown, always on)

---

## 3. Stage 2 — Environment Setup

### Purpose

Configure the scenario inputs and the agent roster before running the simulation.

### Scenario inputs

Four editable inputs displayed as pills with inline sliders:

| Input | Default | Unit |
|---|---|---|
| Oil shock | +5% | % change from current Brent |
| USD/PHP shift | +2% | % change from live rate |
| BSP rate | 6.5% | absolute % |
| Demand index | 72 | 0–100 index |

Each pill shows a drag slider. Changed values highlight with a dark border. The current live values (from Stage 1 financial strip) are shown as the baseline.

### Agent roster

**Default 3 agents:**

| Agent | Icon | Focus | Default RAG sources |
|---|---|---|---|
| Market Analyst | trend line | Price signals, news, short-term pass-through | DOE, Reuters, BusinessWorld, Yahoo Finance |
| Policy Expert | monitor | Monetary policy, FX transmission, regulatory | BSP, NEDA, BSP.gov |
| Risk Assessor | triangle warning | Tail risks, remittances, demand shocks, OPEC | OPEC, Manila Bulletin, NEDA |

**Agent configuration:**
- Each agent card shows: name, role description, RAG source toggles (per-agent, showing which sources it reads), editable system prompt preview
- "Add custom agent" button — opens a dialog to name the agent, set its role, and assign RAG sources
- Maximum 10 agents. Warning shown at >6: "Estimated run time: ~X min"

**Run button:** "Run Simulation →" — disabled until at least 1 source has finished indexing.

**Estimated run time formula:**

```
calls = agent_count × rounds
time_sec = calls × 10  # conservative for RTX 5050
```

Shown as "~N min estimated" next to the Run button.

---

## 4. Stage 3 — Simulation Execution

### Purpose

Run the agent debate and synthesize outputs via HistGBM. The canvas animates live as agents respond.

### Debate engine

**Rounds:** 3 (default). Auto-reduces to 2 if agent count > 7.

**Per round, per agent:**
1. Retrieve top-5 TF-IDF chunks from that agent's assigned sources, scored against the current scenario text
2. Build prompt: system prompt + RAG chunks + scenario inputs + previous agents' responses from this round + this agent's own responses from prior rounds
3. Call `ollama.chat(model='deepseek-r1:8b', messages=prompt, stream=True)`
4. Stream tokens to the agent's speech bubble on the canvas
5. Parse `<think>...</think>` tokens → animate as "thinking..." dots; content after `</think>` → display as the agent's statement
6. Extract the agent's price estimate (regex: `₱\d+\.\d+` or `+₱\d+` pattern)
7. Store response in debate history for next agent/round to read

**HistGBM synthesis:** After each complete round, run `build_features()` with scenario-adjusted inputs → `model.predict()` → update the engine node on the canvas with the current forecast.

**Final synthesis:** After round 3, weighted average of all agent final-round estimates combined with HistGBM prediction → final forecast price. Agent weight = normalized response length (longer, more detailed responses weighted higher). HistGBM prediction is weighted at 40%, agents collectively at 60%.

### Agent network canvas (QGraphicsScene / QGraphicsView)

- Dot-grid background: `radial-gradient` dots at 22px spacing, `#D1D5DB`, 50% opacity
- Agent nodes: white rounded rect cards, `border: 1.5px #EAECF0`, shadow. Active node gets `border: 1.5px #1C1E26`, stronger shadow
- HistGBM engine node: centered, dark (`#1C1E26` background), shows live forecast price
- RAG source mini-nodes: small pills in canvas corners showing assigned source names
- Connection lines: dashed gray from agents to engine; debate arrows between agents (gray for done rounds, solid black for active round direction)
- Speech bubbles: appear below each agent node, stream tokens live
- Thinking animation: three pulsing dots while `<think>` tokens are streaming

**Toolbar above canvas:**
- Scenario summary (compact pill text)
- Round badges: 1, 2, 3 — filled black when done, outlined when active, gray when pending
- Live dot + "Running" label
- Stop button — cancels mid-run, generates partial report from completed agents

**Right panel (alongside canvas):**
- Debate log: per-agent per-round summary (agent name, key reasoning, price estimate)
- Emerging output: forecast price updates after each round, per-agent confidence bars, "finalizing..." for metrics not yet computed

---

## 5. Stage 4 — Report Generation

### Purpose

Present the complete simulation output. No user action required to reach this stage — it activates automatically when the simulation completes.

### Left panel — Debate Summary

- **Consensus block:** weighted average estimate, low/high range, overall confidence % — defined as the share of agents whose final-round estimate falls within ±₱0.20/L of the weighted average
- **Per-agent verdicts:** name, final estimate, one-paragraph reasoning summary, confidence bar
- **Dissent note:** if any agent's estimate differs from consensus by more than ±₱0.30/L, flag them by name with their specific objection

### Right panel — Final Outputs

**Metric cards (2×2 grid):**
- Gas price forecast (₱/L) — dark card
- Pressure index (0–100)
- CV-RMSE (±₱X model uncertainty band)
- 6-month trend (%)

**6-month forecast sparkline:** matplotlib SVG-style line with ±CV-RMSE confidence band, month labels on x-axis, price labels on endpoints.

**Feature importance bars:** horizontal bar chart, top 5 drivers by HistGBM gain importance.

**Top cited RAG sources:** 3 most-retrieved chunks across all agents, shown as source + excerpt.

**Export PDF button:** generates a PDF snapshot of the full report (debate summary + metrics + charts) using `reportlab`. Saved to user's Downloads folder.

---

## 6. Stage 5 — Deep Interaction

### Purpose

Allow users to modify the simulation without starting over. Three independent interaction modes accessible via tabs within Stage 5.

### Tab 1 — Adjust & Re-run

- Scenario input sliders (same as Stage 2 but inline, no navigation needed)
- Changed inputs highlighted with dark border
- Diff block: shows what changed vs. last run + rough expected forecast shift (pre-computed heuristic before re-running)
- "Re-run Simulation →" button — goes back through Stage 3 with new inputs, then auto-returns to Stage 4

### Tab 2 — Ask an Agent

- Agent selector chips (one per configured agent)
- Freeform chat interface: user types a follow-up question, agent answers using its debate context + RAG
- Agent answers are a single Ollama call (no multi-round debate for follow-ups)
- Chat history persists for the session

### Tab 3 — Toggle RAG Sources

- Toggle list of all indexed sources (on/off per source)
- Turning a source off removes its chunks from the TF-IDF index (re-fit on toggle)
- Impact note: plain-English explanation of what toggling that source does to each agent's context
- "Add new source URL or PDF" button — same as Stage 1 add, but inline
- "Re-run with current sources →" button

---

## 7. Data Flow (End to End)

```
Stage 1                Stage 2              Stage 3                Stage 4
──────────────         ──────────           ─────────────────      ──────────
9 parallel fetches  →  Scenario inputs  →   Agent debate loop  →   Consensus
PDF loader          →  Agent roster     →   deepseek-r1:8b     →   Forecast
TF-IDF index built  →  Top-5 chunks/    →   HistGBM engine     →   Report
                        agent assigned      (build_features)       Export PDF
                                            ↑
                                        Stage 5 (re-run loop back here)
```

---

## 8. New Dependencies

| Package | Purpose | Install |
|---|---|---|
| `ollama` | Python client for local Ollama server | `pip install ollama` |
| `beautifulsoup4` | HTML parsing for web fetch | `pip install beautifulsoup4` |
| `pymupdf` (`fitz`) | PDF text extraction | `pip install pymupdf` |
| `reportlab` | PDF export | `pip install reportlab` |

Existing dependencies already in use: `requests`, `sklearn`, `PyQt6`, `matplotlib`, `pandas`, `numpy`.

**Ollama prerequisite:** User must have Ollama installed and `deepseek-r1:8b` pulled (`ollama pull deepseek-r1:8b`). The app checks for Ollama on startup and shows a setup banner if not found.

---

## 9. File Structure (new and changed)

```
ph_economic_ai/
├── main.py                          # entry point — showMaximized(), wire stages
├── engine/
│   ├── rag.py                       # NEW: fetcher, chunker, TF-IDF index
│   ├── debate.py                    # NEW: debate loop, Ollama calls, streaming
│   └── model.py                     # EXTRACTED from main.py + dashboard.py: build_features, train, predict
├── ui/
│   ├── sidebar.py                   # NEW: left nav sidebar
│   ├── stage1_rag.py                # NEW: Stage 1 panel
│   ├── stage2_setup.py              # NEW: Stage 2 panel
│   ├── stage3_canvas.py             # NEW: agent network canvas
│   ├── stage4_report.py             # NEW: report panel
│   ├── stage5_interact.py           # NEW: interaction tabs
│   └── dashboard.py                 # RETIRED: replaced by stages above
└── assets/
    └── corpus/
        └── neda_2024_2026.txt       # pre-bundled NEDA corpus
```

---

## 10. Out of Scope

- Persistent simulation history across sessions (no database)
- User accounts or authentication
- Cloud LLM fallback (Ollama only)
- Mobile or web version
- Agent memory across separate simulation runs
- Real-time streaming price data subscriptions (5-min poll is sufficient)
