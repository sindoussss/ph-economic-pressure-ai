# Spec — Strata Pressure Monitor (Forum → Tournament)

**Status:** M1–M5 complete + post-review upgrades, full suite 656 green; published to
`origin/pressure-monitor`. Post-M5 upgrades landed:
- **Named cast + live forum feed + debate-map graph** (`ui/forum_graph.py`) — agents
  stream as chat cards with their RAG source nodes; the graph grows turn by turn.
- **Anchor magnitude-guard wired** — `auto_assemble` sets a per-sector anchor and the
  forum clamps the consensus (§6.6), so a weak-model "+5%/month" food read no longer
  reaches the card.
- **Differentiated, challenging agents** — each channel (social/news/market) stays in
  its lane and is told not to repeat; confidence is corroboration-scaled.
- **Judged deliberation** — a judge synthesises each sector's debate into one present
  read (like the swarm's master judge), not a plain average.
- **Hybrid social sourcing** (`engine/live_social.py`) — best-effort live Reddit/Trends
  refresh with a frozen-snapshot fallback; the validated benchmark stays frozen.
- **Home "Run" chains** the Monitor forum, then the tournament swarm.

Both former pending items are now **done** (2026-07-26):
- **Google Trends refreshed** and the keystone run: `sentiment_nowcast.json` reports a
  clean null — search interest does not nowcast PH fuel/food inflation beyond naive.
  One gotcha, recorded because it nearly produced a vacuous null: Trends normalises all
  terms in a *single* query against one shared 0–100 scale, which crushed the
  low-volume Tagalog terms to near-constant zero. Query **one term per payload**.
  Reddit was dropped — its API now gates access behind a research proposal, and the
  forum runs on Trends + news + market without it.
- **Mean-baseline correction adopted canonically.** The verdict gate did tighten as
  predicted: `outlook.sector_basis` now returns `efficient` for every sector, because
  no target in the frozen report beats naive any more.
One-click, monitor-led expansion of the Strata app, inspired by BettaFish's moderated
forum but bounded by the Strata benchmark.

## Intent
Add an autonomous "Run" flow that **leads with a present-pressure read** (the
Monitor) and keeps the forecast a quiet, bounded, monthly footnote. Strata becomes
a *pressure monitor*, not a *price predictor*.

Flow: `Run → auto_assemble() → Forum debate = Pressure Brief (hero) → Tournament
debate = Forecast (secondary, bounded)`. Two debates back-to-back; the Monitor is
the *output of the first debate*, not a separate phase.

## Non-negotiables (why this stays honest)
1. **Layer isolation.** Social *text* + Forum are exploratory. The social *numeric*
   signal (Google Trends) enters the validated benchmark as a frozen CSV and is
   DM-tested like everything else. `test_benchmark_isolation` must stay green.
2. **Offline & reproducible.** Live pulls live ONLY in `tools/refresh_social.py`,
   which writes frozen snapshots. The app and benchmark read only frozen data — no
   API key on the run path. Snapshots are dated; the UI shows the "as of" date.
3. **The moderator is the benchmark's voice.** On efficient targets (fuel/FX) the
   Forum converges on "no edge → naive + interval"; the forecast never becomes the
   headline and never claims a weekly/daily number (monthly horizon only).

## Output contract
- **HERO — Pressure Monitor (present):** per sector (gas/food/electricity), "as of
  <date>": direction now, drivers/events, confidence, sources, window chips
  (today / this week / this month apply to the *present read only*).
- **Secondary — Next-month outlook (bounded):** the audit verdict first
  (fuel/FX efficient → naive+interval; electricity → mechanical channel; food →
  own-dynamics), then the tournament forecast + interval, spread labeled
  "agreement, not probability".

## The honest keystone
`benchmark/sentiment_nowcast.py`: add the Trends column to the nowcast frame and
run the **same** walk-forward + `mom_verdict` (baseline pool including `mean`).
Expected: a null — "social search interest does not nowcast PH fuel/food inflation
beyond naive." Reported like the other nulls; ties the flashy layer to the spine.

## Milestones
1. **M1 — Snapshots (this increment).** `tools/refresh_social.py` (pytrends + praw,
   manual, defensive) → `benchmark/data/google_trends_monthly.csv` (numeric) and
   `assets/corpus/social/reddit_<date>.jsonl` (text). `engine/social_snapshot.py`
   loads/slices/registers the frozen text into the RagEngine via `add_text` (never
   `RagEngine.SOURCES`). Test with synthetic fixtures; graceful when absent.
2. **M2 — Benchmark sentiment test** (`sentiment_nowcast.py` + artifact). Keystone.
3. **M3 — Monitor** (`auto_assemble` + Forum + moderator → Pressure Brief). Shippable
   on its own — the present-pressure read, no forecast yet.
4. **M4 — Outlook** (tournament seeded by the brief + verdict gate). Additive, bounded.
5. **M5 — One-click UI (monitor-led) + docs.**

## Data sources (free-tier only)
Google Trends (`pytrends`, terms: presyo ng gas / diesel price / Meralco bill /
bigas presyo) and Reddit (`praw`, r/Philippines + r/phinvest). **Skip X and
Facebook** (cost / no public API / ToS). News already covered by the RagEngine RSS.

## Budget
Forum ≈ 8 local calls; Tournament = 39. Full run ≈ ~50 calls / 4–6 min on 8 GB.
`fast` mode (2 agents, 1 round) for iteration. Monitor alone (M3) ≈ 8 calls.

## Will NOT claim
Weekly/daily *forecast* numbers; any forecast that beats the benchmark; any live
dependency in the validated layer.
