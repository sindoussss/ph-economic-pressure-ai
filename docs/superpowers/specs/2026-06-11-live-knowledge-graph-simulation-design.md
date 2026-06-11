# ph_economic_ai — Live MiroFish Simulation (Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Builds on:** SP3 (the knowledge-graph engine + `KnowledgeGraphCanvas`, post-run only). This makes the graph the **live** Simulation view.

---

## 1. Problem & Goal

Today the Simulation screen shows the old structured "arena" animation during a run; the MiroFish knowledge graph is built only at `swarm_complete` and is then *buried* (the app auto-jumps to the Report, so the user rarely sees it). The user expected the Simulation to **look like MiroFish while it runs**.

**Goal:** make the live Simulation view **be** the MiroFish knowledge graph — it grows as the swarm works (sources → agents → claims → evidence → judges → master), densifies with entities near the end, and **stays on screen at completion** with a "View report →" button instead of auto-jumping.

**Approved decisions:** (1) replace the old arena canvas entirely with the live KG; (2) on completion, stay on the finished graph + a "View report →" button (no auto-navigate).

---

## 2. Scope

### In scope
- `ui/stage3_swarm_canvas.py` (`Stage3SwarmPanel`): host only the `KnowledgeGraphCanvas` as the live view; rewire the existing thread-signal handlers to incrementally build a `KnowledgeGraphBuilder` and refresh the canvas (throttled); add a `view_report_requested` signal + a "View report →" button shown at completion.
- `ui/main_window.py`: stop auto-navigating to the Report in `_on_swarm_complete`; connect `view_report_requested` → navigate to the Report.
- Live build covers gas-swarm agents/claims/judges/master + RAG sources + data inputs + food/electricity sector agents/claims; evidence via in-memory `rag.query`; entities via the existing background `EntityExtractWorker`.

### Out of scope
- Deleting the old QGraphics node classes (`_AgentNode`, `_RegionalNode`, … ~1k lines) — left **unused**, flagged for a separate cleanup so this change stays focused.
- Any change to the swarm/debate computation, the engine KG/extractor (reused as-is), the Report/landing screens.
- Incremental/animated force-layout (the canvas re-lays-out per snapshot; throttling keeps it smooth at this node count — true incremental layout is a future optimization).

### Non-negotiables
- **Honesty:** every live node maps to a real signal payload (agent claim) or real in-memory retrieval (evidence) or grounded extraction (entities). No fabricated nodes.
- **Robustness:** the entire live path is wrapped so a malformed signal / rag / extraction failure can never crash a run; if the live build errors, the run still completes and the post-run `show_knowledge_graph` (SP3) still renders the full graph as a fallback.

---

## 3. Architecture

### 3.1 Live builder (in `Stage3SwarmPanel`, helper logic factored into `ui/kg_live.py`)
A thin, testable module `ui/kg_live.py` with pure functions that mutate a `KnowledgeGraphBuilder` from signal payloads (no Qt), so the wiring is unit-testable:
```python
def seed(builder, sources: list[str], scenario: dict) -> None       # sources + data_inputs
def add_agent_round(builder, responses, region_for, rag, scenario) -> None  # agents+claims+evidence
def add_regional(builder, region: str, estimate) -> None            # judge + master->judge edge
def add_master(builder, final_estimate) -> None
def add_sector_agent(builder, name, sector, estimate, statement) -> None    # food/elec
```
(`add_agent_round` re-queries `rag.query(scenario_text, sources=...)` per agent — in-memory, cheap — and adds evidence + retrieved edges; reuses the SP3 `KnowledgeGraphBuilder` + `kg_assemble` helpers where possible.)

### 3.2 Panel wiring (`Stage3SwarmPanel`)
- `_build_main_row`: the `KnowledgeGraphCanvas` (`self._kg_canvas`) is the primary widget (the old `self._canvas` is no longer added to the layout). The verdict sidebar + console stay.
- New: `self._kg_builder` (created/reset on `reset()`), a `QTimer` `self._kg_refresh` (~1.5 s, coalescing), and a dirty flag. Each thread handler calls the matching `kg_live.*`, sets dirty; the timer fires → `self._kg_canvas.set_snapshot(*self._kg_builder.snapshot())` when dirty.
- `connect_thread`/`reset`: seed the builder (rag sources + scenario), start the refresh timer.
- Existing handlers repurposed: `_on_group_round_done` → `add_agent_round`; `_on_regional_done` → `add_regional`; `_on_swarm_complete` → `add_master` + start `EntityExtractWorker` (entities densify) + reveal the "View report →" button + final snapshot. Sector handlers (`_on_food_agent_done`/`_on_elec_agent_done`) → `add_sector_agent`. (Old-canvas calls like `store_response`/`mark_eliminated` are removed.)
- New signal `view_report_requested = pyqtSignal()` + a "View report →" button (hidden until `swarm_complete`).
- The panel needs `rag` + `scenario` for live evidence: capture them in `connect_thread(thread)` (the thread holds `_rag`/`scenario`) — store `self._rag`, `self._scenario`, and build agents/region map via `build_swarm_agents` for the region-for mapping (as the SP3 adapter does).

### 3.3 `main_window`
- `_on_swarm_complete`: remove `self._stack.setCurrentIndex(<Report>)` (and the related `set_active`) so the user stays on the Simulation graph. Keep building the post-run graph as the SP3 fallback only if the live build produced nothing (guard: if `self._stage3_swarm` has a populated live builder, skip the redundant post-run `show_knowledge_graph`; else call it).
- Connect `self._stage3_swarm.view_report_requested` → navigate to the Report (the index used today), reusing the existing unlock/`set_active`/`setCurrentIndex` logic.

## 4. Data flow
```
run start  -> kg_live.seed(builder, rag.all_source_names, scenario)        [sources + data]
group_round_done(responses) -> kg_live.add_agent_round(...)                [agent+claim+evidence]
regional_done(verdict)      -> kg_live.add_regional(...)                    [judge]
food/elec agent_done        -> kg_live.add_sector_agent(...)               [sector cluster]
swarm_complete(mv)          -> kg_live.add_master(...); EntityExtractWorker [entities densify]
                            -> show 'View report ->'  (NO auto-nav)
every change -> dirty; ~1.5s timer -> kg_canvas.set_snapshot(builder.snapshot())
'View report ->' click -> view_report_requested -> main_window -> Report
```

## 5. Error handling / robustness
- Each handler's `kg_live.*` call is wrapped (`try/except`, logged) — a bad payload skips that node, never raises into the Qt event loop.
- `rag.query` failure in `add_agent_round` → that agent simply has no evidence node (caught).
- Entity extraction already degrades to empty (SP3).
- If the live builder is empty at completion (e.g. live path disabled/errored), main_window falls back to the SP3 post-run `show_knowledge_graph` so the graph still appears.
- Throttle prevents layout thrash; final `set_snapshot` on completion guarantees the finished graph is shown.

## 6. Testing
- `test_kg_live.py` (pure, no Qt): feed a scripted sequence — `seed` then two `add_agent_round` calls (mock rag returning chunks) then `add_regional` then `add_master` — assert the builder's snapshot has the expected source/agent/claim/evidence/judge/master nodes + edges, and that a second identical round is idempotent.
- `test_stage3_swarm` (offscreen, smoke): the panel builds with the KG canvas as the main view; a synthetic `group_round_done` + `swarm_complete` grows the graph (canvas node count > 0) and reveals the "View report →" button; clicking it emits `view_report_requested`.
- `test_main_window` (smoke): a completed run does NOT change the stack to the Report automatically; `view_report_requested` navigates to it.
- Full suite green.

## 7. Deliverables (definition of done)
1. `ui/kg_live.py` — pure live-build helpers; tested.
2. `Stage3SwarmPanel` shows the live KG (old arena removed from view), grows it from the thread signals (incl. food/elec), densifies with entities at completion, and exposes `view_report_requested` + a "View report →" button.
3. `main_window` no longer auto-jumps to the Report; the button navigates there; SP3 post-run render kept as a guarded fallback.
4. Tests per §6; full suite green; honest + robust per §2/§5.

## 8. Why it matters
It delivers the thing the user actually pictured: the Simulation **is** the MiroFish graph, alive and growing as the swarm reasons — every node traceable — and the finished graph stays on screen instead of being buried. The centerpiece is finally front-and-centre, during *and* after the run.
