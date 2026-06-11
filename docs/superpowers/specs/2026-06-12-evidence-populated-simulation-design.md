# ph_economic_ai — Evidence-Populated Simulation Canvas (Design)

**Date:** 2026-06-12
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Context:** The bare force-graph KG ("dot cloud") was rejected — the user wants the *old* structured/labelled simulation canvas (`_SwarmCanvas`) back (already restored), **but more populated**. This adds real retrieved-evidence satellite nodes to it. Entities are explicitly deferred to a later phase.

---

## 1. Problem & Goal

The old `_SwarmCanvas` (region clusters + RAG sources + agents + judges + sector clusters, ~37 nodes) is the look the user wants — structured, labelled, "line vibe" — but too sparse (lots of whitespace). 

**Goal:** make it visibly **populated** by hanging each agent's *real* retrieved RAG chunks off it as small satellite nodes (~37 → ~100+ nodes), keeping the existing structure/labels/edges intact, and keeping it honest (each new node is a real chunk, clickable to its source + text).

---

## 2. Scope

### In scope (`ui/stage3_swarm_canvas.py`)
- A new `_EvidenceNode` QGraphics item (tiny faint dot, clickable → details card shows source + chunk text).
- `_SwarmCanvas.add_evidence_layer(rag, scenario, top_k=3)` — adds evidence satellites + faint edges around each existing agent node.
- Panel wiring: call `add_evidence_layer` from `connect_thread` (where `rag`/`scenario` are available), so the canvas is dense from run start.

### Out of scope
- **Entity nodes** — deferred to a later phase (slower, LLM-dependent).
- The bare KG force-graph live view (already reverted), the report/landing, any swarm/engine change.
- The `_kg_*` live-builder code on the branch is left as-is (unused for the view; harmless) — a separate tidy-up.

### Non-negotiables
- **Honesty:** every evidence node is a *real* chunk from `rag.query` (the same retrieval the swarm uses); clicking it shows the actual source + text. No fabricated/decorative nodes presented as data.
- **Don't break the old canvas:** the existing clusters/agents/judges/RAG/sector layout + animations + trust badges stay exactly as they are; evidence is purely additive.
- **Robustness:** the whole layer is wrapped — a `rag.query` failure means that agent simply gets no satellites; never crashes the run.

---

## 3. Components (`ui/stage3_swarm_canvas.py`)

### 3.1 `_EvidenceNode(QGraphicsObject)`
- Tiny dot (r ≈ 3 px), faint fill (a muted gray or a faint tint of its source's colour), low z-order so it reads as background texture under the agents.
- Stores `source` + `text`; `clicked = pyqtSignal(...)` (or reuse the canvas's existing `node_clicked` mechanism) so a click routes to the panel's details card showing `source` + `text[:300]` (provenance).
- Non-interactive hover is fine; no animation (keeps it lightweight at ~60–80 instances).

### 3.2 `_SwarmCanvas.add_evidence_layer(rag, scenario, top_k=3)`
- The agent nodes created in `_build` already carry their `rag_sources` and have a scene position. Keep a reference to them (e.g. a list `self._agent_nodes` populated in `_build`).
- For each agent node: `chunks = rag.query(_scenario_text(scenario), top_k=top_k, sources=node.rag_sources)`; for each chunk, create an `_EvidenceNode(source, text)` positioned in a small ring (radius ≈ 22–30 px) around the agent's position, add a faint edge agent→evidence, `addItem` both.
- Dedupe identical `(source, text)` chunks across agents (one node, multiple edges) to avoid clutter — or keep per-agent (simpler); pick per-agent satellites for the "each agent has its evidence" look, dedupe only exact repeats on the same agent.
- Reuse `_scenario_text` from `engine.kg_swarm_adapter` for the query text.
- Idempotent / guarded: if called twice, clear prior evidence items first; wrap in `try/except`.

### 3.3 Panel wiring (`Stage3SwarmPanel`)
- In `connect_thread(thread)` (already captures the run context), after the existing wiring, call:
  `self._canvas.add_evidence_layer(getattr(thread, '_rag', None), getattr(thread, '_scenario', {}))`.
- Route `_EvidenceNode` clicks to the existing details card (reuse `_on_node_clicked` / the canvas `node_clicked` signal) to show the chunk's source + text.

## 4. Data flow
```
connect_thread(thread) -> canvas.add_evidence_layer(thread._rag, thread._scenario)
   per agent node -> rag.query(scenario_text, sources=node.rag_sources)
     per chunk -> _EvidenceNode(source,text) in a ring around the agent + faint edge
click evidence -> details card shows source + text (provenance)
```
No network (rag is in-memory), no engine change; dense from run start.

## 5. Error handling / robustness
- `add_evidence_layer` wrapped in `try/except`; per-agent `rag.query` wrapped so one failure skips only that agent's satellites.
- If `rag` is `None` (e.g. a test/edge), the method is a no-op.
- Evidence items are low z-order and lightweight; ~60–80 dots + faint edges stay smooth.

## 6. Testing
- `test_evidence_layer` (offscreen Qt): build a `_SwarmCanvas`, call `add_evidence_layer(mock_rag, scenario)` where `mock_rag.query` returns N chunks; assert the scene gains `_EvidenceNode` items (count ≈ agents × N) and agent→evidence edges; assert an `_EvidenceNode` carries the chunk's `source`/`text`.
- `test_stage3_swarm` smoke: the panel still builds; `connect_thread` with a stub thread (carrying `_rag`/`_scenario`) populates evidence without error.
- Full suite green; existing canvas tests unaffected.

## 7. Deliverables (definition of done)
1. `_EvidenceNode` + `_SwarmCanvas.add_evidence_layer` (real chunks, ring layout, faint edges, clickable provenance, guarded).
2. Panel wires it in `connect_thread`; clicks show source + text.
3. The simulation reads as populated (~100+ nodes) in the old structured/labelled style.
4. Tests per §6; full suite green; old canvas behaviour intact.

## 8. Why it matters
It gives the user exactly what they asked for: the old simulation's structured, labelled "line vibe" — now densely populated with the agents' *real* evidence, every added node traceable to its source. Density that's earned, in the look they like.
