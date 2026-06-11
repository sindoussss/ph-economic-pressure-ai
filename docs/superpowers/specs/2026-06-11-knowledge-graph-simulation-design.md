# ph_economic_ai — MiroFish Knowledge-Graph Simulation (SP3 Design)

**Date:** 2026-06-11
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Inspiration:** MiroFish "Graph Relationship Visualization" (`.mirofish_refs/mf2.png`, `mf6.png`) — a light, dense, organic force-directed knowledge cloud with a red relationship fan on the focused node, a floating Node Details card, and a legend.
**Program context:** Sub-project 3 of "make Strata a real, public app." SP1 (repo safety + packaging) and SP2 (editorial restyle) are deferred; this spec covers SP3 only. A 2-minute `.gitignore` hardening is folded in as the plan's task 0 (the repo is already public with untracked personal folders).

---

## 1. Problem & Goal

The Simulation screen's agent graph (`stage3_swarm_canvas.py`) reads as **plain**: ~37 sparse nodes in neat labelled clusters on a washed-out white canvas, lots of empty space, an unused console bar, and food/electricity stranded in the corners. The user wants it to look **like MiroFish** — a dense, organic, light knowledge cloud.

**Goal:** turn the simulation into a **real knowledge graph** of what the swarm actually did — every agent, the evidence it retrieved, the sources, the claims, *and* named entities extracted from the evidence — rendered MiroFish-style. Density must be **honest**: every node traces to a real artifact (a retrieved chunk, an estimate) or to a grounded extraction with provenance. No fabricated nodes.

---

## 2. Scope

### In scope
- `engine/knowledge_graph.py` — the structured knowledge-graph model + builder (the reliable core).
- `engine/entity_extractor.py` — a grounded GraphRAG-style entity/relation extraction layer over real chunk text (additive density).
- Live wiring in the swarm run so the graph builds progressively.
- `stage3_swarm_canvas.py` re-render in MiroFish style (organic cloud, type-colored dots, red relationship fan on focus, Node Details with provenance, legend, metrics, live console stream).

### Out of scope
- The report/workbench/landing restyle (SP2), forecast charts (SP2), repo packaging (SP1).
- The debate-mode canvas (`stage3_canvas.py`) — swarm mode only (the screen the user uses).
- Any change to the forecasting/benchmark logic.

### Non-negotiables
- **Grounding:** every node maps to a real artifact or a provenance-bearing extraction. Extracted entities are visually marked and carry the source chunk they came from.
- **Graceful degradation:** the structured core always renders; the LLM extraction is purely additive and must never break the graph if it is slow, errors, or returns garbage.

---

## 3. Data model (`engine/knowledge_graph.py`)

```python
@dataclass
class KGNode:
    id: str                      # stable unique id (e.g. 'src:DOE', 'ev:DOE#3', 'agent:FCST-NCR', 'ent:diesel')
    kind: str                    # 'source'|'evidence'|'agent'|'judge'|'master'|'data_input'|'claim'|'entity'
    label: str                   # short display label
    payload: dict                # real content: chunk text+source / estimate / extraction provenance
    cluster: str | None = None   # region or sector for layout/colour ('NCR','food','elec','gas',...)

@dataclass
class KGEdge:
    src: str                     # KGNode.id
    dst: str                     # KGNode.id
    kind: str                    # 'retrieved'|'from_source'|'in_region'|'aggregates'|'claims'|'references'|'mentions'|'relates'
    weight: float = 1.0
```

**Node kinds & their real source:**
| kind | from | payload |
|---|---|---|
| `source` | the ~14 RAG feeds (`rag.all_source_names`) | name, url |
| `evidence` | each retrieved `Chunk` | source, url, text |
| `agent` | swarm agents (role, region) | name, role, region, estimate |
| `judge` | regional judges | region, estimate |
| `master` | master judge | final_estimate |
| `data_input` | scenario inputs (oil_price, usd_php, …) | key, value |
| `claim` | an agent's estimate+statement | estimate, statement |
| `entity` | grounded extraction (§4) | name, type, provenance:{chunk_id, source} |

**Edge kinds:** `evidence → from_source → source`; `agent → retrieved → evidence`; `agent → in_region → judge`; `judge → aggregates → agent` (and `master → aggregates → judge`); `agent → claims → claim`; `claim → references → data_input`; `entity → mentions → evidence` (provenance); `entity → relates → entity` (extracted relation).

### `KnowledgeGraphBuilder`
- `add_source(name, url)`, `add_evidence(chunk)`, `add_agent(name, role, region, estimate)`, `add_judge(...)`, `add_master(...)`, `add_data_input(k, v)`, `add_claim(agent_id, estimate, statement)`, `add_entity(name, etype, chunk_id, source)`, `add_relation(ent_a, ent_b, kind)`.
- Idempotent by `id` (re-adding a node merges payload; re-adding an edge is a no-op).
- `snapshot() -> (list[KGNode], list[KGEdge])` for the canvas; `node(id)` and `edges_of(id)` for focus/Node-Details.
- Pure/standalone (no Qt, no ollama) → unit-testable.

---

## 4. Grounded entity extraction (`engine/entity_extractor.py`)

- `extract(chunk_text: str, source: str) -> ExtractionResult` where `ExtractionResult = {entities: [{name, type}], relations: [{a, b, kind}]}`.
- Calls an **existing** local model (`qwen2.5:3b`, already pulled for the swarm — no new dependency) with a strict prompt: "From this Philippine economic text, list named entities (commodity, agency, place, policy, figure) and any relations, as `ENTITY: <name> | <type>` and `REL: <a> -> <b> | <kind>` lines." Deterministic-ish (low `num_predict`).
- A **pure parser** `parse_extraction(text) -> ExtractionResult` turns the model lines into the structured result (regex; ignores malformed lines). This is the unit-tested seam (mock the model, test the parser).
- **Provenance:** the caller tags every entity/relation with the `chunk_id` + `source` it came from, so `add_entity(...)` records where it was extracted. Entities mentioned in ≥1 chunk link to each via `mentions` edges (so a recurring entity like "diesel" connects multiple evidence nodes → real hubs, real density).
- **Robustness:** wrapped so any exception/timeout/empty result yields zero entities for that chunk and is logged — never raised. Extraction runs on a background worker; the structured graph is already on-screen before any entity arrives.

---

## 5. Live wiring (swarm run)

As the swarm executes (in `stage3_swarm_canvas.py`'s controller / the swarm thread bridge):
```
on swarm start      -> builder.add_source(...) for each feed; add_data_input(...) for scenario
per agent activation -> add_agent(...); for each rag.query chunk: add_evidence(chunk) + edge(agent retrieved evidence)
per agent estimate   -> add_claim(agent, estimate, statement); edges claim->references->data_input
per new evidence     -> enqueue chunk for EntityExtractor (background); on result: add_entity(...) + mentions/relates edges
on judges/master     -> add_judge/add_master + aggregates edges
each builder change   -> canvas receives an incremental snapshot and re-lays-out (throttled)
```
The existing swarm/debate computation is untouched; this only *observes* its artifacts.

---

## 6. Render — MiroFish style (`stage3_swarm_canvas.py`)

- **Canvas:** light (`#FCFCFC`), organic force-directed layout (reuse `force-directed-v2`); fade node opacity toward the edges; remove the big empty cluster halos in favour of soft sector tints.
- **Nodes:** tiny dots sized by degree; colour by `kind`/`cluster` — gray `evidence`/`entity` majority, red gas agents, blue judges, near-black master, green food, amber electricity, blue source pills (kept as labelled pills at the top, as today).
- **Edges:** faint gray mass edges (`#D8DCE4`, low opacity); on focus/active, the focused node's edges render as the **red relationship fan** (`#E5484D`), with neighbours emphasised.
- **Node Details card** (reuse/restyle `_NodeDetailCard`): on click/focus, show `label`, `kind`, payload (for `evidence`: source + chunk text; for `entity`: type + "extracted from <source>: '<text>'"; for `agent`/`claim`: estimate + statement), and "N relationships". This is the honesty surface — provenance is visible.
- **Legend** (bottom-left): node-type colour key incl. an explicit "extracted entity" swatch.
- **Metrics** (bottom-right): nodes / edges / clusters / density — fed from the real graph.
- **Console** (the empty bottom bar): stream short live lines ("FCST·NCR retrieved DOE chunk", "extracted: diesel, Meralco GC", "REGL converged -1.9") — fills the dead space, signals life.

---

## 7. Error handling / robustness
- Extraction failures/timeouts/garbage → 0 entities for that chunk, logged, never raised (§4).
- If ollama is unavailable, the **structured graph still renders fully** (agents, evidence, sources, claims, judges) — the entity layer is simply empty. The screen is never broken by the optional layer.
- Canvas re-layout is throttled (coalesce rapid builder updates) to stay responsive as hundreds of nodes arrive.
- All node ids are stable strings; duplicate adds merge (no duplicate nodes from repeated retrievals of the same chunk).

## 8. Testing
- `test_knowledge_graph.py`: builder adds the right nodes/edges from mock retrievals/estimates/judges; idempotency (re-add merges); `snapshot`/`edges_of`/`node` correct; a recurring entity links multiple evidence nodes.
- `test_entity_extractor.py`: `parse_extraction` parses well-formed lines, ignores malformed ones, returns empty on garbage; `extract` swallows a raised mock-model error and returns empty (graceful degradation).
- `test_swarm_canvas` (offscreen Qt, smoke): canvas builds from a snapshot, focusing a node highlights its edges, Node Details shows the node's provenance text; renders with an empty entity layer (ollama-absent path).
- Full suite stays green; honest-surface intact.

## 9. Deliverables (definition of done)
1. `KnowledgeGraphBuilder` — structured core, pure, tested.
2. `EntityExtractor` — grounded extraction + pure parser, graceful, tested.
3. Live wiring — graph builds progressively during a swarm run, extraction on a background worker.
4. MiroFish-style canvas — organic cloud, type colours, red relationship fan, Node Details with provenance, legend, live metrics + console.
5. Tests per §8; structured graph renders even with no ollama; full suite green.

## 10. Why it matters
It turns the plainest screen into the centerpiece: a live, dense, MiroFish-grade knowledge graph of the swarm's *actual* reasoning — agents pulling real evidence, citing real sources, surfacing real entities — with every node clickable back to its source. Impressive *and* honest: the density is earned, not faked, which is exactly the bar ("real app, not BS") the project is held to.
