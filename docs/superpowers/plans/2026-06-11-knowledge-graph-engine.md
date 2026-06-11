# Knowledge-Graph Engine (SP3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure, honest knowledge-graph engine behind the MiroFish-style simulation: a structured graph of the swarm's real artifacts + a grounded entity-extraction layer with provenance.

**Architecture:** Two standalone, Qt-free, ollama-optional modules + an assembler — all unit-tested. `KnowledgeGraphBuilder` holds nodes/edges (idempotent, snapshot-able). `EntityExtractor` turns real chunk text into grounded entities/relations (pure parser + ollama call that degrades to empty on any failure). `assemble.py` builds a full graph from plain swarm data and folds in extractions. The Qt canvas render + live wiring is a separate follow-on plan (SP3b) written against this engine.

**Tech Stack:** Python 3.10, pytest. Optional: ollama (graceful if absent).

**Spec:** `docs/superpowers/specs/2026-06-11-knowledge-graph-simulation-design.md` (§3 builder, §4 extractor; §5/§6 live wiring + render are SP3b).

**Prereqs (on `master`):** `ph_economic_ai/engine/` exists; ollama is used elsewhere as `ollama.chat(model=, messages=, options=)`. `rag.query(...) -> list[dict]` with keys `source`, `text` (and `url`). No existing `knowledge_graph.py` / `entity_extractor.py` / `assemble.py`.

**Conventions:** Tests in `ph_economic_ai/tests/` with the path shim `sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))`. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

---

## Task 0: Branch + harden `.gitignore` (the 2-minute safety step)

**Files:** Modify `.gitignore`

- [ ] **Step 1: Branch**
```bash
git checkout master && git pull && git checkout -b feature/knowledge-graph-engine
```

- [ ] **Step 2: Append a personal-folder guard to `.gitignore`**

Append (duplicates are harmless if any already exist):
```gitignore

# --- Personal / unrelated working-tree items: never publish (repo is public) ---
/Personal/
/Character AI/
/Project_Maria/
/Writes/
/AFP/
/.claude/
/.superpowers/
/.mirofish_refs/
_ohms_final.py
launch_and_run_swarm.py
navigate_swarm.py
ph_economic_ai/cache/
```

- [ ] **Step 3: Verify those paths are now ignored**

Run: `git status --short --untracked-files=normal | grep -E "Personal/|Character AI/|Project_Maria/|Writes/|AFP/|\.claude/|\.superpowers/" | head`
Expected: **no output** (they're ignored now). If any still show, fix the pattern.

- [ ] **Step 4: Commit**
```bash
git add .gitignore
git commit -m "chore: gitignore personal/unrelated folders (public repo safety)"
```

---

## Task 1: `KnowledgeGraphBuilder` (structured core)

**Files:**
- Create: `ph_economic_ai/engine/knowledge_graph.py`
- Test: `ph_economic_ai/tests/test_knowledge_graph.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_knowledge_graph.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder, KGNode, KGEdge


def test_evidence_links_to_source_and_is_idempotent():
    b = KnowledgeGraphBuilder()
    b.add_source('DOE', 'http://doe')
    e1 = b.add_evidence('DOE', 3, 'diesel up 1.2', 'http://doe')
    e2 = b.add_evidence('DOE', 3, 'diesel up 1.2', 'http://doe')   # same chunk again
    assert e1 == e2
    nodes, edges = b.snapshot()
    assert len([n for n in nodes if n.kind == 'evidence']) == 1     # merged, not duplicated
    assert KGEdge(e1, 'src:DOE', 'from_source') in edges
    assert b.node(e1).payload['text'] == 'diesel up 1.2'


def test_agent_claim_and_aggregation_edges():
    b = KnowledgeGraphBuilder()
    b.add_master(-1.8)
    b.add_judge('NCR', -1.7)
    b.add_edge('master', 'judge:NCR', 'aggregates')
    a = b.add_agent('FCST-NCR', 'Forecaster', 'NCR', -1.9)
    b.add_edge('judge:NCR', a, 'aggregates')
    c = b.add_claim(a, -1.9, 'pump easing')
    assert b.node(c).payload['estimate'] == -1.9
    assert any(e.kind == 'claims' and e.src == a for e in b.edges_of(a))
    assert any(e.kind == 'aggregates' and e.dst == a for e in b.edges_of(a))


def test_entity_provenance_accumulates_across_chunks():
    b = KnowledgeGraphBuilder()
    ev1 = b.add_evidence('DOE', 1, 'diesel ...')
    ev2 = b.add_evidence('CRUDE', 2, 'diesel ...')
    n1 = b.add_entity('Diesel', 'commodity', ev1, 'DOE')
    n2 = b.add_entity('diesel', 'commodity', ev2, 'CRUDE')   # case-insensitive same entity
    assert n1 == n2
    node = b.node(n1)
    assert len(node.payload['provenance']) == 2
    assert {e.dst for e in b.edges_of(n1) if e.kind == 'mentions'} == {ev1, ev2}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_knowledge_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: ...knowledge_graph`.

- [ ] **Step 3: Implement**

Create `ph_economic_ai/engine/knowledge_graph.py`:
```python
"""Honest knowledge graph of a swarm run — nodes/edges that each map to a real
artifact (retrieved chunk, claim) or a grounded extraction (entity + provenance).
Pure: no Qt, no ollama. The canvas (SP3b) renders a snapshot()."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KGNode:
    id: str
    kind: str          # source|evidence|agent|judge|master|data_input|claim|entity
    label: str
    payload: dict = field(default_factory=dict)
    cluster: Optional[str] = None


@dataclass(frozen=True)
class KGEdge:
    src: str
    dst: str
    kind: str          # retrieved|from_source|in_region|aggregates|claims|references|mentions|relates
    weight: float = 1.0


class KnowledgeGraphBuilder:
    def __init__(self):
        self._nodes: dict[str, KGNode] = {}
        self._edges: dict[tuple, KGEdge] = {}

    # -- primitives --
    def add_node(self, id: str, kind: str, label: str,
                 payload: Optional[dict] = None, cluster: Optional[str] = None) -> str:
        if id in self._nodes:
            if payload:
                self._nodes[id].payload.update(payload)
        else:
            self._nodes[id] = KGNode(id, kind, label, dict(payload or {}), cluster)
        return id

    def add_edge(self, src: str, dst: str, kind: str, weight: float = 1.0) -> None:
        key = (src, dst, kind)
        if key not in self._edges:
            self._edges[key] = KGEdge(src, dst, kind, weight)

    # -- convenience (stable ids) --
    def add_source(self, name: str, url: str = '') -> str:
        return self.add_node(f'src:{name}', 'source', name, {'url': url})

    def add_evidence(self, source: str, idx: int, text: str, url: str = '') -> str:
        nid = f'ev:{source}#{idx}'
        self.add_node(nid, 'evidence', f'{source} #{idx}',
                      {'source': source, 'text': text, 'url': url}, cluster=source)
        self.add_edge(nid, f'src:{source}', 'from_source')
        return nid

    def add_agent(self, name: str, role: str = '', region: str = '',
                  estimate: Optional[float] = None) -> str:
        nid = f'agent:{name}'
        return self.add_node(nid, 'agent', name,
                             {'role': role, 'region': region, 'estimate': estimate},
                             cluster=region or None)

    def add_judge(self, region: str, estimate: Optional[float] = None) -> str:
        return self.add_node(f'judge:{region}', 'judge', region,
                             {'estimate': estimate}, cluster=region)

    def add_master(self, final_estimate: Optional[float] = None) -> str:
        return self.add_node('master', 'master', 'MASTER',
                             {'final_estimate': final_estimate})

    def add_data_input(self, key: str, value) -> str:
        return self.add_node(f'data:{key}', 'data_input', key, {'value': value})

    def add_claim(self, agent_id: str, estimate: Optional[float], statement: str = '') -> str:
        nid = f'claim:{agent_id}'
        label = f'{estimate:+.2f}' if estimate is not None else '—'
        self.add_node(nid, 'claim', label, {'estimate': estimate, 'statement': statement})
        self.add_edge(agent_id, nid, 'claims')
        return nid

    def add_entity(self, name: str, etype: str, chunk_id: str, source: str) -> str:
        nid = f'ent:{name.strip().lower()}'
        prov = {'chunk_id': chunk_id, 'source': source}
        node = self._nodes.get(nid)
        if node:
            node.payload.setdefault('provenance', []).append(prov)
        else:
            self.add_node(nid, 'entity', name.strip(),
                          {'type': etype, 'provenance': [prov]})
        self.add_edge(nid, chunk_id, 'mentions')
        return nid

    def add_relation(self, ent_a: str, ent_b: str, kind: str = 'relates') -> None:
        self.add_edge(f'ent:{ent_a.strip().lower()}', f'ent:{ent_b.strip().lower()}', kind)

    # -- read --
    def snapshot(self):
        return list(self._nodes.values()), list(self._edges.values())

    def node(self, id: str) -> Optional[KGNode]:
        return self._nodes.get(id)

    def edges_of(self, id: str) -> list:
        return [e for e in self._edges.values() if e.src == id or e.dst == id]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_knowledge_graph.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add ph_economic_ai/engine/knowledge_graph.py ph_economic_ai/tests/test_knowledge_graph.py
git commit -m "feat(engine): KnowledgeGraphBuilder — honest structured graph of a swarm run"
```

---

## Task 2: `EntityExtractor` (grounded extraction + pure parser)

**Files:**
- Create: `ph_economic_ai/engine/entity_extractor.py`
- Test: `ph_economic_ai/tests/test_entity_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_entity_extractor.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import ph_economic_ai.engine.entity_extractor as ee


def test_parse_extraction_reads_entities_and_relations():
    text = (
        'ENTITY: diesel | commodity\n'
        'ENTITY: Meralco | organization\n'
        'garbage line that should be ignored\n'
        'REL: diesel -> pump price | drives\n'
    )
    res = ee.parse_extraction(text)
    assert {'name': 'diesel', 'type': 'commodity'} in res['entities']
    assert len(res['entities']) == 2
    assert res['relations'] == [{'a': 'diesel', 'b': 'pump price', 'kind': 'drives'}]


def test_parse_extraction_empty_on_garbage():
    assert ee.parse_extraction('no structured lines here') == {'entities': [], 'relations': []}


def test_extract_degrades_to_empty(monkeypatch):
    # model raising -> empty, never propagates
    def boom(*a, **k):
        raise RuntimeError('ollama down')
    monkeypatch.setattr(ee, 'ollama', type('M', (), {'chat': staticmethod(boom)}))
    assert ee.extract('some real chunk text', 'DOE') == {'entities': [], 'relations': []}
    # no ollama at all -> empty
    monkeypatch.setattr(ee, 'ollama', None)
    assert ee.extract('text', 'DOE') == {'entities': [], 'relations': []}
    # empty input -> empty (no call)
    assert ee.extract('   ', 'DOE') == {'entities': [], 'relations': []}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_entity_extractor.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `ph_economic_ai/engine/entity_extractor.py`:
```python
"""Grounded entity/relation extraction over REAL chunk text. Pure parser is the
tested seam; the ollama call degrades to an empty result on any failure so the
structured graph is never broken by this optional layer."""
import logging
import re

try:
    import ollama
except Exception:                       # ollama not installed / importable
    ollama = None

_MODEL = 'qwen2.5:3b'                    # already pulled for the swarm — no new dep
_PROMPT = (
    'Extract named entities and relations from the Philippine economic text. '
    'Output ONLY lines in these exact formats, nothing else:\n'
    'ENTITY: <name> | <type>   (type one of: commodity, agency, place, policy, figure)\n'
    'REL: <a> -> <b> | <relation>\n'
)

_ENTITY_RE = re.compile(r'^\s*ENTITY:\s*(.+?)\s*\|\s*(.+?)\s*$')
_REL_RE = re.compile(r'^\s*REL:\s*(.+?)\s*->\s*(.+?)\s*\|\s*(.+?)\s*$')


def parse_extraction(text: str) -> dict:
    entities, relations = [], []
    for line in (text or '').splitlines():
        m = _ENTITY_RE.match(line)
        if m and m.group(1).strip():
            entities.append({'name': m.group(1).strip(), 'type': m.group(2).strip()})
            continue
        m = _REL_RE.match(line)
        if m and m.group(1).strip() and m.group(2).strip():
            relations.append({'a': m.group(1).strip(), 'b': m.group(2).strip(),
                              'kind': m.group(3).strip()})
    return {'entities': entities, 'relations': relations}


def extract(chunk_text: str, source: str = '', model: str = _MODEL) -> dict:
    if ollama is None or not (chunk_text or '').strip():
        return {'entities': [], 'relations': []}
    try:
        resp = ollama.chat(
            model=model,
            messages=[{'role': 'system', 'content': _PROMPT},
                      {'role': 'user', 'content': chunk_text[:1500]}],
            options={'num_predict': 256, 'temperature': 0.1},
        )
        return parse_extraction(resp['message']['content'])
    except Exception as exc:                      # noqa: BLE001 — must never raise
        logging.warning('entity extraction failed for %s: %s', source, exc)
        return {'entities': [], 'relations': []}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_entity_extractor.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**
```bash
git add ph_economic_ai/engine/entity_extractor.py ph_economic_ai/tests/test_entity_extractor.py
git commit -m "feat(engine): grounded EntityExtractor (pure parser + graceful ollama)"
```

---

## Task 3: `assemble.py` — build a full graph from plain swarm data

**Files:**
- Create: `ph_economic_ai/engine/kg_assemble.py`
- Test: `ph_economic_ai/tests/test_kg_assemble.py`

Plain-data inputs (no swarm/Qt coupling) so it's pure-testable; SP3b adapts the real `MasterVerdict`/agents/`rag` into these shapes.

- [ ] **Step 1: Write the failing test**

Create `ph_economic_ai/tests/test_kg_assemble.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.engine.kg_assemble import assemble_structured, apply_extraction


def _inputs():
    return dict(
        sources=['DOE', 'CRUDE'],
        data_inputs={'oil_price': 82.0, 'usd_php': 57.0},
        regionals=[{'region': 'NCR', 'estimate': -1.7}],
        agents=[{'name': 'FCST-NCR', 'role': 'Forecaster', 'region': 'NCR',
                 'estimate': -1.9, 'statement': 'easing'}],
        retrievals={'FCST-NCR': [{'source': 'DOE', 'idx': 1, 'text': 'diesel down'}]},
        master_estimate=-1.8,
    )


def test_assemble_structured_builds_real_graph():
    b = KnowledgeGraphBuilder()
    assemble_structured(b, **_inputs())
    nodes, edges = b.snapshot()
    kinds = {n.id: n.kind for n in nodes}
    assert kinds['master'] == 'master'
    assert kinds['judge:NCR'] == 'judge'
    assert kinds['agent:FCST-NCR'] == 'agent'
    assert kinds['ev:DOE#1'] == 'evidence'
    assert kinds['data:oil_price'] == 'data_input'
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('master', 'judge:NCR', 'aggregates') in ek
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek
    assert ('agent:FCST-NCR', 'ev:DOE#1', 'retrieved') in ek
    assert ('ev:DOE#1', 'src:DOE', 'from_source') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek


def test_apply_extraction_grounds_entities():
    b = KnowledgeGraphBuilder()
    ev = b.add_evidence('DOE', 1, 'diesel down')
    apply_extraction(b, ev, 'DOE', {
        'entities': [{'name': 'diesel', 'type': 'commodity'}],
        'relations': [{'a': 'diesel', 'b': 'pump', 'kind': 'drives'}],
    })
    assert b.node('ent:diesel').payload['provenance'][0] == {'chunk_id': ev, 'source': 'DOE'}
    ek = {(e.src, e.dst, e.kind) for e in b.edges_of('ent:diesel')}
    assert ('ent:diesel', ev, 'mentions') in ek
    assert ('ent:diesel', 'ent:pump', 'relates') in ek
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_kg_assemble.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement**

Create `ph_economic_ai/engine/kg_assemble.py`:
```python
"""Assemble a KnowledgeGraph from plain swarm data + fold in grounded extractions.
Inputs are lists/dicts (not swarm dataclasses) so this stays pure and testable;
SP3b adapts the real MasterVerdict / agents / rag into these shapes."""
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder


def assemble_structured(builder: KnowledgeGraphBuilder, *, sources, data_inputs,
                        regionals, agents, retrievals, master_estimate=None) -> None:
    builder.add_master(master_estimate)
    for s in sources:
        builder.add_source(s)
    for k, v in (data_inputs or {}).items():
        builder.add_data_input(k, v)
    for r in regionals:
        jid = builder.add_judge(r['region'], r.get('estimate'))
        builder.add_edge('master', jid, 'aggregates')
    for a in agents:
        aid = builder.add_agent(a['name'], a.get('role', ''), a.get('region', ''),
                                a.get('estimate'))
        region = a.get('region')
        if region:
            builder.add_edge(f'judge:{region}', aid, 'aggregates')
        if a.get('estimate') is not None:
            cid = builder.add_claim(aid, a['estimate'], a.get('statement', ''))
            for k in (data_inputs or {}):
                builder.add_edge(cid, f'data:{k}', 'references')
        for ev in retrievals.get(a['name'], []):
            evid = builder.add_evidence(ev['source'], ev['idx'], ev['text'],
                                        ev.get('url', ''))
            builder.add_edge(aid, evid, 'retrieved')


def apply_extraction(builder: KnowledgeGraphBuilder, chunk_id: str, source: str,
                     result: dict) -> None:
    for e in result.get('entities', []):
        builder.add_entity(e['name'], e.get('type', ''), chunk_id, source)
    for r in result.get('relations', []):
        builder.add_relation(r['a'], r['b'], r.get('kind', 'relates'))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_kg_assemble.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**
```bash
git add ph_economic_ai/engine/kg_assemble.py ph_economic_ai/tests/test_kg_assemble.py
git commit -m "feat(engine): assemble knowledge graph from swarm data + ground extractions"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (adds the new engine tests; no UI touched).

- [ ] **End-to-end engine smoke (no ollama needed)**

Run:
```bash
python -c "from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder; from ph_economic_ai.engine.kg_assemble import assemble_structured; b=KnowledgeGraphBuilder(); assemble_structured(b, sources=['DOE'], data_inputs={'oil_price':82}, regionals=[{'region':'NCR','estimate':-1.7}], agents=[{'name':'FCST','region':'NCR','estimate':-1.9}], retrievals={'FCST':[{'source':'DOE','idx':1,'text':'x'}]}, master_estimate=-1.8); n,e=b.snapshot(); print('nodes',len(n),'edges',len(e))"
```
Expected: prints node/edge counts > 0 — a real graph assembles with zero entity extraction (graceful-degradation path proven).

---

## Hand-off to SP3b

After this engine is on `master`, the next plan **SP3b — MiroFish canvas render + live wiring** will: (1) adapt the real `MasterVerdict` / `DEFAULT_AGENTS` / `rag` into the plain shapes above + record per-agent retrievals during the swarm run; (2) run `EntityExtractor` on a background worker, calling `apply_extraction` as results arrive; (3) re-render `stage3_swarm_canvas.py` MiroFish-style (organic cloud, type colours, red relationship fan on focus, Node Details showing provenance, legend, live metrics + console) from `builder.snapshot()`. SP3b is written against this concrete engine API + a full read of the canvas.

---

## Self-Review (completed by plan author)

**Spec coverage:** §3 data model (KGNode/KGEdge kinds, builder methods, idempotency, snapshot/node/edges_of) → Task 1. §4 extractor (pure `parse_extraction`, `extract` grounded + graceful) → Task 2. Assembler from real artifacts + `apply_extraction` provenance/mentions/relates → Task 3. §7 graceful degradation (structured graph assembles with zero entities) → Task 3 + final smoke. §2 `.gitignore` safety → Task 0. §5/§6 live wiring + render are explicitly deferred to SP3b (hand-off section).

**Placeholder scan:** none — every code/test step is complete.

**Type consistency:** node ids are stable strings used consistently — `src:<name>`, `ev:<source>#<idx>`, `agent:<name>`, `judge:<region>`, `claim:agent:<name>` (claim id = `f'claim:{agent_id}'` where `agent_id='agent:<name>'`), `data:<key>`, `ent:<lower-name>`, `master`. Task 3's `assemble_structured` builds `claim:agent:FCST-NCR` and the test asserts exactly that. `add_evidence` returns `ev:<source>#<idx>` and Task 3 passes that id to `retrieved`/`mentions` edges. `KGEdge` is `frozen` so `KGEdge(...) in edges` equality checks in Task 1's test work. `apply_extraction` uses `add_entity`/`add_relation` whose lowercase-id behaviour matches Task 1's provenance test. `extract` reads `resp['message']['content']` (the ollama chat shape used elsewhere in the codebase).
```
