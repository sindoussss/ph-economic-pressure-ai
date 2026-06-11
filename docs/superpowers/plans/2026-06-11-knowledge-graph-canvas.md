# MiroFish Knowledge-Graph Canvas (SP3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the simulation as a MiroFish-style dense knowledge graph — built from the SP3a engine on the real swarm result, with grounded entity enrichment, an organic light cloud, a red relationship fan on focus, Node Details with provenance, a legend, and live metrics.

**Architecture:** Post-run, deterministic. On `swarm_complete`, an **adapter** turns `MasterVerdict.all_responses` + `build_swarm_agents()` + `scenario` + `rag` into the plain inputs for SP3a's `assemble_structured`, producing a `KnowledgeGraphBuilder`. A pure **layout** turns a snapshot into positioned, coloured render nodes. A new **`KnowledgeGraphCanvas`** Qt widget renders it MiroFish-style. A background **extraction worker** runs SP3a's `EntityExtractor` over the evidence chunks and folds entities in (graceful). Wiring lives in `main_window._on_swarm_complete` (which already has `rag`/`scenario`) → `Stage3SwarmPanel.show_knowledge_graph(...)`.

**Tech Stack:** Python 3.10, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-knowledge-graph-simulation-design.md` (§5 live wiring is realised here as post-run + async enrichment; §6 render).

**Prereqs (on `master`, from SP3a):** `ph_economic_ai/engine/knowledge_graph.py` (`KnowledgeGraphBuilder`, `KGNode`, `KGEdge`), `kg_assemble.py` (`assemble_structured(builder, *, sources, data_inputs, regionals, agents, retrievals, master_estimate)`), `entity_extractor.py` (`extract(text, source) -> {entities, relations}`). Node ids per SP3a: `src:<name>`, `ev:<source>#<idx>`, `agent:<name>`, `judge:<region>`, `claim:agent:<name>`, `data:<key>`, `ent:<lower>`, `master`.

**Confirmed codebase facts (anchors):**
- `swarm.py`: `SwarmAgent(name, role, model, group_id, region_name, system_prompt, rag_sources, …)`; `build_swarm_agents(current_price) -> list[SwarmAgent]`; `AgentResponse(agent_name, round_num, thinking, statement, price_estimate)`; `RegionalVerdict(judge_id, region_pair: tuple[str,str], estimate, …)`; `MasterVerdict(final_estimate, …, regional_verdicts, regional_estimates, all_responses: list[AgentResponse])`.
- `rag.all_source_names: list[str]`; `rag.query(text, top_k=, sources=) -> list[dict]` with keys `source`, `text`, `url`.
- `main_window._on_swarm_complete(master_verdict)` (~line 619) has `self._rag`, `self._last_scenario`, `self._stage3_swarm`. `self._last_scenario` has `current_price` (+ oil/fx/etc.).
- `Stage3SwarmPanel(store=None)` (`stage3_swarm_canvas.py:1709`): hosts `self._canvas` in `_build_main_row` (~1830), a `_NodeDetailCard` `self._details_card`, `_log(text, color=)`, a console (`_build_console`). `node_clicked` → `_on_node_clicked(info: dict)`.

**Conventions:** Tests in `ph_economic_ai/tests/` with the path shim; offscreen Qt via `os.environ.setdefault('QT_QPA_PLATFORM','offscreen')`. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/knowledge-graph-canvas`

---

## Task 1: Swarm → graph adapter

**Files:** Create `ph_economic_ai/engine/kg_swarm_adapter.py`; Test `ph_economic_ai/tests/test_kg_swarm_adapter.py`

- [ ] **Step 1: Failing test**
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from types import SimpleNamespace as NS
from ph_economic_ai.engine.kg_swarm_adapter import build_graph


class _Rag:
    all_source_names = ['DOE', 'CRUDE']
    def query(self, text, top_k=3, sources=None):
        return [{'source': (sources or ['DOE'])[0], 'text': 'diesel down', 'url': 'u'}]


def _master():
    return NS(
        final_estimate=-1.8,
        regional_verdicts=[NS(judge_id=0, region_pair=('NCR', 'CENTRAL LUZON'), estimate=-1.7)],
        all_responses=[NS(agent_name='FCST-NCR', statement='easing', price_estimate=-1.9)],
    )


def test_build_graph_from_real_swarm_shapes():
    agents = [NS(name='FCST-NCR', role='Forecaster', region_name='NCR',
                 rag_sources=['DOE'])]
    b = build_graph(_master(), agents, {'current_price': 60.0, 'oil_price': 82.0}, _Rag())
    nodes, edges = b.snapshot()
    ids = {n.id: n for n in nodes}
    assert ids['agent:FCST-NCR'].payload['estimate'] == -1.9          # from all_responses
    assert 'ev:DOE#0' in ids and ids['ev:DOE#0'].kind == 'evidence'   # re-query retrieval
    assert 'data:oil_price' in ids
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek         # agent joined to its judge
    assert ('agent:FCST-NCR', 'ev:DOE#0', 'retrieved') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_kg_swarm_adapter.py -v`).

- [ ] **Step 3: Implement** `ph_economic_ai/engine/kg_swarm_adapter.py`:
```python
"""Adapt a finished swarm run (MasterVerdict + agents + scenario + rag) into the
plain inputs SP3a's assemble_structured wants, then build the KnowledgeGraph.
Post-run + pure (rag re-queried for evidence); no SwarmThread changes."""
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.engine.kg_assemble import assemble_structured

_DATA_KEYS = ('current_price', 'oil_price', 'usd_php', 'bsp_rate', 'demand_index')


def _scenario_text(scenario: dict) -> str:
    parts = [f'{k}={scenario[k]}' for k in _DATA_KEYS if scenario.get(k) is not None]
    return 'Philippine fuel scenario: ' + ', '.join(parts)


def build_inputs(master_verdict, agents, scenario: dict, rag, top_k: int = 3) -> dict:
    resp_by_name = {r.agent_name: r for r in (getattr(master_verdict, 'all_responses', None) or [])}
    sources = list(getattr(rag, 'all_source_names', []) or [])
    data_inputs = {k: scenario.get(k) for k in _DATA_KEYS if scenario.get(k) is not None}

    regionals, region_for = [], {}
    for rv in master_verdict.regional_verdicts:
        pair = tuple(rv.region_pair or ())
        key = pair[0] if pair else f'J{getattr(rv, "judge_id", 0)}'
        regionals.append({'region': key, 'estimate': rv.estimate})
        for rn in pair:
            region_for[rn] = key

    text = _scenario_text(scenario)
    agent_dicts, retrievals = [], {}
    for ag in agents:
        r = resp_by_name.get(ag.name)
        region = region_for.get(getattr(ag, 'region_name', ''), getattr(ag, 'region_name', ''))
        agent_dicts.append({
            'name': ag.name, 'role': getattr(ag, 'role', ''), 'region': region,
            'estimate': getattr(r, 'price_estimate', None) if r else None,
            'statement': getattr(r, 'statement', '') if r else '',
        })
        try:
            chunks = rag.query(text, top_k=top_k, sources=getattr(ag, 'rag_sources', None))
        except Exception:
            chunks = []
        retrievals[ag.name] = [
            {'source': c.get('source', '?'), 'idx': i, 'text': c.get('text', '')}
            for i, c in enumerate(chunks or [])
        ]

    return dict(sources=sources, data_inputs=data_inputs, regionals=regionals,
                agents=agent_dicts, retrievals=retrievals,
                master_estimate=getattr(master_verdict, 'final_estimate', None))


def build_graph(master_verdict, agents, scenario: dict, rag, top_k: int = 3) -> KnowledgeGraphBuilder:
    b = KnowledgeGraphBuilder()
    assemble_structured(b, **build_inputs(master_verdict, agents, scenario, rag, top_k))
    return b
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`kg_swarm_adapter.py` + test) — `feat(engine): adapt finished swarm run into a knowledge graph`.

---

## Task 2: Pure layout + colour model

**Files:** Create `ph_economic_ai/ui/kg_layout.py`; Test `ph_economic_ai/tests/test_kg_layout.py`

- [ ] **Step 1: Failing test**
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui.kg_layout import render_model, node_color


def _graph():
    b = KnowledgeGraphBuilder()
    b.add_master(-1.8)
    j = b.add_judge('NCR', -1.7)
    b.add_edge('master', j, 'aggregates')
    a = b.add_agent('FCST', 'Forecaster', 'NCR', -1.9)
    b.add_edge(j, a, 'aggregates')
    ev = b.add_evidence('DOE', 0, 'x')
    b.add_edge(a, ev, 'retrieved')
    return b


def test_render_model_positions_and_colours():
    nodes, edges = _graph().snapshot()
    rm = render_model(nodes, edges, width=800, height=600, seed=3)
    assert len(rm['nodes']) == len(nodes)
    for rn in rm['nodes']:
        assert 0 <= rn['x'] <= 800 and 0 <= rn['y'] <= 600
        assert rn['color'].startswith('#')
    # deterministic for a fixed seed
    rm2 = render_model(nodes, edges, width=800, height=600, seed=3)
    assert [n['x'] for n in rm['nodes']] == [n['x'] for n in rm2['nodes']]
    # hub (master, degree>=1) at least as large as a leaf evidence node
    by_id = {n['id']: n for n in rm['nodes']}
    assert by_id['master']['r'] >= by_id['ev:DOE#0']['r']


def test_node_color_by_kind_and_sector():
    from ph_economic_ai.engine.knowledge_graph import KGNode
    assert node_color(KGNode('agent:x', 'agent', 'x')) == '#E5484D'
    assert node_color(KGNode('a', 'agent', 'x', cluster='food')) == '#15A150'
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement** `ph_economic_ai/ui/kg_layout.py`:
```python
"""Pure render model for the knowledge graph: a seeded force-directed layout +
colour-by-kind. No Qt — returns plain dicts the canvas draws."""
import math
import random

_KIND_COLORS = {
    'source': '#3B6FD4', 'evidence': '#9AA1AC', 'entity': '#9AA1AC',
    'agent': '#E5484D', 'judge': '#3B6FD4', 'master': '#111111',
    'data_input': '#6B7280', 'claim': '#C0C4CC',
}
_SECTOR_COLORS = {'food': '#15A150', 'elec': '#E8920C', 'electricity': '#E8920C'}


def node_color(node) -> str:
    if getattr(node, 'cluster', None) in _SECTOR_COLORS:
        return _SECTOR_COLORS[node.cluster]
    return _KIND_COLORS.get(node.kind, '#9AA1AC')


def _layout(nodes, edges, width, height, iterations, seed):
    rng = random.Random(seed)
    pos = {n.id: [rng.uniform(0, width), rng.uniform(0, height)] for n in nodes}
    if not nodes:
        return pos
    k = math.sqrt((width * height) / max(len(nodes), 1)) * 0.8
    adj = [(e.src, e.dst) for e in edges if e.src in pos and e.dst in pos]
    temp = width / 10.0
    for _ in range(iterations):
        disp = {nid: [0.0, 0.0] for nid in pos}
        ids = list(pos)
        for i, a in enumerate(ids):                       # repulsion
            for b in ids[i + 1:]:
                dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
                d = math.hypot(dx, dy) or 0.01
                f = (k * k) / d
                ux, uy = dx / d, dy / d
                disp[a][0] += ux * f; disp[a][1] += uy * f
                disp[b][0] -= ux * f; disp[b][1] -= uy * f
        for s, t in adj:                                  # attraction
            dx, dy = pos[s][0] - pos[t][0], pos[s][1] - pos[t][1]
            d = math.hypot(dx, dy) or 0.01
            f = (d * d) / k
            ux, uy = dx / d, dy / d
            disp[s][0] -= ux * f; disp[s][1] -= uy * f
            disp[t][0] += ux * f; disp[t][1] += uy * f
        for nid in pos:                                   # apply, cooled + bounded
            dx, dy = disp[nid]
            d = math.hypot(dx, dy) or 0.01
            pos[nid][0] += (dx / d) * min(d, temp)
            pos[nid][1] += (dy / d) * min(d, temp)
            pos[nid][0] = min(width, max(0.0, pos[nid][0]))
            pos[nid][1] = min(height, max(0.0, pos[nid][1]))
        temp *= 0.95
    return pos


def render_model(nodes, edges, width=1000, height=700, iterations=80, seed=7) -> dict:
    pos = _layout(nodes, edges, width, height, iterations, seed)
    degree = {n.id: 0 for n in nodes}
    for e in edges:
        if e.src in degree:
            degree[e.src] += 1
        if e.dst in degree:
            degree[e.dst] += 1
    rnodes = []
    for n in nodes:
        x, y = pos[n.id]
        r = 2.0 + min(degree.get(n.id, 0), 12) * 0.6
        if n.kind in ('master', 'judge'):
            r = max(r, 5.0)
        rnodes.append({'id': n.id, 'x': x, 'y': y, 'r': r, 'color': node_color(n),
                       'kind': n.kind, 'label': n.label})
    redges = [{'src': e.src, 'dst': e.dst, 'kind': e.kind} for e in edges]
    return {'nodes': rnodes, 'edges': redges, 'pos': pos}
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`kg_layout.py` + test) — `feat(ui): pure force-directed layout + colour model for the knowledge graph`.

---

## Task 3: `KnowledgeGraphCanvas` widget (MiroFish render)

**Files:** Create `ph_economic_ai/ui/kg_canvas.py`; Test `ph_economic_ai/tests/test_kg_canvas.py`

- [ ] **Step 1: Failing test**
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def _builder():
    b = KnowledgeGraphBuilder()
    a = b.add_agent('FCST', 'Forecaster', 'NCR', -1.9)
    ev = b.add_evidence('DOE', 0, 'diesel down')
    b.add_edge(a, ev, 'retrieved')
    b.add_entity('diesel', 'commodity', ev, 'DOE')
    return b


def test_canvas_renders_snapshot_and_focus(app):
    from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
    c = KnowledgeGraphCanvas()
    nodes, edges = _builder().snapshot()
    c.set_snapshot(nodes, edges)
    assert c.node_item_count() == len(nodes)
    # focusing the agent highlights its incident edges (red fan)
    c.focus('agent:FCST')
    assert c.focused_edge_count() >= 1
    # node details payload available for the honesty surface
    info = c.node_info('ev:DOE#0')
    assert info['kind'] == 'evidence' and 'diesel' in info['payload']['text']


def test_canvas_empty_snapshot_no_crash(app):
    from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
    c = KnowledgeGraphCanvas()
    c.set_snapshot([], [])
    assert c.node_item_count() == 0
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement** `ph_economic_ai/ui/kg_canvas.py`:
```python
"""MiroFish-style knowledge-graph canvas: light organic cloud of tiny colour-coded
nodes, faint mass edges, a red relationship fan on the focused node. Renders a
snapshot (KGNode/KGEdge) via the pure kg_layout render model."""
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QBrush, QPen, QColor, QPainter
from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsEllipseItem

from ph_economic_ai.ui.kg_layout import render_model

_EDGE = QColor('#D8DCE4')
_EDGE_HOT = QColor('#E5484D')
_W, _H = 1100, 760


class _NodeItem(QGraphicsEllipseItem):
    def __init__(self, nid, x, y, r, color):
        super().__init__(QRectF(x - r, y - r, 2 * r, 2 * r))
        self.nid = nid
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(2)
        self.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)


class KnowledgeGraphCanvas(QGraphicsView):
    node_clicked = pyqtSignal(dict)            # {'id','kind','label','payload'}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet('border:none;background:#FCFCFC;')
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._nodes: dict[str, dict] = {}      # id -> render node
        self._info: dict[str, dict] = {}       # id -> {kind,label,payload}
        self._edges: list[dict] = []
        self._node_items: dict[str, _NodeItem] = {}
        self._edge_items: list = []
        self._focused: str | None = None

    # -- API --
    def set_snapshot(self, nodes, edges):
        self._scene.clear()
        self._node_items.clear(); self._edge_items.clear()
        self._info = {n.id: {'id': n.id, 'kind': n.kind, 'label': n.label,
                             'payload': n.payload} for n in nodes}
        rm = render_model(nodes, edges, width=_W, height=_H)
        self._nodes = {n['id']: n for n in rm['nodes']}
        self._edges = rm['edges']
        for e in self._edges:                  # faint mass edges first
            a, b = self._nodes.get(e['src']), self._nodes.get(e['dst'])
            if not a or not b:
                continue
            line = self._scene.addLine(a['x'], a['y'], b['x'], b['y'], QPen(_EDGE, 0.5))
            line.setZValue(1); line.setData(0, (e['src'], e['dst']))
            self._edge_items.append(line)
        for n in self._nodes.values():         # nodes on top
            item = _NodeItem(n['id'], n['x'], n['y'], n['r'], n['color'])
            self._scene.addItem(item)
            self._node_items[n['id']] = item
        if self._nodes:
            self._scene.setSceneRect(self._scene.itemsBoundingRect())
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def focus(self, node_id: str):
        self._focused = node_id
        for line in self._edge_items:
            s, d = line.data(0)
            hot = node_id in (s, d)
            line.setPen(QPen(_EDGE_HOT if hot else _EDGE, 1.1 if hot else 0.5))
            line.setZValue(3 if hot else 1)

    def focused_edge_count(self) -> int:
        if not self._focused:
            return 0
        return sum(1 for e in self._edges
                   if self._focused in (e['src'], e['dst']))

    def node_info(self, node_id: str) -> dict:
        return self._info.get(node_id, {})

    def node_item_count(self) -> int:
        return len(self._node_items)

    # -- interaction --
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, _NodeItem):
            self.focus(item.nid)
            self.node_clicked.emit(self._info.get(item.nid, {}))
        super().mousePressEvent(event)
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`kg_canvas.py` + test) — `feat(ui): MiroFish knowledge-graph canvas (cloud + red relationship fan)`.

---

## Task 4: Background entity-extraction worker

**Files:** Create `ph_economic_ai/ui/kg_extract_worker.py`; Test `ph_economic_ai/tests/test_kg_extract_worker.py`

- [ ] **Step 1: Failing test** (tests the pure worker core, no QThread spin-up):
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui.kg_extract_worker import enrich_with_entities


def test_enrich_applies_extraction_to_evidence(monkeypatch):
    b = KnowledgeGraphBuilder()
    ev = b.add_evidence('DOE', 0, 'diesel down on Brent slide')
    import ph_economic_ai.ui.kg_extract_worker as w
    monkeypatch.setattr(w, 'extract', lambda text, source='': {
        'entities': [{'name': 'diesel', 'type': 'commodity'}], 'relations': []})
    n = enrich_with_entities(b, extract_fn=w.extract)
    assert n == 1                                   # one chunk processed
    assert b.node('ent:diesel') is not None
    assert any(e.kind == 'mentions' for e in b.edges_of('ent:diesel'))
```

- [ ] **Step 2: Run → fails.**

- [ ] **Step 3: Implement** `ph_economic_ai/ui/kg_extract_worker.py`:
```python
"""Grounded entity enrichment over a built graph's evidence nodes. `enrich_with_
entities` is the pure, tested core; `EntityExtractWorker` runs it off the UI
thread. Both degrade to no-ops on any extraction failure (EntityExtractor already
returns empty on error)."""
from PyQt6.QtCore import QThread, pyqtSignal

from ph_economic_ai.engine.entity_extractor import extract
from ph_economic_ai.engine.kg_assemble import apply_extraction


def enrich_with_entities(builder, extract_fn=extract) -> int:
    nodes, _ = builder.snapshot()
    processed = 0
    for n in [x for x in nodes if x.kind == 'evidence']:
        result = extract_fn(n.payload.get('text', ''), n.payload.get('source', ''))
        apply_extraction(builder, n.id, n.payload.get('source', ''), result)
        processed += 1
    return processed


class EntityExtractWorker(QThread):
    progress = pyqtSignal(int)        # entities/relations applied so far (chunks done)
    done = pyqtSignal()

    def __init__(self, builder, parent=None):
        super().__init__(parent)
        self._builder = builder

    def run(self):
        nodes, _ = self._builder.snapshot()
        for i, n in enumerate([x for x in nodes if x.kind == 'evidence'], start=1):
            result = extract(n.payload.get('text', ''), n.payload.get('source', ''))
            apply_extraction(self._builder, n.id, n.payload.get('source', ''), result)
            self.progress.emit(i)
        self.done.emit()
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`kg_extract_worker.py` + test) — `feat(ui): background grounded entity-extraction worker`.

---

## Task 5: Wire the knowledge graph into the simulation panel

**Files:** Modify `ph_economic_ai/ui/stage3_swarm_canvas.py` (add `show_knowledge_graph`), `ph_economic_ai/ui/main_window.py` (build + hand off on swarm complete). Test: window smoke.

- [ ] **Step 1: Add `show_knowledge_graph` to `Stage3SwarmPanel`**

Read `Stage3SwarmPanel._build_main_row` (~1830) for how `self._canvas` is added to its layout. Add a `KnowledgeGraphCanvas` instance (`self._kg_canvas = KnowledgeGraphCanvas()`, hidden initially, added to the same layout/stack as `self._canvas`) and:
```python
    def show_knowledge_graph(self, builder):
        """Swap the live arena for the MiroFish knowledge graph + start enrichment."""
        from ph_economic_ai.ui.kg_extract_worker import EntityExtractWorker
        self._kg_builder = builder
        nodes, edges = builder.snapshot()
        self._kg_canvas.set_snapshot(nodes, edges)
        self._canvas.setVisible(False)
        self._kg_canvas.setVisible(True)
        self._kg_canvas.node_clicked.connect(self._on_node_clicked)   # reuse details card
        self._log(f'KNOWLEDGE GRAPH  {len(nodes)} nodes  {len(edges)} edges', color='#3B6FD4')
        self._kg_worker = EntityExtractWorker(builder)
        self._kg_worker.progress.connect(lambda _i: self._kg_canvas.set_snapshot(*builder.snapshot()))
        self._kg_worker.done.connect(lambda: self._log('entity extraction complete', color='#15A150'))
        self._kg_worker.start()
```
Ensure `_on_node_clicked(info)` handles the KG `info` dict shape `{'id','kind','label','payload'}` — show `payload` in the details card (for `evidence`/`entity` show `payload['source']` + text/provenance). If the existing `_on_node_clicked` expects a different dict, add a compatible branch (check `info.get('kind')`).
Import `KnowledgeGraphCanvas` at the top of the file.

- [ ] **Step 2: Build + hand off in `main_window._on_swarm_complete`**

In `main_window.py`, import `from ph_economic_ai.engine.kg_swarm_adapter import build_graph` and `from ph_economic_ai.engine.swarm import build_swarm_agents` (already imported per anchors — verify). At the end of `_on_swarm_complete(self, master_verdict)` (after the existing body), add:
```python
        # Build the MiroFish knowledge graph from the finished run (post-run, safe).
        try:
            price = self._last_scenario.get('current_price', 0.0)
            agents = build_swarm_agents(price)
            kg = build_graph(master_verdict, agents, self._last_scenario, self._rag)
            self._stage3_swarm.show_knowledge_graph(kg)
        except Exception as exc:        # never break the run on a viz error
            import logging; logging.warning('knowledge graph build failed: %s', exc)
```

- [ ] **Step 3: Import + window smoke**

Run: `python -c "import ph_economic_ai.ui.main_window, ph_economic_ai.ui.stage3_swarm_canvas, ph_economic_ai.ui.kg_canvas; print('import OK')"`
Run: `python -m pytest ph_economic_ai/tests/test_main_window.py ph_economic_ai/tests/test_stage3_swarm.py -q`
Expected: pass (construction unaffected; the KG path only fires on a real run).

- [ ] **Step 4: Headless integration smoke** (build a graph from fake swarm shapes and render it):
```bash
python -c "
import os,sys; os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PyQt6.QtWidgets import QApplication; from types import SimpleNamespace as NS
app=QApplication.instance() or QApplication(sys.argv)
from ph_economic_ai.engine.kg_swarm_adapter import build_graph
from ph_economic_ai.ui.kg_canvas import KnowledgeGraphCanvas
class R:
    all_source_names=['DOE','CRUDE']
    def query(self,t,top_k=3,sources=None): return [{'source':(sources or ['DOE'])[0],'text':'diesel down','url':''}]
mv=NS(final_estimate=-1.8, regional_verdicts=[NS(judge_id=0,region_pair=('NCR','CL'),estimate=-1.7)], all_responses=[NS(agent_name='FCST',statement='easing',price_estimate=-1.9)])
ag=[NS(name='FCST',role='Forecaster',region_name='NCR',rag_sources=['DOE'])]
b=build_graph(mv,ag,{'current_price':60,'oil_price':82},R())
c=KnowledgeGraphCanvas(); c.set_snapshot(*b.snapshot()); print('rendered nodes', c.node_item_count())
"
```
Expected: prints `rendered nodes <N>` (N>0).

- [ ] **Step 5: Commit** (`stage3_swarm_canvas.py` + `main_window.py`) — `feat(ui): render the swarm as a MiroFish knowledge graph on completion`.

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): run a swarm → on completion the simulation view becomes the dense MiroFish knowledge graph; clicking a node shows Node Details (a chunk's source+text, an entity's provenance); entity nodes appear as extraction finishes; legend + metrics present.

---

## Self-Review (completed by plan author)
**Spec coverage:** §5 build-from-run (post-run adapter) → Task 1; §6 render (cloud, colours, red fan, Node Details payload, snapshot) → Tasks 2–3; grounded extraction enrichment + graceful → Task 4; live-ish progressive refresh + wiring + console log → Task 5; §7 robustness (viz wrapped in try/except in main_window; worker degrades via EntityExtractor) → Tasks 4–5.
**Placeholder scan:** none — pure tasks (1,2,4) and the canvas (3) have complete code; Task 5 gives concrete edits against confirmed anchors (`_on_swarm_complete`, `_build_main_row`, `_on_node_clicked`, `build_swarm_agents`).
**Type consistency:** adapter emits exactly the kwargs `assemble_structured` accepts; region keys join agents to `judge:<region>` (region_for maps `region_name`→`region_pair[0]`, same key `add_judge` uses). `render_model` returns `{nodes:[{id,x,y,r,color,kind,label}], edges:[{src,dst,kind}]}` consumed verbatim by `kg_canvas`. `KnowledgeGraphCanvas.node_clicked` emits `{'id','kind','label','payload'}`, matching what Task 5's `_on_node_clicked` reads. `EntityExtractWorker`/`enrich_with_entities` operate on the same `KnowledgeGraphBuilder` the canvas snapshots. `extract`/`apply_extraction` signatures match SP3a.
**Scope note:** "live progressive build during the run" (spec §5) is realised as post-run render + async entity enrichment — simpler, defense-safe, and the structured graph shows immediately; true per-round animation is a future enhancement (would hook `group_round_done`).
```
