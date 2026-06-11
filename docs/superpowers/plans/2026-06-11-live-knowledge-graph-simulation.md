# Live MiroFish Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Simulation screen the **live** MiroFish knowledge graph — it grows from the swarm's signals during the run, densifies with entities at the end, and stays on screen with a "View report →" button (no auto-jump).

**Architecture:** A pure `ui/kg_live.py` feeds a `KnowledgeGraphBuilder` from signal payloads (testable, no Qt). `Stage3SwarmPanel` shows the `KnowledgeGraphCanvas` as the live view, keeps the old `_SwarmCanvas` **constructed but hidden** (so trust badges / sector-typing / console keep working untouched), feeds the builder alongside its existing handlers, and refreshes the canvas on a ~1.5 s coalescing timer. `main_window` drops the auto-jump and wires a new `view_report_requested` signal; the SP3 post-run render stays as a guarded fallback.

**Tech Stack:** Python 3.10, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-11-live-knowledge-graph-simulation-design.md`.

**Confirmed anchors:**
- `engine/knowledge_graph.py`: `KnowledgeGraphBuilder` (`add_master/add_source/add_data_input/add_agent/add_judge/add_claim/add_evidence/add_edge/snapshot`; node ids `master`, `src:<n>`, `data:<k>`, `agent:<name>`, `judge:<region>`, `ev:<src>#<i>`).
- `engine/kg_swarm_adapter.py`: `_scenario_text(scenario)`, `_DATA_KEYS`.
- `engine/swarm.py`: `SwarmThread(rag, scenario, …)` → attrs `self._rag`, `self._scenario`; `build_swarm_agents(current_price) -> [SwarmAgent(name, role, region_name, rag_sources, …)]`; `AgentResponse(agent_name, statement, price_estimate, …)`; `RegionalVerdict(region_pair, estimate, …)`.
- `ui/stage3_swarm_canvas.py` `Stage3SwarmPanel`: `_build_main_row` (~1832) builds `self._canvas = _SwarmCanvas()` (visible) + `self._kg_canvas = KnowledgeGraphCanvas()` (hidden), both `addWidget(stretch=1)`; handlers `_on_group_round_done(gid, rnd, responses)` (~2148), `_on_regional_done(jid, verdict)` (~2174), `_on_swarm_complete(mv)` (~2184, ends `self.swarm_complete.emit(mv)`), sector `_on_food_agent_done(resp)`/`_on_elec_agent_done(resp)`; `connect_thread(thread)` (~2303); `reset()` (~2313); `show_knowledge_graph(builder)` (~2064, post-run + EntityExtractWorker). `KnowledgeGraphCanvas.set_snapshot(nodes, edges)`.
- `ui/main_window.py` `_on_swarm_complete` (~592): builds `kg = build_graph(...)` then `self._stage3_swarm.show_knowledge_graph(kg)` (~654) and navigates to the Report (`setCurrentIndex`/`set_active`, the index used for Report).

**Conventions:** Tests in `ph_economic_ai/tests/`, offscreen Qt. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`. Never add `self.show()`; tests use `not widget.isHidden()`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/live-knowledge-graph`

---

## Task 1: `ui/kg_live.py` — pure live-build helpers

**Files:** Create `ph_economic_ai/ui/kg_live.py`; Test `ph_economic_ai/tests/test_kg_live.py`

- [ ] **Step 1: Failing test** — create `ph_economic_ai/tests/test_kg_live.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from types import SimpleNamespace as NS
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder
from ph_economic_ai.ui import kg_live


class _Rag:
    all_source_names = ['DOE', 'CRUDE']
    def query(self, text, top_k=3, sources=None):
        return [{'source': (sources or ['DOE'])[0], 'text': 'diesel down', 'url': ''}]


def test_live_build_sequence():
    b = KnowledgeGraphBuilder()
    scenario = {'current_price': 60.0, 'oil_price': 82.0}
    kg_live.seed(b, ['DOE', 'CRUDE'], scenario)
    meta = {'FCST-NCR': NS(role='Forecaster', region_name='NCR', rag_sources=['DOE'])}
    resp = [NS(agent_name='FCST-NCR', statement='easing', price_estimate=-1.9)]
    kg_live.add_round(b, resp, meta, _Rag(), scenario)
    kg_live.add_round(b, resp, meta, _Rag(), scenario)            # idempotent re-emit
    kg_live.add_regional(b, ('NCR', 'CENTRAL LUZON'), -1.7, meta)
    kg_live.add_master(b, -1.8)
    nodes, edges = b.snapshot()
    ids = {n.id: n for n in nodes}
    assert 'src:DOE' in ids and 'data:oil_price' in ids
    assert ids['agent:FCST-NCR'].payload['estimate'] == -1.9
    assert ids['master'].payload['final_estimate'] == -1.8
    assert 'ev:DOE#0' in ids and ids['ev:DOE#0'].kind == 'evidence'
    ek = {(e.src, e.dst, e.kind) for e in edges}
    assert ('agent:FCST-NCR', 'ev:DOE#0', 'retrieved') in ek
    assert ('agent:FCST-NCR', 'claim:agent:FCST-NCR', 'claims') in ek
    assert ('judge:NCR', 'agent:FCST-NCR', 'aggregates') in ek
    assert ('master', 'judge:NCR', 'aggregates') in ek
    # idempotent: one evidence node, not two
    assert len([n for n in nodes if n.kind == 'evidence']) == 1


def test_sector_agent():
    b = KnowledgeGraphBuilder()
    kg_live.add_sector_agent(b, 'AGRI', 'food', -2.6, 'rice eases')
    n = b.node('agent:AGRI')
    assert n.cluster == 'food' and n.payload['estimate'] == -2.6
    assert any(e.kind == 'claims' for e in b.edges_of('agent:AGRI'))
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_kg_live.py -v`).

- [ ] **Step 3: Implement** — create `ph_economic_ai/ui/kg_live.py`:
```python
"""Pure helpers to build the knowledge graph LIVE from swarm signal payloads.
No Qt — the panel calls these from its thread handlers, then refreshes the canvas."""
from ph_economic_ai.engine.kg_swarm_adapter import _DATA_KEYS, _scenario_text


def seed(builder, sources, scenario: dict) -> None:
    """At run start: master placeholder + RAG sources + scenario data inputs."""
    builder.add_master(None)
    for s in (sources or []):
        builder.add_source(s)
    for k in _DATA_KEYS:
        v = (scenario or {}).get(k)
        if v is not None:
            builder.add_data_input(k, v)


def add_round(builder, responses, agent_meta, rag, scenario, top_k: int = 3) -> None:
    """A group round: each agent + its claim + its retrieved evidence (live rag)."""
    text = _scenario_text(scenario or {})
    for r in (responses or []):
        name = getattr(r, 'agent_name', None)
        if not name:
            continue
        meta = (agent_meta or {}).get(name)
        region = getattr(meta, 'region_name', '') if meta else ''
        est = getattr(r, 'price_estimate', None)
        aid = builder.add_agent(name, getattr(meta, 'role', '') if meta else '', region, est)
        if est is not None:
            builder.add_claim(aid, est, getattr(r, 'statement', ''))
        rs = getattr(meta, 'rag_sources', None) if meta else None
        try:
            chunks = rag.query(text, top_k=top_k, sources=rs) if rag is not None else []
        except Exception:
            chunks = []
        for i, c in enumerate(chunks or []):
            ev = builder.add_evidence(c.get('source', '?'), i, c.get('text', ''))
            builder.add_edge(aid, ev, 'retrieved')


def add_regional(builder, region_pair, estimate, agent_meta) -> None:
    """A regional judge: judge node + master->judge + judge->its agents."""
    pair = tuple(region_pair or ())
    key = pair[0] if pair else 'region'
    jid = builder.add_judge(key, estimate)
    builder.add_edge('master', jid, 'aggregates')
    for name, meta in (agent_meta or {}).items():
        if getattr(meta, 'region_name', None) in pair:
            builder.add_edge(jid, f'agent:{name}', 'aggregates')


def add_master(builder, final_estimate) -> None:
    builder.add_master(final_estimate)          # idempotent: merges the estimate in


def add_sector_agent(builder, name, sector, estimate, statement: str = '') -> None:
    aid = builder.add_agent(name, '', sector, estimate)
    if estimate is not None:
        builder.add_claim(aid, estimate, statement)
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`kg_live.py` + test) — `feat(ui): pure live knowledge-graph build helpers`.

---

## Task 2: Live wiring in `Stage3SwarmPanel`

**Files:** Modify `ph_economic_ai/ui/stage3_swarm_canvas.py`; Test `ph_economic_ai/tests/test_stage3_swarm.py` (append)

- [ ] **Step 1: Append the smoke test**
```python
def test_live_graph_grows_and_view_report(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel
    p = Stage3SwarmPanel()
    # seed as connect_thread would (without a real thread)
    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None): return [{'source': 'DOE', 'text': 'x'}]
    p._begin_live_graph(_Rag(), {'current_price': 60.0}, {})
    assert not p._kg_canvas.isHidden()                     # KG is the live view
    p._on_group_round_done(0, 1, [NS(agent_name='FCST', statement='s', price_estimate=-1.8)])
    p._flush_kg()
    assert p._kg_canvas.node_item_count() > 0              # graph grew
    fired = []
    p.view_report_requested.connect(lambda: fired.append(True))
    p._on_swarm_complete(NS(final_estimate=-1.8, confidence_pct=80, regional_verdicts=[],
                            dissenting_regions=[], all_responses=[]))
    assert not p._view_report_btn.isHidden()               # button revealed
    p._view_report_btn.click()
    assert fired == [True]
```

- [ ] **Step 2: Run → fails** (no `_begin_live_graph`/`view_report_requested`/`_view_report_btn` yet).

- [ ] **Step 3: Implement** — in `stage3_swarm_canvas.py`:

(a) Imports (top): `from ph_economic_ai.ui import kg_live as _kg_live`, `from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder`, `from ph_economic_ai.engine.swarm import build_swarm_agents`, and ensure `QTimer`, `QPushButton`, `pyqtSignal` are imported (they are used elsewhere in the file).

(b) Class signal (next to the existing `swarm_complete = pyqtSignal(object)`):
```python
    view_report_requested = pyqtSignal()
```

(c) In `_build_main_row`, after the `self._kg_canvas` lines, make the KG the visible view and add the button (initially hidden). Change `self._kg_canvas.setVisible(False)` to start hidden but flip in `_begin_live_graph`; add a small overlay button below the row is complex — instead add the button to the verdict sidebar or as a panel-level widget. Simplest: add it to the panel's main layout. In `__init__`/`_build` (where the main row is added to the panel layout), after adding the main row add:
```python
        self._view_report_btn = QPushButton('View report →')
        self._view_report_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border:none;border-radius:8px;'
            'padding:8px 16px;font-family:Consolas,monospace;font-size:11px;font-weight:700;}')
        self._view_report_btn.setVisible(False)
        self._view_report_btn.clicked.connect(self.view_report_requested.emit)
        <panel_layout>.addWidget(self._view_report_btn, alignment=Qt.AlignmentFlag.AlignRight)
```
(Use the actual panel layout variable from `_build`; read it.)

(d) In `__init__`/`_build`, initialise live state: `self._kg_builder = KnowledgeGraphBuilder()`, `self._agent_meta = {}`, `self._rag = None`, `self._scenario = {}`, `self._kg_dirty = False`, and a coalescing timer:
```python
        self._kg_refresh = QTimer(self)
        self._kg_refresh.setInterval(1500)
        self._kg_refresh.timeout.connect(self._flush_kg)
```

(e) Add the live-build methods:
```python
    def _begin_live_graph(self, rag, scenario, agent_meta):
        """Start a fresh live knowledge graph: seed it and show the KG canvas."""
        self._rag, self._scenario, self._agent_meta = rag, scenario or {}, agent_meta or {}
        self._kg_builder = KnowledgeGraphBuilder()
        try:
            srcs = list(getattr(rag, 'all_source_names', []) or [])
            _kg_live.seed(self._kg_builder, srcs, self._scenario)
        except Exception:
            pass
        self._canvas.setVisible(False)
        self._kg_canvas.setVisible(True)
        try:
            self._kg_canvas.node_clicked.connect(self._on_node_clicked)
        except Exception:
            pass
        self._view_report_btn.setVisible(False)
        self._flush_kg()
        if not self._kg_refresh.isActive():
            self._kg_refresh.start()

    def _flush_kg(self):
        try:
            self._kg_canvas.set_snapshot(*self._kg_builder.snapshot())
        except Exception:
            pass
        self._kg_dirty = False

    def has_live_graph(self) -> bool:
        try:
            return len(self._kg_builder.snapshot()[0]) > 3
        except Exception:
            return False
```

(f) Feed the builder from the existing handlers — at the END of each (keep all existing old-canvas/sidebar/console lines unchanged), add a wrapped live-build call + dirty flag:
- `_on_group_round_done(self, group_id, round_num, responses)` → append:
  ```python
        try:
            _kg_live.add_round(self._kg_builder, responses, self._agent_meta,
                               self._rag, self._scenario)
            self._kg_dirty = True
        except Exception:
            pass
  ```
- `_on_regional_done(self, judge_id, verdict)` → append:
  ```python
        try:
            _kg_live.add_regional(self._kg_builder, getattr(verdict, 'region_pair', ()),
                                  getattr(verdict, 'estimate', None), self._agent_meta)
            self._kg_dirty = True
        except Exception:
            pass
  ```
- `_on_food_agent_done(self, resp)` → append `try: _kg_live.add_sector_agent(self._kg_builder, resp.agent_name, 'food', resp.price_estimate, getattr(resp,'statement','')); self._kg_dirty = True\n except Exception: pass`.
- `_on_elec_agent_done(self, resp)` → same with `'elec'`.
- `_on_swarm_complete(self, master_verdict)` → before the final `self.swarm_complete.emit(master_verdict)`, append:
  ```python
        try:
            _kg_live.add_master(self._kg_builder, getattr(master_verdict, 'final_estimate', None))
            self._flush_kg()
            from ph_economic_ai.ui.kg_extract_worker import EntityExtractWorker
            self._kg_worker = EntityExtractWorker(self._kg_builder)
            self._kg_worker.progress.connect(lambda _i: self._kg_canvas.set_snapshot(*self._kg_builder.snapshot()))
            self._kg_worker.start()
            self._kg_refresh.stop()
            self._view_report_btn.setVisible(True)
        except Exception:
            pass
  ```

(g) `connect_thread(self, thread)` — at the top, capture rag/scenario + build agent meta + begin the live graph:
```python
        meta = {}
        try:
            price = (getattr(thread, '_scenario', {}) or {}).get('current_price', 0.0)
            meta = {a.name: a for a in build_swarm_agents(price)}
        except Exception:
            pass
        self._begin_live_graph(getattr(thread, '_rag', None),
                               getattr(thread, '_scenario', {}), meta)
```
(then the existing `thread.agent_typing.connect(...)` etc. unchanged).

(h) `reset()` — append: `self._kg_builder = KnowledgeGraphBuilder(); self._kg_dirty = False; self._kg_refresh.stop()` and `if hasattr(self, '_view_report_btn'): self._view_report_btn.setVisible(False)`.

- [ ] **Step 4: Run → passes**: `python -m pytest ph_economic_ai/tests/test_stage3_swarm.py -v`.

- [ ] **Step 5: Commit** (`stage3_swarm_canvas.py` + `test_stage3_swarm.py`) — `feat(ui): live MiroFish knowledge graph as the simulation view`.

---

## Task 3: `main_window` — stop the auto-jump, wire "View report →"

**Files:** Modify `ph_economic_ai/ui/main_window.py`; Test `ph_economic_ai/tests/test_main_window.py` (append)

- [ ] **Step 1: Append the smoke test**
```python
def test_completion_stays_on_simulation_then_button_navigates(window):
    from types import SimpleNamespace as NS
    # the Report stack index is where the workbench lives (3)
    window._stack.setCurrentIndex(2)               # on Simulation
    # simulate the panel asking to view the report
    window._stage3_swarm.view_report_requested.emit()
    assert window._stack.currentIndex() == 3        # navigated to Report on demand
```

- [ ] **Step 2: Run → fails** (no `view_report_requested` wiring yet) — or passes only after Step 3.

- [ ] **Step 3: Implement** — in `main_window.py`:
- In `_on_swarm_complete`, change the post-run KG build to a **guarded fallback** and remove the auto-navigation: keep building `kg` but only call `show_knowledge_graph` if the panel has no live graph, and delete the `set_active(<report>)` / `setCurrentIndex(<report>)` lines from this method:
  ```python
          try:
              from ph_economic_ai.engine.kg_swarm_adapter import build_graph
              if not self._stage3_swarm.has_live_graph():
                  price = self._last_scenario.get('current_price', 0.0)
                  agents = build_swarm_agents(price)
                  kg = build_graph(master_verdict, agents, self._last_scenario, self._rag)
                  self._stage3_swarm.show_knowledge_graph(kg)
          except Exception as exc:
              import logging; logging.warning('knowledge graph fallback failed: %s', exc)
  ```
  (Remove ONLY the Report-navigation lines in `_on_swarm_complete`; leave the rest — sector debate kickoff, sidebar unlocks, etc. — intact. The food/elec debates must still start.)
- Where `self._stage3_swarm` is created/connected (near `self._stage3_swarm.swarm_complete.connect(self._on_swarm_complete)`), add:
  ```python
          self._stage3_swarm.view_report_requested.connect(self._goto_report)
  ```
  and add a small method reusing today's report-nav logic:
  ```python
      def _goto_report(self):
          self._sidebar.unlock_stages([2, 3])
          self._sidebar.set_active(3)
          self._stack.setCurrentIndex(3)
  ```
  (Match the exact unlock/index values the old `_on_swarm_complete` used for the Report.)

- [ ] **Step 4: Run → passes**: `python -m pytest ph_economic_ai/tests/test_main_window.py -v`.

- [ ] **Step 5: Commit** (`main_window.py` + `test_main_window.py`) — `feat(ui): stay on the live graph at completion; 'View report' navigates`.

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): start a run → the Simulation **is** the knowledge graph, growing as agents act (sources → agents → claims → evidence → judges → master), food/elec clusters included; at completion it stays on the (now entity-dense) graph with a "View report →" button that opens the Report.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 `kg_live` pure helpers → Task 1; §3.2 panel (KG as live view, builder + ~1.5s timer, seed in connect_thread, handlers feed builder incl. food/elec, `view_report_requested` + button, `has_live_graph`, finalize+entity worker on complete) → Task 2; §3.3 main_window (drop auto-jump, wire button to Report, guarded SP3 fallback) → Task 3; §5 robustness (every live call wrapped; fallback if live empty) → Tasks 2–3; §6 testing → all tasks.
**Placeholder scan:** none — `kg_live.py` complete; Task 2 gives exact append-points per handler + the new methods; Task 3 gives the guarded fallback + nav method. The two `<panel_layout>` / `<report index>` spots are explicitly "read the real value" (the panel's `_build` layout var; the Report index the old `_on_swarm_complete` used = 3) — concrete, not vague.
**Type consistency:** `kg_live.*` operate on `KnowledgeGraphBuilder` (same ids as SP3). `add_round` reads `resp.agent_name/price_estimate/statement` (real `AgentResponse`) + `agent_meta[name].region_name/role/rag_sources` (real `SwarmAgent`). Region link: `add_agent(region=region_name)` + `add_regional` joins `judge:<pair[0]>`→agents whose `region_name in pair` — keys line up. The old `_canvas` stays constructed (all existing `self._canvas.*`/trust-badge/sector-typing calls keep working); only its visibility flips off. `has_live_graph` gates the main_window fallback so the graph always renders. `view_report_requested` (panel) → `_goto_report` (main_window) reuses the Report index/unlocks from the old auto-jump.
