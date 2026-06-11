# Evidence-Populated Simulation Canvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the old structured simulation canvas (`_SwarmCanvas`) visibly *populated* by hanging each agent's REAL retrieved RAG chunks off it as small satellite nodes (~37 → ~100+ nodes), keeping its labelled "line vibe", every added node clickable to its source + text.

**Architecture:** Add a lightweight `_EvidenceNode` QGraphics item + a `_SwarmCanvas.add_evidence_layer(rag, scenario)` method that queries the in-memory rag per agent and drops `top_k` satellite dots (with faint edges) around each agent. The panel calls it from `connect_thread`. Evidence clicks reuse the panel's existing `node_clicked` → `_on_node_clicked` `kind=='evidence'` rendering. Purely additive — the existing clusters/agents/judges/RAG/animations are untouched.

**Tech Stack:** Python 3.10, PyQt6, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-12-evidence-populated-simulation-design.md`.

**Confirmed anchors (`ui/stage3_swarm_canvas.py`):**
- `_AgentNode(name, role, group_id, region, rag_sources)` stores `self._rag_sources`; node has `.pos()`. Class pattern: `clicked = pyqtSignal(str)`, `QGraphicsObject`, `paint`/`boundingRect`/`mousePressEvent`.
- `_RagNode` (≈508) is the small-node reference; `_Edge` (≈151) has `.set_path_between(x1,y1,x2,y2)` + `.set_state('dead')` (faint dotted).
- `_SwarmCanvas` (≈799): `node_clicked = pyqtSignal(dict)` (≈800); `_build` creates agents into `self._agents[name] = node` (≈918-926); `_emit_rag_click(source)` (≈1166) emits `{'type':'rag','source':source}`; `_scatter(seed_key, anchor, radius, min_r=0)` (≈125) returns an (x,y) jittered around `anchor`.
- Panel `_on_node_clicked(info)` (≈2035) already renders `info` with `kind=='evidence'` + `payload={'source','text'}` (calls `_details_card.show_agent(...)`). So emitting that dict shape needs NO panel rendering change.
- Panel `connect_thread(thread)` captures the run; `thread._rag`, `thread._scenario` exist.
- Imports already present in the file: `QGraphicsObject, pyqtSignal, Qt, QRectF, QColor, QPen, QBrush, QPainter` (used by sibling node classes).

**Conventions:** Tests in `ph_economic_ai/tests/`, offscreen Qt. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`. Never add `self.show()`; tests use `not widget.isHidden()`.

**Task 0 (branch):** Continue on the current branch `feature/live-knowledge-graph` (already checked out; it holds the revert to the old canvas). Confirm: `git branch --show-current` → `feature/live-knowledge-graph`.

---

## Task 1: `_EvidenceNode` + `add_evidence_layer` (the canvas)

**Files:** Modify `ph_economic_ai/ui/stage3_swarm_canvas.py`; Test `ph_economic_ai/tests/test_evidence_layer.py` (create)

- [ ] **Step 1: Write the failing test** — create `ph_economic_ai/tests/test_evidence_layer.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


class _Rag:
    def query(self, text, top_k=3, sources=None):
        src = (sources or ['DOE'])[0]
        return [{'source': src, 'text': f'{src} chunk {i}'} for i in range(top_k)]


def test_add_evidence_layer_populates(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import _SwarmCanvas, _EvidenceNode
    c = _SwarmCanvas()
    n_agents = len(c._agents)
    assert n_agents > 0
    c.add_evidence_layer(_Rag(), {'current_price': 60.0}, top_k=3)
    ev = [it for it in c._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev) == n_agents * 3                      # each agent gained 3 real chunks
    assert ev[0]._source and ev[0]._text               # carries provenance
    # idempotent: calling again clears + rebuilds (not doubles)
    c.add_evidence_layer(_Rag(), {'current_price': 60.0}, top_k=3)
    ev2 = [it for it in c._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev2) == n_agents * 3


def test_evidence_click_emits_provenance(app):
    from ph_economic_ai.ui.stage3_swarm_canvas import _SwarmCanvas
    c = _SwarmCanvas()
    got = []
    c.node_clicked.connect(lambda d: got.append(d))
    c._emit_evidence_click('DOE', 'diesel eases 0.20')
    assert got and got[0]['kind'] == 'evidence'
    assert got[0]['payload'] == {'source': 'DOE', 'text': 'diesel eases 0.20'}
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_evidence_layer.py -v`) — no `_EvidenceNode` / `add_evidence_layer`.

- [ ] **Step 3: Implement** — in `stage3_swarm_canvas.py`:

(a) Add the `_EvidenceNode` class near the other node classes (after `_RagNode`):
```python
class _EvidenceNode(QGraphicsObject):
    """A tiny faint dot = one real retrieved RAG chunk. Click -> source + text."""
    clicked = pyqtSignal(str, str)            # (source, text)
    _R = 3.0

    def __init__(self, source: str, text: str, parent=None):
        super().__init__(parent)
        self._source = source or '?'
        self._text = text or ''
        self.setZValue(1)                     # below agents (z=5), above edges
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(f'{self._source}: {self._text[:80]}')

    def boundingRect(self) -> QRectF:
        r = self._R + 1.0
        return QRectF(-r, -r, 2 * r, 2 * r)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor('#C7CBD1')))          # faint background texture
        p.drawEllipse(QRectF(-self._R, -self._R, 2 * self._R, 2 * self._R))

    def mousePressEvent(self, ev):
        self.clicked.emit(self._source, self._text)
        ev.accept()
```

(b) Add `_emit_evidence_click` + `add_evidence_layer` as methods on `_SwarmCanvas` (e.g. right after `_emit_rag_click`):
```python
    def _emit_evidence_click(self, source: str, text: str):
        self.node_clicked.emit({
            'kind': 'evidence',
            'label': source,
            'payload': {'source': source, 'text': text},
        })

    def add_evidence_layer(self, rag, scenario: dict, top_k: int = 3):
        """Hang each agent's REAL retrieved chunks off it as satellite dots.
        Re-callable: clears any prior evidence first. Guarded — never raises."""
        from ph_economic_ai.engine.kg_swarm_adapter import _scenario_text
        # clear prior evidence (idempotent)
        for it in getattr(self, '_evidence_items', []):
            try:
                self._scene.removeItem(it)
            except Exception:
                pass
        self._evidence_items = []
        if rag is None:
            return
        try:
            text = _scenario_text(scenario or {})
        except Exception:
            text = ''
        import math
        for node in list(self._agents.values()):
            try:
                chunks = rag.query(text, top_k=top_k,
                                   sources=getattr(node, '_rag_sources', None)) or []
            except Exception:
                continue
            ax, ay = node.pos().x(), node.pos().y()
            seen = set()
            n = len(chunks)
            for i, c in enumerate(chunks):
                src, txt = c.get('source', '?'), c.get('text', '')
                if (src, txt) in seen:
                    continue
                seen.add((src, txt))
                ang = (2 * math.pi * i / max(n, 1)) - math.pi / 2
                ex, ey = ax + 26.0 * math.cos(ang), ay + 26.0 * math.sin(ang)
                edge = _Edge()
                edge.set_path_between(ax, ay, ex, ey)
                edge.set_state('dead')                 # faint dotted line
                self._scene.addItem(edge)
                self._evidence_items.append(edge)
                ev = _EvidenceNode(src, txt)
                ev.setPos(ex, ey)
                ev.clicked.connect(self._emit_evidence_click)
                self._scene.addItem(ev)
                self._evidence_items.append(ev)
```
(Also initialise `self._evidence_items = []` in `_SwarmCanvas.__init__`/`_build` so the first clear is safe — or rely on the `getattr(..., [])` default shown above, which is sufficient.)

- [ ] **Step 4: Run → passes** (`python -m pytest ph_economic_ai/tests/test_evidence_layer.py -v`).

- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/stage3_swarm_canvas.py ph_economic_ai/tests/test_evidence_layer.py && git commit -m "feat(ui): evidence satellite nodes populate the simulation canvas"`

---

## Task 2: Wire it into the panel (`connect_thread`)

**Files:** Modify `ph_economic_ai/ui/stage3_swarm_canvas.py` (`Stage3SwarmPanel.connect_thread`); Test `ph_economic_ai/tests/test_stage3_swarm.py` (append)

- [ ] **Step 1: Append the smoke test**
```python
def test_connect_thread_populates_evidence(app):
    from types import SimpleNamespace as NS
    from ph_economic_ai.ui.stage3_swarm_canvas import Stage3SwarmPanel, _EvidenceNode
    p = Stage3SwarmPanel()

    class _Rag:
        all_source_names = ['DOE']
        def query(self, t, top_k=3, sources=None):
            return [{'source': 'DOE', 'text': f'c{i}'} for i in range(top_k)]

    thread = NS(_rag=_Rag(), _scenario={'current_price': 60.0},
                agent_typing=NS(connect=lambda *a: None),
                agent_done_typing=NS(connect=lambda *a: None),
                group_round_done=NS(connect=lambda *a: None),
                group_eliminated=NS(connect=lambda *a: None),
                group_survivor=NS(connect=lambda *a: None),
                regional_done=NS(connect=lambda *a: None),
                swarm_complete=NS(connect=lambda *a: None))
    p.connect_thread(thread)
    ev = [it for it in p._canvas._scene.items() if isinstance(it, _EvidenceNode)]
    assert len(ev) > 0                                  # canvas populated from run start
```
(If `connect_thread` references more thread signals, add matching `NS(connect=...)` stubs so the stub thread satisfies it.)

- [ ] **Step 2: Run → fails** (no evidence added in `connect_thread`).

- [ ] **Step 3: Implement** — in `Stage3SwarmPanel.connect_thread(self, thread)`, after the existing wiring (the `_begin_live_graph` call + `.connect(...)` lines), add:
```python
        try:
            self._canvas.add_evidence_layer(getattr(thread, '_rag', None),
                                            getattr(thread, '_scenario', {}))
        except Exception:
            pass
```

- [ ] **Step 4: Run → passes** (`python -m pytest ph_economic_ai/tests/test_stage3_swarm.py -v`). Then import smoke: `python -c "import ph_economic_ai.ui.stage3_swarm_canvas; print('import OK')"`.

- [ ] **Step 5: Commit** `git add ph_economic_ai/ui/stage3_swarm_canvas.py ph_economic_ai/tests/test_stage3_swarm.py && git commit -m "feat(ui): populate evidence layer on run start (connect_thread)"`

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): start a run → the Simulation shows the old structured clusters/labels/RAG/line vibe, now with a dense cloud of small evidence dots around each agent (~100+ nodes); clicking a dot shows its source + text in the details card.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 `_EvidenceNode` → Task 1(a); §3.2 `add_evidence_layer` (per-agent `rag.query`, ring layout radius 26, faint `_Edge('dead')`, per-agent dedupe, idempotent clear, guarded) → Task 1(b); §3.3 panel wiring in `connect_thread` + click→details reuse → Task 2 + the existing `_on_node_clicked` `kind=='evidence'` path (so `_emit_evidence_click` emits exactly that dict shape). §5 robustness (None-rag no-op, per-agent try/except, whole-layer guard) → Task 1(b) + Task 2 wrap. §6 testing → both tasks.
**Placeholder scan:** none — full code for the node, the layer method, both emit/wiring, and the tests. Node count is deterministic (agents × top_k, minus exact per-agent dupes) → the test asserts `n_agents * 3` with a distinct-chunk mock.
**Type consistency:** `_EvidenceNode.clicked = pyqtSignal(str, str)` → `_emit_evidence_click(source, text)` → `node_clicked.emit({'kind':'evidence','label','payload':{'source','text'}})`, matching the panel's existing evidence renderer. `add_evidence_layer(rag, scenario, top_k=3)` matches the spec + the Task 2 call. Reuses `_Edge.set_path_between/set_state`, `self._agents`, `_scenario_text` — all confirmed to exist. Purely additive: no existing node/edge/animation/trust code touched.
