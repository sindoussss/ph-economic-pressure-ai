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
