"""Adapt the live forum debate into a KnowledgeGraph the existing canvas renders.

Pure (no Qt): each agent turn adds its sector hub, the named agent (clustered by
sector), and its claim, with edges agent->sector and agent->claim. The Monitor
panel feeds this to KnowledgeGraphCanvas.set_snapshot() as the debate grows, so
the graph fills in turn by turn alongside the chat feed.
"""
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder

_SECTOR_LABEL = {'gas': 'GAS', 'food': 'FOOD', 'electricity': 'ELECTRICITY'}


def add_forum_turn(builder: KnowledgeGraphBuilder, name: str, occupation: str,
                   sector: str, estimate=None, statement: str = '') -> KnowledgeGraphBuilder:
    """Add one agent's turn to the graph: sector hub + named agent + its claim."""
    sid = builder.add_node(f'sector:{sector}', 'master',
                           _SECTOR_LABEL.get(sector, (sector or '?').upper()),
                           {'sector': sector})
    aid = builder.add_agent(name or '?', role=occupation or '', region=sector,
                            estimate=estimate)
    builder.add_edge(aid, sid, 'in_region')
    builder.add_claim(aid, estimate, (statement or '')[:200])
    return builder


def build_forum_graph(turns) -> KnowledgeGraphBuilder:
    """Build a graph from an iterable of turn dicts
    (name / occupation / sector / estimate / message)."""
    b = KnowledgeGraphBuilder()
    for t in turns:
        add_forum_turn(b, t.get('name', ''), t.get('occupation', ''), t.get('sector', ''),
                       t.get('estimate'), t.get('message', ''))
    return b
