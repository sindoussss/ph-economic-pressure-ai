"""Adapt the live forum debate into a KnowledgeGraph the existing canvas renders.

Pure (no Qt): each agent turn adds its sector hub, the named agent (clustered by
sector), and its claim, with edges agent->sector and agent->claim. The Monitor
panel feeds this to KnowledgeGraphCanvas.set_snapshot() as the debate grows, so
the graph fills in turn by turn alongside the chat feed.
"""
from ph_economic_ai.engine.knowledge_graph import KnowledgeGraphBuilder

_SECTOR_LABEL = {'gas': 'GAS', 'food': 'FOOD', 'electricity': 'ELECTRICITY'}

# Short, readable labels for the RAG sources an agent reads.
_SOURCE_SHORT = {
    'RedditPH': 'Reddit', 'GoogleTrends': 'Trends',
    'YahooFinanceCrude': 'Brent', 'YahooFinanceForex': 'USD/PHP',
    'ManilaBulletin': 'Manila Bulletin', 'BusinessWorld': 'BusinessWorld',
    'PHRetailFuel': 'Pump price', 'DOEBulletin': 'DOE',
    'NFARiceRetail': 'NFA rice', 'PAGASAWeather': 'PAGASA',
    'WBPhilFood': 'WB Food', 'MeralcoCharge': 'Meralco', 'WESMSpot': 'WESM',
    'EIAElectricity': 'EIA',
}


def add_forum_turn(builder: KnowledgeGraphBuilder, name: str, occupation: str,
                   sector: str, estimate=None, statement: str = '',
                   sources=None) -> KnowledgeGraphBuilder:
    """Add one agent's turn to the graph: sector hub + named agent + its claim +
    the RAG sources it read (Reddit, Trends, news, market feeds)."""
    sid = builder.add_node(f'sector:{sector}', 'master',
                           _SECTOR_LABEL.get(sector, (sector or '?').upper()),
                           {'sector': sector})
    aid = builder.add_agent(name or '?', role=occupation or '', region=sector,
                            estimate=estimate)
    builder.add_edge(aid, sid, 'in_region')
    builder.add_claim(aid, estimate, (statement or '')[:200])
    for src in (sources or []):
        if not src:
            continue
        srcid = builder.add_node(f'src:{src}', 'source',
                                 _SOURCE_SHORT.get(src, src), {'source': src})
        builder.add_edge(aid, srcid, 'retrieved')
    return builder


def seed_sectors(builder: KnowledgeGraphBuilder, sectors) -> KnowledgeGraphBuilder:
    """Add just the sector hubs, so the debate map shows the three anchors
    immediately on Run (before any agent has spoken)."""
    for s in sectors:
        builder.add_node(f'sector:{s}', 'master',
                         _SECTOR_LABEL.get(s, (s or '?').upper()), {'sector': s})
    return builder


def build_forum_graph(turns) -> KnowledgeGraphBuilder:
    """Build a graph from an iterable of turn dicts
    (name / occupation / sector / estimate / message / sources)."""
    b = KnowledgeGraphBuilder()
    for t in turns:
        add_forum_turn(b, t.get('name', ''), t.get('occupation', ''), t.get('sector', ''),
                       t.get('estimate'), t.get('message', ''), t.get('sources'))
    return b
