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
