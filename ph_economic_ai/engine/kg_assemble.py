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
        builder.add_relation(r['a'], r['b'])
