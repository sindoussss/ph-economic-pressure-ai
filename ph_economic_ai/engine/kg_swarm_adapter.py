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
