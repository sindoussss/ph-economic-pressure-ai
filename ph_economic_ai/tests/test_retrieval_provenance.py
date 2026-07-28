"""The graph must show what an agent read, not what it would read now.

`RSK-019`: evidence edges were rebuilt by re-querying RAG when the graph was
drawn. That answers a different question than the one the graph appears to
answer, and the two diverge as soon as the corpus, the embeddings, or top_k move.
A graph showing evidence the agent never saw is worse than one showing none,
because it looks like provenance.
"""
from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.ui import kg_live


class _Builder:
    """Minimal stand-in recording what the graph would draw."""

    def __init__(self):
        self.evidence = []
        self.edges = []

    def add_agent(self, name, role, region, est):
        return f'agent:{name}'

    def add_claim(self, aid, est, statement):
        return f'claim:{aid}'

    def add_evidence(self, source, i, text):
        eid = f'ev:{source}:{i}'
        self.evidence.append({'id': eid, 'source': source, 'text': text})
        return eid

    def add_edge(self, src, dst, kind):
        self.edges.append({'src': src, 'dst': dst, 'kind': kind})


class _Rag:
    """A RAG whose corpus has moved on since the run."""

    def __init__(self):
        self.calls = 0

    def query(self, text, top_k=3, sources=None):
        self.calls += 1
        return [{'source': 'TODAYS_NEWS', 'text': 'something published after the run'}]


def _response(retrieval=None):
    return AgentResponse(agent_name='Market Analyst', round_num=1, thinking='',
                         statement='ESTIMATE: +0.50', price_estimate=0.5,
                         retrieval=retrieval or [])


def test_response_defaults_to_empty_retrieval():
    assert _response().retrieval == []


def test_preserved_retrieval_is_used_and_rag_is_not_requeried():
    read = [{'source': 'DOE_2026_06', 'text': 'what the agent actually read'}]
    builder, rag = _Builder(), _Rag()
    kg_live.add_round(builder, [_response(read)], {}, rag, {'fuel_type': 'gasoline'})

    assert rag.calls == 0, 'RAG was re-queried despite the response carrying retrieval'
    assert [e['source'] for e in builder.evidence] == ['DOE_2026_06']
    assert [e['text'] for e in builder.evidence] == ['what the agent actually read']


def test_preserved_retrieval_is_labelled_retrieved():
    read = [{'source': 'DOE_2026_06', 'text': 'x'}]
    builder = _Builder()
    kg_live.add_round(builder, [_response(read)], {}, _Rag(), {})
    kinds = {e['kind'] for e in builder.edges}
    assert 'retrieved' in kinds
    assert 'reconstructed' not in kinds


def test_a_response_without_retrieval_falls_back_but_is_labelled_reconstructed():
    """Older responses predate stored retrieval. They may still be drawn, but the
    edge must not claim to be what the agent read."""
    builder, rag = _Builder(), _Rag()
    kg_live.add_round(builder, [_response(None)], {}, rag, {})

    assert rag.calls == 1
    kinds = {e['kind'] for e in builder.edges}
    assert 'reconstructed' in kinds
    assert 'retrieved' not in kinds


def test_the_two_kinds_are_labelled_differently_in_the_ui():
    from ph_economic_ai.ui.forum_graph import _EDGE_LABEL
    assert _EDGE_LABEL['retrieved'] != _EDGE_LABEL['reconstructed'], (
        'preserved and re-derived evidence must be distinguishable on screen')


def test_stale_corpus_cannot_masquerade_as_provenance():
    """The regression this exists to prevent, stated end to end: RAG now returns
    something the agent never saw, and the graph must still show the real thing."""
    read = [{'source': 'DOE_2026_06', 'text': 'the price notice the agent read'}]
    builder = _Builder()
    kg_live.add_round(builder, [_response(read)], {}, _Rag(), {})
    sources = {e['source'] for e in builder.evidence}
    assert 'TODAYS_NEWS' not in sources
    assert sources == {'DOE_2026_06'}
