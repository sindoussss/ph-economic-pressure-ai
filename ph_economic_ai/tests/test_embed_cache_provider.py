"""The embedding cache must not survive a change of embedding model.

A cache built under Ollama held 145 vectors of 768 dimensions from
`nomic-embed-text`, and was stamped `gemini-embedding-001`, because
`_embed_model_name` returned the Gemini name whatever the provider was. The
stamp exists precisely to stop vectors from two models being mixed, and it was
reporting the wrong model, so it never fired.

Switching to a hosted provider then produced two different crashes from that one
cause, both of which disabled semantic retrieval and fell back to keyword
matching with a valid API key and no sign that anything was wrong:

    could not write embedding cache (all input arrays must have the same shape)
    matmul: ... (size 3072 is different from 768)

The first is `np.stack` on a mixed-width cache at refit. The second is a stale
uniform-width matrix meeting a live query vector. Both are pinned here.
"""
import logging

import numpy as np
import pytest

from ph_economic_ai.engine import rag as rag_module
from ph_economic_ai.engine.rag import RagEngine, _embed_model_name


@pytest.fixture(autouse=True)
def _cache_in_tmp(tmp_path, monkeypatch):
    """Never touch the real cache/embeddings.npz from a test."""
    monkeypatch.setattr(rag_module, '_EMBED_CACHE_PATH', tmp_path / 'embeddings.npz')
    yield tmp_path / 'embeddings.npz'


def _bare_engine() -> RagEngine:
    r = RagEngine.__new__(RagEngine)
    r._chunks = []
    r._dirty = False
    r._use_embeddings = True
    r._embed_cache = {}
    r._embed_vecs = None
    r._embed_active = []
    r._vectorizer = None
    r._matrix = None
    return r


# ── The stamp names the model that actually embedded ──────────────────────────

def test_local_inference_stamps_the_ollama_model(monkeypatch):
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: True)
    assert _embed_model_name() == 'nomic-embed-text'


def test_hosted_inference_stamps_the_gemini_model(monkeypatch):
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: False)
    assert _embed_model_name() == 'gemini-embedding-001'


def test_the_two_providers_do_not_share_a_stamp(monkeypatch):
    """The regression. Both used to return the Gemini name, so an Ollama cache
    loaded happily into a Gemini session and corrupted retrieval."""
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: True)
    local = _embed_model_name()
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: False)
    assert local != _embed_model_name()


def test_the_env_override_still_wins(monkeypatch):
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: False)
    monkeypatch.setenv('STRATA_LLM_EMBED_MODEL', 'text-embedding-004')
    assert _embed_model_name() == 'text-embedding-004'


def test_a_cache_written_locally_is_rejected_by_a_hosted_session(monkeypatch, _cache_in_tmp):
    """End to end through the real save and load."""
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: True)
    r = _bare_engine()
    r._embed_cache = {f'k{i}': np.zeros(768, dtype=np.float32) for i in range(5)}
    r._save_embed_cache()
    assert _cache_in_tmp.exists()

    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: False)
    fresh = _bare_engine()
    fresh._load_embed_cache()
    assert fresh._embed_cache == {}, 'stale local vectors loaded into a hosted session'


def test_a_cache_written_locally_is_accepted_by_another_local_session(monkeypatch):
    monkeypatch.setattr(rag_module.llm, 'is_local', lambda: True)
    r = _bare_engine()
    r._embed_cache = {f'k{i}': np.zeros(768, dtype=np.float32) for i in range(5)}
    r._save_embed_cache()

    fresh = _bare_engine()
    fresh._load_embed_cache()
    assert len(fresh._embed_cache) == 5


# ── Mixed widths cost the strays, not all retrieval ───────────────────────────

def test_a_mixed_width_cache_keeps_the_majority():
    r = _bare_engine()
    r._embed_cache = {f'old{i}': np.zeros(768, dtype=np.float32) for i in range(145)}
    r._embed_cache.update({f'new{i}': np.zeros(3072, dtype=np.float32) for i in range(6)})
    r._drop_mismatched_dims()
    assert len(r._embed_cache) == 145
    assert {len(v) for v in r._embed_cache.values()} == {768}
    np.stack(list(r._embed_cache.values()))          # the call that used to raise


def test_a_uniform_cache_is_left_alone():
    r = _bare_engine()
    r._embed_cache = {f'k{i}': np.zeros(3072, dtype=np.float32) for i in range(4)}
    r._drop_mismatched_dims()
    assert len(r._embed_cache) == 4


def test_an_empty_cache_does_not_raise():
    r = _bare_engine()
    r._drop_mismatched_dims()
    assert r._embed_cache == {}


# ── A width mismatch at query time self-heals ─────────────────────────────────

_MATMUL_ERROR = (
    'matmul: Input operand 1 has a mismatch in its core dimension 0, with '
    'gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3072 is different from 768)'
)


def test_a_query_width_mismatch_drops_the_cache_and_retries(_cache_in_tmp, caplog):
    """Without this the bad vectors stay on disk, so every restart reloads them
    and fails in the same place: one provider switch, permanent keyword search."""
    _cache_in_tmp.write_bytes(b'stale')
    r = _bare_engine()
    r._embed_cache = {'k': np.zeros(768, dtype=np.float32)}
    r._embed_vecs = np.zeros((1, 768), dtype=np.float32)
    r._embed_active = [object()]

    calls = []

    def _mismatch(text, top_k, sources):
        calls.append(text)
        raise ValueError(_MATMUL_ERROR)

    r._query_embeddings = _mismatch
    r._query_tfidf = lambda t, k, s: [{'source': 'tfidf', 'text': ''}]
    r._refit = lambda: None

    with caplog.at_level(logging.WARNING):
        out = r.query('why did fuel prices go down', top_k=2)

    assert len(calls) == 2, 'it must retry once after clearing, not give up'
    assert r._embed_cache == {}
    assert not _cache_in_tmp.exists(), 'the stale file must be removed, not just the dict'
    assert r._dirty
    assert out == [{'source': 'tfidf', 'text': ''}]
    assert 'width mismatch' in caplog.text


def test_an_unrelated_query_error_does_not_wipe_the_cache(caplog):
    """Self-healing must not become "delete the cache on any error"."""
    r = _bare_engine()
    r._embed_cache = {'k': np.zeros(3072, dtype=np.float32)}
    r._embed_vecs = np.zeros((1, 3072), dtype=np.float32)
    r._embed_active = [object()]
    r._query_embeddings = lambda t, k, s: (_ for _ in ()).throw(ValueError('bad input'))
    r._query_tfidf = lambda t, k, s: [{'source': 'tfidf', 'text': ''}]

    with caplog.at_level(logging.WARNING):
        r.query('anything', top_k=2)

    assert len(r._embed_cache) == 1, 'an unrelated failure must not clear the cache'
    assert 'width mismatch' not in caplog.text


def test_a_network_failure_still_falls_back_quietly(caplog):
    r = _bare_engine()
    r._embed_cache = {'k': np.zeros(3072, dtype=np.float32)}
    r._embed_vecs = np.zeros((1, 3072), dtype=np.float32)
    r._embed_active = [object()]
    r._query_embeddings = lambda t, k, s: (_ for _ in ()).throw(RuntimeError('timeout'))
    r._query_tfidf = lambda t, k, s: [{'source': 'tfidf', 'text': ''}]

    with caplog.at_level(logging.WARNING):
        out = r.query('anything', top_k=2)

    assert out == [{'source': 'tfidf', 'text': ''}]
    assert len(r._embed_cache) == 1
