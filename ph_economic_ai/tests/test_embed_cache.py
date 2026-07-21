"""Tests for the persistent, content-keyed embedding cache in rag.py.

The cache is what makes app startup cheap: without it every launch re-embedded
the whole corpus. These tests pin the two properties that matter — entries
survive a restart, and vectors from a different model are never reused.
"""
import numpy as np
import pytest

from ph_economic_ai.engine import rag
from ph_economic_ai.engine.rag import Chunk


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp file so tests never touch the real one."""
    monkeypatch.setattr(rag, '_EMBED_CACHE_PATH', tmp_path / 'embeddings.npz')
    yield tmp_path / 'embeddings.npz'


def _chunk(text: str, source: str = 'DOE') -> Chunk:
    return Chunk(source=source, url='http://x', timestamp='2026-01-01', text=text)


# ── Keying ────────────────────────────────────────────────────────────────────

def test_key_is_stable_for_identical_text():
    assert rag._embed_key('hello') == rag._embed_key('hello')


def test_key_differs_for_different_text():
    assert rag._embed_key('hello') != rag._embed_key('world')


def test_key_ignores_text_beyond_the_truncation_point():
    """_refit hashes full text while the batch path hashes the truncated text;
    if those disagreed the cache would never hit."""
    base = 'x' * rag._EMBED_MAX_CHARS
    assert rag._embed_key(base) == rag._embed_key(base + 'ignored tail')


def test_key_is_not_identity_based():
    """The old cache keyed on id(chunk); two equal-text chunks must share a key."""
    a, b = _chunk('same text'), _chunk('same text')
    assert a is not b
    assert rag._embed_key(a.text) == rag._embed_key(b.text)


# ── Normalisation ─────────────────────────────────────────────────────────────

def test_normalise_produces_unit_vector():
    assert np.linalg.norm(rag._normalise([3.0, 4.0])) == pytest.approx(1.0)


def test_normalise_tolerates_zero_vector():
    assert not np.isnan(rag._normalise([0.0, 0.0])).any()


# ── Persistence ───────────────────────────────────────────────────────────────

def test_cache_survives_a_restart(monkeypatch):
    """The whole point: a second RagEngine must not re-embed anything."""
    monkeypatch.setattr(rag.llm, 'embed', lambda texts: [[1.0, 0.0] for _ in texts])

    first = rag.RagEngine()
    first._chunks = [_chunk('alpha'), _chunk('beta')]
    first._dirty = True
    first._refit()
    assert len(first._embed_cache) == 2

    calls: list = []

    def _boom(texts):
        calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(rag.llm, 'embed', _boom)
    second = rag.RagEngine()
    second._chunks = [_chunk('alpha'), _chunk('beta')]
    second._dirty = True
    second._refit()

    assert calls == [], 'restart re-embedded chunks that were already cached'
    assert len(second._embed_cache) == 2


def test_only_uncached_chunks_are_embedded(monkeypatch):
    seen: list[list[str]] = []

    def _embed(texts):
        seen.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(rag.llm, 'embed', _embed)

    engine = rag.RagEngine()
    engine._chunks = [_chunk('alpha')]
    engine._dirty = True
    engine._refit()

    engine._chunks = [_chunk('alpha'), _chunk('gamma')]
    engine._dirty = True
    engine._refit()

    assert seen[-1] == ['gamma'], 'already-cached chunk was re-embedded'


def test_cache_from_a_different_model_is_discarded(monkeypatch):
    """Vector dimensions are model-specific — reusing them across models would
    corrupt retrieval silently instead of failing loudly."""
    monkeypatch.setattr(rag.llm, 'embed', lambda texts: [[1.0, 0.0] for _ in texts])
    engine = rag.RagEngine()
    engine._chunks = [_chunk('alpha')]
    engine._dirty = True
    engine._refit()
    assert rag._EMBED_CACHE_PATH.exists()

    monkeypatch.setenv('STRATA_LLM_EMBED_MODEL', 'some-other-embedding-model')
    reloaded = rag.RagEngine()
    assert reloaded._embed_cache == {}


def test_corrupt_cache_file_is_not_fatal(monkeypatch):
    rag._EMBED_CACHE_PATH.write_bytes(b'not an npz file')
    engine = rag.RagEngine()          # must not raise
    assert engine._embed_cache == {}


def test_refit_falls_back_to_tfidf_when_embedding_fails(monkeypatch):
    """No API key must degrade to lexical search, never crash the app."""
    def _fail(texts):
        raise rag.llm.LLMError('no key')

    monkeypatch.setattr(rag.llm, 'embed', _fail)
    engine = rag.RagEngine()
    engine._chunks = [_chunk('alpha')]
    engine._dirty = True
    engine._refit()

    assert engine._use_embeddings is False
    assert engine._matrix is not None, 'TF-IDF fallback index was not built'


def test_batching_respects_the_batch_size(monkeypatch):
    sizes: list[int] = []

    def _embed(texts):
        sizes.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(rag.llm, 'embed', _embed)
    monkeypatch.setattr(rag, '_EMBED_BATCH', 2)

    engine = rag.RagEngine()
    engine._embed_chunks_batched([_chunk(f'c{i}') for i in range(5)])

    assert sizes == [2, 2, 1]
