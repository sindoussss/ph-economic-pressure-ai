"""Local agents, hosted judges, and a fallback that cannot lie about itself.

Free tiers throttle on TOKENS per minute, not requests. The 32 agent calls spend
about 44,800 fast-tier tokens against Groq's 6,000/min ceiling, which is a run
that dies on HTTP 429 with the daily allowance untouched — measured live, with
14,399 of 14,400 daily requests still remaining. The 7 judge calls spend about
24,300 against the deep tier's 12,000/min, which is a run that finishes.

So the split is not "hosted is better", it is "spend the hosted budget where the
verdict is decided and stop fighting the wrong ceiling".

Three things have to hold, and the third is the one that bites quietly:

1. The deep tier can be overridden without moving the base.
2. A blocked hosted tier falls back rather than killing the run.
3. A run that fell back is never confused with one that did not.
"""
import os

import pytest

from ph_economic_ai.engine import llm


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ('STRATA_LLM_PROVIDER', 'STRATA_LLM_DEEP_PROVIDER',
                'GROQ_API_KEY', 'GEMINI_API_KEY'):
        monkeypatch.delenv(var, raising=False)
    llm.reset_provenance()
    llm.clear_quota_listeners()
    yield
    llm.reset_provenance()
    llm.clear_quota_listeners()


# ── Per-tier providers ────────────────────────────────────────────────────────

def test_without_an_override_both_tiers_follow_the_base(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    assert llm.provider_for(llm.FAST) == 'ollama'
    assert llm.provider_for(llm.DEEP) == 'ollama'


def test_the_deep_override_moves_only_the_judges(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    assert llm.provider_for(llm.FAST) == 'ollama'
    assert llm.provider_for(llm.DEEP) == 'groq'


def test_the_base_stays_local_so_embeddings_do_not_move(monkeypatch):
    """`is_local()` routes `embed()`. A hosted base sends every RAG query to
    Gemini's endpoint, capped at 9 requests/min — a fresh bottleneck replacing
    the one the split just removed."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    assert llm.is_local() is True


def test_an_unknown_override_is_rejected_loudly(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'nonsense')
    with pytest.raises(llm.LLMError, match='STRATA_LLM_DEEP_PROVIDER'):
        llm.provider_for(llm.DEEP)


def test_the_models_follow_the_tier_provider(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    assert llm.model_for(llm.FAST, llm.provider_for(llm.FAST)) == 'qwen2.5:3b'
    assert llm.model_for(llm.DEEP, llm.provider_for(llm.DEEP)) == 'llama-3.3-70b-versatile'


# ── Fallback ──────────────────────────────────────────────────────────────────

def test_a_hosted_tier_falls_back_to_the_base(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    assert llm.fallback_provider(llm.DEEP) == 'ollama'


def test_a_tier_already_on_the_base_has_nowhere_to_fall(monkeypatch):
    """Retrying the provider that just failed is not a fallback."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    assert llm.fallback_provider(llm.DEEP) is None
    assert llm.fallback_provider(llm.FAST) is None


def test_stream_falls_back_when_the_primary_is_blocked(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    seen = []

    def fake(provider, messages, tier=llm.FAST, max_tokens=None, json_mode=False,
             temperature=None, seed=None):
        seen.append(provider)
        if provider == 'groq':
            raise llm.LLMError('groq deep tier rate limited')
        yield 'local answer'

    monkeypatch.setattr(llm, '_stream_via', fake)
    out = ''.join(llm.stream([{'role': 'user', 'content': 'x'}], tier=llm.DEEP))
    assert out == 'local answer'
    assert seen == ['groq', 'ollama']


def test_a_fallback_is_recorded_not_hidden(monkeypatch):
    """A verdict decided by a 7b local judge after a 429 looks identical on the
    report to one decided by a 70b hosted judge. The difference has to be
    recoverable from the run itself."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')

    def fake(provider, messages, tier=llm.FAST, max_tokens=None, json_mode=False,
             temperature=None, seed=None):
        if provider == 'groq':
            raise llm.LLMError('blocked')
        yield 'x'

    monkeypatch.setattr(llm, '_stream_via', fake)
    ''.join(llm.stream([{'role': 'user', 'content': 'x'}], tier=llm.DEEP))
    entry = llm.last_provenance()[llm.DEEP]
    assert entry['fell_back'] is True
    assert entry['provider'] == 'ollama'


def test_a_failure_with_no_fallback_still_raises(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')

    def fake(provider, messages, tier=llm.FAST, max_tokens=None, json_mode=False,
             temperature=None, seed=None):
        raise llm.LLMError('ollama is down')
        yield  # pragma: no cover

    monkeypatch.setattr(llm, '_stream_via', fake)
    with pytest.raises(llm.LLMError, match='ollama is down'):
        list(llm.stream([{'role': 'user', 'content': 'x'}], tier=llm.DEEP))


def test_a_mid_stream_failure_does_not_splice_two_models(monkeypatch):
    """Once tokens have been yielded, switching provider would join two models'
    prose into one statement — worse than the failure it avoids."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')

    def fake(provider, messages, tier=llm.FAST, max_tokens=None, json_mode=False,
             temperature=None, seed=None):
        if provider == 'groq':
            yield 'half an ans'
            raise llm.LLMError('died mid-stream')
        yield 'local'

    monkeypatch.setattr(llm, '_stream_via', fake)
    with pytest.raises(llm.LLMError, match='mid-stream'):
        list(llm.stream([{'role': 'user', 'content': 'x'}], tier=llm.DEEP))


def test_an_explicit_provider_argument_disables_fallback(monkeypatch):
    """Callers naming a provider mean it — the ablation harness pins one."""
    def fake(provider, messages, tier=llm.FAST, max_tokens=None, json_mode=False,
             temperature=None, seed=None):
        raise llm.LLMError('nope')
        yield  # pragma: no cover

    monkeypatch.setattr(llm, '_stream_via', fake)
    with pytest.raises(llm.LLMError):
        list(llm.stream([{'role': 'user', 'content': 'x'}], provider='groq'))


# ── Provenance drives the recall key ──────────────────────────────────────────

def test_a_fallback_run_gets_a_different_identity(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')

    llm._record_provenance(llm.FAST, 'ollama', 'qwen2.5:3b')
    llm._record_provenance(llm.DEEP, 'groq', 'llama-3.3-70b-versatile')
    clean = llm.provenance_id()

    llm.reset_provenance()
    llm._record_provenance(llm.FAST, 'ollama', 'qwen2.5:3b')
    llm._record_provenance(llm.DEEP, 'ollama', 'qwen2.5:7b', fell_back=True)
    assert llm.provenance_id() != clean


def test_provenance_before_a_run_reports_the_configured_models(monkeypatch):
    """The recall LOOKUP happens before the first call, so it has to describe
    what the run intends to use."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('STRATA_LLM_DEEP_PROVIDER', 'groq')
    ident = llm.provenance_id()
    assert 'ollama' in ident and 'groq' in ident


def test_reset_clears_the_previous_run(monkeypatch):
    llm._record_provenance(llm.DEEP, 'groq', 'x', fell_back=True)
    llm.reset_provenance()
    assert llm.last_provenance() == {}
