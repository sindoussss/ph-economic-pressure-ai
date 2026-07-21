"""Tests for the hosted-LLM client. Everything here runs without a network or
an API key — the transport is stubbed and the pure seams are exercised directly.
"""
import threading
import time

import pytest

from ph_economic_ai.engine import llm


# ── Config resolution ─────────────────────────────────────────────────────────

def test_active_provider_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'gemini')
    monkeypatch.setenv('GROQ_API_KEY', 'x')
    assert llm.active_provider() == 'gemini'


def test_active_provider_falls_back_to_whichever_key_exists(monkeypatch):
    monkeypatch.delenv('STRATA_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', 'x')
    assert llm.active_provider() == 'gemini'


def test_active_provider_raises_actionable_error_with_no_keys(monkeypatch):
    monkeypatch.delenv('STRATA_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(llm.LLMError, match='No LLM provider configured'):
        llm.active_provider()


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'openai')
    with pytest.raises(llm.LLMError, match='Unknown STRATA_LLM_PROVIDER'):
        llm.active_provider()


def test_is_configured_never_raises(monkeypatch):
    monkeypatch.delenv('STRATA_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert llm.is_configured() is False


def test_deep_and_fast_tiers_differ_so_judges_get_the_stronger_model():
    assert llm.model_for(llm.FAST, 'groq') != llm.model_for(llm.DEEP, 'groq')


def test_model_id_is_env_overridable(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_FAST_MODEL', 'some-new-free-model')
    assert llm.model_for(llm.FAST, 'groq') == 'some-new-free-model'


def test_unknown_tier_is_rejected():
    with pytest.raises(llm.LLMError, match='Unknown tier'):
        llm.model_for('medium', 'groq')


# ── Gemini message translation ────────────────────────────────────────────────

def test_system_prompt_is_hoisted_to_system_instruction():
    payload = llm.to_gemini_payload(
        [{'role': 'system', 'content': 'be terse'},
         {'role': 'user', 'content': 'hello'}],
        max_tokens=None,
    )
    assert payload['systemInstruction']['parts'][0]['text'] == 'be terse'
    assert [c['role'] for c in payload['contents']] == ['user']


def test_assistant_role_is_renamed_to_model():
    payload = llm.to_gemini_payload(
        [{'role': 'user', 'content': 'a'}, {'role': 'assistant', 'content': 'b'}],
        max_tokens=None,
    )
    assert [c['role'] for c in payload['contents']] == ['user', 'model']


def test_multiple_system_messages_are_joined_not_dropped():
    payload = llm.to_gemini_payload(
        [{'role': 'system', 'content': 'one'},
         {'role': 'system', 'content': 'two'},
         {'role': 'user', 'content': 'q'}],
        max_tokens=None,
    )
    assert payload['systemInstruction']['parts'][0]['text'] == 'one\n\ntwo'


def test_consecutive_user_turns_are_preserved():
    """The swarm builds a multi-turn debate transcript; collapsing turns would
    silently rewrite what the agents actually said to each other."""
    payload = llm.to_gemini_payload(
        [{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}],
        max_tokens=None,
    )
    assert len(payload['contents']) == 2


def test_max_tokens_maps_to_generation_config():
    payload = llm.to_gemini_payload([{'role': 'user', 'content': 'q'}], max_tokens=750)
    assert payload['generationConfig']['maxOutputTokens'] == 750


# ── Stream token extraction ───────────────────────────────────────────────────

def test_extract_groq_token():
    assert llm.extract_groq_token(
        {'choices': [{'delta': {'content': 'hi'}}]}) == 'hi'


def test_extract_groq_token_tolerates_empty_delta():
    assert llm.extract_groq_token({'choices': [{'delta': {}}]}) == ''


def test_extract_gemini_token_joins_parts():
    assert llm.extract_gemini_token(
        {'candidates': [{'content': {'parts': [{'text': 'a'}, {'text': 'b'}]}}]}) == 'ab'


def test_extract_gemini_token_tolerates_missing_content():
    assert llm.extract_gemini_token({'candidates': [{}]}) == ''


@pytest.mark.parametrize('line,expected', [
    ('data: {"a":1}', '{"a":1}'),
    ('data: [DONE]', None),
    ('data:', None),
    (': keepalive', None),
    ('event: message', None),
])
def test_parse_sse_line(line, expected):
    assert llm._parse_sse_line(line) == expected


# ── Rate limiter ──────────────────────────────────────────────────────────────

def test_limiter_allows_up_to_rpm_without_blocking():
    limiter = llm._RateLimiter(rpm=5, window=60.0)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire()
    assert time.monotonic() - start < 0.5


def test_limiter_blocks_the_request_over_the_cap():
    """The swarm fans out across threads; without this the first parallel round
    would burn straight through the free-tier RPM ceiling into HTTP 429."""
    limiter = llm._RateLimiter(rpm=2, window=0.5)
    limiter.acquire()
    limiter.acquire()
    start = time.monotonic()
    limiter.acquire()               # must wait for the window to slide
    assert time.monotonic() - start >= 0.3


def test_limiter_is_thread_safe():
    """Concurrent acquires must never exceed the cap — this is the property the
    ThreadPoolExecutor in GroupArena.run depends on."""
    limiter = llm._RateLimiter(rpm=10, window=5.0)
    errors: list[str] = []

    def worker():
        try:
            limiter.acquire()
        except Exception as exc:            # pragma: no cover - failure path
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(limiter._hits) == 10


def test_limiter_blocks_when_the_token_cap_is_hit():
    """Tokens/minute, not requests/minute, is what actually throttles a swarm
    run: ~44K fast-tier tokens against a 6K/min free-tier ceiling."""
    limiter = llm._RateLimiter(rpm=1000, tpm=100, window=0.5)
    limiter.acquire(tokens=90)
    start = time.monotonic()
    limiter.acquire(tokens=50)          # 140 > 100 → must wait for the window
    assert time.monotonic() - start >= 0.3


def test_limiter_ignores_tokens_when_tpm_is_zero():
    limiter = llm._RateLimiter(rpm=1000, tpm=0, window=60.0)
    start = time.monotonic()
    for _ in range(5):
        limiter.acquire(tokens=10_000)
    assert time.monotonic() - start < 0.5


def test_oversized_single_call_does_not_deadlock():
    """A prompt larger than the whole per-minute budget can never fit; it must
    go through and let the provider's 429 handling deal with it."""
    limiter = llm._RateLimiter(rpm=100, tpm=100, window=60.0)
    start = time.monotonic()
    limiter.acquire(tokens=5_000)
    assert time.monotonic() - start < 0.5


def test_estimate_tokens_counts_prompt_and_reserved_completion():
    messages = [{'role': 'user', 'content': 'x' * 400}]
    assert llm.estimate_tokens(messages, max_tokens=750) == 400 // 4 + 750


def test_estimate_tokens_handles_missing_content():
    assert llm.estimate_tokens([{'role': 'user'}], max_tokens=None) == 0


def test_tiers_get_separate_limiters(monkeypatch):
    """Fast and deep have independent quotas; sharing a limiter would let bulk
    agent traffic consume the judges' budget."""
    monkeypatch.setattr(llm, '_limiters', {})
    fast = llm._limiter_for('groq', llm.FAST)
    deep = llm._limiter_for('groq', llm.DEEP)
    assert fast is not deep


def test_tpm_override_is_respected(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_TPM', '1234')
    assert llm.tpm_for('groq', llm.FAST) == 1234


def test_backoff_grows_and_is_jittered():
    assert llm._backoff(0) < llm._backoff(3)
    assert llm._backoff(99) <= 16.0 + 0.75      # capped


# ── stream() wiring ───────────────────────────────────────────────────────────

def test_stream_yields_tokens_from_groq(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'groq')
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    chunks = [
        {'choices': [{'delta': {'content': 'hel'}}]},
        {'choices': [{'delta': {'content': 'lo'}}]},
    ]
    monkeypatch.setattr(llm, '_post_sse', lambda *a, **k: iter(chunks))
    assert ''.join(llm.stream([{'role': 'user', 'content': 'q'}])) == 'hello'


def test_stream_yields_tokens_from_gemini(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'gemini')
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    chunks = [{'candidates': [{'content': {'parts': [{'text': 'hey'}]}}]}]
    monkeypatch.setattr(llm, '_post_sse', lambda *a, **k: iter(chunks))
    assert ''.join(llm.stream([{'role': 'user', 'content': 'q'}])) == 'hey'


def test_complete_joins_the_stream(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'groq')
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    chunks = [
        {'choices': [{'delta': {'content': 'a'}}]},
        {'choices': [{'delta': {'content': 'b'}}]},
    ]
    monkeypatch.setattr(llm, '_post_sse', lambda *a, **k: iter(chunks))
    assert llm.complete([{'role': 'user', 'content': 'q'}]) == 'ab'


def test_deep_tier_selects_the_deep_model(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'groq')
    monkeypatch.setenv('GROQ_API_KEY', 'k')
    monkeypatch.delenv('STRATA_LLM_DEEP_MODEL', raising=False)
    seen: dict = {}

    def fake_post(url, headers, payload, provider, *args, **kwargs):
        seen.update(payload)
        return iter(())

    monkeypatch.setattr(llm, '_post_sse', fake_post)
    list(llm.stream([{'role': 'user', 'content': 'q'}], tier=llm.DEEP))
    assert seen['model'] == llm._DEFAULT_MODELS['groq'][llm.DEEP]


def test_embed_without_gemini_key_raises_so_caller_can_fall_back(monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    with pytest.raises(llm.LLMError, match='GEMINI_API_KEY'):
        llm.embed(['some text'])


def test_embed_of_nothing_makes_no_request(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'k')
    monkeypatch.setattr(llm.requests, 'post', _boom)
    assert llm.embed([]) == []


def _boom(*a, **k):                      # pragma: no cover - must never run
    raise AssertionError('no HTTP request should be made')
