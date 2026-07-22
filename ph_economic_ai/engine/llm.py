"""Provider-agnostic LLM client for the exploratory layer.

Three providers: **ollama** (local, the default), plus **groq** and **gemini**
for anyone who wants hosted inference. Ollama is the default deliberately —
the app then runs with no API key, no quota and no internet, which matters for
a thesis demo where the network cannot be relied on.

Three things drive the design:

1. **Call sites name a _tier_, not a model.** The swarm asks for `'fast'` or
   `'deep'`; each provider maps those onto its own model IDs. Switching
   provider is then an env change, not a code change, and prompts port
   untouched. Every ID is env-overridable, so a model swap is a config fix.

2. **Model size must fit the GPU.** The local tiers are 3B/7B, not the 14B the
   project used to reach for. A 9GB model on an 8GB card does not fail — it
   silently spills to CPU and runs an order of magnitude slower, which is the
   single easiest way to make local inference look hopeless.

3. **Rate limiting applies to hosted providers only.** Free tiers cap requests
   and tokens per minute, and the swarm fans out across threads (see
   `GroupArena.run`), so without a shared limiter the first parallel round
   trips straight into HTTP 429. Local Ollama has no such quota: throttling it
   would be pure invented latency, so the limiter is bypassed entirely.

The network layer is quarantined in `_post_sse` / `_ollama_stream`; the message
translation and the limiter are pure and directly testable.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Iterable, Iterator, Optional

import requests

# ── Tiers ─────────────────────────────────────────────────────────────────────
# 'fast'  — the 20 bulk swarm agents. Cheap, high daily quota.
# 'deep'  — judges and synthesis. Stronger, much tighter daily quota.
FAST = 'fast'
DEEP = 'deep'

_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    'ollama': {
        # The fast tier carries 32 of the 39 calls and must stay small enough
        # to sit in VRAM alongside its context — a model that does not fit is
        # not slow, it is unusable, which is what qwen2.5:14b judges did to
        # this project before.
        FAST: 'qwen2.5:3b',          # ~2GB
        # The deep tier is deliberately large. It runs only the 7 judge calls,
        # and those decide the master verdict. qwen2.5:7b consistently produced
        # fuel estimates several times too large — the ±₱8/L plausibility guard
        # was discarding whole regional verdicts. A reasoning model is worth
        # ~55s per call here; at 7 calls that is a bounded cost.
        DEEP: 'deepseek-r1:14b',     # ~9GB, partially CPU-offloaded on 8GB
    },
    'groq': {
        # 14.4K requests/day — this is what absorbs the bulk agent traffic.
        FAST: 'llama-3.1-8b-instant',
        # Only ~1K requests/day, so reserve it for the handful of judge calls.
        DEEP: 'llama-3.3-70b-versatile',
    },
    'gemini': {
        FAST: 'gemini-2.5-flash-lite',
        DEEP: 'gemini-2.5-flash',
    },
}

# Providers that run locally: no quota, no key, no rate limiting.
_LOCAL_PROVIDERS = frozenset({'ollama'})

# Local inference runs on a single GPU, which cannot genuinely serve many
# requests at once. The swarm's parallel rounds otherwise fire ~12 concurrent
# calls (agents x groups), each needing a chat model AND an embedding model in
# VRAM — on an 8GB card with a 9GB judge in the mix, that thrashes memory and
# has been observed to stall Ollama outright. Cap concurrent local calls; a
# small GPU is effectively serial anyway. Set STRATA_OLLAMA_CONCURRENCY=1 to
# fully serialize if a machine still struggles.
_local_sem: Optional[threading.Semaphore] = None
_local_sem_lock = threading.Lock()

# Hard wall-clock ceiling on one local call, streaming included. The per-read
# socket timeout only catches a fully silent connection; a model trickling one
# token every few seconds would slip past it forever. This bounds the whole
# call so a single stuck request surfaces as an error instead of freezing the
# run behind a clock that ticks to infinity.
_LOCAL_CALL_DEADLINE = float(os.getenv('STRATA_OLLAMA_CALL_DEADLINE', '300'))


def _local_gate() -> threading.Semaphore:
    """Process-wide limiter on concurrent local requests. Lazily sized."""
    global _local_sem
    with _local_sem_lock:
        if _local_sem is None:
            n = max(1, int(os.getenv('STRATA_OLLAMA_CONCURRENCY', '2')))
            _local_sem = threading.Semaphore(n)
        return _local_sem

# Free-tier requests/minute, per provider. Deliberately a touch under the
# published ceiling: the published number is a hard cutoff, not a target, and
# clock skew between us and the provider makes the boundary fuzzy.
_DEFAULT_RPM: dict[str, int] = {'groq': 28, 'gemini': 9}

# Free-tier tokens/minute, per provider/tier. This — not requests/minute — is
# usually what actually throttles a swarm run: one run spends roughly 44K
# tokens on the fast tier, most of it on completions, against a 6K/min cap.
_DEFAULT_TPM: dict[str, dict[str, int]] = {
    'groq':   {FAST: 6_000, DEEP: 12_000},
    'gemini': {FAST: 250_000, DEEP: 250_000},
}

# Tokens are estimated, never measured — we must decide whether to wait
# *before* sending. Four characters per token is the usual English rule of
# thumb and errs slightly high on this corpus, which is the safe direction.
_CHARS_PER_TOKEN = 4

_TIMEOUT = 120        # seconds; free tiers can queue under load
_MAX_RETRIES = 4


class LLMError(RuntimeError):
    """Raised when a provider call cannot be completed."""


# ── Rate limiting ─────────────────────────────────────────────────────────────

class _RateLimiter:
    """Process-wide sliding-window limiter on requests *and* tokens.

    A sliding window rather than a token bucket because free-tier quotas are
    enforced as "no more than N in any 60s", which a bucket's burst allowance
    would violate on the swarm's first parallel fan-out.

    Both caps matter and they bind at different times: requests/minute stops a
    burst of small calls, tokens/minute stops a handful of large ones. A swarm
    run spends ~44K fast-tier tokens against a 6K/min cap, so in practice the
    token cap is the one that actually throttles.
    """

    def __init__(self, rpm: int, tpm: int = 0, window: float = 60.0):
        self._rpm = max(1, rpm)
        self._tpm = max(0, tpm)          # 0 disables the token cap
        self._window = window
        self._hits: list[tuple[float, int]] = []   # (timestamp, tokens)
        self._lock = threading.Lock()

    def _spent(self) -> int:
        return sum(tokens for _, tokens in self._hits)

    def acquire(self, tokens: int = 0) -> None:
        """Block until issuing a request of `tokens` stays within both caps."""
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self._window
                self._hits = [h for h in self._hits if h[0] > cutoff]

                over_requests = len(self._hits) >= self._rpm
                over_tokens = bool(self._tpm) and self._spent() + tokens > self._tpm
                # A single call larger than the whole per-minute budget can
                # never fit; let it through rather than deadlock, and let the
                # provider's own 429 handling deal with it.
                impossible = bool(self._tpm) and tokens > self._tpm

                if not over_requests and (not over_tokens or impossible):
                    self._hits.append((now, tokens))
                    return

                wait = self._hits[0][0] - cutoff if self._hits else 0.01
            time.sleep(max(wait, 0.01))


_limiters: dict[str, _RateLimiter] = {}
_limiters_lock = threading.Lock()


def _rpm_for(provider: str) -> int:
    return int(os.getenv('STRATA_LLM_RPM', '0')) or _DEFAULT_RPM.get(provider, 10)


def tpm_for(provider: str, tier: str) -> int:
    override = int(os.getenv('STRATA_LLM_TPM', '0'))
    if override:
        return override
    return _DEFAULT_TPM.get(provider, {}).get(tier, 0)


def estimate_tokens(messages: list[dict], max_tokens: Optional[int]) -> int:
    """Approximate the token cost of a call, prompt plus reserved completion."""
    chars = sum(len(m.get('content', '') or '') for m in messages)
    return chars // _CHARS_PER_TOKEN + (max_tokens or 0)


def _limiter_for(provider: str, tier: str = FAST) -> _RateLimiter:
    """One limiter per provider+tier — the tiers have separate quotas.

    Local providers never reach here (they use their own transport), so every
    limiter in this map governs a real hosted quota.
    """
    key = f'{provider}:{tier}'
    with _limiters_lock:
        if key not in _limiters:
            _limiters[key] = _RateLimiter(_rpm_for(provider), tpm_for(provider, tier))
        return _limiters[key]


def effective_rpm() -> int:
    """The active requests-per-minute cap, or 0 when there is none.

    Never raises — UI-safe. Returns 0 for local providers: Ollama has no quota,
    and pretending otherwise would make the run-time estimate invent a
    rate-limit floor that does not exist.
    """
    try:
        provider = active_provider()
    except LLMError:
        return min(_DEFAULT_RPM.values())
    if provider in _LOCAL_PROVIDERS:
        return 0
    return _rpm_for(provider)


# ── Config ────────────────────────────────────────────────────────────────────

def active_provider() -> str:
    """Resolve the provider from env.

    Order: an explicit choice wins; otherwise a hosted key if one is present;
    otherwise local Ollama. Ollama last in precedence but first in practice —
    it needs no key, so the app works offline out of the box.
    """
    explicit = (os.getenv('STRATA_LLM_PROVIDER') or '').strip().lower()
    if explicit:
        if explicit not in _DEFAULT_MODELS:
            raise LLMError(
                f"Unknown STRATA_LLM_PROVIDER {explicit!r}. "
                f"Supported: {', '.join(sorted(_DEFAULT_MODELS))}."
            )
        return explicit
    if os.getenv('GROQ_API_KEY'):
        return 'groq'
    if os.getenv('GEMINI_API_KEY'):
        return 'gemini'
    return 'ollama'


def is_local(provider: Optional[str] = None) -> bool:
    """True when inference runs on this machine — no key, no quota."""
    try:
        return (provider or active_provider()) in _LOCAL_PROVIDERS
    except LLMError:
        return False


def _api_key(provider: str) -> str:
    if provider in _LOCAL_PROVIDERS:
        return ''
    key = os.getenv('GROQ_API_KEY' if provider == 'groq' else 'GEMINI_API_KEY')
    if not key:
        raise LLMError(f'{provider} selected but its API key env var is unset.')
    return key


def ollama_host() -> str:
    return os.getenv('OLLAMA_HOST', 'http://localhost:11434').rstrip('/')


def model_for(tier: str, provider: Optional[str] = None) -> str:
    """Resolve tier -> concrete model ID, honouring env overrides."""
    provider = provider or active_provider()
    if tier not in (FAST, DEEP):
        raise LLMError(f"Unknown tier {tier!r}; expected {FAST!r} or {DEEP!r}.")
    override = os.getenv(f'STRATA_LLM_{tier.upper()}_MODEL')
    if override:
        return override
    return _DEFAULT_MODELS[provider][tier]


def is_configured() -> bool:
    """True when a provider resolves and has whatever credentials it needs.

    Cheap and never raises — it sits on UI paths. It does NOT prove a call will
    succeed: for Ollama it cannot tell whether the daemon is running or the
    model is pulled. Use `probe()` for that.
    """
    try:
        _api_key(active_provider())
        return True
    except LLMError:
        return False


def probe() -> tuple[bool, str]:
    """Actually check that a call would work. Returns (ok, human message).

    Makes one real request, so this belongs in diagnostics and setup screens —
    never in a hot path.
    """
    try:
        provider = active_provider()
    except LLMError as exc:
        return False, str(exc)

    if provider in _LOCAL_PROVIDERS:
        try:
            resp = requests.get(f'{ollama_host()}/api/tags', timeout=5)
            resp.raise_for_status()
            installed = {m['name'] for m in resp.json().get('models', [])}
        except requests.RequestException as exc:
            return False, (
                f'Ollama is not reachable at {ollama_host()} ({exc}). '
                'Start the Ollama app and try again.'
            )
        wanted = {model_for(FAST, provider), model_for(DEEP, provider)}
        missing = {m for m in wanted if not _has_model(m, installed)}
        if missing:
            pulls = '  '.join(f'ollama pull {m}' for m in sorted(missing))
            return False, f'Ollama is running but these models are missing: {pulls}'
        return True, f'Ollama ready at {ollama_host()} with {", ".join(sorted(wanted))}.'

    if not is_configured():
        return False, f'{provider} selected but its API key env var is unset.'
    try:
        complete([{'role': 'user', 'content': 'ping'}], max_tokens=1)
        return True, f'{provider} reachable.'
    except LLMError as exc:
        return False, str(exc)


def _has_model(wanted: str, installed: set) -> bool:
    """Ollama reports 'qwen2.5:3b'; tolerate a missing ':latest' either way."""
    candidates = {wanted, f'{wanted}:latest', wanted.removesuffix(':latest')}
    return bool(candidates & installed)


def describe_model(tier: str) -> str:
    """Best-effort model label for provenance records. Never raises.

    Recording *which* model produced a claim must not itself be able to fail a
    run, so an unconfigured provider degrades to the tier name rather than
    raising the way `model_for` does.
    """
    try:
        return model_for(tier)
    except LLMError:
        return tier


# ── Message translation (pure) ────────────────────────────────────────────────

def to_gemini_payload(
    messages: list[dict],
    max_tokens: Optional[int],
    json_mode: bool = False,
) -> dict:
    """Convert OpenAI-style messages to Gemini's schema.

    Gemini splits the system prompt out into `systemInstruction` and calls the
    assistant role 'model'. Consecutive same-role turns are left as-is; Gemini
    tolerates them and merging would distort the debate transcript the swarm
    deliberately builds up.
    """
    contents: list[dict] = []
    system_parts: list[str] = []
    for msg in messages:
        role = msg.get('role', 'user')
        text = msg.get('content', '') or ''
        if role == 'system':
            system_parts.append(text)
            continue
        contents.append({
            'role': 'model' if role == 'assistant' else 'user',
            'parts': [{'text': text}],
        })
    payload: dict = {'contents': contents}
    if system_parts:
        payload['systemInstruction'] = {'parts': [{'text': '\n\n'.join(system_parts)}]}
    gen_config: dict = {}
    if max_tokens:
        gen_config['maxOutputTokens'] = max_tokens
    if json_mode:
        gen_config['responseMimeType'] = 'application/json'
    if gen_config:
        payload['generationConfig'] = gen_config
    return payload


def _parse_sse_line(line: str) -> Optional[str]:
    """Extract the JSON payload from one SSE `data:` line, or None to skip."""
    if not line.startswith('data:'):
        return None
    data = line[len('data:'):].strip()
    if not data or data == '[DONE]':
        return None
    return data


def extract_groq_token(obj: dict) -> str:
    """Pull the incremental text out of one Groq/OpenAI stream chunk."""
    for choice in obj.get('choices', []):
        delta = choice.get('delta') or {}
        if 'content' in delta and delta['content']:
            return delta['content']
    return ''


def extract_gemini_token(obj: dict) -> str:
    """Pull the incremental text out of one Gemini stream chunk."""
    out = ''
    for cand in obj.get('candidates', []):
        for part in (cand.get('content') or {}).get('parts', []):
            out += part.get('text', '') or ''
    return out


def extract_ollama_token(obj: dict) -> str:
    """Pull the incremental text out of one Ollama stream chunk.

    Ollama streams newline-delimited JSON rather than SSE, and each frame
    carries the whole message object with an incremental `content`.
    """
    return (obj.get('message') or {}).get('content', '') or ''


def extract_ollama_thinking(obj: dict) -> str:
    """Pull reasoning text out of one Ollama frame.

    Reasoning models (deepseek-r1 and friends) return their chain of thought in
    a separate `thinking` field rather than inline, so `content` alone silently
    drops it.
    """
    return (obj.get('message') or {}).get('thinking', '') or ''


def wrap_ollama_thinking(frames: Iterable[dict]) -> Iterator[str]:
    """Yield tokens with reasoning re-wrapped as inline <think>...</think>.

    The rest of the codebase splits reasoning from answer with
    `debate._parse_think`, which expects inline tags — the convention older
    Ollama used. Restoring that shape here keeps the agent "thinking" panels
    working with reasoning models without changing anything downstream.
    """
    in_think = False
    for obj in frames:
        thought = extract_ollama_thinking(obj)
        if thought:
            if not in_think:
                in_think = True
                yield '<think>'
            yield thought
        token = extract_ollama_token(obj)
        if token:
            if in_think:
                in_think = False
                yield '</think>'
            yield token
    if in_think:
        yield '</think>'          # model emitted only reasoning before stopping


# ── Transport ─────────────────────────────────────────────────────────────────

def _post_sse(
    url: str,
    headers: dict,
    payload: dict,
    provider: str,
    tier: str = FAST,
    tokens: int = 0,
) -> Iterator[dict]:
    """POST and yield decoded SSE JSON objects, retrying on 429/5xx.

    Retries are the caller's only defence against a free tier's burst limits,
    so 429 is honoured via Retry-After when the provider supplies it rather
    than guessed at.
    """
    last_err: Optional[str] = None
    for attempt in range(_MAX_RETRIES):
        _limiter_for(provider, tier).acquire(tokens)
        try:
            resp = requests.post(
                url, headers=headers, json=payload,
                stream=True, timeout=_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_err = f'network error: {exc}'
            time.sleep(_backoff(attempt))
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get('Retry-After')
            resp.close()
            last_err = f'HTTP {resp.status_code}'
            delay = float(retry_after) if _is_number(retry_after) else _backoff(attempt)
            time.sleep(delay)
            continue

        if resp.status_code != 200:
            detail = resp.text[:400]
            resp.close()
            raise LLMError(f'{provider} returned HTTP {resp.status_code}: {detail}')

        with resp:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                data = _parse_sse_line(raw)
                if data is None:
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue      # partial/keepalive frame; the stream continues
        return

    raise LLMError(
        f'{provider} unavailable after {_MAX_RETRIES} attempts ({last_err}). '
        'Free-tier quota may be exhausted — check your daily request limit.'
    )


def _ollama_stream(
    messages: list[dict],
    model: str,
    max_tokens: Optional[int],
    json_mode: bool,
) -> Iterator[dict]:
    """POST to a local Ollama and yield decoded NDJSON frames.

    No retries: a local failure (daemon down, model not pulled) is a setup
    problem the user needs to see immediately, not have masked by slow retries.
    But two guards apply, because "local" does not mean "reliable" on a GPU the
    swarm over-commits: a concurrency gate so parallel agents don't thrash VRAM,
    and a wall-clock deadline so a stalled stream fails loudly.
    """
    payload: dict = {'model': model, 'messages': messages, 'stream': True}
    if max_tokens:
        payload['options'] = {'num_predict': max_tokens}
    if json_mode:
        payload['format'] = 'json'

    gate = _local_gate()
    gate.acquire()
    started = time.monotonic()
    try:
        try:
            resp = requests.post(
                f'{ollama_host()}/api/chat', json=payload,
                stream=True, timeout=_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(
                f'Cannot reach Ollama at {ollama_host()} ({exc}). '
                'Is the Ollama app running?'
            ) from exc

        if resp.status_code == 404:
            resp.close()
            raise LLMError(
                f"Ollama has no model {model!r}. Pull it first:  ollama pull {model}"
            )
        if resp.status_code != 200:
            detail = resp.text[:300]
            resp.close()
            raise LLMError(f'Ollama returned HTTP {resp.status_code}: {detail}')

        with resp:
            try:
                for raw in resp.iter_lines(decode_unicode=True):
                    if time.monotonic() - started > _LOCAL_CALL_DEADLINE:
                        raise LLMError(
                            f'Ollama call exceeded {_LOCAL_CALL_DEADLINE:.0f}s and was '
                            f'treated as stalled ({model}). Restarting the Ollama app '
                            'usually clears this.'
                        )
                    if not raw:
                        continue
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        continue
            except requests.RequestException as exc:
                # A silent connection trips the socket read-timeout mid-stream;
                # convert it so callers see one error type, not two.
                raise LLMError(
                    f'Ollama stream interrupted ({model}): {exc}. '
                    'Restarting the Ollama app usually clears this.'
                ) from exc
    finally:
        gate.release()


def _ollama_embed(texts: list[str]) -> list[list[float]]:
    """Embed via a local Ollama. Batches natively through /api/embed.

    Shares the concurrency gate with chat: embeddings and completions compete
    for the same GPU, and mixing them under load is what stalled the swarm.
    """
    model = os.getenv('STRATA_LLM_OLLAMA_EMBED_MODEL', 'nomic-embed-text')
    gate = _local_gate()
    gate.acquire()
    try:
        try:
            resp = requests.post(
                f'{ollama_host()}/api/embed',
                json={'model': model, 'input': texts},
                timeout=_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(f'Cannot reach Ollama at {ollama_host()} ({exc}).') from exc

        if resp.status_code == 404:
            raise LLMError(
                f"Ollama has no embedding model {model!r}. "
                f"Pull it first:  ollama pull {model}"
            )
        if resp.status_code != 200:
            raise LLMError(f'Ollama embedding failed: HTTP {resp.status_code} {resp.text[:200]}')
        return resp.json().get('embeddings', [])
    finally:
        gate.release()


def _is_number(value: Optional[str]) -> bool:
    try:
        float(value)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter, so parallel agents don't resynchronise."""
    return min(2.0 ** attempt, 16.0) + random.uniform(0, 0.75)


# ── Public API ────────────────────────────────────────────────────────────────

def stream(
    messages: list[dict],
    tier: str = FAST,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    json_mode: bool = False,
) -> Iterator[str]:
    """Yield response text incrementally. Mirrors the old ollama stream shape.

    `json_mode` constrains the reply to a single JSON object — the replacement
    for ollama's `format='json'`, which several callers parse directly.
    """
    provider = provider or active_provider()
    model = model_for(tier, provider)
    key = _api_key(provider)
    cost = estimate_tokens(messages, max_tokens)

    if provider == 'ollama':
        frames = _ollama_stream(messages, model, max_tokens, json_mode)
        if json_mode:
            # Callers of json_mode parse the result directly, so reasoning must
            # be dropped rather than wrapped — a <think> preamble would make the
            # payload unparseable.
            for obj in frames:
                token = extract_ollama_token(obj)
                if token:
                    yield token
        else:
            yield from wrap_ollama_thinking(frames)
        return

    if provider == 'groq':
        url = 'https://api.groq.com/openai/v1/chat/completions'
        headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        payload: dict = {'model': model, 'messages': messages, 'stream': True}
        if max_tokens:
            payload['max_tokens'] = max_tokens
        if json_mode:
            payload['response_format'] = {'type': 'json_object'}
        for obj in _post_sse(url, headers, payload, provider, tier, cost):
            token = extract_groq_token(obj)
            if token:
                yield token
        return

    # gemini
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{model}:streamGenerateContent?alt=sse'
    )
    headers = {'x-goog-api-key': key, 'Content-Type': 'application/json'}
    payload = to_gemini_payload(messages, max_tokens, json_mode=json_mode)
    for obj in _post_sse(url, headers, payload, provider, tier, cost):
        token = extract_gemini_token(obj)
        if token:
            yield token


def strip_json_fence(text: str) -> str:
    """Unwrap ```json ... ``` fencing around a JSON payload.

    Requesting JSON mode is a strong hint, not a guarantee: reasoning models in
    particular still wrap their answer in a markdown fence, which turns a valid
    payload into a `json.loads` failure. Observed on deepseek-r1 via Ollama and
    occasionally on hosted models, so this is applied to every provider.
    """
    cleaned = (text or '').strip()
    if not cleaned.startswith('```'):
        return cleaned
    cleaned = re.sub(r'^```[a-zA-Z]*\s*', '', cleaned)
    return re.sub(r'\s*```$', '', cleaned).strip()


def complete(
    messages: list[dict],
    tier: str = FAST,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    """Collect a full response. Convenience wrapper over `stream`.

    In JSON mode the reply is de-fenced, so callers can `json.loads` it
    directly rather than each re-implementing the same cleanup.
    """
    text = ''.join(stream(
        messages, tier=tier, max_tokens=max_tokens,
        provider=provider, json_mode=json_mode,
    ))
    return strip_json_fence(text) if json_mode else text


def embed(texts: Iterable[str]) -> list[list[float]]:
    """Embed texts, locally via Ollama or hosted via Gemini.

    Groq serves no embedding model, so a Groq user falls through to Gemini if
    they have that key. Callers must be prepared for `LLMError` and fall back
    to the lexical TF-IDF path — embeddings are an upgrade, never a hard
    requirement.
    """
    batch = list(texts)
    if not batch:
        return []

    if is_local():
        return _ollama_embed(batch)

    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise LLMError('Embeddings need GEMINI_API_KEY (free: aistudio.google.com).')
    model = os.getenv('STRATA_LLM_EMBED_MODEL', 'gemini-embedding-001')
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{model}:batchEmbedContents'
    )
    payload = {
        'requests': [
            {'model': f'models/{model}', 'content': {'parts': [{'text': t}]}}
            for t in batch
        ]
    }
    _limiter_for('gemini').acquire()
    resp = requests.post(
        url,
        headers={'x-goog-api-key': key, 'Content-Type': 'application/json'},
        json=payload,
        timeout=_TIMEOUT,
    )
    if resp.status_code != 200:
        raise LLMError(f'embedding request failed: HTTP {resp.status_code} {resp.text[:300]}')
    return [e['values'] for e in resp.json().get('embeddings', [])]
