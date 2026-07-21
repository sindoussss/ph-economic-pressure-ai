"""Provider-agnostic hosted-LLM client for the exploratory layer.

Replaces the old local-Ollama dependency. Only free-tier providers are wired
up (Groq, Gemini), so the app stays runnable by anyone with a free API key and
no GPU.

Two things drive the design:

1. **Call sites name a _tier_, not a model.** The swarm asks for `'fast'` or
   `'deep'`; each provider maps those onto its own model IDs. Swapping Groq for
   Gemini is then an env change, not a code change, and the prompts port
   untouched. Free-tier model IDs also get retired often, so every ID is
   env-overridable — a deprecation is a config fix, not a patch.

2. **Rate limiting is not optional.** Free tiers are ~30 requests/minute, and
   the swarm fans out across threads (see `GroupArena.run`). Without a shared
   limiter the first parallel round trips straight into HTTP 429. The limiter
   here is process-wide and thread-safe precisely because the callers are not.

The network layer is deliberately thin and quarantined in `_post_sse`; the
message translation and the limiter are pure and directly testable.
"""
from __future__ import annotations

import json
import os
import random
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
    """One limiter per provider+tier — the tiers have separate quotas."""
    key = f'{provider}:{tier}'
    with _limiters_lock:
        if key not in _limiters:
            _limiters[key] = _RateLimiter(_rpm_for(provider), tpm_for(provider, tier))
        return _limiters[key]


def effective_rpm() -> int:
    """The active requests-per-minute cap. Never raises — UI-safe.

    Falls back to the most restrictive default when no provider is configured,
    so a time estimate shown before setup errs on the pessimistic side.
    """
    try:
        return _rpm_for(active_provider())
    except LLMError:
        return min(_DEFAULT_RPM.values())


# ── Config ────────────────────────────────────────────────────────────────────

def active_provider() -> str:
    """Resolve the provider from env, falling back to whichever key exists."""
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
    raise LLMError(
        'No LLM provider configured. Set GROQ_API_KEY (free: console.groq.com) '
        'or GEMINI_API_KEY (free: aistudio.google.com), and optionally '
        'STRATA_LLM_PROVIDER to pick between them.'
    )


def _api_key(provider: str) -> str:
    key = os.getenv('GROQ_API_KEY' if provider == 'groq' else 'GEMINI_API_KEY')
    if not key:
        raise LLMError(f'{provider} selected but its API key env var is unset.')
    return key


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
    """True when a call would have credentials. Never raises — UI-safe."""
    try:
        active_provider()
        return True
    except LLMError:
        return False


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


def complete(
    messages: list[dict],
    tier: str = FAST,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    json_mode: bool = False,
) -> str:
    """Collect a full response. Convenience wrapper over `stream`."""
    return ''.join(stream(
        messages, tier=tier, max_tokens=max_tokens,
        provider=provider, json_mode=json_mode,
    ))


def embed(texts: Iterable[str]) -> list[list[float]]:
    """Embed texts via Gemini's free embedding endpoint.

    Groq serves no embedding model, so this always targets Gemini regardless of
    the configured chat provider. Callers must be prepared for `LLMError` and
    fall back to the lexical TF-IDF path — embeddings are an upgrade, never a
    hard requirement.
    """
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        raise LLMError('Embeddings need GEMINI_API_KEY (free: aistudio.google.com).')
    model = os.getenv('STRATA_LLM_EMBED_MODEL', 'gemini-embedding-001')
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/'
        f'{model}:batchEmbedContents'
    )
    batch = list(texts)
    if not batch:
        return []
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
