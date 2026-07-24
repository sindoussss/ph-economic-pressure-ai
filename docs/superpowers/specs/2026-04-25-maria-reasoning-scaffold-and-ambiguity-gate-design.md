# Maria — ReasoningScaffold + AmbiguityGate Design

**Date:** 2026-04-25  
**File:** `Project_Maria/Maria_App_original.py`  
**Goal:** Close the quality gap between Maria (7B) and larger models on three fronts:
- **B** — weak answers on complex/multi-step queries
- **C** — Maria never asks clarifying questions; guesses and charges ahead
- **D** (partial) — tone drift in long responses deferred to a future session (ToneConsistencyCheck)

---

## Architecture

Two new components slot into the existing pipeline additively. Nothing is removed.

```
[Intent classified]
        ↓
  ① AmbiguityGate          ← NEW  (pre-LLM; may short-circuit everything)
        ↓ (not ambiguous)
  [System prompt built]
        ↓
  ② ReasoningScaffold      ← NEW  (injected into system prompt)
        ↓
  [Main LLM call]
        ↓
  [Existing post-processing unchanged]
        ↓
  [clean_output → filler strip → emit]
```

---

## Component 1 — ReasoningScaffold

### Purpose
Inject intent-specific structured reasoning guidance into the system prompt so the 7B model
produces step-by-step, balanced, and complete answers instead of shallow or incomplete ones.

### Location in code
One new block in `_run_impl_inner`, appended to the system prompt immediately **after** the
existing LENGTH RULE injection block (~line 17462). No new class or function needed.

### Detection
- Intent-based: uses the already-computed `_query_mode`
- Comparison detection: one new module-level compiled regex `_COMPARISON_SCAFFOLD_RE`

```python
_COMPARISON_SCAFFOLD_RE = re.compile(
    r'\b(compare|vs\.?|versus|difference between|pros and cons|which is better|better than)\b',
    re.IGNORECASE
)
```

### Scaffolds injected per trigger

| Trigger | Scaffold text (appended to system prompt) |
|---|---|
| `_INTENT_CODE` | `REASONING: Break the solution into: (1) understand the problem, (2) plan the approach, (3) write the code, (4) check edge cases. Keep code clean and correct.` |
| `_INTENT_PLANNING` | `REASONING: List every constraint the user mentioned first. Build the plan addressing each one explicitly. Do not drop any constraint.` |
| `_INTENT_EXPLAINER` | `REASONING: Structure your answer — core concept first, then how/why it works, then a concrete example. Do not skip the example.` |
| `_COMPARISON_SCAFFOLD_RE` match | `REASONING: Structure — key differences first, then key similarities, then your assessment. Be balanced; don't favour one side without reason.` |
| `is_math == True` (already detected) | Strengthen the existing MathSpecialist domain hint: append `Show each step explicitly. State which rule or formula you are using at each step. Verify the final answer by substituting back or checking units.` |

### Non-firing cases
- `_INTENT_CASUAL`, `_INTENT_EMOTIONAL`, `_INTENT_REACTION` — never scaffold; keep responses natural
- `_is_clean_mode` — skip (short casual replies)
- A query can match both a comparison scaffold AND a code/planning/explainer scaffold; both are appended (they don't conflict)

---

## Component 2 — AmbiguityGate

### Purpose
Before building the system prompt or calling the main LLM, detect queries that are too vague
to answer well. If ambiguous, generate ONE natural clarifying question in Maria's voice using
a fast LLM call, emit it as Maria's response, and return early — skipping the main LLM call entirely.

### Location in code
In `_run_impl_inner`, after `_query_mode` is set and `_active_ctx` is resolved, but **before**
the system prompt construction block. Pattern:

```python
if _is_ambiguous_query(self.user_text, self.history):
    _clarification = _generate_clarification(self.user_text, MODEL_FAST)
    # emit _clarification as Maria's response and return early
```

### Detection — `_is_ambiguous_query(query: str, history: list) -> bool`

Two signals. Either signal alone is sufficient to mark the query ambiguous, subject to the
hard bypasses below.

**Signal 1 — Vague pronoun with no prior context:**
- Query word count ≤ 8
- AND query matches `_VAGUE_PRONOUN_RE`:
  ```python
  _VAGUE_PRONOUN_RE = re.compile(
      r'\b(this|that|it|these|those|the thing|the one)\b', re.IGNORECASE
  )
  ```
- AND history contains no prior assistant messages (nothing to refer back to)

Rationale: "explain this" with zero history is guaranteed ambiguous. "explain this" after
Maria gave an answer is a valid follow-up — do not block it.

**Signal 2 — Bare vague task request:**
- Query matches `_VAGUE_REQUEST_RE`:
  ```python
  _VAGUE_REQUEST_RE = re.compile(
      r'^(help me with|fix this|improve this|make this|check this|review this|'
      r'explain this|what about this|can you help|patulong|tulungan mo ako|'
      r'i-explain mo|ano ba ito)\b',
      re.IGNORECASE
  )
  ```
- AND query word count ≤ 10
- AND no code block (` ``` `) in the message (if they pasted code, the request is not vague)

### Hard bypasses — gate never fires when any of these are true

| Bypass | Reason |
|---|---|
| `_query_mode in {_INTENT_CASUAL, _INTENT_EMOTIONAL, _INTENT_REACTION}` | These are fine to answer naturally; asking "what do you mean?" kills the vibe |
| `_active_ctx.is_active()` | Mid-task short messages are follow-ups, not ambiguous queries |
| Query contains a clear question word + noun: `re.search(r'\b(who|what|when|where|why|how)\b.{3,}', query)` | Structured questions are not vague |
| `len(query.strip()) > 120` | Long queries almost always contain enough context |

### Clarification generator — `_generate_clarification(query: str, model: str) -> str`

One LLM call using `MODEL_FAST`.

```
System:
  You are Maria Clara. The user sent a message that is too vague to answer well.
  Ask ONE short clarifying question in Maria's natural Taglish/English voice.
  Under 20 words. No filler openers like "Sure!", "Of course!", or "Great question!".
  Do not answer the question — only ask for clarification.

User:
  "{query}"
```

- `temperature`: 0.3
- `num_predict`: 50
- `num_ctx`: 512 (tiny context — no history needed)

Output is run through `clean_output()` only (strip artifacts), then emitted directly as
Maria's response. The hallucination scan, Filipino flavor, and SelfCritiqueLoopEngine are
all skipped — it's just one question.

The clarification appears in chat history as a normal assistant message. On the next turn,
the user answers it and Maria proceeds through the full pipeline normally. No special state
tracking required.

---

## What is NOT in scope (deferred)

- **ToneConsistencyCheck** (Component ③): post-generation tone drift fix for long responses.
  Deferred to the next session after ① and ② are stable.
- **Persistent cross-session memory**: separate initiative.
- Changes to the intent router, SelfCritiqueLoopEngine, or web search pipeline.

---

## Testing checklist

### ReasoningScaffold
- [ ] Code question → system prompt contains `REASONING: Break the solution into`
- [ ] "Compare Python vs JavaScript" → system prompt contains comparison scaffold
- [ ] Math question → MathSpecialist hint + verification line both present
- [ ] Casual greeting → no scaffold injected
- [ ] Planning question with multiple constraints → planning scaffold injected

### AmbiguityGate
- [ ] "explain this" with no history → clarification question emitted, main LLM not called
- [ ] "explain this" after a prior exchange → gate bypassed, normal answer
- [ ] "patulong" with no context → clarification emitted
- [ ] "what is Python?" → gate bypassed (clear question structure)
- [ ] Active task + short follow-up → gate bypassed
- [ ] Emotional message → gate bypassed
- [ ] Message with pasted code block + "fix this" → gate bypassed
