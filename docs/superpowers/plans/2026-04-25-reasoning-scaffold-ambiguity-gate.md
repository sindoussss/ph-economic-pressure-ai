# ReasoningScaffold + AmbiguityGate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured reasoning scaffolds for complex queries and a pre-LLM ambiguity gate that asks one clarifying question instead of guessing on vague messages.

**Architecture:** Three injection points in `Maria_App_original.py`: (1) new module-level regex constants near line 1999, (2) two new helper functions before `_classify_query_intent` (~line 3138), and (3) two inline code blocks in `_run_impl_inner` — the AmbiguityGate hook right after intent is decided (~line 15841) and the ReasoningScaffold appended to `system_prompt` after the `elif is_code:` block (~line 17529).

**Tech Stack:** Python 3, PyQt6, Ollama (`ollama.chat`), `re` module, `pytest`

---

## File Map

| Action | Location |
|---|---|
| **Modify** `Project_Maria/Maria_App_original.py` line ~1999 | Add 3 new module-level compiled regexes |
| **Modify** `Project_Maria/Maria_App_original.py` line ~3136 | Add `_is_ambiguous_query` and `_generate_clarification` functions |
| **Modify** `Project_Maria/Maria_App_original.py` line ~15841 | Add AmbiguityGate hook (early return) |
| **Modify** `Project_Maria/Maria_App_original.py` line ~17529 | Add ReasoningScaffold injection block |
| **Create** `Project_Maria/test_ambiguity_gate.py` | pytest tests for constants + `_is_ambiguous_query` |
| **Create** `Project_Maria/test_reasoning_scaffold.py` | pytest tests for `_COMPARISON_SCAFFOLD_RE` |

---

## Task 1: Add module-level pattern constants

**Files:**
- Modify: `Project_Maria/Maria_App_original.py:1999`

These constants are depended on by both the test files (via marker extraction) and the new functions. They must exist before tests can load them.

- [ ] **Step 1: Find the insertion point**

Open `Maria_App_original.py` and locate this exact line (currently line 1999):
```
_TASK_MAX_PASSTHROUGH = 3    # max consecutive pass-through user turns before task expires
```
The new block goes on the blank line immediately after that line, before the `# Per-domain continuation vocabulary.` comment.

- [ ] **Step 2: Insert the constants block**

Insert this block after line 1999:

```python
# ── ReasoningScaffold + AmbiguityGate patterns ──────────────────────────────
_COMPARISON_SCAFFOLD_RE = re.compile(
    r'\b(compare|vs\.?|versus|difference between|pros and cons|which is better|better than)\b',
    re.IGNORECASE
)
_VAGUE_PRONOUN_RE = re.compile(
    r'\b(this|that|it|these|those|the thing|the one)\b', re.IGNORECASE
)
_VAGUE_REQUEST_RE = re.compile(
    r'^(help me with|fix this|improve this|make this|check this|review this|'
    r'explain this|what about this|can you help|patulong|tulungan mo ako|'
    r'i-explain mo|ano ba ito)\b',
    re.IGNORECASE
)
```

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected output: `SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add ReasoningScaffold + AmbiguityGate pattern constants"
```

---

## Task 2: Write failing tests for `_is_ambiguous_query`

**Files:**
- Create: `Project_Maria/test_ambiguity_gate.py`

Write the tests BEFORE the function exists. They will fail at import time — that's expected.

- [ ] **Step 1: Create the test file**

Create `Project_Maria/test_ambiguity_gate.py` with this content:

```python
"""
AmbiguityGate unit tests.
Run: cd PythonProject && python -m pytest Project_Maria/test_ambiguity_gate.py -v
"""
import re
from pathlib import Path
import pytest


def _slice_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _load_ambiguity_ns() -> dict:
    src = Path(__file__).with_name("Maria_App_original.py").read_text(encoding="utf-8")
    code = "import re\n\n" + "\n\n".join([
        _slice_between(
            src,
            "# ── ReasoningScaffold + AmbiguityGate patterns ──",
            "# ── AmbiguityGate helpers ──",
        ),
        _slice_between(
            src,
            "# ── AmbiguityGate helpers ──",
            "def _classify_query_intent(",
        ),
    ])
    ns: dict = {"re": re}
    exec(code, ns)
    return ns


NS = _load_ambiguity_ns()
is_ambiguous = NS["_is_ambiguous_query"]

EMPTY_HISTORY: list = []
WITH_HISTORY = [
    {"role": "user", "content": "what is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
]


# ── Regex constant sanity checks ─────────────────────────────────────────────

class TestVaguePronounRE:
    RE = NS["_VAGUE_PRONOUN_RE"]

    def test_matches_this(self):
        assert self.RE.search("explain this")

    def test_matches_it(self):
        assert self.RE.search("what is it")

    def test_matches_the_thing(self):
        assert self.RE.search("what about the thing")

    def test_no_match_on_clear_query(self):
        assert not self.RE.search("explain Python decorators")


class TestVagueRequestRE:
    RE = NS["_VAGUE_REQUEST_RE"]

    def test_matches_help_me_with(self):
        assert self.RE.search("help me with this")

    def test_matches_fix_this(self):
        assert self.RE.search("fix this")

    def test_matches_patulong(self):
        assert self.RE.search("patulong")

    def test_no_match_mid_sentence(self):
        # Pattern uses ^ so it must start the query
        assert not self.RE.search("I need you to fix this code for me")


# ── _is_ambiguous_query logic ────────────────────────────────────────────────

class TestIsAmbiguousQuery:
    def test_vague_pronoun_no_history_is_ambiguous(self):
        assert is_ambiguous("explain this", EMPTY_HISTORY)

    def test_vague_pronoun_with_history_not_ambiguous(self):
        # "explain this" after a prior exchange = valid follow-up
        assert not is_ambiguous("explain this", WITH_HISTORY)

    def test_bare_vague_request_no_code_is_ambiguous(self):
        assert is_ambiguous("help me with this", EMPTY_HISTORY)

    def test_vague_request_with_code_block_not_ambiguous(self):
        assert not is_ambiguous("fix this\n```python\nx = 1\n```", EMPTY_HISTORY)

    def test_long_query_not_ambiguous(self):
        long_q = "I need help understanding how async/await works in Python " * 3
        assert not is_ambiguous(long_q, EMPTY_HISTORY)

    def test_clear_question_word_not_ambiguous(self):
        assert not is_ambiguous("what is recursion", EMPTY_HISTORY)
        assert not is_ambiguous("how does threading work", EMPTY_HISTORY)

    def test_clear_factual_query_not_ambiguous(self):
        assert not is_ambiguous("ano ang capital ng Pilipinas", EMPTY_HISTORY)

    def test_patulong_no_context_is_ambiguous(self):
        assert is_ambiguous("patulong", EMPTY_HISTORY)

    def test_patulong_with_history_is_still_ambiguous(self):
        # "patulong" alone is always vague regardless of history (Signal 2)
        assert is_ambiguous("patulong", WITH_HISTORY)
```

- [ ] **Step 2: Run — confirm failure**

```bash
cd "C:\Users\user\PycharmProjects\PythonProject" && python -m pytest Project_Maria/test_ambiguity_gate.py -v 2>&1 | head -30
```
Expected: `ERROR` or `ValueError` from `_load_ambiguity_ns` because `# ── AmbiguityGate helpers ──` marker doesn't exist yet.

---

## Task 3: Implement `_is_ambiguous_query` and `_generate_clarification`

**Files:**
- Modify: `Project_Maria/Maria_App_original.py:3136`

- [ ] **Step 1: Find the insertion point**

Locate this exact text near line 3136 in `Maria_App_original.py`:
```python
})


def _classify_query_intent(query: str, is_filipino: bool = False,
```
The `})` closes a frozenset of trivial social words. Insert the new block in the blank lines between `})` and `def _classify_query_intent(`.

- [ ] **Step 2: Insert the helpers block**

```python
# ── AmbiguityGate helpers ────────────────────────────────────────────────────
def _is_ambiguous_query(query: str, history: list) -> bool:
    """Return True when the query is too vague to answer without clarification."""
    q = query.strip()
    wc = len(q.split())

    if wc > 120:
        return False
    # Clear question structure is never ambiguous
    if re.search(r'\b(who|what|when|where|why|how)\b.{3,}', q, re.IGNORECASE):
        return False

    _has_prior_assistant = any(m.get('role') == 'assistant' for m in history)

    # Signal 1: vague pronoun with no prior context
    if wc <= 8 and _VAGUE_PRONOUN_RE.search(q) and not _has_prior_assistant:
        return True

    # Signal 2: bare vague task request (no code block, short message)
    if wc <= 10 and _VAGUE_REQUEST_RE.search(q) and '```' not in q:
        return True

    return False


def _generate_clarification(query: str, model: str) -> str:
    """One fast LLM call → ONE clarifying question in Maria's voice."""
    _system = (
        "You are Maria Clara. The user sent a message that is too vague to answer well.\n"
        "Ask ONE short clarifying question in Maria's natural Taglish/English voice.\n"
        "Under 20 words. No filler openers like 'Sure!', 'Of course!', or 'Great question!'.\n"
        "Do not answer the question — only ask for clarification."
    )
    try:
        with _OLLAMA_SEMAPHORE:
            _resp = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": _system},
                    {"role": "user",   "content": query},
                ],
                options={
                    "temperature": 0.3,
                    "num_predict": 50,
                    "num_ctx": 512,
                    "num_gpu": _NUM_GPU_LAYERS,
                },
            )
        _text = ""
        if isinstance(_resp, dict):
            _text = _resp.get("message", {}).get("content", "").strip()
        elif hasattr(_resp, "message"):
            _text = getattr(_resp.message, "content", "").strip()
        return clean_output(_text) if _text else "Anong ibig mong sabihin exactly? Give me a bit more context."
    except Exception as _e:
        print(f"   ⚠️ AmbiguityGate clarification failed: {_e}")
        return "Hmm, gusto ko siguraduhing naiintindihan ko — can you give me more context?"
```

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`

- [ ] **Step 4: Run tests — confirm they pass**

```bash
python -m pytest Project_Maria/test_ambiguity_gate.py -v
```
Expected: All tests `PASSED`. If any fail, check the marker strings in the test file match the comments exactly as inserted in Step 2.

- [ ] **Step 5: Commit**

```bash
git add Project_Maria/Maria_App_original.py Project_Maria/test_ambiguity_gate.py
git commit -m "feat: add _is_ambiguous_query and _generate_clarification helpers"
```

---

## Task 4: Add AmbiguityGate hook in `_run_impl_inner`

**Files:**
- Modify: `Project_Maria/Maria_App_original.py:~15841`

- [ ] **Step 1: Find the insertion point**

Locate this exact block (around line 15840–15842):
```python
            _task_family = _query_mode if _query_mode in _TASK_INTENTS else ""
            self.routed_intent_decided.emit(self.message_id, self.session_id, _task_family)

            _ic_result_detail = (f"Routing decided after "
```
Insert the AmbiguityGate block in the blank line between `self.routed_intent_decided.emit(...)` and `_ic_result_detail`.

- [ ] **Step 2: Insert the hook**

```python
            # ── AmbiguityGate: ask one clarifying question for vague queries ──
            if (_query_mode not in {_INTENT_CASUAL, _INTENT_EMOTIONAL, _INTENT_REACTION}
                    and not _active_ctx.is_active()
                    and _is_ambiguous_query(self.user_text, self.history)):
                print(f"   ❓ AmbiguityGate fired — query too vague, asking clarification")
                _clarification = _generate_clarification(self.user_text, MODEL_FAST)
                self.response_ready.emit(_clarification, self.message_id, self.session_id)
                self.finished_processing.emit(self.message_id, self.session_id)
                return
```

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add AmbiguityGate hook in _run_impl_inner"
```

---

## Task 5: Write failing test for ReasoningScaffold regex

**Files:**
- Create: `Project_Maria/test_reasoning_scaffold.py`

- [ ] **Step 1: Create the test file**

```python
"""
ReasoningScaffold unit tests — validates the comparison detection regex.
Run: cd PythonProject && python -m pytest Project_Maria/test_reasoning_scaffold.py -v
"""
import re
from pathlib import Path
import pytest


def _slice_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _load_scaffold_ns() -> dict:
    src = Path(__file__).with_name("Maria_App_original.py").read_text(encoding="utf-8")
    # Extract just the constants block (up to the AmbiguityGate helpers marker)
    code = "import re\n\n" + _slice_between(
        src,
        "# ── ReasoningScaffold + AmbiguityGate patterns ──",
        "# ── AmbiguityGate helpers ──",
    )
    ns: dict = {"re": re}
    exec(code, ns)
    return ns


NS = _load_scaffold_ns()


class TestComparisonScaffoldRE:
    RE = NS["_COMPARISON_SCAFFOLD_RE"]

    def test_matches_compare(self):
        assert self.RE.search("compare Python vs JavaScript")

    def test_matches_versus(self):
        assert self.RE.search("Python versus JavaScript for backend")

    def test_matches_difference_between(self):
        assert self.RE.search("what's the difference between Redis and Memcached")

    def test_matches_pros_and_cons(self):
        assert self.RE.search("pros and cons of using React")

    def test_matches_which_is_better(self):
        assert self.RE.search("which is better, SQL or NoSQL?")

    def test_matches_better_than(self):
        assert self.RE.search("is Postgres better than MySQL for this use case")

    def test_matches_vs_dot(self):
        assert self.RE.search("React vs. Vue")

    def test_no_match_on_explain(self):
        assert not self.RE.search("explain how Python works")

    def test_no_match_on_casual(self):
        assert not self.RE.search("kumusta ka")

    def test_no_match_on_plain_question(self):
        assert not self.RE.search("what is machine learning")
```

- [ ] **Step 2: Run — confirm it passes**

Since the constants already exist (added in Task 1), this test should already pass:

```bash
python -m pytest Project_Maria/test_reasoning_scaffold.py -v
```
Expected: All tests `PASSED`. (These tests validate the constant we added in Task 1.)

---

## Task 6: Implement ReasoningScaffold injection

**Files:**
- Modify: `Project_Maria/Maria_App_original.py:~17529`

- [ ] **Step 1: Find the insertion point**

Locate the end of the `elif is_code:` block. It ends with this line (approximately line 17529):
```python
Structure: explain approach → code → key explanations → examples. Never dump unexplained code.
"""
            # ── 3.1 UPGRADE: Enhanced Debug Intake Template ───────────────────
```
Insert the ReasoningScaffold block in the blank line between the closing `"""` and the `# ── 3.1 UPGRADE` comment.

- [ ] **Step 2: Insert the scaffold block**

```python
            # ── ReasoningScaffold — structured reasoning guidance ─────────────
            # Appended after math/code blocks (which already have strong scaffolds).
            # Planning, explainer, and comparison queries get intent-specific structure.
            if _query_mode == _INTENT_PLANNING:
                system_prompt += (
                    "\nREASONING: List every constraint the user mentioned first. "
                    "Build the plan addressing each one explicitly. Do not drop any constraint."
                )
            elif _query_mode == _INTENT_EXPLAINER:
                system_prompt += (
                    "\nREASONING: Structure your answer — core concept first, "
                    "then how/why it works, then a concrete example. Do not skip the example."
                )
            if _COMPARISON_SCAFFOLD_RE.search(self.user_text):
                system_prompt += (
                    "\nREASONING: Structure — key differences first, then key similarities, "
                    "then your assessment. Be balanced; don't favour one side without reason."
                )
```

Note: The comparison scaffold uses a standalone `if` (not `elif`) so it fires even when combined with a planning or explainer intent (e.g., "compare these two study plans").

- [ ] **Step 3: Syntax check**

```bash
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py Project_Maria/test_reasoning_scaffold.py
git commit -m "feat: add ReasoningScaffold injection for planning/explainer/comparison intents"
```

---

## Task 7: Run full test suite and verify

**Files:** None modified.

- [ ] **Step 1: Run all new tests**

```bash
python -m pytest Project_Maria/test_ambiguity_gate.py Project_Maria/test_reasoning_scaffold.py -v
```
Expected: All tests `PASSED`, no warnings about missing markers.

- [ ] **Step 2: Run existing routing regression tests**

```bash
python -m pytest Project_Maria/test_routing_regressions.py -v
```
Expected: All tests `PASSED`. (Confirms the new code didn't break existing routing logic.)

- [ ] **Step 3: Syntax check one final time**

```bash
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```
Expected: `SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "test: verify ReasoningScaffold + AmbiguityGate pass all regression tests"
```

---

## Manual Smoke Tests (run the app)

After all tasks pass, start Maria and verify:

**AmbiguityGate:**
1. Open a fresh session, type `explain this` → Maria should ask a clarifying question, not answer
2. Have a conversation, then type `explain this` → Maria should answer normally (history bypass)
3. Type `help me with this` with no prior context → Maria asks for clarification
4. Type `fix this` but paste a code block → Maria answers, not asks (code block bypass)
5. Type `ano ang capital ng Pilipinas` → Maria answers directly (clear question bypass)

**ReasoningScaffold:**
6. Ask `compare Python vs JavaScript for web backend` → answer should address differences, similarities, then give a verdict
7. Ask `help me make a study plan for my exam on Saturday, I only have 3 hours a day and I need to cover 5 chapters` → answer should list constraints before building the plan
8. Ask `explain how recursion works` → answer should cover concept → mechanism → concrete example
