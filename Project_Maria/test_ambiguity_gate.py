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
