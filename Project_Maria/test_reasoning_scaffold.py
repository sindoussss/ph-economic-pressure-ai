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
