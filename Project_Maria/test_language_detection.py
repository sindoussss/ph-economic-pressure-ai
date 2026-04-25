"""
Language detection unit tests — verifies Filipino/Taglish detection works for
sentences that previously fell through to English due to missing function words.

Run: cd PythonProject && python -m pytest Project_Maria/test_language_detection.py -v
"""
import re
import threading
from collections import OrderedDict
from pathlib import Path
import pytest


def _slice_between(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def _load_detect_ns() -> dict:
    src = Path(__file__).with_name("Maria_App_original.py").read_text(encoding="utf-8")

    # Extract precisely the pieces detect_language() needs:
    #   1. _RE_WORDS compiled regex (one line)
    #   2. Unicode range helper functions (_has_cjk, _has_hira_kata, …)
    #   3. Language marker tables (_FILIPINO_HARD, _FILIPINO_SOFT, etc.)
    #   4. Cache + precomputed sets + detect_language body
    parts = [
        _slice_between(src, "_RE_WORDS          = re.compile", "\n# ── Module-level compiled"),
        _slice_between(src, "def _has_cjk(", "# ─────────────────────────────────────────────────────────────────────────────\n#  LANGUAGE MARKER TABLES"),
        _slice_between(src, "# ─────────────────────────────────────────────────────────────────────────────\n#  LANGUAGE MARKER TABLES", "_lang_detect_cache: OrderedDict"),
        _slice_between(src, "_lang_detect_cache: OrderedDict", "def is_mixed_or_wrong_language("),
    ]

    code = "\n".join([
        "import re, threading",
        "from collections import OrderedDict",
        "detect = None",   # stub langdetect — tests must not rely on it
    ] + parts)

    ns: dict = {}
    exec(code, ns)
    return ns


NS = _load_detect_ns()
detect_language = NS["detect_language"]


# ── Tests: pure Tagalog sentences that were previously broken ─────────────────

class TestCoreFilipinoPronounsAndArticles:
    """Sentences using the newly-added function words."""

    def test_mga_plural_marker(self):
        assert detect_language("Mga bata ang naglalaro.") == "tl"

    def test_namin_genitive(self):
        assert detect_language("Ito ang bahay namin.") == "tl"

    def test_nila_genitive(self):
        assert detect_language("Nandoon na sila.") == "tl"

    def test_niya_genitive(self):
        assert detect_language("Kinuha niya ang libro.") == "tl"

    def test_sila_subject(self):
        assert detect_language("Sila ay masasaya.") == "tl"

    def test_kayo_subject(self):
        assert detect_language("Kayo na ba?") == "tl"

    def test_tayo_subject(self):
        assert detect_language("Tayo na!") == "tl"

    def test_ako_first_person(self):
        assert detect_language("Ako ay mag-aaral.") == "tl"

    def test_ka_second_person(self):
        assert detect_language("Saan ka pupunta?") == "tl"

    def test_ko_genitive(self):
        assert detect_language("Ang libro ko ay nawala.") == "tl"

    def test_mo_genitive(self):
        assert detect_language("Ibibigay ko sa iyo ang sagot mo.") == "tl"

    def test_ang_article(self):
        assert detect_language("Ang buhay ay maganda.") == "tl"

    def test_ng_particle(self):
        assert detect_language("Kumain ng kanin.") == "tl"

    def test_sa_locative(self):
        assert detect_language("Nandito ako sa bahay.") == "tl"


class TestShortFilipinoPhrases:
    """Very short Filipino messages that previously fell through to English."""

    def test_kumain_ka_na(self):
        assert detect_language("Kumain ka na?") == "tl"

    def test_saan_ka_pupunta(self):
        assert detect_language("Saan ka pupunta?") == "tl"

    def test_ito_ang_sagot(self):
        assert detect_language("Ito ang sagot.") == "tl"

    def test_ang_ganda_mo(self):
        assert detect_language("Ang ganda mo!") == "tl"

    def test_anong_oras_na(self):
        # "anong" is in HARD — should work pre-fix too, but verifying
        assert detect_language("Anong oras na?") == "tl"

    def test_mga_tanong(self):
        assert detect_language("Mga tanong?") == "tl"

    def test_sila_ba(self):
        assert detect_language("Sila ba yun?") == "tl"

    def test_tayo_na(self):
        assert detect_language("Tayo na!") == "tl"


class TestTaglishMixed:
    """Taglish messages — should still detect as Filipino (tl)."""

    def test_taglish_ako_thinking(self):
        assert detect_language("Ako ay thinking about it.") == "tl"

    def test_taglish_sana_work(self):
        assert detect_language("Sana gumana yung code mo.") == "tl"

    def test_taglish_nga_pala(self):
        assert detect_language("Nga pala, can you help me?") == "tl"

    def test_taglish_mga_files(self):
        assert detect_language("Mga files natin ay nasa folder.") == "tl"


class TestEnglishNotAffected:
    """Pure English messages must stay English — no false positives."""

    def test_pure_english_question(self):
        assert detect_language("Can you help me with this?") == "en"

    def test_pure_english_statement(self):
        assert detect_language("The quick brown fox jumps over the lazy dog.") == "en"

    def test_english_code_question(self):
        assert detect_language("How do I fix this Python error?") == "en"

    def test_english_short(self):
        assert detect_language("What is Python?") == "en"

    def test_english_single_word_sa(self):
        # "sa" alone is ambiguous — should not cause false positive for English messages
        # that happen to contain "sa" as part of a word (word-boundary safe)
        assert detect_language("The data is available.") == "en"
