import re
import json
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _recall(topic, history):
    _STOP_WORDS = frozenset({
        "earlier","before","anyway","that","it","this","there","here",
        "something","anything","everything","what","which","said","say",
        "tell","told","mentioned","talked","discussed","just","again",
        "now","then","so","right","okay","ok","really","actually",
        "yung","mo","ka","ba","nga","naman","kasi","sabi","sinabi","kanina","dati","noon",
    })
    topic_lower = topic.lower().strip()
    meaningful_words = [w for w in topic_lower.split()[:8]
                        if w not in _STOP_WORDS and len(w) > 2]
    if not meaningful_words:
        return "GENERIC"

    candidates = history[-30:]
    corpus = [msg["content"] for msg in candidates]
    query = " ".join(meaningful_words)
    vec = TfidfVectorizer(min_df=1).fit(corpus + [query])
    matrix = vec.transform(corpus)
    q_vec = vec.transform([query])
    scores = cosine_similarity(q_vec, matrix)[0]
    ranked = sorted([(i, s) for i, s in enumerate(scores) if s >= 0.08],
                    key=lambda x: x[1], reverse=True)[:5]
    if ranked:
        return [candidates[i]["content"] for i, _ in ranked]
    return []


HISTORY = [
    {"role": "user",      "content": "I need to study for my exam next Monday"},
    {"role": "assistant", "content": "Sure! What subject is the exam on?"},
    {"role": "user",      "content": "It's calculus - derivatives and integrals"},
    {"role": "assistant", "content": "Got it. I recommend spacing your review over 3 days."},
    {"role": "user",      "content": "I only have 2 hours tonight though"},
    {"role": "assistant", "content": "Okay, focus on derivatives first - highest exam weight."},
]


def test_context_limit_is_32768_and_no_duplicate():
    txt = open("Project_Maria/Maria_App_original.py", encoding="utf-8").read()
    hits = re.findall(r"CONTEXT_LIMIT\s*=\s*\d+", txt)
    assert hits == ["CONTEXT_LIMIT = 32768"], f"expected exactly one definition at 32768, got {hits}"


def test_recall_finds_exam_turn():
    # "exam" appears in 3 history messages — all should score >= 0.08
    results = _recall("what exam did I mention", HISTORY)
    assert any("exam" in r for r in results), f"expected exam turn in results, got {results}"


def test_recall_finds_calculus_turn():
    results = _recall("calculus derivatives", HISTORY)
    assert any("calculus" in r.lower() for r in results), f"expected calculus turn, got {results}"


def test_recall_finds_review_schedule_by_shared_words():
    # TF-IDF is lexical — query must share actual words with target message.
    # "spacing review days" shares 3 words with "I recommend spacing your review over 3 days."
    results = _recall("spacing review days", HISTORY)
    assert any("3 days" in r or "review" in r for r in results), f"expected review/days turn, got {results}"


def test_recall_generic_topic_returns_generic_flag():
    # all stop-words -> should trigger generic-recall path
    result = _recall("ba nga kasi kanina", HISTORY)
    assert result == "GENERIC"


def test_recall_unrelated_topic_returns_empty():
    results = _recall("figure count total number", HISTORY)
    assert results == []


# ── Compression JSON parsing ──────────────────────────────────────────────────

def _parse_compression_output(raw):
    """Mirrors the JSON-parse logic in _compress_history."""
    text = raw
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    data = json.loads(text)
    parts = []
    for key, label in (
        ("topics",         "Topics"),
        ("decisions",      "Decisions"),
        ("corrections",    "Corrections"),
        ("code_artifacts", "Code/Artifacts"),
        ("open_questions", "Open questions"),
    ):
        items = data.get(key) or []
        if items:
            parts.append(f"{label}: " + " | ".join(str(x) for x in items))
    return "\n".join(parts)


def test_compression_parses_clean_json():
    raw = json.dumps({
        "topics": ["study plan for calculus exam"],
        "decisions": ["focus on derivatives first"],
        "corrections": [],
        "code_artifacts": [],
        "open_questions": ["how many hours available tomorrow?"],
    })
    result = _parse_compression_output(raw)
    assert "Topics: study plan for calculus exam" in result
    assert "Decisions: focus on derivatives first" in result
    assert "Open questions: how many hours available tomorrow?" in result
    assert "Corrections" not in result  # empty list → omitted
    assert "Code/Artifacts" not in result


def test_compression_strips_markdown_fences():
    raw = '```json\n{"topics": ["calculus exam"], "decisions": [], "corrections": [], "code_artifacts": [], "open_questions": []}\n```'
    result = _parse_compression_output(raw)
    assert "Topics: calculus exam" in result


def test_compression_malformed_json_raises():
    # Confirms that malformed JSON triggers JSONDecodeError (so the fallback path activates in the app)
    with pytest.raises(json.JSONDecodeError):
        _parse_compression_output("here are some bullet points\n- thing one\n- thing two")
