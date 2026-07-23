"""Frozen social-media snapshot — the offline, reproducible substrate for the
Pressure Monitor's Forum debate.

The Forum reads only *frozen* social text, never a live feed — exactly as the
benchmark reads only frozen CSVs. `tools/refresh_social.py` does the live pulls
and writes the snapshot; this module loads it, slices it by recency window
(today / this week / this month), and registers it with the RagEngine through
`add_text` so it never enters `RagEngine.SOURCES` (which would turn it into a live
fetch). An absent snapshot yields an empty result and the Monitor degrades
gracefully rather than erroring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

CORPUS_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'corpus' / 'social'

# Recency windows in days, keyed by the monitor's labels. Rolling (last-N-days)
# rather than calendar buckets, so a sparse feed never yields an empty "today"
# at the start of a month.
WINDOWS: dict[str, int] = {'today': 1, 'this_week': 7, 'this_month': 30}


@dataclass
class SocialPost:
    date: str            # ISO 'YYYY-MM-DD'
    source: str          # e.g. 'RedditPH', 'GoogleTrends'
    title: str
    text: str
    score: float = 0.0   # upvotes / search interest / salience
    url: str = ''


def _snapshot_files(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.glob('*.jsonl')) if corpus_dir.exists() else []


def load_social_snapshot(corpus_dir: Path = CORPUS_DIR) -> list[SocialPost]:
    """Load every frozen social post from *.jsonl. Missing dir/files -> [].

    A malformed line is skipped, not fatal — a snapshot is a best-effort corpus,
    and one bad record should never sink a run.
    """
    posts: list[SocialPost] = []
    for f in _snapshot_files(corpus_dir):
        for line in f.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                posts.append(SocialPost(
                    date=str(d['date']), source=str(d['source']),
                    title=str(d.get('title', '')), text=str(d.get('text', '')),
                    score=float(d.get('score', 0) or 0), url=str(d.get('url', ''))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return posts


def _as_of(as_of: Optional[date]) -> date:
    return as_of or date.today()


def window_slice(posts: list[SocialPost], window: str,
                 as_of: Optional[date] = None) -> list[SocialPost]:
    """Posts within the recency window ('today' | 'this_week' | 'this_month')."""
    days = WINDOWS.get(window)
    if days is None:
        raise ValueError(f'unknown window {window!r}; use one of {list(WINDOWS)}')
    ref = _as_of(as_of)
    cutoff = ref - timedelta(days=days - 1)
    out = []
    for p in posts:
        try:
            d = datetime.strptime(p.date[:10], '%Y-%m-%d').date()
        except (ValueError, TypeError):
            continue
        if cutoff <= d <= ref:
            out.append(p)
    return out


def to_rag_text(posts: list[SocialPost], source: str) -> str:
    """Bundle one source's posts into a single text blob for RagEngine.add_text."""
    lines = []
    for p in posts:
        if p.source != source:
            continue
        tag = f'[{p.date}]'
        head = p.title.strip()
        body = p.text.strip()
        lines.append(f'{tag} {head}\n{body}'.strip() if body else f'{tag} {head}'.strip())
    return '\n\n'.join(l for l in lines if l)


def social_sources(posts: list[SocialPost]) -> list[str]:
    return sorted({p.source for p in posts})


def register_social_sources(rag, corpus_dir: Path = CORPUS_DIR,
                            window: Optional[str] = None,
                            as_of: Optional[date] = None) -> dict[str, int]:
    """Load the frozen snapshot and feed it to the RagEngine via `add_text` — never
    `RagEngine.SOURCES`, so no live fetch is ever triggered. Optionally restrict to
    a recency window. Returns {source: post_count}. An empty snapshot returns {} and
    the Forum simply runs without social context.
    """
    posts = load_social_snapshot(corpus_dir)
    if window:
        posts = window_slice(posts, window, as_of)
    counts: dict[str, int] = {}
    for src in social_sources(posts):
        text = to_rag_text(posts, src)
        if text:
            rag.add_text(src, text, url=f'social-snapshot://{src}')
            counts[src] = sum(1 for p in posts if p.source == src)
    return counts
