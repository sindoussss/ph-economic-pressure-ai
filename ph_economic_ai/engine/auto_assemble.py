"""Auto-assemble the present-pressure context — no user input.

This is what the one "Run" button calls first. It registers the frozen social
snapshot (windowed) into the RagEngine, counts what is available per sector, and
attaches the benchmark's own verdict as an honesty note the moderator will carry
into the debate. It performs NO live fetch of its own — market/news context comes
from the RagEngine (the exploratory app's existing behaviour); the social layer
is read from the frozen snapshot only.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.social_snapshot import (
    CORPUS_DIR, WINDOWS, load_social_snapshot, register_social_sources, window_slice)

# Import the location rather than re-deriving it: the engine already depends on
# the report's schema, so a path constant is strictly less coupling than a
# hand-built path that breaks silently if the artifacts directory moves.
from ph_economic_ai.benchmark.paths import ACCURACY_REPORT as _REPORT

# Sectors, their reporting unit, and the RAG source channels each capability
# agent draws on. Social sources are frozen-snapshot-backed; the rest are the
# RagEngine's existing sources.
SECTOR_UNIT = {'gas': '₱/L', 'food': '%', 'electricity': '₱/kWh'}

def sector_corpus(sector: str) -> list[str]:
    """Every source for a sector, regardless of channel.

    An agent may READ the whole sector corpus and still SPEAK only to its channel.
    Channels exist so three agents do not write the same paragraph; they were never
    meant to hide evidence from each other. A social agent that cannot see the
    price data disagrees with the market agent because it is blind, not because it
    read the same facts differently.
    """
    out: list[str] = []
    for channel_sources in SECTOR_SOURCES.get(sector, {}).values():
        for s in channel_sources:
            if s not in out:
                out.append(s)
    return out


SECTOR_SOURCES: dict[str, dict[str, list[str]]] = {
    'gas': {
        'social': ['RedditPH', 'GoogleTrends'],
        'news': ['ManilaBulletin', 'BusinessWorld', 'PHRetailFuel', 'DOEBulletin'],
        'market': ['YahooFinanceCrude', 'YahooFinanceForex'],
    },
    'food': {
        'social': ['RedditPH', 'GoogleTrends'],
        'news': ['NFARiceRetail', 'ManilaBulletin', 'PAGASAWeather'],
        'market': ['WBPhilFood', 'YahooFinanceCrude'],
    },
    'electricity': {
        'social': ['RedditPH', 'GoogleTrends'],
        'news': ['MeralcoCharge', 'WESMSpot'],
        'market': ['YahooFinanceCrude', 'EIAElectricity'],
    },
}

_DEFAULT_NOTE = ('This is a present-pressure read (a nowcast). Keep the discussion '
                 'on what is happening NOW, not a forward forecast.')

# Sector keywords (Tagalog + English) used to count how much of the social snapshot
# is actually ABOUT each sector. Without this every sector is told the same total,
# so a snapshot full of rice chatter would read as gas evidence.
_SECTOR_TERMS: dict[str, tuple[str, ...]] = {
    'gas': ('gas', 'diesel', 'gasolina', 'petrol', 'fuel', 'krudo', 'oil', 'pump',
            'presyo ng gas'),
    'food': ('bigas', 'rice', 'food', 'pagkain', 'palengke', 'gulay', 'vegetable',
             'agri', 'grocery'),
    'electricity': ('meralco', 'kuryente', 'electric', 'power rate', 'kwh', 'bill',
                    'brownout'),
}


# Whole-word matching is essential here, not substring: 'gas' appears inside
# 'bigas' (rice) and 'rice' inside 'price', so naive containment cross-labels the
# sectors — rice chatter counted as fuel evidence and vice versa.
_SECTOR_RE = {s: re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in terms) + r')\b')
              for s, terms in _SECTOR_TERMS.items()}


def _sector_posts(posts, sector: str):
    """Posts whose title/text mention the sector. Unknown sector -> everything."""
    rx = _SECTOR_RE.get(sector)
    if rx is None:
        return posts
    return [p for p in posts if rx.search(f'{p.title} {p.text}'.lower())]


@dataclass
class SectorContext:
    sector: str
    unit: str
    verdict_note: str                      # the moderator's honesty steer
    social_counts: dict[str, int] = field(default_factory=dict)  # per-window post counts
    anchor: Optional[float] = None         # magnitude guard (wired in M4)
    scenario: dict = field(default_factory=dict)


@dataclass
class AssembledContext:
    as_of: str
    window: str
    contexts: list[SectorContext]


def _verdict_notes(report_path: Path = _REPORT) -> dict[str, str]:
    """Turn the frozen audit verdicts into per-sector honesty notes. Absent/broken
    report -> defaults, so the Monitor never depends on the report existing."""
    notes: dict[str, str] = {}
    try:
        rep = json.loads(Path(report_path).read_text(encoding='utf-8'))
        audit = {a.get('target'): a for a in rep.get('audit', []) if isinstance(a, dict)}
        if audit.get('fuel', {}).get('verdict') == 'efficient':
            notes['gas'] = (
                'Benchmark verdict: 1-month fuel is informationally EFFICIENT. '
                'Describe present pressure; do NOT imply a confident next-month call — '
                'the forecast stays naive + interval.')
    except Exception:
        pass
    return notes


def _sector_anchor(sector: str) -> float:
    """A magnitude-guard anchor for the present read. Food uses its own-trend
    persistence (§6.6); gas/electricity guard around zero here, because the Monitor
    has no within-month oil/FX shock to feed the pass-through — the forecast stage
    applies the full mechanical anchor with that shock."""
    if sector == 'food':
        return anchoring.food_persistence_anchor([])
    return 0.0


def auto_assemble(rag=None, corpus_dir: Path = CORPUS_DIR,
                  as_of: Optional[date] = None, window: str = 'this_week',
                  sectors=('gas', 'food', 'electricity'),
                  report_path: Path = _REPORT) -> AssembledContext:
    """Build the present-pressure context for every sector. If `rag` is given, the
    windowed social snapshot is registered into it (via add_text — never a live
    fetch). Returns the assembled context the Forum debates over."""
    if window not in WINDOWS:
        raise ValueError(f'unknown window {window!r}; use one of {list(WINDOWS)}')
    ref = as_of or date.today()

    if rag is not None:
        register_social_sources(rag, corpus_dir, window=window, as_of=ref)

    posts = load_social_snapshot(corpus_dir)
    notes = _verdict_notes(report_path)

    contexts = []
    for s in sectors:
        relevant = _sector_posts(posts, s)      # only chatter about THIS sector
        counts = {w: len(window_slice(relevant, w, ref)) for w in WINDOWS}
        contexts.append(SectorContext(
            sector=s, unit=SECTOR_UNIT.get(s, ''),
            verdict_note=notes.get(s, _DEFAULT_NOTE),
            social_counts=counts, anchor=_sector_anchor(s),
            scenario={'as_of': ref.isoformat(), 'window': window}))
    return AssembledContext(as_of=ref.isoformat(), window=window, contexts=contexts)
