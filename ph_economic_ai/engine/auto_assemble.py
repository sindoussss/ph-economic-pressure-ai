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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine.social_snapshot import (
    CORPUS_DIR, WINDOWS, load_social_snapshot, register_social_sources,
    social_sources, window_slice)

_REPORT = (Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts'
           / 'accuracy_report.json')

# Sectors, their reporting unit, and the RAG source channels each capability
# agent draws on. Social sources are frozen-snapshot-backed; the rest are the
# RagEngine's existing sources.
SECTOR_UNIT = {'gas': '₱/L', 'food': '%', 'electricity': '₱/kWh'}

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
    social_posts = [p for p in posts if p.source in set(social_sources(posts))]
    notes = _verdict_notes(report_path)

    contexts = []
    for s in sectors:
        counts = {w: len(window_slice(social_posts, w, ref)) for w in WINDOWS}
        contexts.append(SectorContext(
            sector=s, unit=SECTOR_UNIT.get(s, ''),
            verdict_note=notes.get(s, _DEFAULT_NOTE),
            social_counts=counts,
            scenario={'as_of': ref.isoformat(), 'window': window}))
    return AssembledContext(as_of=ref.isoformat(), window=window, contexts=contexts)
