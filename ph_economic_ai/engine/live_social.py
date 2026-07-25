"""Best-effort LIVE social refresh for the hybrid Monitor.

The Monitor is hybrid: it pulls fresh Google Trends search-interest (and Reddit, if
credentials happen to be configured) into the EXPLORATORY social snapshot when the
optional deps and network are available, and otherwise silently leaves the frozen
snapshot in place. This never raises and never blocks the run for long.

Trends needs no credentials and is the practical default source; Reddit's API now
sits behind a researcher-approval process, so the no-creds path is the normal one.

It writes ONLY the exploratory social corpus. The validated benchmark's Trends CSV
is never touched here, so the benchmark stays reproducible — live data enriches the
debate, it does not leak into the audit.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
from pathlib import Path

from ph_economic_ai.engine.social_snapshot import CORPUS_DIR


def _have(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def _trends_to_posts(corpus_dir: Path) -> int:
    """Fetch recent Google Trends interest and write it as social 'GoogleTrends'
    posts the Social analyst can read — NOT the validated benchmark CSV."""
    from pytrends.request import TrendReq

    from ph_economic_ai.tools.refresh_social import _TRENDS_TERMS
    today = dt.date.today().isoformat()
    cached = corpus_dir / f'trends_{today}.jsonl'
    if cached.exists() and cached.stat().st_size > 0:
        # Already pulled today. Google rate-limits anonymous Trends calls hard
        # (HTTP 429), so re-fetching on every Run mostly buys a throttle.
        return sum(1 for ln in cached.read_text(encoding='utf-8').splitlines()
                   if ln.strip())
    py = TrendReq(hl='en-US', tz=480)
    py.build_payload(_TRENDS_TERMS, geo='PH', timeframe='now 7-d')
    df = py.interest_over_time()
    if df is None or df.empty:
        return 0
    latest = df.iloc[-1]
    posts = []
    for term in _TRENDS_TERMS:
        if term in latest:
            posts.append({
                'date': today, 'source': 'GoogleTrends',
                'title': f'Search interest "{term}": {int(latest[term])}/100 (7-day, PH)',
                'text': '', 'score': float(latest[term])})
    if not posts:
        return 0
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / f'trends_{today}.jsonl').write_text(
        '\n'.join(json.dumps(p, ensure_ascii=False) for p in posts), encoding='utf-8')
    return len(posts)


def refresh_social_snapshot(corpus_dir: Path = CORPUS_DIR) -> dict:
    """Best-effort live refresh of the exploratory social snapshot.

    Returns a status dict; never raises. Skips silently when deps/creds/network are
    unavailable, so the Monitor falls back to whatever frozen snapshot exists. Only
    the exploratory corpus is written — the benchmark is untouched.
    """
    corpus_dir = Path(corpus_dir)
    status = {'reddit': 0, 'trends': 0, 'live': False, 'notes': []}
    if not (_have('praw') or _have('pytrends')):
        status['notes'].append('live social libs not installed; using frozen snapshot')
        return status
    if _have('praw'):
        try:
            from ph_economic_ai.tools.refresh_social import _reddit_creds, refresh_reddit
            if all(_reddit_creds()):
                status['reddit'] = refresh_reddit(out_dir=corpus_dir)
            else:
                # Reddit now gates API access behind a researcher review, so the
                # normal state is "no creds" — skip quietly instead of announcing
                # a failure on every single Run.
                status['notes'].append('reddit: not configured')
        except Exception as e:
            status['notes'].append(f'reddit skipped: {type(e).__name__}')
    if _have('pytrends'):
        try:
            status['trends'] = _trends_to_posts(corpus_dir)
        except Exception as e:
            status['notes'].append(f'trends skipped: {type(e).__name__}')
    status['live'] = (status['reddit'] > 0 or status['trends'] > 0)
    return status
