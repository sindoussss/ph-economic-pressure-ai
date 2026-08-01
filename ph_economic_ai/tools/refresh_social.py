"""Refresh the frozen social snapshot for the Pressure Monitor.

This is the ONLY place live social/search data is pulled — the Monitor and the
benchmark read only what this writes. Run it MANUALLY between milestones, never on
the app's run path, mirroring `benchmark/refresh_data.py`.

Writes:
  * benchmark/data/google_trends_monthly.csv     — numeric, validated-layer input
  * assets/corpus/social/reddit_<YYYY-MM-DD>.jsonl — text, exploratory Forum RAG

Defensive by design: missing deps or network failures are reported and skipped;
a partial file is never written (the lesson from tools/anchor_backtest.py).

Requires (only when actually refreshing):  pip install pytrends praw
Reddit needs credentials in env (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET) or
~/.ph_economic_ai/config.json ({"reddit_client_id": ..., "reddit_client_secret": ...}).

    python -m ph_economic_ai.tools.refresh_social --trends --reddit
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from ph_economic_ai.benchmark.provenance import write_record

_ROOT = Path(__file__).resolve().parents[1]
_TRENDS_CSV = _ROOT / 'benchmark' / 'data' / 'google_trends_monthly.csv'
_SOCIAL_DIR = _ROOT / 'assets' / 'corpus' / 'social'

# Tagalog + English so the search-interest signal spans both how Filipinos and
# the English-language press phrase price pressure.
_TRENDS_TERMS = ['presyo ng gas', 'diesel price', 'Meralco bill', 'bigas presyo']
#: Google rate-limits bursts with a 429. One term per request means four
#: requests, so they are spaced and retried with exponential backoff.
_TRENDS_ATTEMPTS = 4
_TRENDS_BACKOFF = 20.0
_TRENDS_SPACING = 12.0

_REDDIT_SUBS = ['Philippines', 'phinvest']
_REDDIT_QUERIES = ['gas price', 'fuel price', 'meralco', 'rice price', 'inflation']


# ── Google Trends (numeric → validated benchmark CSV) ─────────────────────────

def refresh_trends(terms=_TRENDS_TERMS, geo: str = 'PH', out: Path = _TRENDS_CSV) -> int:
    """Monthly search-interest series per term. Returns months written (0 = skipped)."""
    try:
        from pytrends.request import TrendReq
    except ImportError:
        print('trends: pytrends not installed (pip install pytrends) — skipped')
        return 0
    # ONE TERM PER REQUEST. A shared payload rescales every term against the most
    # popular one, so a low-volume term is reported as a fraction of `diesel
    # price` rather than against its own history, and sparse terms flatten to
    # zero. The provenance record has always said this is how the file is built
    # and the committed file matches it -- every column reaches 100, which only
    # happens when each term is scaled on its own -- but the CODE had drifted to
    # a single shared `build_payload(terms, ...)`. Running it would have degraded
    # the data AND written a record claiming a method it did not use.
    #
    # Requests are spaced, because four in a row is what earns a 429.
    import time

    import pandas as pd
    frames, failed = [], []
    for i, term in enumerate(terms):
        for attempt in range(_TRENDS_ATTEMPTS):
            try:
                py = TrendReq(hl='en-US', tz=480)
                py.build_payload([term], geo=geo,
                                 timeframe=f'2016-01-01 {date.today().isoformat()}')
                one = py.interest_over_time()
                if one is not None and not one.empty:
                    frames.append(one[[term]])
                break
            except Exception as e:
                if attempt == _TRENDS_ATTEMPTS - 1:
                    failed.append(f'{term} ({type(e).__name__})')
                else:
                    time.sleep(_TRENDS_BACKOFF * (2 ** attempt))
        if i < len(terms) - 1:
            time.sleep(_TRENDS_SPACING)

    if failed:
        # Partial coverage is a different series, not a shorter one: the nowcast
        # would silently lose a driver. All or nothing.
        print(f'trends: {len(failed)} of {len(terms)} terms failed '
              f'({"; ".join(failed)}) — nothing written')
        return 0
    if not frames:
        print('trends: empty response — nothing written')
        return 0
    df = pd.concat(frames, axis=1)
    monthly = df.resample('MS').mean().round(2)
    monthly.index = monthly.index.strftime('%Y-%m')
    monthly.index.name = 'date'
    out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(out)
    write_record(out, source='Google Trends via pytrends',
                 params={'geo': 'PH', 'granularity': 'monthly',
                         'payload': 'one term per request'},
                 transformations=['request each term separately, because a shared '
                                  'payload rescales all terms to the most popular one '
                                  'and flattened low-volume terms to zero',
                                  'monthly interest index per term'],
                 units='Google Trends interest index (0-100 per term)',
                 notes='Sentiment nowcast input. Never pass retries= to TrendReq: '
                       'urllib3 v2 removed Retry(method_whitelist=).')
    print(f'trends: wrote {len(monthly)} months x {len(terms)} terms -> {out}')
    return len(monthly)


# ── Reddit (text → exploratory Forum RAG corpus) ──────────────────────────────

def _reddit_creds():
    cid = os.environ.get('REDDIT_CLIENT_ID')
    sec = os.environ.get('REDDIT_CLIENT_SECRET')
    if cid and sec:
        return cid.strip(), sec.strip()
    cfg = Path.home() / '.ph_economic_ai' / 'config.json'
    if cfg.exists():
        try:
            d = json.loads(cfg.read_text(encoding='utf-8'))
            if d.get('reddit_client_id') and d.get('reddit_client_secret'):
                return d['reddit_client_id'].strip(), d['reddit_client_secret'].strip()
        except Exception:
            pass
    return None, None


def refresh_reddit(subs=_REDDIT_SUBS, queries=_REDDIT_QUERIES, limit: int = 25,
                   out_dir: Path = _SOCIAL_DIR) -> int:
    """Search recent threads about price pressure. Returns posts written (0 = skipped)."""
    try:
        import praw
    except ImportError:
        print('reddit: praw not installed (pip install praw) — skipped')
        return 0
    cid, sec = _reddit_creds()
    if not (cid and sec):
        print('reddit: no credentials (set REDDIT_CLIENT_ID/SECRET) — skipped')
        return 0
    try:
        reddit = praw.Reddit(client_id=cid, client_secret=sec,
                             user_agent='strata-pressure-monitor/0.1')
    except Exception as e:
        print(f'reddit: client init failed ({type(e).__name__}: {e}) — skipped')
        return 0

    posts: list[dict] = []
    for sub in subs:
        for q in queries:
            try:
                for s in reddit.subreddit(sub).search(
                        q, sort='new', time_filter='month', limit=limit):
                    posts.append({
                        'date': datetime.fromtimestamp(
                            s.created_utc, tz=timezone.utc).strftime('%Y-%m-%d'),
                        'source': 'RedditPH',
                        'title': (s.title or '')[:300],
                        'text': (getattr(s, 'selftext', '') or '')[:1500],
                        'score': float(getattr(s, 'score', 0) or 0),
                        'url': f'https://reddit.com{s.permalink}',
                    })
            except Exception as e:
                print(f'reddit: search {sub}/{q} failed ({type(e).__name__}: {e})')

    seen, uniq = set(), []
    for p in posts:
        if p['url'] in seen:
            continue
        seen.add(p['url'])
        uniq.append(p)
    if not uniq:
        print('reddit: no posts fetched — nothing written')
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f'reddit_{date.today().isoformat()}.jsonl'
    out.write_text('\n'.join(json.dumps(p, ensure_ascii=False) for p in uniq),
                   encoding='utf-8')
    print(f'reddit: wrote {len(uniq)} posts -> {out}')
    return len(uniq)


def main() -> int:
    ap = argparse.ArgumentParser(description='Refresh the frozen social snapshot.')
    ap.add_argument('--trends', action='store_true', help='refresh Google Trends CSV')
    ap.add_argument('--reddit', action='store_true', help='refresh Reddit corpus')
    args = ap.parse_args()
    if not (args.trends or args.reddit):
        ap.print_help()
        return 0
    if args.trends:
        refresh_trends()
    if args.reddit:
        refresh_reddit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
