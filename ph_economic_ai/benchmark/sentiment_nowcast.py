"""Does social search interest nowcast Philippine inflation — beyond naive?

The Pressure Monitor's Forum reasons over social/search chatter. Before the app is
allowed to imply that chatter carries signal, the *validated* benchmark tests it
with the same machinery as everything else: the frozen Google Trends series
(`benchmark/data/google_trends_monthly.csv`, written by `tools/refresh_social.py`)
is added to the MoM nowcast frame and run through the identical walk-forward +
Diebold-Mariano `mom_verdict`, against the corrected baseline pool that now
**includes the historical mean** (see docs/defense/mean-baseline-finding.md).

Three questions, per target, all judged against the best naive (incl. mean):
  * sentiment_only          — do the trend terms ALONE beat naive?
  * drivers_plus_sentiment  — does adding trends to the drivers beat naive?
  * drivers_only            — the reference (already a null for headline/food).

Expected outcome: a null — "social search interest does not nowcast PH fuel/food
inflation beyond naive." That is the finding, and it is what licenses the Forum to
sense present sentiment while the forecast stays humble. Reproduce:

    python -m ph_economic_ai.benchmark.sentiment_nowcast
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from ph_economic_ai.benchmark.nowcast import (
    PANEL_METHODS, build_nowcast_frame, run_mom_nowcast)
from ph_economic_ai.benchmark.targets import load_inflation_mom
from ph_economic_ai.benchmark.food_nowcast import _build_food_frame, load_food_features

MIN_TRAIN = 24
_DATA = Path(__file__).resolve().parent / 'data'
_TRENDS_CSV = _DATA / 'google_trends_monthly.csv'
_OUT = Path(__file__).resolve().parent / 'artifacts' / 'sentiment_nowcast.json'

# Corrected pool: the mean is the strong naive for a mean-reverting rate series.
POOL = ('random_walk', 'seasonal_naive', 'drift', 'mean')
# Feature-using candidates + baselines + mean (arima/ets ignore features, so they
# are irrelevant to whether *sentiment* adds anything).
METHODS = ['random_walk', 'seasonal_naive', 'drift', 'mean', 'ridge', 'hgb']

_PREV = 'prev_mom'


def load_trends(csv: Path = _TRENDS_CSV) -> Optional[pd.DataFrame]:
    """Frozen monthly search-interest series (date-indexed). None if not yet built."""
    if not Path(csv).exists():
        return None
    df = pd.read_csv(csv, dtype={'date': str}).set_index('date').sort_index()
    return df.apply(pd.to_numeric, errors='coerce')


def _attach_trends(frame: pd.DataFrame, trends: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join the trend columns onto a nowcast frame (inner join on the date index),
    renamed to a safe `trend_*` namespace. Returns (joined_frame, trend_cols)."""
    t = trends.copy()
    t.columns = ['trend_' + re.sub(r'[^0-9a-zA-Z]+', '_', str(c)).strip('_').lower()
                 for c in t.columns]
    joined = frame.join(t, how='inner').dropna()
    trend_cols = [c for c in joined.columns if c.startswith('trend_')]
    return joined, trend_cols


def _verdict(frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Run the MoM nowcast on exactly `feature_cols` (+ target) through the shared
    machinery, against the mean-inclusive pool. Slim verdict dict."""
    sub = frame[feature_cols + ['target']]
    r = run_mom_nowcast(MIN_TRAIN, baseline_pool=POOL, frame=sub, methods=METHODS)
    return {k: r.get(k) for k in
            ('verdict', 'best_method', 'best_naive', 'best_skill_vs_naive', 'dm_p', 'n')}


def run_target(frame: pd.DataFrame, trend_cols: list[str]) -> dict:
    """Three verdicts for one target frame that already carries trend columns."""
    driver_cols = [c for c in frame.columns
                   if c not in ('target', _PREV) and c not in trend_cols]
    sentiment_only = _verdict(frame, trend_cols)
    drivers_only = _verdict(frame, driver_cols)
    drivers_plus = _verdict(frame, driver_cols + trend_cols)
    edge = (sentiment_only['verdict'] == 'beats_best_naive'
            or drivers_plus['verdict'] == 'beats_best_naive')
    return {
        'n': sentiment_only.get('n'),
        'trend_cols': trend_cols,
        'sentiment_only': sentiment_only,
        'drivers_only': drivers_only,
        'drivers_plus_sentiment': drivers_plus,
        'sentiment_edge': bool(edge),
    }


def _target_frames(trends: pd.DataFrame) -> dict:
    """Attach trends to the headline and food MoM nowcast frames."""
    headline = build_nowcast_frame(target_loader=load_inflation_mom, prev_col=_PREV)
    food = _build_food_frame(load_food_features())
    out = {}
    for name, base in (('headline', headline), ('food', food)):
        joined, tcols = _attach_trends(base, trends)
        if len(joined) >= MIN_TRAIN + 5 and tcols:
            out[name] = (joined, tcols)
    return out


def run(trends: Optional[pd.DataFrame] = None) -> dict:
    """Test whether the Trends signal nowcasts inflation beyond naive. If the frozen
    Trends CSV does not exist yet (refresh_social not run), returns a clear status."""
    trends = load_trends() if trends is None else trends
    if trends is None or trends.empty:
        result = {'status': 'no_trends_data',
                  'note': 'run `python -m ph_economic_ai.tools.refresh_social --trends` first'}
        return result

    frames = _target_frames(trends)
    targets = {name: run_target(fr, tc) for name, (fr, tc) in frames.items()}
    any_edge = any(t['sentiment_edge'] for t in targets.values())
    result = {
        'status': 'computed',
        'baseline_pool': list(POOL),
        'trend_terms': list(trends.columns),
        'targets': targets,
        'sentiment_edge_anywhere': bool(any_edge),
        'finding': (
            'social search interest does NOT nowcast PH fuel/food inflation beyond '
            'the naive (mean) baseline' if not any_edge else
            'a sentiment edge survived — inspect per-target verdicts before claiming it'),
    }
    return result


def _write(result: dict) -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(result, indent=2), encoding='utf-8')


def _main() -> int:
    result = run()
    _write(result)
    if result.get('status') != 'computed':
        print(f"sentiment_nowcast: {result['status']} — {result.get('note', '')}")
        print(f"Wrote {_OUT}")
        return 0
    print(f"Sentiment nowcast (pool incl. mean) — trend terms: {result['trend_terms']}\n")
    for name, t in result['targets'].items():
        print(f"  {name} (n={t['n']}):")
        for k in ('sentiment_only', 'drivers_plus_sentiment', 'drivers_only'):
            v = t[k]
            print(f"    {k:24} {v['verdict']:22} skill={v['best_skill_vs_naive']}  DM p={v['dm_p']}")
    print(f"\n  {result['finding']}")
    print(f"Wrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())
