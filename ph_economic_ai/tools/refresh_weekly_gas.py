"""Build `benchmark/data/weekly_gas_features.csv` — the weekly backtest panel.

The network half of `benchmark/weekly_gas.py`. Split out for the reason every
other panel in `benchmark/data` is: the tests must run offline against a frozen
input, so a Yahoo outage cannot turn a red suite into a mystery.

Joins two already-committed sources rather than introducing any new dependency:

  * the target  -- `doe_regional_ron95.csv`, DOE's weekly city price panel,
    differenced as a MATCHED panel (per-city change, then median) so the number
    is a price move rather than a change of which cities filed a report;
  * the features -- Brent (BZ=F), RBOB (RB=F) and USD/PHP (PHP=X), the same
    three tickers `refresh_data.build_features_csv` already fetches at daily
    interval and then discards by aggregating to months.

Each commodity feature is the % change over the week ending two days BEFORE the
DOE cycle date. That lag is the whole experiment: it is what makes the input
information that was PUBLISHED before the price moved, rather than a description
of it afterwards.

    python -m ph_economic_ai.tools.refresh_weekly_gas
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import time
import urllib.request

from ph_economic_ai.benchmark.paths import data
from ph_economic_ai.benchmark.provenance import write_record
from ph_economic_ai.benchmark.weekly_gas import (
    COMMODITY_COLS, FEATURE_LAG_DAYS, FEATURE_WINDOW_DAYS, MIN_CITIES,
    matched_panel_changes, prior_window_change,
)

PANEL_CSV = data('doe_regional_ron95.csv')
OUT = data('weekly_gas_features.csv')

YAHOO_CHART = 'https://query1.finance.yahoo.com/v8/finance/chart/<ticker>'
TICKERS = {'brent_pct': 'BZ=F', 'rbob_pct': 'RB=F', 'usdphp_pct': 'PHP=X'}
_HEADERS = {'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/120.0.0.0 Safari/537.36'),
            'Accept': 'application/json'}


def fetch_daily(ticker: str, start: dt.date, end: dt.date) -> dict:
    """{date: close} at daily interval. Daily, not weekly: the feature window is
    defined in days relative to a cycle date that is not always the same weekday,
    and a weekly bar would silently round that boundary.
    """
    p1 = int(dt.datetime.combine(start, dt.time()).timestamp())
    p2 = int(dt.datetime.combine(end, dt.time()).timestamp())
    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
           f'?period1={p1}&period2={p2}&interval=1d')
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode())
    result = payload['chart']['result'][0]
    closes = result['indicators']['quote'][0]['close']
    out = {}
    for stamp, close in zip(result['timestamp'], closes):
        if close is None:
            continue
        out[dt.datetime.fromtimestamp(stamp, dt.timezone.utc).date()] = float(close)
    if not out:
        raise ValueError(f'Yahoo returned no usable closes for {ticker!r}')
    return out


def build() -> None:
    with open(PANEL_CSV, encoding='utf-8') as fh:
        panel_rows = list(csv.DictReader(fh))
    changes = matched_panel_changes(panel_rows, min_cities=MIN_CITIES)
    if not changes:
        raise SystemExit('no matched weekly changes — check the DOE panel')
    cycles = sorted(changes)
    lo = cycles[0] - dt.timedelta(days=FEATURE_LAG_DAYS + FEATURE_WINDOW_DAYS + 21)
    hi = cycles[-1] + dt.timedelta(days=3)
    print(f'target: {len(cycles)} matched weeks, {cycles[0]} .. {cycles[-1]}')

    series = {}
    for col, ticker in TICKERS.items():
        series[col] = fetch_daily(ticker, lo, hi)
        print(f'  {col:11s} {ticker:6s} n={len(series[col])}')
        time.sleep(0.5)

    rows = []
    for day in cycles:
        feats = {c: prior_window_change(series[c], day) for c in COMMODITY_COLS}
        if any(math.isnan(v) for v in feats.values()):   # no usable window
            continue
        rows.append({'cycle': day.isoformat(),
                     'pump_change': f'{changes[day]:.4f}',
                     **{c: f'{feats[c]:.6f}' for c in COMMODITY_COLS}})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.DictWriter(fh, fieldnames=['cycle', 'pump_change', *COMMODITY_COLS],
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    write_record(
        OUT,
        source=('DOE Oil Industry Management Bureau weekly city price panel '
                '(doe_regional_ron95.csv) joined to Yahoo Finance daily closes'),
        params={'tickers': TICKERS, 'interval': '1d',
                'feature_lag_days': FEATURE_LAG_DAYS,
                'feature_window_days': FEATURE_WINDOW_DAYS,
                'min_cities': MIN_CITIES},
        transformations=[
            'target: matched panel — per-city price change across consecutive '
            '7-day DOE cycles, then median across cities present in both weeks',
            'target: drop weeks with fewer than min_cities matched, and any '
            '|change| > 20 PHP/L as a parse artifact',
            'features: % change of each daily close over the 7-day window '
            'ending FEATURE_LAG_DAYS before the cycle date (strictly prior '
            'information — this lag is the experiment)',
        ],
        units='pump_change in PHP/L per week; *_pct in percent',
        notes=('Weekly backtest panel. The lag is load-bearing: a window '
               'reaching the cycle date would read the answer it predicts.'),
    )
    print(f'Wrote {OUT.name} ({len(rows)} rows, '
          f'{rows[0]["cycle"]}..{rows[-1]["cycle"]}) + provenance')


if __name__ == '__main__':
    build()
