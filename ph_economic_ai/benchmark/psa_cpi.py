"""PSA OpenSTAT Transport-CPI gold loader (free, official, citable).

Fetches monthly Transport CPI (by commodity group, 2018=100) from the PSA
OpenSTAT PX-Web API and freezes it as a committed CSV. The MoM transform of this
series is the nowcast target for the fuel->inflation pass-through.
"""
import re
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.targets import cpi_to_mom

HERE = Path(__file__).parent
TRANSPORT_CSV = HERE / 'data' / 'psa_transport_cpi_monthly.csv'

_MONTHS = {m.lower(): i for i, m in enumerate(
    ['January', 'February', 'March', 'April', 'May', 'June', 'July',
     'August', 'September', 'October', 'November', 'December'], start=1)}
_ABBR = {m[:3].lower(): i for m, i in _MONTHS.items()}


def _label_to_ym(label: str):
    """Normalise a PX-Web time label to 'YYYY-MM', or None if unparseable.

    Handles: '1994M01', '1994 M01', 'January 1994', '1994 January', '2018-03'."""
    s = str(label).strip()
    m = re.fullmatch(r'(\d{4})-(\d{2})', s)
    if m:
        return f'{m.group(1)}-{m.group(2)}'
    m = re.fullmatch(r'(\d{4})\s*M(\d{1,2})', s, re.IGNORECASE)
    if m:
        return f'{m.group(1)}-{int(m.group(2)):02d}'
    m = re.fullmatch(r'([A-Za-z]+)\s+(\d{4})', s)
    if m:
        mo = _MONTHS.get(m.group(1).lower()) or _ABBR.get(m.group(1)[:3].lower())
        if mo:
            return f'{m.group(2)}-{mo:02d}'
    m = re.fullmatch(r'(\d{4})\s+([A-Za-z]+)', s)
    if m:
        mo = _MONTHS.get(m.group(2).lower()) or _ABBR.get(m.group(2)[:3].lower())
        if mo:
            return f'{m.group(1)}-{mo:02d}'
    return None


def load_transport_cpi(csv_path: Path = TRANSPORT_CSV) -> pd.Series:
    """Monthly Transport CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['transport_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_transport_mom(csv_path: Path = TRANSPORT_CSV) -> pd.Series:
    """Month-over-month Transport inflation % from the committed gold."""
    return cpi_to_mom(load_transport_cpi(csv_path))


# ---------------------------------------------------------------------------
# Network fetch — PSA OpenSTAT PX-Web (json format; json-stat2 value array is
# sparse/broken on this API instance so we use the tabular 'json' format).
# ---------------------------------------------------------------------------

PSA_TRANSPORT_URL_BACKCAST = (
    'https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/'
    '2018NEW/0012M4ACP28.px'
)
PSA_TRANSPORT_URL_CURRENT = (
    'https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB/2M/PI/CPI/'
    '2018NEW/0012M4ACP22.px'
)

# Variable metadata confirmed from live GET on 2026-06-10:
#   Geolocation: '0' = 'PHILIPPINES'
#   Commodity Description: '203' = '07 - TRANSPORT'
#   Year: '0' = '1994' ... '23' = '2017'  (backcasted table)
#          '0' = '2018' ... '8' = '2026'  (current table)
#   Period: '0'='Jan' ... '11'='Dec', '12'='Ave'  (Ave = annual average, skip)

_PERIOD_TO_MM = {str(i): f'{i + 1:02d}' for i in range(12)}  # '0'->'01' .. '11'->'12'

_PSA_HEADERS = {'User-Agent': 'Mozilla/5.0'}


def _fetch_px_table(url: str, first_year: int) -> dict:
    """POST a PSA PX-Web table and return {YYYY-MM: float}.

    Uses the tabular 'json' format (not json-stat2) because the json-stat2
    value array returned by this PSA PX-Web instance is sparse/incorrect.

    Parameters
    ----------
    url : str
        PX-Web .px endpoint URL.
    first_year : int
        Calendar year corresponding to year value-id '0' in this table.
    """
    import json
    import requests

    meta = requests.get(url, headers=_PSA_HEADERS, timeout=30).json()
    by_code = {v['code']: v for v in meta['variables']}
    year_var = by_code['Year']
    period_var = by_code['Period']

    # All year ids + only month ids (skip '12' = 'Ave')
    all_year_ids = year_var['values']
    month_ids = [pid for pid in period_var['values'] if pid != '12']

    body = {
        'query': [
            {'code': 'Geolocation',
             'selection': {'filter': 'item', 'values': ['0']}},
            {'code': 'Commodity Description',
             'selection': {'filter': 'item', 'values': ['203']}},
            {'code': 'Year',
             'selection': {'filter': 'item', 'values': all_year_ids}},
            {'code': 'Period',
             'selection': {'filter': 'item', 'values': month_ids}},
        ],
        'response': {'format': 'json'},
    }
    resp = requests.post(url, json=body, headers=_PSA_HEADERS, timeout=60)
    resp.raise_for_status()
    data = json.loads(resp.content.decode('utf-8-sig'))

    # Build year-id -> YYYY map
    year_lbl = {year_var['values'][i]: str(first_year + i)
                for i in range(len(year_var['values']))}

    result = {}
    for row in data['data']:
        year_id, period_id = row['key'][2], row['key'][3]
        raw_val = row['values'][0]
        if raw_val in ('..', '', None):
            continue
        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            continue
        yyyy = year_lbl.get(year_id)
        mm = _PERIOD_TO_MM.get(period_id)
        if yyyy and mm:
            result[f'{yyyy}-{mm}'] = val
    return result


def fetch_transport_cpi(out_csv: Path = TRANSPORT_CSV) -> None:
    """Fetch monthly Transport CPI from PSA OpenSTAT and freeze to CSV.

    Combines two PX-Web tables:
    - 0012M4ACP28.px : Jan 1994 – Dec 2017 (backcasted 2018-base values)
    - 0012M4ACP22.px : Jan 2018 – present  (official 2018-base series)

    The tables are fetched using the tabular 'json' response format; the
    json-stat2 value array is sparse/incorrect on this PSA PX-Web instance.
    """
    series_back = _fetch_px_table(PSA_TRANSPORT_URL_BACKCAST, first_year=1994)
    series_curr = _fetch_px_table(PSA_TRANSPORT_URL_CURRENT, first_year=2018)

    # Merge; current table takes precedence for any overlap (2018 overlap)
    combined = {**series_back, **series_curr}

    if len(combined) < 100:
        raise ValueError(
            f'transport CPI series too short ({len(combined)} rows) — '
            'check PX-Web selection'
        )

    df = (
        pd.DataFrame(sorted(combined.items()), columns=['date', 'transport_cpi'])
        .sort_values('date')
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f'Wrote psa_transport_cpi_monthly.csv ({len(df)} rows, '
        f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})'
    )
