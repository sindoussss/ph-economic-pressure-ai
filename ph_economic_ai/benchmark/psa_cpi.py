"""PSA OpenSTAT Transport-CPI gold loader (free, official, citable).

Fetches monthly Transport CPI (by commodity group, 2018=100) from the PSA
OpenSTAT PX-Web API and freezes it as a committed CSV. The MoM transform of this
series is the nowcast target for the fuel->inflation pass-through.
"""
import re
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.targets import cpi_to_mom
from ph_economic_ai.benchmark.provenance import write_record

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


def _resolve_commodity_id(commodity_var: dict, coicop_prefix: str) -> str:
    """Return the value id whose label is the COICOP division `coicop_prefix`
    (e.g. '01' -> '01 - FOOD AND NON-ALCOHOLIC BEVERAGES', not '01.1 - FOOD')."""
    needle = f'{coicop_prefix} -'
    for vid, txt in zip(commodity_var['values'], commodity_var['valueTexts']):
        if txt.strip().startswith(needle):
            return vid
    raise ValueError(f"no commodity matching '{needle}'; "
                     f"available: {commodity_var['valueTexts'][:12]}")


def _fetch_px_table(url: str, first_year: int, coicop_prefix: str) -> dict:
    """POST a PSA PX-Web table and return {YYYY-MM: float}.

    Uses the tabular 'json' format (not json-stat2) because the json-stat2
    value array returned by this PSA PX-Web instance is sparse/incorrect.

    Parameters
    ----------
    url : str
        PX-Web .px endpoint URL.
    first_year : int
        Calendar year corresponding to year value-id '0' in this table.
    coicop_prefix : str
        COICOP division code (e.g. '07' for Transport, '01' for Food).
    """
    import json
    import requests

    meta = requests.get(url, headers=_PSA_HEADERS, timeout=30).json()
    by_code = {v['code']: v for v in meta['variables']}
    commodity_id = _resolve_commodity_id(by_code['Commodity Description'], coicop_prefix)
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
             'selection': {'filter': 'item', 'values': [commodity_id]}},
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


def fetch_cpi_subcategory(coicop_prefix: str, out_csv: Path, column_name: str,
                          source_label: str, min_rows: int = 50) -> None:
    """Fetch one PSA OpenSTAT COICOP series (backcast + current tables spliced
    on the overlap) and freeze it to CSV with a provenance sidecar. Shared by
    every fetch_X_cpi() wrapper in this module -- extracted so a new series is
    a four-line wrapper, not a ~25-line copy of the same fetch/splice/write
    logic three functions already duplicated."""
    series_back = _fetch_px_table(PSA_TRANSPORT_URL_BACKCAST, first_year=1994, coicop_prefix=coicop_prefix)
    series_curr = _fetch_px_table(PSA_TRANSPORT_URL_CURRENT, first_year=2018, coicop_prefix=coicop_prefix)

    # Merge; current table takes precedence for any overlap (2018 overlap)
    combined = {**series_back, **series_curr}

    if len(combined) < min_rows:
        raise ValueError(f'{column_name} series too short ({len(combined)} rows) — '
                         'check PX-Web selection')

    df = (pd.DataFrame(sorted(combined.items()), columns=['date', column_name])
          .sort_values('date'))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, lineterminator='\n')
    write_record(out_csv, source=source_label,
                 params={'coicop_prefix': coicop_prefix, 'base_year': '2018=100',
                         'endpoints': 'backcast + current PX-Web tables'},
                 transformations=['fetch both PX-Web tables',
                                  'filter to the COICOP prefix',
                                  'splice backcast and current on the overlap',
                                  'label months YYYY-MM, sort'],
                 units='CPI index (2018=100)',
                 notes=f'PSA gold target for the {column_name} nowcast.')
    print(f'Wrote {out_csv.name} ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def fetch_transport_cpi(out_csv: Path = TRANSPORT_CSV) -> None:
    """Fetch monthly Transport CPI from PSA OpenSTAT and freeze to CSV.

    Combines two PX-Web tables:
    - 0012M4ACP28.px : Jan 1994 – Dec 2017 (backcasted 2018-base values)
    - 0012M4ACP22.px : Jan 2018 – present  (official 2018-base series)
    """
    fetch_cpi_subcategory('07', out_csv, 'transport_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 07 Transport',
                          min_rows=100)


# ---------------------------------------------------------------------------
# Food CPI (COICOP 01) — monthly index (2018=100)
# ---------------------------------------------------------------------------

FOOD_CSV = HERE / 'data' / 'psa_food_cpi_monthly.csv'


def fetch_food_cpi(out_csv: Path = FOOD_CSV) -> None:
    """Fetch monthly Food (COICOP 01) CPI from PSA OpenSTAT and freeze to CSV."""
    fetch_cpi_subcategory('01', out_csv, 'food_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01 Food and non-alcoholic beverages',
                          min_rows=100)


def load_food_cpi(csv_path: Path = FOOD_CSV) -> pd.Series:
    """Monthly Food CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['food_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_food_mom(csv_path: Path = FOOD_CSV) -> pd.Series:
    """Month-over-month Food inflation % from the committed gold."""
    return cpi_to_mom(load_food_cpi(csv_path))


# ---------------------------------------------------------------------------
# Electricity CPI (COICOP 04.5.1) — monthly index (2018=100)
# ---------------------------------------------------------------------------

ELECTRICITY_CSV = HERE / 'data' / 'psa_electricity_cpi_monthly.csv'


def fetch_electricity_cpi(out_csv: Path = ELECTRICITY_CSV) -> None:
    """Fetch monthly Electricity (COICOP 04.5.1) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('04.5.1', out_csv, 'electricity_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 04.5.1 Electricity',
                          min_rows=50)


def load_electricity_cpi(csv_path: Path = ELECTRICITY_CSV) -> pd.Series:
    """Monthly Electricity CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['electricity_cpi'].astype(float).values,
                  index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_electricity_mom(csv_path: Path = ELECTRICITY_CSV) -> pd.Series:
    """Month-over-month Electricity inflation % from the committed gold."""
    return cpi_to_mom(load_electricity_cpi(csv_path))


# ---------------------------------------------------------------------------
# Food sub-categories (COICOP 01.1.x) — monthly index (2018=100)
#
# Confirmed live against openstat.psa.gov.ph's own "Commodity Description"
# dimension (not assumed from documentation): these six COICOP codes exist
# as directly selectable series in the same PX-Web table the top-level food
# series already comes from. PSA's CPI does not split poultry from
# beef/pork — '01.1.2 Meat' is one category at every COICOP depth this API
# exposes, a permanent limitation, not a gap to work around here.
# ---------------------------------------------------------------------------

RICE_CSV = HERE / 'data' / 'psa_rice_cpi_monthly.csv'
MEAT_CSV = HERE / 'data' / 'psa_meat_cpi_monthly.csv'
FISH_CSV = HERE / 'data' / 'psa_fish_cpi_monthly.csv'
DAIRY_EGGS_CSV = HERE / 'data' / 'psa_dairy_eggs_cpi_monthly.csv'
VEGETABLES_CSV = HERE / 'data' / 'psa_vegetables_cpi_monthly.csv'
SUGAR_CSV = HERE / 'data' / 'psa_sugar_cpi_monthly.csv'


def fetch_rice_cpi(out_csv: Path = RICE_CSV) -> None:
    """Fetch monthly Rice (COICOP 01.1.1.12) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.1.12', out_csv, 'rice_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.1.12 Rice',
                          min_rows=100)


def load_rice_cpi(csv_path: Path = RICE_CSV) -> pd.Series:
    """Monthly Rice CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['rice_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_rice_mom(csv_path: Path = RICE_CSV) -> pd.Series:
    """Month-over-month Rice inflation % from the committed gold."""
    return cpi_to_mom(load_rice_cpi(csv_path))


def fetch_meat_cpi(out_csv: Path = MEAT_CSV) -> None:
    """Fetch monthly Meat (COICOP 01.1.2) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.2', out_csv, 'meat_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.2 Meat',
                          min_rows=100)


def load_meat_cpi(csv_path: Path = MEAT_CSV) -> pd.Series:
    """Monthly Meat CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['meat_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_meat_mom(csv_path: Path = MEAT_CSV) -> pd.Series:
    """Month-over-month Meat inflation % from the committed gold."""
    return cpi_to_mom(load_meat_cpi(csv_path))


def fetch_fish_cpi(out_csv: Path = FISH_CSV) -> None:
    """Fetch monthly Fish and other seafood (COICOP 01.1.3) CPI from PSA OpenSTAT -> CSV."""
    fetch_cpi_subcategory('01.1.3', out_csv, 'fish_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.3 Fish and other seafood',
                          min_rows=100)


def load_fish_cpi(csv_path: Path = FISH_CSV) -> pd.Series:
    """Monthly Fish CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['fish_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_fish_mom(csv_path: Path = FISH_CSV) -> pd.Series:
    """Month-over-month Fish inflation % from the committed gold."""
    return cpi_to_mom(load_fish_cpi(csv_path))


def fetch_dairy_eggs_cpi(out_csv: Path = DAIRY_EGGS_CSV) -> None:
    """Fetch monthly Milk, dairy products & eggs (COICOP 01.1.4) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.4', out_csv, 'dairy_eggs_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.4 Milk, other dairy products and eggs',
                          min_rows=100)


def load_dairy_eggs_cpi(csv_path: Path = DAIRY_EGGS_CSV) -> pd.Series:
    """Monthly Dairy & eggs CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['dairy_eggs_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_dairy_eggs_mom(csv_path: Path = DAIRY_EGGS_CSV) -> pd.Series:
    """Month-over-month Dairy & eggs inflation % from the committed gold."""
    return cpi_to_mom(load_dairy_eggs_cpi(csv_path))


def fetch_vegetables_cpi(out_csv: Path = VEGETABLES_CSV) -> None:
    """Fetch monthly Vegetables, tubers, plantains & pulses (COICOP 01.1.7) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.7', out_csv, 'vegetables_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.7 Vegetables, tubers, plantains, cooking bananas and pulses',
                          min_rows=100)


def load_vegetables_cpi(csv_path: Path = VEGETABLES_CSV) -> pd.Series:
    """Monthly Vegetables CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['vegetables_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_vegetables_mom(csv_path: Path = VEGETABLES_CSV) -> pd.Series:
    """Month-over-month Vegetables inflation % from the committed gold."""
    return cpi_to_mom(load_vegetables_cpi(csv_path))


def fetch_sugar_cpi(out_csv: Path = SUGAR_CSV) -> None:
    """Fetch monthly Sugar, confectionery & desserts (COICOP 01.1.8) CPI -> CSV."""
    fetch_cpi_subcategory('01.1.8', out_csv, 'sugar_cpi',
                          'PSA OpenStat PX-Web CPI by commodity group, COICOP 01.1.8 Sugar, confectionery and desserts',
                          min_rows=100)


def load_sugar_cpi(csv_path: Path = SUGAR_CSV) -> pd.Series:
    """Monthly Sugar CPI index (2018=100) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['sugar_cpi'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_sugar_mom(csv_path: Path = SUGAR_CSV) -> pd.Series:
    """Month-over-month Sugar inflation % from the committed gold."""
    return cpi_to_mom(load_sugar_cpi(csv_path))
