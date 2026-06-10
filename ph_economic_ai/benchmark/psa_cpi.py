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
