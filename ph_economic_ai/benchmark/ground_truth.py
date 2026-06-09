"""Load the committed World Bank RON95 monthly gold series.

The CSV at WB_CSV is the frozen, citable ground truth used by the backtest. It
is populated by `refresh_data.py` from the World Bank Global Fuel Prices DB
(manual workbook download — see that module). Until populated, callers that need
the series should handle FileNotFoundError.
"""
from pathlib import Path

import pandas as pd

WB_CSV = Path(__file__).parent / 'data' / 'world_bank_ron95.csv'


def load_world_bank_ron95(csv_path: Path = WB_CSV) -> pd.Series:
    """Return RON95 PHP/liter as a Series indexed by 'YYYY-MM' (sorted ascending).

    Raises FileNotFoundError if the gold CSV has not been populated yet.
    """
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(
        df['ron95_php_per_liter'].astype(float).values,
        index=df['date'].astype(str).values,
    )
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.index.name = 'date'
    return s
