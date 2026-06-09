"""Target registry for the predictability audit. Each Target abstracts a
Philippine economic series so the same panel + DM machinery can audit it.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ph_economic_ai.benchmark.features import build_target_frame
from ph_economic_ai.benchmark.ground_truth import load_world_bank_ron95

DATA = Path(__file__).parent / 'data'
FEATURES_CSV = DATA / 'features_monthly.csv'
FX_CSV = DATA / 'usd_php_monthly.csv'
CPI_CSV = DATA / 'ph_cpi_monthly.csv'


@dataclass
class Target:
    name: str
    load_gold: Callable[[], pd.Series]
    build_frame: Callable[[], pd.DataFrame]
    has_mechanism: bool = False


def _features() -> pd.DataFrame:
    return pd.read_csv(FEATURES_CSV, dtype={'date': str}).set_index('date').sort_index()


def load_fx(csv_path: Path = FX_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['usd_php'].astype(float).values, index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def cpi_to_yoy(cpi_index: pd.Series) -> pd.Series:
    """Convert a monthly CPI index to year-on-year inflation %, dropping first 12."""
    s = cpi_index.sort_index()
    yoy = (s / s.shift(12) - 1.0) * 100.0
    return yoy.dropna()


def load_inflation(csv_path: Path = CPI_CSV) -> pd.Series:
    df = pd.read_csv(csv_path, dtype={'date': str})
    cpi = pd.Series(df['cpi_index'].astype(float).values, index=df['date'].astype(str).values)
    cpi = cpi[~cpi.index.duplicated(keep='last')]
    return cpi_to_yoy(cpi)


def _fuel_frame() -> pd.DataFrame:
    gold = load_world_bank_ron95()
    drivers = _features()
    return build_target_frame(gold, drivers, 'fuel',
                              ['oil_price', 'usd_php', 'gas_price', 'demand_index'])


def _fx_frame() -> pd.DataFrame:
    fx = load_fx()
    feats = _features()
    drivers = pd.DataFrame({'oil': feats['oil_price']})
    drivers = drivers.join(load_inflation().rename('inflation'), how='outer')
    return build_target_frame(fx, drivers, 'fx', ['oil', 'inflation'])


def _inflation_frame() -> pd.DataFrame:
    infl = load_inflation()
    feats = _features()
    drivers = pd.DataFrame({'fuel': feats['gas_price'], 'fx': feats['usd_php']})
    return build_target_frame(infl, drivers, 'inflation', ['fuel', 'fx'])


TARGETS = {
    'fuel': Target('fuel', load_world_bank_ron95, _fuel_frame, has_mechanism=True),
    'fx': Target('fx', load_fx, _fx_frame, has_mechanism=False),
    'inflation': Target('inflation', load_inflation, _inflation_frame, has_mechanism=False),
}


def cpi_to_mom(cpi_index: pd.Series) -> pd.Series:
    """Convert a monthly CPI index to month-over-month inflation %, dropping the
    first (undefined) month."""
    s = cpi_index.sort_index()
    mom = (s / s.shift(1) - 1.0) * 100.0
    return mom.dropna()


def load_inflation_mom(csv_path: Path = CPI_CSV) -> pd.Series:
    """Load the committed CPI index and return month-over-month inflation %."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    cpi = pd.Series(df['cpi_index'].astype(float).values, index=df['date'].astype(str).values)
    cpi = cpi[~cpi.index.duplicated(keep='last')]
    return cpi_to_mom(cpi)
