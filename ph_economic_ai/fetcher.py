import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

CACHE_PATH = Path(__file__).parent / 'cache' / 'data.json'
CACHE_TTL_HOURS = 24
FETCH_TIMEOUT = 8

_YAHOO_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json',
}


def fetch_dataset() -> tuple[pd.DataFrame, str]:
    raise NotImplementedError


def _fetch_all() -> pd.DataFrame:
    raise NotImplementedError


def _load_cache(cache_path: Path = CACHE_PATH) -> tuple[Optional[pd.DataFrame], bool]:
    raise NotImplementedError


def _save_cache(df: pd.DataFrame, cache_path: Path = CACHE_PATH) -> None:
    raise NotImplementedError


def _fetch_yahoo(ticker: str) -> pd.Series:
    raise NotImplementedError


def _fetch_doe_prices() -> pd.Series:
    raise NotImplementedError


def _compute_demand(dates: list[str]) -> list[float]:
    raise NotImplementedError
