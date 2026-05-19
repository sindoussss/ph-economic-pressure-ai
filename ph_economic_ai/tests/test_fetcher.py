import json
import sys
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.fetcher import _compute_demand, _load_cache, _save_cache


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'date': ['2024-01', '2024-02'],
        'oil_price': [80.0, 82.0],
        'usd_php': [56.0, 56.5],
        'demand_index': [72.0, 68.0],
        'gas_price': [65.0, 66.0],
    })


def test_compute_demand_range():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    assert all(55.0 <= v <= 90.0 for v in values), f'Out of range: {values}'


def test_compute_demand_peaks():
    dates = [f'2024-{m:02d}' for m in range(1, 13)]
    values = _compute_demand(dates)
    june = values[5]   # index 5 = June
    assert values[2] > june, f'March ({values[2]:.1f}) should be > June ({june:.1f})'
    assert values[11] > june, f'December ({values[11]:.1f}) should be > June ({june:.1f})'


def test_cache_roundtrip(tmp_path):
    cache_file = tmp_path / 'data.json'
    df = _sample_df()
    _save_cache(df, cache_path=cache_file)
    loaded_df, _ = _load_cache(cache_path=cache_file)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True),
        loaded_df.reset_index(drop=True),
        check_dtype=False,
    )


def test_fresh_cache_is_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    _save_cache(_sample_df(), cache_path=cache_file)
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert is_fresh


def test_stale_cache_is_not_fresh(tmp_path):
    cache_file = tmp_path / 'data.json'
    payload = {
        'fetched_at': (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat(),
        'data': _sample_df().to_dict(orient='records'),
    }
    cache_file.write_text(json.dumps(payload), encoding='utf-8')
    _, is_fresh = _load_cache(cache_path=cache_file)
    assert not is_fresh
