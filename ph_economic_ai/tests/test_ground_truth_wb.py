import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import pytest

from ph_economic_ai.benchmark.ground_truth import load_world_bank_ron95, WB_CSV


def _write_csv(tmp_path, rows):
    p = tmp_path / 'wb.csv'
    lines = ['date,ron95_php_per_liter'] + [f'{d},{v}' for d, v in rows]
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return p


def test_loader_parses_and_sorts(tmp_path):
    # Deliberately out of order + a duplicate date (keep last)
    p = _write_csv(tmp_path, [
        ('2020-03', 45.10), ('2020-01', 44.00), ('2020-02', 44.50), ('2020-02', 44.55),
    ])
    s = load_world_bank_ron95(p)
    assert isinstance(s, pd.Series)
    assert list(s.index) == ['2020-01', '2020-02', '2020-03']
    assert s.index.is_monotonic_increasing
    assert s['2020-02'] == pytest.approx(44.55)   # duplicate -> last wins
    assert (s > 0).all()


def test_loader_index_is_year_month_strings(tmp_path):
    p = _write_csv(tmp_path, [('2019-11', 43.2), ('2019-12', 43.9)])
    s = load_world_bank_ron95(p)
    assert all(len(str(i)) == 7 and str(i)[4] == '-' for i in s.index)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_world_bank_ron95(tmp_path / 'does_not_exist.csv')


@pytest.mark.skipif(not WB_CSV.exists(),
                    reason='gold series not populated yet (run refresh_data.py)')
def test_committed_gold_series_is_backtest_ready():
    """Once the real World Bank series is committed, it must be long enough for a
    meaningful 1-month walk-forward backtest (min_train=24)."""
    s = load_world_bank_ron95()
    assert len(s) >= 24, 'gold series too short for a meaningful backtest'
    assert (s > 0).all()
    assert s.index.is_monotonic_increasing
