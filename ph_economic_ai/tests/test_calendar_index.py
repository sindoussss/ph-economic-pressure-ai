"""Calendar-aware lagging.

The defect these guard against: frames are indexed 'YYYY-MM' and were lagged with
`shift(n)`, which moves by ROW. On a gapped index a one-row lag silently becomes a
two- or three-month lag, and nothing raises.
"""
import numpy as np
import pandas as pd
import pytest

from ph_economic_ai.benchmark.calendar_index import (
    calendar_lag, is_complete, missing_months, reindex_complete, to_periods)


def _s(pairs):
    return pd.Series([v for _, v in pairs], index=[k for k, _ in pairs], dtype=float)


COMPLETE = _s([('2020-01', 1.0), ('2020-02', 2.0), ('2020-03', 3.0), ('2020-04', 4.0)])
# 2020-03 absent: the row after the gap is 2020-04, whose true predecessor is missing.
GAPPED = _s([('2020-01', 1.0), ('2020-02', 2.0), ('2020-04', 4.0), ('2020-05', 5.0)])


def test_missing_months_finds_the_hole():
    assert missing_months(GAPPED.index) == ['2020-03']
    assert missing_months(COMPLETE.index) == []
    assert is_complete(COMPLETE.index) and not is_complete(GAPPED.index)


def test_missing_months_handles_empty_and_single():
    assert missing_months([]) == []
    assert missing_months(['2020-01']) == []


def test_missing_months_spans_a_year_boundary():
    idx = ['2019-11', '2019-12', '2020-02']
    assert missing_months(idx) == ['2020-01']


def test_lag_matches_shift_when_the_index_is_complete():
    """The whole point: on clean data this changes nothing."""
    pd.testing.assert_series_equal(calendar_lag(COMPLETE, 1), COMPLETE.shift(1))
    pd.testing.assert_series_equal(calendar_lag(COMPLETE, 2), COMPLETE.shift(2))


def test_lag_across_a_gap_is_nan_not_a_wrong_value():
    """The regression. `shift(1)` hands 2020-04 the value from 2020-02 -- a two-month
    lag wearing a one-month label. The calendar lag refuses and yields NaN."""
    assert GAPPED.shift(1).loc['2020-04'] == 2.0        # the defect, still reproducible
    got = calendar_lag(GAPPED, 1)
    assert np.isnan(got.loc['2020-04'])                 # 2020-03 does not exist
    assert got.loc['2020-02'] == 1.0                    # real predecessors survive
    assert got.loc['2020-05'] == 4.0
    assert np.isnan(got.loc['2020-01'])                 # nothing before the start


def test_gapped_rows_are_dropped_rather_than_kept_wrong():
    frame = pd.DataFrame({'target': GAPPED})
    frame['prev'] = calendar_lag(frame['target'], 1)
    kept = frame.dropna()
    assert list(kept.index) == ['2020-02', '2020-05']
    assert list(kept['prev']) == [1.0, 4.0]


def test_seasonal_lag_of_twelve_months():
    idx = pd.period_range('2020-01', '2021-12', freq='M').strftime('%Y-%m')
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    got = calendar_lag(s, 12)
    assert np.isnan(got.iloc[:12]).all()
    assert got.loc['2021-01'] == s.loc['2020-01']
    assert got.loc['2021-12'] == s.loc['2020-12']


def test_seasonal_lag_respects_a_hole_twelve_months_back():
    idx = [p for p in pd.period_range('2020-01', '2021-06', freq='M').strftime('%Y-%m')
           if p != '2020-05']
    s = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    got = calendar_lag(s, 12)
    assert np.isnan(got.loc['2021-05'])                 # 2020-05 is absent
    assert got.loc['2021-06'] == s.loc['2020-06']


def test_lag_preserves_index_name_and_order():
    s = COMPLETE.rename('target')
    got = calendar_lag(s, 1)
    assert got.name == 'target'
    assert list(got.index) == list(s.index)


def test_lag_on_an_unsorted_index_uses_the_calendar_not_position():
    shuffled = GAPPED.iloc[[3, 0, 2, 1]]
    got = calendar_lag(shuffled, 1)
    assert got.loc['2020-05'] == 4.0
    assert np.isnan(got.loc['2020-04'])
    assert got.loc['2020-02'] == 1.0


def test_reindex_complete_makes_absence_visible_without_inventing_values():
    frame = pd.DataFrame({'a': GAPPED})
    out = reindex_complete(frame)
    assert list(out.index) == ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05']
    assert np.isnan(out.loc['2020-03', 'a'])
    assert out['a'].notna().sum() == 4


def test_to_periods_keeps_only_the_month_part():
    """Documented leniency: a daily string collapses to its month."""
    assert list(to_periods(['2020-01-15']).astype(str)) == ['2020-01']


def test_a_duplicated_month_fails_loudly_rather_than_misaligning():
    """A daily index collapses many rows onto one period. Silent misalignment there
    would be the same class of bug this module exists to remove, so it must raise."""
    s = pd.Series([1.0, 2.0], index=['2020-01-05', '2020-01-20'])
    with pytest.raises(ValueError, match='duplicate'):
        calendar_lag(s, 1)


def _shipped_frame(builder):
    if builder == 'nowcast':
        from ph_economic_ai.benchmark.nowcast import build_nowcast_frame
        from ph_economic_ai.benchmark.targets import load_inflation_mom
        return (build_nowcast_frame(target_loader=load_inflation_mom, prev_col='prev_mom'),
                load_inflation_mom())
    if builder == 'food':
        from ph_economic_ai.benchmark.food_nowcast import (
            _build_food_frame, load_food_features, load_food_mom)
        return _build_food_frame(load_food_features()), load_food_mom()
    from ph_economic_ai.benchmark.electricity_nowcast import (
        _build_electricity_frame, load_electricity_features, load_electricity_mom)
    return _build_electricity_frame(load_electricity_features()), load_electricity_mom()


@pytest.mark.parametrize('builder', ['nowcast', 'food', 'electricity'])
def test_shipped_frames_carry_no_fabricated_lag(builder):
    """Every retained row's prev_mom must be the value of the month immediately
    before it in the SOURCE target series.

    Not in the frame's own index: a real predecessor may itself have been dropped
    for lacking a predecessor. What must never happen is prev_mom carrying a value
    from two or three months back, which is exactly what the row shift did.
    """
    frame, source = _shipped_frame(builder)
    src = pd.Series(source.to_numpy(dtype=float), index=to_periods(source.index))
    assert not src.index.has_duplicates

    for period, claimed in zip(to_periods(frame.index), frame['prev_mom'].to_numpy()):
        true_prev = period - 1
        assert true_prev in src.index, (
            f'{builder}: {period} kept, but {true_prev} is absent from the source')
        assert claimed == pytest.approx(src.loc[true_prev]), (
            f'{builder}: {period} carries {claimed}, but {true_prev} is '
            f'{src.loc[true_prev]}')
