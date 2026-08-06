"""The band that replaces the single number.

Three point estimates would not improve accuracy; a calibrated interval improves
calibration, which is the thing that was actually missing. These tests pin the
coverage property and, just as importantly, that an uncalibrated band never
claims to be calibrated.
"""
import numpy as np
import pytest

from ph_economic_ai.engine import interval as iv


def test_halfwidth_grows_with_the_level():
    errors = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.4, 2.0, 2.5, 3.0]
    assert (iv.conformal_halfwidth(errors, 0.5)
            <= iv.conformal_halfwidth(errors, 0.8)
            <= iv.conformal_halfwidth(errors, 0.9))


def test_halfwidth_uses_the_finite_sample_rank():
    """ceil((n + 1) * level), not a plain percentile. This is what gives conformal
    its small-sample coverage guarantee."""
    errors = [1.0, 2.0, 3.0, 4.0]           # n = 4, level 0.5 -> rank ceil(2.5) = 3
    assert iv.conformal_halfwidth(errors, 0.5) == 3.0


def test_too_few_points_for_the_level_returns_the_largest_error():
    errors = [1.0, 2.0]                      # rank ceil(3 * 0.95) = 3 > n
    assert iv.conformal_halfwidth(errors, 0.95) == 2.0


def test_halfwidth_rejects_an_empty_history_rather_than_returning_zero():
    """A zero-width band would claim perfect precision."""
    with pytest.raises(ValueError):
        iv.conformal_halfwidth([], 0.5)


@pytest.mark.parametrize('level', [-0.1, 0.0, 1.0, 1.5])
def test_halfwidth_rejects_a_nonsense_level(level):
    with pytest.raises(ValueError):
        iv.conformal_halfwidth([0.5, 1.0], level)


def test_empirical_coverage_is_at_least_the_nominal_level():
    """The property the whole thing rests on: an 80 percent band should contain at
    least 80 percent of held-out errors."""
    rng = np.random.default_rng(4)
    calib = np.abs(rng.normal(0, 1.0, 500))
    future = np.abs(rng.normal(0, 1.0, 2000))
    for level in (0.5, 0.8, 0.9):
        half = iv.conformal_halfwidth(calib, level)
        covered = float(np.mean(future <= half))
        assert covered >= level - 0.03, f'level {level} covered only {covered:.3f}'


def test_band_is_centred_and_symmetric():
    errors = [0.2] * 20
    b = iv.band(1.15, errors, level=0.5)
    assert b['central'] == 1.15
    assert b['low'] == pytest.approx(0.95)
    assert b['high'] == pytest.approx(1.35)
    assert b['calibrated'] is True


def test_a_short_history_is_flagged_uncalibrated():
    """The UI must be able to tell the two apart, so this cannot silently pass."""
    b = iv.band(1.15, [0.3, 0.4], level=0.5)
    assert b['calibrated'] is False
    assert b['n_graded'] == 2
    assert 'graded runs needed' in b['source']


def test_no_history_still_produces_a_usable_band():
    b = iv.band(1.15, [], level=0.5)
    assert b['calibrated'] is False
    assert b['low'] < b['central'] < b['high']


def test_a_worse_track_record_widens_the_band():
    """The band must respond to being wrong, otherwise it is decoration."""
    accurate = [0.1] * 20
    poor = [1.5] * 20
    assert (iv.band(1.0, poor, 0.5)['half_width']
            > iv.band(1.0, accurate, 0.5)['half_width'])


def test_bands_returns_the_lead_and_the_expanded_level():
    out = iv.bands(1.15, [0.3] * 20)
    assert set(out) == {'0.5', '0.9'}
    assert out['0.9']['half_width'] >= out['0.5']['half_width']


def test_format_band_reads_as_a_range_not_a_point():
    line = iv.format_band(iv.band(1.15, [0.5] * 20, 0.5))
    assert 'to' in line and '50% band' in line


def test_nan_errors_are_ignored_rather_than_poisoning_the_quantile():
    """NaN is dropped, so the quantile is taken over the 3 real values:
    rank = ceil((3 + 1) * 0.5) = 2, giving the second smallest."""
    assert iv.conformal_halfwidth([0.2, float('nan'), 0.4, 0.6], 0.5) == pytest.approx(0.4)
    assert iv.conformal_halfwidth([0.2, 0.4, 0.6], 0.5) == pytest.approx(0.4)


def test_store_graded_errors_feed_the_band(tmp_path):
    """End to end: the app's own horizon-matched grades calibrate its band."""
    from ph_economic_ai.engine.store import AgentTrustStore
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    assert s.get_graded_errors() == []

    for i in range(14):
        rid = s.save_run(scenario={'current_price': 60.0}, final_estimate=1.0,
                         confidence_pct=70, horizon_days=-1.0)
        s.apply_ground_truth_grade(rid, actual_change=1.0 + (i % 5) * 0.1)

    errors = s.get_graded_errors()
    assert len(errors) == 14
    b = iv.band(1.15, errors, level=0.5)
    assert b['calibrated'] is True
    assert b['n_graded'] == 14


# ── Per-sector units ─────────────────────────────────────────────────────────
# The store grades only fuel, against DOE peso-per-litre prices. Applying that
# error history to a percentage sector would produce a band in the wrong unit and
# label it "calibrated", which is worse than showing no band at all.

def test_an_ungraded_sector_never_reports_calibrated():
    fuel_errors = [0.3] * 20
    b = iv.band(0.31, fuel_errors, 0.5, sector='food')
    assert b['calibrated'] is False
    assert b['n_graded'] == 0
    assert 'no graded outcome series' in b['source']


def test_fuel_errors_do_not_leak_into_a_percentage_band():
    """The bug this guards: PHP/L widths wrapped around a percent estimate."""
    fuel_errors = [2.5] * 20                     # wide, in PHP/L
    food = iv.band(0.31, fuel_errors, 0.5, sector='food')
    assert food['half_width'] == iv.FALLBACK_HALFWIDTH['food'][0.5]
    assert food['half_width'] < 1.0


def test_the_graded_sector_still_calibrates():
    b = iv.band(1.15, [0.3] * 20, 0.5, sector='gas')
    assert b['calibrated'] is True


def test_each_sector_has_its_own_scale():
    errs = []
    gas = iv.band(1.0, errs, 0.5, sector='gas')['half_width']
    food = iv.band(1.0, errs, 0.5, sector='food')['half_width']
    elec = iv.band(1.0, errs, 0.5, sector='electricity')['half_width']
    assert gas != food and food != elec


def test_an_unknown_sector_is_refused_rather_than_given_another_unit():
    """This asserted the opposite, by name: "falls back without crashing".

    What it was protecting was a silent substitution. An unknown sector received
    the `gas` prior, +/-0.60 PHP/L, and a card would render that beside whatever
    unit the sector actually uses. The three real sectors differ by more than a
    factor of two, so the substitution is a wrong number rather than a rough one.

    `band` already refuses the mirror image -- it drops graded fuel errors for a
    percentage sector, "so that is refused rather than trusted to call sites" --
    and the fallback table was the same hazard left open. Both live call sites
    pass a real sector, so nothing in the app changes; a typo now says so.
    """
    with pytest.raises(ValueError) as e:
        iv.band(1.0, [], 0.5, sector='mystery')
    assert 'wrong unit' in str(e.value)

    for sector in ('gas', 'food', 'electricity'):
        b = iv.band(1.0, [], 0.5, sector=sector)
        assert b['half_width'] > 0 and b['calibrated'] is False


def test_bands_passes_the_sector_through():
    out = iv.bands(0.31, [0.3] * 20, sector='food')
    assert all(not v['calibrated'] for v in out.values())
