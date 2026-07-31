"""The level-premium test that corrected nine multipliers.

Pre-registered at `docs/preregistration/2026-08-01-regional-level-premiums.md`,
which also discloses that two of the twelve ratios had already been seen.

Phase 2 asked whether a freight premium should scale a price CHANGE and closed as
infeasible, needing 527 to 779 years of weekly data to resolve 5 percent of a
slope. This asks whether the constants are right as the LEVEL premiums they are
documented to be, where the signal is about seventy times larger, and answers it
from 2023 to 2026 of DOE prices.

Nine of eleven testable multipliers were refuted and corrected. **MIMAROPA and
Bicol were tested and left alone**, and they carry the weight here: an estimator
that flattened every region toward 1.00 would have flattened those two as well.
"""
import datetime as dt

import pytest

from ph_economic_ai.tools import regional_level_premiums as lp


# ── The decision rule is the pre-registered one ──────────────────────────────

def test_a_multiplier_inside_its_corrected_interval_is_left_alone():
    """MIMAROPA's shape: claimed 1.08, measured 1.092, corrected interval
    [1.079, 1.104] covers it. The dispersion here is set to match that real
    interval width; a tighter fixture would exclude 1.08 and test nothing but
    its own arithmetic."""
    ratios = [1.092 + 0.05 * (i % 9 - 4) for i in range(150)]
    out = lp.assess(ratios, 1.08, family=11)
    assert out['verdict'] == 'CONSISTENT'
    assert out['proposed'] == 1.08, 'a consistent multiplier is never rewritten'
    assert out['corrected_ci_low'] <= 1.08 <= out['corrected_ci_high']


def test_a_multiplier_outside_its_interval_is_replaced_by_the_median():
    """The pre-registered correction policy: median ratio, two decimals."""
    ratios = [0.962 + 0.003 * (i % 5 - 2) for i in range(120)]
    out = lp.assess(ratios, 1.05, family=11)
    assert out['verdict'] == 'WRONG SIZE'
    assert out['proposed'] == 0.96


def test_a_region_indistinguishable_from_the_base_is_called_that():
    ratios = [1.000 + 0.003 * (i % 5 - 2) for i in range(120)]
    out = lp.assess(ratios, 1.05, family=11)
    assert out['verdict'] == 'NO PREMIUM DETECTABLE'
    assert out['proposed'] == 1.0


def test_the_correction_is_rounded_to_the_precision_the_table_is_written_in():
    """Two decimals, because proposing 0.9617 would imply a precision the weekly
    medians do not carry."""
    ratios = [0.96173 + 0.0001 * (i % 3) for i in range(120)]
    assert lp.assess(ratios, 1.05, family=11)['proposed'] == 0.96


# ── The interval widens with the family, and is not a p-value ────────────────

def test_a_larger_family_widens_the_interval():
    """Bonferroni applied to the interval, per the pre-registration, so a bigger
    family makes correction HARDER rather than easier."""
    ratios = [0.99 + 0.01 * (i % 7 - 3) for i in range(150)]
    narrow = lp.assess(ratios, 1.05, family=1)
    wide = lp.assess(ratios, 1.05, family=11)
    span = lambda r: r['corrected_ci_high'] - r['corrected_ci_low']
    assert span(wide) > span(narrow)


def test_the_uncorrected_interval_is_reported_alongside():
    """Both are printed, so a reader can see what the correction cost."""
    ratios = [0.99 + 0.01 * (i % 7 - 3) for i in range(150)]
    out = lp.assess(ratios, 1.05, family=11)
    assert out['ci_low'] >= out['corrected_ci_low']
    assert out['ci_high'] <= out['corrected_ci_high']


def test_the_bootstrap_is_reproducible():
    """A fixed seed was pre-registered so the interval is not a lottery."""
    ratios = [0.99 + 0.01 * (i % 7 - 3) for i in range(150)]
    assert lp.assess(ratios, 1.05, 11) == lp.assess(ratios, 1.05, 11)


def test_a_wide_interval_is_marked_uninformative():
    """Wider than 0.10 is twice the largest premium in the table, so it cannot
    speak to a 3-to-10 percent claim and must not trigger a correction."""
    ratios = [1.0 + 0.4 * (i % 11 - 5) for i in range(150)]
    assert lp.assess(ratios, 1.05, family=11)['uninformative']


# ── South Luzon names its region in the filename ─────────────────────────────

@pytest.mark.parametrize('name,region', [
    ('petro_sluz_2021-apr-08_laguna', 'CALABARZON'),
    ('petro_sluz_2022-may-10_cavite', 'CALABARZON'),
    ('petro_sluz_2023-jan-03_batangas_rizal_quezon', 'CALABARZON'),
    ('petro_sluz_2024-feb-06_mimaropa', 'MIMAROPA'),
    ('petro_sluz_2024-feb-06_minaropa', 'MIMAROPA'),
    ('petro_sluz_2025-mar-04_bicol_region', 'Bicol Region'),
])
def test_a_south_luzon_file_names_its_region(name, region):
    """These sheets are published per REGION GROUP and carry no province column,
    so those rows would otherwise contribute nothing. Including the three regions
    was pre-registered; this is how they are assigned."""
    assert lp.region_of_south_luzon_file(name) == region


def test_a_south_luzon_file_naming_no_region_is_refused():
    assert lp.region_of_south_luzon_file('petro_sluz_2021-apr-08') is None


# ── Coverage is never inferred ───────────────────────────────────────────────

def test_the_four_regions_with_no_source_carry_empty_province_sets():
    """DOE publishes nothing north of NCR. They are listed rather than omitted so
    a coverage table cannot read as complete (`DEC-044`), and they keep their
    unfitted multipliers because no data can refute them."""
    from ph_economic_ai.tools.doe_price_series import REGION_PROVINCES
    for region in ('Ilocos Region', 'Cagayan Valley', 'Central Luzon', 'CAR'):
        assert REGION_PROVINCES[region] == frozenset()

    from ph_economic_ai.engine import swarm
    live = {g['name']: g['multiplier'] for g in swarm.ALL_REGIONS}
    assert live['Central Luzon'] == 1.02, 'untestable, so untouched'
    assert live['CAR'] == 1.08


def test_every_engine_region_appears_in_the_province_map():
    """Otherwise a region could be silently unreachable by any future test."""
    from ph_economic_ai.engine import swarm
    from ph_economic_ai.tools.doe_price_series import REGION_PROVINCES
    assert {g['name'] for g in swarm.ALL_REGIONS} == set(REGION_PROVINCES)


# ── What was actually changed ────────────────────────────────────────────────

def test_the_corrected_multipliers_are_what_the_rule_proposed():
    """Pins the nine corrections against the measured medians, so a later edit
    cannot drift them back toward the unfitted originals."""
    from ph_economic_ai.engine import swarm
    live = {g['name']: g['multiplier'] for g in swarm.ALL_REGIONS}
    for region, value in {
        'CALABARZON': 0.98, 'Western Visayas': 1.00, 'Central Visayas': 1.00,
        'Eastern Visayas': 1.00, 'Zamboanga': 0.98, 'Northern Mindanao': 0.99,
        'Davao Region': 0.96, 'SOCCSKSARGEN': 0.99, 'Caraga': 1.02,
    }.items():
        assert live[region] == value, region


def test_the_positive_control_regions_were_left_at_their_original_values():
    """The single most important assertion here. MIMAROPA and Bicol were tested
    on the same data by the same estimator and came back CONSISTENT, at 1.092 and
    1.049 against claims of 1.08 and 1.06. If a later change flattens these two
    toward 1.00, the nine corrections stop being evidence about freight and start
    being evidence about the method."""
    from ph_economic_ai.engine import swarm
    live = {g['name']: g['multiplier'] for g in swarm.ALL_REGIONS}
    assert live['MIMAROPA'] == 1.08
    assert live['Bicol Region'] == 1.06
