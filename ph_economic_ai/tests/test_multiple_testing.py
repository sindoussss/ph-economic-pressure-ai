"""Multiple-comparison correction — the M1 reviewer defense.

Verifies the Bonferroni/BH machinery against known cases, and that the family it
corrects is the one the benchmark actually tested. The second half exists because
the record was empty for real: `build_family` read only `accuracy_report.json`,
every verdict there is a null under the corrected pool, and so the artifact behind
the project's one positive result (`fuel_audit`, `confirmed_on_holdout`) said
`n_tests = 0` while `selection_holdout.json` held 23 untouched DM tests.
"""
import json
from pathlib import Path

import pytest

from ph_economic_ai.benchmark import multiple_testing as mt


# ── Corrections against known cases ───────────────────────────────────────────

def test_bonferroni_threshold():
    # m=2, alpha=0.05 -> threshold 0.025
    assert mt.bonferroni_reject([0.01, 0.02], 0.05) == [True, True]
    assert mt.bonferroni_reject([0.03, 0.04], 0.05) == [False, False]


def test_bonferroni_adjusted_is_p_times_m_capped():
    assert mt.bonferroni_adjusted([0.01, 0.2, 0.5]) == pytest.approx([0.03, 0.6, 1.0])


def test_bh_step_up_textbook_case():
    """Benjamini–Hochberg (1995) worked example shape: reject the k smallest."""
    ps = [0.005, 0.01, 0.03, 0.06]           # crit (i/4)·0.05 = .0125,.025,.0375,.05
    # 0.005≤.0125, 0.01≤.025, 0.03≤.0375, 0.06>.05 -> reject first 3
    assert mt.benjamini_hochberg_reject(ps, 0.05) == [True, True, True, False]


def test_bh_is_at_least_as_powerful_as_bonferroni():
    ps = [0.0005, 0.001, 0.0011, 0.0046, 0.0214, 0.0323]
    bonf = mt.benjamini_hochberg_reject(ps, 0.05)
    bh = mt.benjamini_hochberg_reject(ps, 0.05)
    # BH never rejects fewer than Bonferroni
    assert sum(bh) >= sum(mt.bonferroni_reject(ps, 0.05))


def test_bh_adjusted_is_monotone_and_capped():
    q = mt.benjamini_hochberg_adjusted([0.0005, 0.02, 0.5, 0.9])
    assert all(0.0 <= x <= 1.0 for x in q)
    ordered = [q[i] for i in sorted(range(len(q)), key=lambda i: [0.0005, 0.02, 0.5, 0.9][i])]
    assert ordered == sorted(ordered)        # non-decreasing in p


def test_empty_family_does_not_crash():
    assert mt.benjamini_hochberg_reject([], 0.05) == []
    assert mt.bonferroni_reject([], 0.05) == []


# ── The actual family from the frozen report ──────────────────────────────────

_CANDIDATE_PATHS = [
    ('nowcast_mom',), ('mom_longsample', 'mom'), ('food_nowcast', 'mom'),
    ('electricity_nowcast', 'mom'), ('electricity_nowcast', 'driver_ablation'),
    ('transport_nowcast', 'driver_ablation'),
]


def _node(report, path):
    node = report
    for key in path:
        node = node.get(key, {}) if isinstance(node, dict) else {}
    return node


def test_family_is_the_confirmatory_tests_only():
    """The family is DERIVED from the report's verdicts, never hardcoded — so this
    stays honest whichever way the verdicts fall."""
    report = json.loads(
        (Path(mt._ARTIFACTS) / 'accuracy_report.json').read_text())
    family = mt.build_family(report)
    assert all(f['dm_p'] is not None for f in family)
    expected = sum(1 for p in _CANDIDATE_PATHS
                   if _node(report, p).get('verdict') == 'beats_best_naive'
                   and _node(report, p).get('dm_p') is not None)
    assert len(family) == expected      # exactly the real positives, no more


def test_corrected_pool_leaves_no_confirmatory_positives_in_the_report():
    """With the historical mean in the baseline pool (§4.7), every MoM verdict in
    `accuracy_report.json` is a null, so the report contributes nothing. The
    empty case must be handled, not crash."""
    report = json.loads((Path(mt._ARTIFACTS) / 'accuracy_report.json').read_text())
    assert mt.build_family(report) == []


def test_empty_family_correction_does_not_divide_by_zero():
    r = mt.correct([])
    assert r['n_tests'] == 0
    assert r['survive_bonferroni'] == []
    assert r['survive_bh_only'] == []
    assert r['bonferroni_threshold'] is None
    assert r['expected_false_positives'] == 0.0


def test_survivor_logic_still_discriminates_on_a_synthetic_family():
    """Machinery guard: keep proving the correction can separate a strong
    positive from a weak one, so a future genuine finding is classified
    correctly."""
    fam = [{'test': 'strong', 'skill_vs_naive': 0.30, 'dm_p': 0.0005},
           {'test': 'weak', 'skill_vs_naive': 0.10, 'dm_p': 0.032}]
    r = mt.correct(fam)
    assert r['survive_bonferroni'] == ['strong']       # p < 0.05/2
    assert r['survive_bh_only'] == ['weak']            # FDR only


# ── The selection-holdout family ──────────────────────────────────────────────

def _holdout() -> dict:
    return json.loads((Path(mt._ARTIFACTS) / 'selection_holdout.json').read_text())


def test_selection_family_covers_every_holdout_dm_test():
    """Every row `selection.run()` actually tested is a family member. The count
    is derived from the artifact, never hardcoded, so adding a target to
    `selection.run()` widens the family instead of silently escaping it."""
    holdout = _holdout()
    family = mt.build_selection_family(holdout)
    expected = [k for k, v in holdout.items() if v.get('holdout_dm_p') is not None]
    assert [f['key'] for f in family] == expected
    assert all(f['source'] == 'selection_holdout.json' for f in family)


def test_selection_family_skips_rows_with_no_p_value():
    """An `insufficient_data` row was never tested, so it must not inflate the
    denominator every other test is divided by."""
    holdout = {'tested': {'verdict': 'not_confirmed_on_holdout', 'holdout_dm_p': 0.4,
                          'holdout_skill': -0.01},
               'untested': {'verdict': 'insufficient_data', 'n': 30, 'cut': 21}}
    family = mt.build_selection_family(holdout)
    assert [f['key'] for f in family] == ['tested']


def test_selection_family_records_which_side_a_p_value_falls_on():
    """A two-sided DM test can reject because the model is significantly WORSE.
    `dairy_eggs_mom_driver_only` (-35.0%, p = 0.0201) is such a row: recording it
    without direction would let a reader count it as a win."""
    family = {f['key']: f for f in mt.build_selection_family(_holdout())}
    assert family['fuel_audit']['direction'] == 'favours_model'
    assert family['dairy_eggs_mom_driver_only']['direction'] == 'favours_naive'
    assert family['sugar_mom_driver_only']['direction'] == 'favours_naive'


# ── The populated record ──────────────────────────────────────────────────────

def test_the_record_is_not_empty():
    """The regression this file exists for: a repo whose stated purpose is
    refusing to overclaim shipped an empty multiplicity record behind its one
    positive result."""
    result = mt.run()
    # Holdout rows PLUS the weekly-gas hypothesis. Pinned as a sum rather than
    # to a literal so adding a family member is a deliberate edit here, not a
    # number that drifts unnoticed.
    assert result['n_tests'] == len(_holdout()) + 1
    assert result['tests'], 'the multiplicity record must not be empty'
    assert result['bonferroni_threshold'] == pytest.approx(0.05 / result['n_tests'], abs=1e-5)


def test_fuel_audit_survives_neither_correction():
    """The one `confirmed_on_holdout` positive, measured against the family it
    was actually selected from. At 24 tests the Bonferroni threshold is 0.0021
    and p = 0.0296 is an order of magnitude away from it.

    `bonferroni_p` moved from 0.6808 to 0.7104 when weekly gas joined the family,
    because Bonferroni multiplies by m and m went 23 to 24. That is the retroactive
    tightening `docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md`
    warned about when the family grows. It flipped nothing -- both figures are far
    above alpha -- and `test_growing_the_family_flipped_no_verdict` checks that
    directly rather than leaving it to inspection."""
    result = mt.run()
    fuel = next(t for t in result['tests'] if t['key'] == 'fuel_audit')
    assert fuel['dm_p'] == pytest.approx(0.0296)
    assert fuel['bonferroni_p'] == pytest.approx(0.7104, abs=1e-4)
    assert fuel['bh_q'] == pytest.approx(0.2368, abs=1e-4)   # 0.2269 at m = 23
    assert fuel['survives_bonferroni'] is False
    assert fuel['survives_bh'] is False
    assert fuel['test'] in result['survive_neither']
    assert result['survive_bonferroni'] == []
    assert result['survive_bh_only'] == []


def test_the_record_states_how_many_hits_chance_alone_buys():
    """Three of 24 land under 0.05 and ~1.2 are expected by chance. Reporting
    the count without that expectation is how a coin-flip becomes a finding."""
    result = mt.run()
    assert result['expected_false_positives'] == pytest.approx(1.2, abs=0.01)
    nominal = [t for t in result['tests'] if t['nominally_significant']]
    assert len(nominal) == 3
    assert sum(t['direction'] == 'favours_naive' for t in nominal) == 2


# ── The weekly-gas hypothesis joins the family ───────────────────────────────

def test_weekly_gas_enters_as_one_test_not_one_per_spec():
    """The module's own rule: "a panel is one hypothesis tested with K
    candidates, not K hypotheses". Seven specifications were tried on the weekly
    target across PR #31 and PR #36; counting each would inflate m with exactly
    the multiplicity the two-stage holdout already handles.
    """
    validation = {'skill': 0.1469, 'hac_dm_t': -2.093, 'sign_test_p': 0.0078,
                  'holdout': {'skill': 0.1715, 'hac_t': -1.54, 'confirmed': False}}
    family = mt.build_weekly_gas_family(validation)
    assert len(family) == 1
    assert family[0]['key'] == 'weekly_gas'


def test_it_is_scored_on_the_holdout_like_every_other_member():
    """`build_selection_family` uses each row's HOLDOUT DM p, not its
    in-selection p. Scoring the weekly result on its full-sample statistic while
    everyone else is scored on a holdout would give it an easier test than the
    family it is joining.
    """
    validation = {'skill': 0.1469, 'hac_dm_t': -2.093, 'sign_test_p': 0.0078,
                  'holdout': {'skill': 0.1715, 'hac_t': -1.54, 'confirmed': False}}
    entry = mt.build_weekly_gas_family(validation)[0]
    # t = -1.54 two-sided is ~0.12, nowhere near the full-sample t = -2.09 (~0.037)
    assert 0.10 < entry['dm_p'] < 0.15
    assert entry['holdout_verdict'] == 'not_confirmed_on_holdout'


def test_the_stronger_evidence_travels_as_context_not_as_extra_members():
    """The full-sample DM and the 7/7 sign test are real and should be visible,
    but they test the SAME claim. Adding them as rows would charge one
    hypothesis three times -- the error the module already refuses for the
    audit's fuel row."""
    validation = {'skill': 0.1469, 'hac_dm_t': -2.093, 'sign_test_p': 0.0078,
                  'holdout': {'skill': 0.1715, 'hac_t': -1.54, 'confirmed': False}}
    family = mt.build_weekly_gas_family(validation)
    assert len(family) == 1
    entry = family[0]
    assert entry['full_sample_dm_t'] == pytest.approx(-2.093)
    assert entry['sign_test_p'] == pytest.approx(0.0078)


def test_a_missing_validation_artifact_adds_nothing():
    """A fresh checkout that has not run the weekly backtest must not silently
    widen m for everyone else."""
    assert mt.build_weekly_gas_family({}) == []
    assert mt.build_weekly_gas_family(None) == []


def test_the_family_grows_by_exactly_one():
    result = mt.run()
    assert result['n_tests'] == 24
    keys = [t['key'] for t in result['tests']]
    assert keys.count('weekly_gas') == 1


def test_growing_the_family_flipped_no_verdict():
    """The preregistration's actual concern, checked rather than assumed.

    Adding a member raises m, which raises every existing member's Bonferroni p.
    That can only make survival harder, so the risk is not that someone gains a
    finding -- it is that a previously surviving result is retracted by a change
    made for an unrelated reason. Nothing survived at 23 and nothing survives at
    24, so no verdict moved, and this test fails if a future member ever does
    push a survivor out.
    """
    result = mt.run()
    assert result['survive_bonferroni'] == []
    assert result['survive_bh_only'] == []
    assert len(result['survive_neither']) == result['n_tests']
