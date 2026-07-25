"""Multiple-comparison correction — the M1 reviewer defense.

Verifies the Bonferroni/BH machinery against known cases, and that the audit's
headline positives survive the strict correction.
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


def test_corrected_pool_leaves_no_confirmatory_positives():
    """With the historical mean in the baseline pool (§4.7), every MoM verdict is a
    null — so the confirmatory family is empty and there is nothing left to
    correct. The empty family must be handled, not crash."""
    result = mt.run()
    assert result['n_tests'] == 0
    assert result['survive_bonferroni'] == []
    assert result['survive_bh_only'] == []
    assert result['bonferroni_threshold'] is None      # no division by zero


def test_survivor_logic_still_discriminates_on_a_synthetic_family():
    """Machinery guard: now that the real family is empty, keep proving the
    correction can still separate a strong positive from a weak one, so a future
    genuine finding would be classified correctly."""
    fam = [{'test': 'strong', 'skill_vs_naive': 0.30, 'dm_p': 0.0005},
           {'test': 'weak', 'skill_vs_naive': 0.10, 'dm_p': 0.032}]
    r = mt.correct(fam)
    assert r['survive_bonferroni'] == ['strong']       # p < 0.05/2
    assert r['survive_bh_only'] == ['weak']            # FDR only
