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

def test_family_is_the_confirmatory_tests_only():
    report = json.loads(
        (Path(mt._ARTIFACTS) / 'accuracy_report.json').read_text())
    family = mt.build_family(report)
    # every member is a real 'beats naive' test with a p-value
    assert len(family) >= 5
    assert all(f['dm_p'] is not None for f in family)
    labels = {f['test'] for f in family}
    assert any('Electricity MoM' in l for l in labels)
    assert any('Food MoM' in l for l in labels)


def test_headline_positives_survive_bonferroni():
    """The paper's defense: electricity and long-sample MoM survive even the
    strictest family-wise correction."""
    result = mt.run()
    survivors = ' '.join(result['survive_bonferroni'])
    assert 'Electricity MoM' in survivors
    assert 'long, n=143' in survivors
    assert result['n_tests'] == 6


def test_weak_positives_do_not_survive_bonferroni():
    """Honest: short-sample MoM and the transport artifact fail FWER."""
    result = mt.run()
    weak = ' '.join(result['survive_bh_only'])
    assert 'short, n=61' in weak
    assert 'Transport' in weak
