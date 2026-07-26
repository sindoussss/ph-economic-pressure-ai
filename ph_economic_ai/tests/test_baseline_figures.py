import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json

from ph_economic_ai.benchmark import baseline_figures as bf
from ph_economic_ai.benchmark.paths import DOCS_IMG_DIR, FIGURES_DIR, artifact


def _load(name):
    return json.loads(artifact(name).read_text(encoding='utf-8'))


def test_all_three_figures_render_to_both_locations():
    """Each figure is mirrored into artifacts/figures/ and docs/img/ so the
    manuscript and the repo never show different versions of the same result."""
    for fn, art, name in (
        (bf.fig_spurious_skill, 'baseline_theory.json', 'fig5_spurious_skill.png'),
        (bf.fig_size_distortion, 'baseline_size.json', 'fig6_size_distortion.png'),
        (bf.fig_fredmd_exposure, 'vulnerability_survey.json', 'fig7_fredmd_exposure.png'),
    ):
        outs = fn(_load(art))
        assert len(outs) == 2
        assert {p.parent for p in outs} == {FIGURES_DIR, DOCS_IMG_DIR}
        for p in outs:
            assert p.name == name and p.exists() and p.stat().st_size > 5_000


def test_figures_are_driven_by_the_artifacts_not_literals():
    """Regression guard, from the same class of bug as the hardcoded figure
    verdicts: a figure must fail loudly if its artifact key is missing, rather
    than silently drawing a stale number."""
    import pytest
    with pytest.raises((KeyError, TypeError)):
        bf.fig_size_distortion({'alpha': 0.05})          # no 'size' key
    with pytest.raises((KeyError, TypeError)):
        bf.fig_fredmd_exposure({'share_vulnerable': 0.5})  # no 'series' key
