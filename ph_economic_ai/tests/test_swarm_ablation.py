"""Tests for the swarm ablation harness.

The harness reshapes module globals, so the risk is a variant that silently
does nothing — it would then "prove" a cut is safe when the cut was never
applied. These tests check each variant actually changes the swarm, and that
the module is restored afterwards.
"""
import pytest

from ph_economic_ai.engine import swarm
from ph_economic_ai.tools import swarm_ablation as ab


@pytest.fixture(autouse=True)
def _swarm_is_restored():
    """Every test must leave the shared swarm module exactly as it found it."""
    before = (swarm.REGIONS, swarm.REGION_PAIRS, swarm._BRACKET,
              swarm._AGENT_MAX_TOKENS)
    yield
    assert (swarm.REGIONS, swarm.REGION_PAIRS, swarm._BRACKET,
            swarm._AGENT_MAX_TOKENS) == before


def test_applied_restores_globals_even_on_error():
    variant = ab.Variant(name='x', rationale='', regions=['NCR'])
    with pytest.raises(RuntimeError):
        with ab._applied(variant):
            assert swarm.REGIONS == ['NCR']
            raise RuntimeError('boom')
    assert len(swarm.REGIONS) == 4


def test_full_variant_is_the_untouched_baseline():
    baseline = swarm.expected_call_counts()
    full = next(v for v in ab.VARIANTS if v.name == 'full')
    with ab._applied(full):
        assert swarm.expected_call_counts() == baseline


@pytest.mark.parametrize('name', [v.name for v in ab.VARIANTS if v.name != 'full'])
def test_every_variant_actually_changes_something(name):
    """A no-op variant would produce a false 'safe to cut' result."""
    baseline_calls = swarm.expected_call_counts()
    baseline_tokens = swarm._AGENT_MAX_TOKENS
    variant = next(v for v in ab.VARIANTS if v.name == name)

    with ab._applied(variant):
        changed_calls = swarm.expected_call_counts() != baseline_calls
        changed_tokens = swarm._AGENT_MAX_TOKENS != baseline_tokens

    assert changed_calls or changed_tokens, f'{name} is a no-op'


def test_short_completions_lowers_the_token_budget():
    """This variant leaves the call count alone, so only the token estimate
    can show it working."""
    variant = next(v for v in ab.VARIANTS if v.name == 'short_completions')
    full = next(v for v in ab.VARIANTS if v.name == 'full')
    assert ab._estimated_tokens(variant) < ab._estimated_tokens(full)


def test_two_regions_halves_the_agent_calls():
    variant = next(v for v in ab.VARIANTS if v.name == 'two_regions')
    with ab._applied(variant):
        assert swarm.expected_call_counts()['fast'] == 16


def test_one_round_drops_the_elimination_round():
    variant = next(v for v in ab.VARIANTS if v.name == 'one_round')
    with ab._applied(variant):
        # 5 agents x 4 regions, no second round
        assert swarm.expected_call_counts()['fast'] == 20


# ── Overlap logic ─────────────────────────────────────────────────────────────

def _result(name: str, estimates: list[float]) -> ab.VariantResult:
    vr = ab.VariantResult(name=name, rationale='')
    vr.runs = [ab.RunResult(estimate=e, confidence=70, seconds=1.0, calls={})
               for e in estimates]
    return vr


def test_overlapping_ranges_are_detected():
    assert ab._overlaps(_result('a', [1.0, 2.0]), _result('b', [1.5, 3.0]))


def test_disjoint_ranges_are_not_overlapping():
    assert not ab._overlaps(_result('a', [1.0, 2.0]), _result('b', [5.0, 6.0]))


def test_close_means_with_disjoint_ranges_do_not_count_as_agreement():
    """Means can sit close while the distributions clearly disagree — the whole
    reason the harness compares ranges rather than means."""
    a = _result('a', [1.0, 1.1])       # mean 1.05
    b = _result('b', [1.2, 1.3])       # mean 1.25, no overlap
    assert not ab._overlaps(a, b)


def test_unparsed_estimates_never_count_as_agreement():
    assert not ab._overlaps(_result('a', []), _result('b', [1.0]))


def test_summary_reports_spread_not_just_a_mean():
    summary = _result('a', [1.0, 2.0, 3.0]).summary()
    assert summary['estimate_mean'] == 2.0
    assert summary['estimate_min'] == 1.0
    assert summary['estimate_max'] == 3.0
    assert summary['estimate_stdev'] > 0
