"""Call-count derivation and the setup screen's run-time estimate.

The old estimate hardcoded "~371 ollama calls, roughly 15s each" for a run that
actually makes 39. These tests keep the number derived from the swarm's real
shape so it cannot drift again.
"""
import pytest

from ph_economic_ai.engine import llm, swarm
from ph_economic_ai.ui.stage2_setup import estimate_swarm_seconds


# ── Call counts ───────────────────────────────────────────────────────────────

def test_counts_match_the_bracket_and_region_shape():
    counts = swarm.expected_call_counts()
    # 4 regions x (5 agents round 1 + 3 surviving round 2) = 32
    assert counts['fast'] == 32
    # 2 regional judges x 3 calls + 1 master = 7
    assert counts['deep'] == 7
    assert counts['total'] == 39


def test_counts_are_derived_not_hardcoded(monkeypatch):
    """Shrinking the swarm must move the number without touching the UI."""
    monkeypatch.setattr(swarm, 'REGIONS', ['NCR', 'Davao Region'])
    monkeypatch.setattr(swarm, 'REGION_PAIRS', [(0, 1)])
    counts = swarm.expected_call_counts()
    assert counts['fast'] == 16          # 2 regions instead of 4
    assert counts['deep'] == 4           # 1 judge pair + master


def test_deep_tier_stays_a_small_share_of_the_budget():
    """The deep tier has the tighter daily quota — it must not carry bulk work."""
    counts = swarm.expected_call_counts()
    assert counts['deep'] < counts['fast'] / 2


def test_critical_path_counts_round_one_sequentially():
    # 5 roles sequential in round 1, then 1 duration for the parallel round 2
    assert swarm.group_critical_path() == 6


# ── Time estimate ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def groq_configured(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'groq')
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.delenv('STRATA_LLM_RPM', raising=False)


def test_estimate_is_positive_and_finite():
    assert 0 < estimate_swarm_seconds(2) < 3600


def test_more_parallelism_never_increases_the_estimate():
    assert estimate_swarm_seconds(4) <= estimate_swarm_seconds(1)


def test_estimate_is_floored_by_the_rate_limit(monkeypatch):
    """However fast the model is, a free tier cannot exceed its RPM."""
    monkeypatch.setenv('STRATA_LLM_RPM', '2')
    counts = swarm.expected_call_counts()
    expected_floor = counts['total'] / 2 * 60
    assert estimate_swarm_seconds(99) >= expected_floor * 0.99


def test_estimate_is_dominated_by_the_token_cap_on_groq():
    """Groq's free tier is token-bound, not latency-bound: raising parallelism
    cannot help, because the 6K/min ceiling is what sets the floor."""
    assert estimate_swarm_seconds(1) == estimate_swarm_seconds(8)


def test_estimate_tracks_the_token_cap(monkeypatch):
    """Doubling the token allowance should roughly halve the estimate — proof
    the floor really is the token budget and not a constant."""
    monkeypatch.setenv('STRATA_LLM_TPM', '6000')
    tight = estimate_swarm_seconds(4)
    monkeypatch.setenv('STRATA_LLM_TPM', '12000')
    loose = estimate_swarm_seconds(4)
    assert loose < tight


def test_effective_rpm_reflects_the_override(monkeypatch):
    monkeypatch.setenv('STRATA_LLM_RPM', '17')
    assert llm.effective_rpm() == 17


def test_effective_rpm_is_zero_when_running_locally(monkeypatch):
    """No keys means local Ollama, which has no quota. Reporting a cap here
    would make the estimate invent a rate-limit floor that does not exist."""
    monkeypatch.delenv('STRATA_LLM_PROVIDER', raising=False)
    monkeypatch.delenv('GROQ_API_KEY', raising=False)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    assert llm.effective_rpm() == 0


def test_local_estimate_has_no_rate_or_token_floor(monkeypatch):
    """Locally the estimate is pure latency — it must scale with parallelism,
    unlike the Groq case where the token cap flattens it."""
    monkeypatch.setenv('STRATA_LLM_PROVIDER', 'ollama')
    assert estimate_swarm_seconds(4) < estimate_swarm_seconds(1)
