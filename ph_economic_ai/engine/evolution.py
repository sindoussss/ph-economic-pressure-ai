from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

from ph_economic_ai.engine import llm
from ph_economic_ai.engine.debate import Agent
from ph_economic_ai.engine.store import trust_tier
from ph_economic_ai.engine.swarm import SwarmAgent

_COLD_START_RUNS = 3
_DIVERSITY_MIN   = 0.60

#: Roles that are never benched, however low their trust falls.
#:
#: The Critic and the ConfidenceScorer are not contestants, they are the
#: measuring instruments: their `SCORE:` and `CONFIDENCE:` lines are what
#: `compute_combined_score` reads, and when neither is in the room every agent
#: falls back to 0.5, `scores_are_degenerate` fires, and the survivor is picked by
#: a hash tiebreak. Benching them does not remove a weak analyst, it removes the
#: scoring. A live roster had lost NCR Critic, Central Luzon Critic, and two
#: ConfidenceScorers, so two of four groups were eliminating agents at random.
#:
#: They are also the roles most likely to score badly on the internal quality
#: metric, because that metric rewards citations and a causal chain and their job
#: is to grade other agents rather than to write their own analysis.
_UNBENCHABLE_ROLES = frozenset({'Critic', 'ConfidenceScorer'})

_PROMOTED_SUFFIX = (
    ' Your past estimates have been consistently accurate — '
    'trust your data-driven instincts.'
)
_DEMOTED_SUFFIX = (
    ' Previous estimates from your role have diverged from reality — '
    'be more conservative and cite specific data.'
)


def _resolve_tier(base_tier: str, trust_band: str) -> str:
    """Map a trust band onto a model tier.

    The old ladder climbed four Ollama sizes (3b -> 7b -> 14b -> 32b); hosted
    free tiers expose only two, so the ladder collapses to its endpoints. The
    intent is unchanged — agents that have earned trust get the stronger model,
    agents that have not get the cheap one — but a demotion now also protects
    the scarce deep-tier daily quota.
    """
    if trust_band == 'promoted':
        return llm.DEEP
    if trust_band == 'demoted':
        return llm.FAST
    return base_tier


def get_evolved_debate_agents(
    store: 'AgentTrustStore',
    base_agents: list[Agent],
) -> list[Agent]:
    """Return debate agents with model/prompt adjusted by trust scores."""
    if store.total_runs() < _COLD_START_RUNS:
        return list(base_agents)

    trust_map = store.get_all_trust()
    evolved: list[Agent] = []
    for agent in base_agents:
        trust = trust_map.get(agent.name, 0.5)
        band = trust_tier(trust)
        new_tier = _resolve_tier(agent.tier, band)
        new_prompt = agent.system_prompt
        if band == 'promoted':
            new_prompt = new_prompt.rstrip() + _PROMOTED_SUFFIX
        elif band == 'demoted':
            new_prompt = new_prompt.rstrip() + _DEMOTED_SUFFIX
        # Fix: clear is_mini when a mini-agent is promoted to a larger model
        is_mini_new = False if (band == 'promoted' and agent.is_mini) else agent.is_mini
        evolved.append(replace(agent, tier=new_tier, system_prompt=new_prompt,
                                is_mini=is_mini_new))
    return evolved


def get_evolved_swarm_agents(
    store: 'AgentTrustStore',
    base_agents: list[SwarmAgent],
) -> list[SwarmAgent]:
    """Return swarm agents with model adjusted by trust; enforces diversity guard.

    Benching is temporary. Every agent this function benches has its trust decayed
    back toward neutral by `store.recover_benched`, so a bench lasts a couple of
    runs rather than forever. Without that the function could only ever remove
    agents: a benched agent produces no response, `update_trust` is never called
    for it, and its score is frozen below the demotion threshold permanently.
    """
    if store.total_runs() < _COLD_START_RUNS:
        return list(base_agents)

    trust_map = store.get_all_trust()
    # Group by group_id for diversity guard
    groups: dict[int, list[SwarmAgent]] = {}
    for a in base_agents:
        groups.setdefault(a.group_id, []).append(a)

    evolved: list[SwarmAgent] = []
    for group_id, group_agents in groups.items():
        min_active = math.ceil(len(group_agents) * _DIVERSITY_MIN)

        def _band(agent: SwarmAgent) -> str:
            return trust_tier(trust_map.get(agent.name, 0.5))

        def _activate(agent: SwarmAgent) -> SwarmAgent:
            band = _band(agent)
            prompt = agent.system_prompt
            if band == 'promoted':
                prompt = prompt.rstrip() + _PROMOTED_SUFFIX
            elif band == 'demoted':
                prompt = prompt.rstrip() + _DEMOTED_SUFFIX
            return replace(agent, tier=_resolve_tier(agent.tier, band),
                           system_prompt=prompt)

        # The scoring roles are seated first and unconditionally. They still take
        # the tier and prompt their trust has earned; they just cannot lose their
        # seat, because the bracket cannot measure anything without them.
        active = [_activate(a) for a in group_agents
                  if a.role in _UNBENCHABLE_ROLES]
        # Everyone else fills the diversity floor by trust, highest first.
        for agent in sorted(
            (a for a in group_agents if a.role not in _UNBENCHABLE_ROLES),
            key=lambda a: trust_map.get(a.name, 0.5),
            reverse=True,
        ):
            if _band(agent) == 'demoted' and len(active) >= min_active:
                store.recover_benched(agent.name)
                continue
            active.append(_activate(agent))
        evolved.extend(active)
    return evolved
