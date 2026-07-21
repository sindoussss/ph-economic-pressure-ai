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
    """Return swarm agents with model adjusted by trust; enforces diversity guard."""
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
        # Sort by trust descending so we bench lowest-trust first
        scored = sorted(
            group_agents,
            key=lambda a: trust_map.get(a.name, 0.5),
            reverse=True,
        )
        active: list[SwarmAgent] = []
        for agent in scored:
            trust = trust_map.get(agent.name, 0.5)
            band = trust_tier(trust)
            if band == 'demoted' and len(active) >= min_active:
                # bench this agent — skip it
                continue
            new_tier = _resolve_tier(agent.tier, band)
            prompt = agent.system_prompt
            if band == 'promoted':
                prompt = prompt.rstrip() + _PROMOTED_SUFFIX
            elif band == 'demoted':
                prompt = prompt.rstrip() + _DEMOTED_SUFFIX
            active.append(replace(agent, tier=new_tier, system_prompt=prompt))
        evolved.extend(active)
    return evolved
