from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ph_economic_ai.engine.store import AgentTrustStore

from ph_economic_ai.engine.debate import Agent
from ph_economic_ai.engine.store import trust_tier
from ph_economic_ai.engine.swarm import SwarmAgent

_COLD_START_RUNS = 3
_DIVERSITY_MIN   = 0.60

# Model tier maps
_DEBATE_TIERS: dict[str, dict[str, str]] = {
    'deepseek-r1:8b':  {'promoted': 'deepseek-r1:32b', 'demoted': 'qwen2.5:7b'},
    'qwen2.5:3b':      {'promoted': 'qwen2.5:7b',      'demoted': 'qwen2.5:3b'},
    'qwen2.5:7b':      {'promoted': 'qwen2.5:14b',     'demoted': 'qwen2.5:3b'},
    'qwen2.5:14b':     {'promoted': 'deepseek-r1:32b', 'demoted': 'qwen2.5:7b'},
}

_PROMOTED_SUFFIX = (
    ' Your past estimates have been consistently accurate — '
    'trust your data-driven instincts.'
)
_DEMOTED_SUFFIX = (
    ' Previous estimates from your role have diverged from reality — '
    'be more conservative and cite specific data.'
)


def _resolve_model(base_model: str, tier: str) -> str:
    tiers = _DEBATE_TIERS.get(base_model, {})
    if tier == 'promoted':
        return tiers.get('promoted', base_model)
    if tier == 'demoted':
        return tiers.get('demoted', base_model)
    return base_model


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
        tier = trust_tier(trust)
        new_model = _resolve_model(agent.model, tier)
        new_prompt = agent.system_prompt
        if tier == 'promoted':
            new_prompt = new_prompt.rstrip() + _PROMOTED_SUFFIX
        elif tier == 'demoted':
            new_prompt = new_prompt.rstrip() + _DEMOTED_SUFFIX
        # Fix: clear is_mini when a mini-agent is promoted to a larger model
        is_mini_new = False if (tier == 'promoted' and agent.is_mini) else agent.is_mini
        evolved.append(replace(agent, model=new_model, system_prompt=new_prompt,
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
            tier = trust_tier(trust)
            if tier == 'demoted' and len(active) >= min_active:
                # bench this agent — skip it
                continue
            new_model = _resolve_model(agent.model, tier)
            prompt = agent.system_prompt
            if tier == 'promoted':
                prompt = prompt.rstrip() + _PROMOTED_SUFFIX
            elif tier == 'demoted':
                prompt = prompt.rstrip() + _DEMOTED_SUFFIX
            active.append(replace(agent, model=new_model, system_prompt=prompt))
        evolved.extend(active)
    return evolved
