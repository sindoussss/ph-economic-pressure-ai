from __future__ import annotations

import math
import re
import statistics

from ph_economic_ai.engine.debate import AgentResponse

# Regex: matches currency/percentage data-brief citations like $72.40, ₱57.80, 3.8%, 5%
_CITE_RE = re.compile(r'(?:[\$₱â‚±]|PHP|P)\s*\d+\.?\d*|[\+\-]?\d+\.?\d*\s*%')
_CHAIN_FULL_RE = re.compile(
    r'CAUSAL\s+CHAIN\s*:\s*\S+.*?(?:â†’|→|->).*?(?:â†’|→|->).*?(?:â†’|→|->).*?\S',
    re.IGNORECASE,
)
_CHAIN_PARTIAL_RE = re.compile(r'CAUSAL\s+CHAIN\s*:', re.IGNORECASE)

_CONV_SCALE = 2.0  # typical spread for convergence normalization (PHP/L)

_LEN_MEAN  = 400.0
_LEN_SIGMA = 200.0

# Metric weights (must sum to 1.0)
_W_CITE      = 0.30
_W_CONVERGE  = 0.25
_W_CHAIN     = 0.25
_W_LENGTH    = 0.20


class QualityScorer:
    @staticmethod
    def score_responses(
        responses: list[AgentResponse],
        group_estimates: list[float],
    ) -> dict[str, dict]:
        """Score each agent response. Returns {agent_name: metric_dict}. Each agent_name must be unique in responses."""
        valid_ests = [e for e in group_estimates if e is not None]
        median = statistics.median(valid_ests) if valid_ests else 0.0

        results: dict[str, dict] = {}
        for resp in responses:
            stmt = resp.statement or ''
            cites = len(_CITE_RE.findall(stmt))
            citation_score = min(cites / 3.0, 1.0)

            if _CHAIN_FULL_RE.search(stmt):
                chain_score = 1.0
            elif _CHAIN_PARTIAL_RE.search(stmt):
                chain_score = 0.5
            else:
                chain_score = 0.0

            if resp.price_estimate is not None:
                if len(valid_ests) > 1:
                    convergence_score = max(0.0, 1.0 - abs(resp.price_estimate - median) / _CONV_SCALE)
                else:
                    # All estimates are identical — agent is exactly on the median
                    convergence_score = 1.0
            else:
                convergence_score = 0.5

            n = len(stmt)
            length_score = math.exp(-0.5 * ((n - _LEN_MEAN) / _LEN_SIGMA) ** 2)

            overall = (
                _W_CITE     * citation_score
                + _W_CONVERGE * convergence_score
                + _W_CHAIN    * chain_score
                + _W_LENGTH   * length_score
            )
            results[resp.agent_name] = {
                'citation_score':    round(citation_score,    3),
                'convergence_score': round(convergence_score, 3),
                'chain_score':       round(chain_score,       3),
                'length_score':      round(length_score,      3),
                'overall':           round(overall,           3),
                'citation_count':    cites,
                'has_causal_chain':  1 if chain_score >= 1.0 else 0,
            }
        return results

    @staticmethod
    def run_quality(
        responses: list[AgentResponse],
        group_estimates: list[float],
    ) -> float:
        """Overall run quality — average of all agent overall scores."""
        scores = QualityScorer.score_responses(responses, group_estimates)
        if not scores:
            return 0.5
        return round(sum(v['overall'] for v in scores.values()) / len(scores), 3)
