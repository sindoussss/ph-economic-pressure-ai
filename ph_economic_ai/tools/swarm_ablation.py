"""Does the swarm's size actually change its answer?

The swarm costs ~44K fast-tier tokens per run, which is what makes a free-tier
run take minutes. The obvious response is to make it smaller — but cutting
agents is only defensible if the smaller swarm reaches the same verdict. This
harness measures that instead of guessing.

Deliberately lives outside `benchmark/`: the benchmark is the validated half
and must stay reproducible with no API key, a boundary enforced by
`tests/test_benchmark_isolation.py`. This is exploratory tooling and needs a
provider.

Method
------
Agent agreement varies run to run — the README says so and it is the whole
reason a single run proves nothing. So each variant runs `--repeats` times and
we report the spread, not one number. A variant is only "same answer" if its
estimates overlap the full swarm's spread; a variant whose *mean* looks close
but whose spread is wide has not reproduced the result, it has just been
lucky once.

Usage
-----
    set GROQ_API_KEY=...        # or GEMINI_API_KEY
    python -m ph_economic_ai.tools.swarm_ablation --repeats 3

Costs real quota: roughly `repeats x sum(variant call counts)` requests.
Print the plan first with --dry-run.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine import llm, swarm
from ph_economic_ai.engine.rag import RagEngine

ARTIFACT = Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts' / 'swarm_ablation.json'


@dataclass
class Variant:
    """One swarm configuration to measure."""
    name: str
    rationale: str
    regions: Optional[list[str]] = None      # None → leave REGIONS alone
    region_pairs: Optional[list] = None
    bracket: Optional[list] = None           # None → leave _BRACKET alone
    max_tokens: Optional[int] = None         # None → leave the default alone


VARIANTS: list[Variant] = [
    Variant(
        name='full',
        rationale='The shipped configuration — the baseline every cut is judged against.',
    ),
    Variant(
        name='short_completions',
        rationale=(
            'Same structure, max_tokens 750 -> 400. Completions are ~24K of the '
            '~44K token bill, and agents only owe brief reasoning plus one '
            'ESTIMATE line. Cheapest possible cut if the verdict holds.'
        ),
        max_tokens=400,
    ),
    Variant(
        name='two_regions',
        rationale=(
            'NCR + Davao only. Halves agent calls. Tests whether the 4-region '
            'spread carries information or just costs tokens.'
        ),
        regions=['NCR', 'Davao Region'],
        region_pairs=[(0, 1)],
    ),
    Variant(
        name='one_round',
        rationale=(
            'Drops the elimination round; the survivor is picked from round-1 '
            'scores. Tests whether round 2 changes who wins.'
        ),
        bracket=[(1, 4)],
    ),
]


@contextlib.contextmanager
def _applied(variant: Variant):
    """Temporarily reshape the swarm module for one variant."""
    saved = {
        'REGIONS': swarm.REGIONS,
        'REGION_PAIRS': swarm.REGION_PAIRS,
        '_BRACKET': swarm._BRACKET,
        '_AGENT_MAX_TOKENS': swarm._AGENT_MAX_TOKENS,
    }
    try:
        if variant.regions is not None:
            swarm.REGIONS = variant.regions
        if variant.region_pairs is not None:
            swarm.REGION_PAIRS = variant.region_pairs
        if variant.bracket is not None:
            swarm._BRACKET = variant.bracket
        if variant.max_tokens is not None:
            swarm._AGENT_MAX_TOKENS = variant.max_tokens
        yield
    finally:
        for name, value in saved.items():
            setattr(swarm, name, value)


@dataclass
class RunResult:
    estimate: Optional[float]
    confidence: int
    seconds: float
    calls: dict


@dataclass
class VariantResult:
    name: str
    rationale: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def estimates(self) -> list[float]:
        return [r.estimate for r in self.runs if r.estimate is not None]

    def summary(self) -> dict:
        est = self.estimates
        return {
            'variant': self.name,
            'rationale': self.rationale,
            'n_runs': len(self.runs),
            'n_parsed': len(est),
            'estimate_mean': round(statistics.fmean(est), 3) if est else None,
            'estimate_min': round(min(est), 3) if est else None,
            'estimate_max': round(max(est), 3) if est else None,
            'estimate_stdev': round(statistics.pstdev(est), 3) if len(est) > 1 else 0.0,
            'confidence_mean': (
                round(statistics.fmean([r.confidence for r in self.runs]), 1)
                if self.runs else None
            ),
            'seconds_mean': round(statistics.fmean([r.seconds for r in self.runs]), 1)
                if self.runs else None,
            'calls': self.runs[0].calls if self.runs else {},
        }


def _run_once(rag: RagEngine, scenario: dict, variant: Variant) -> RunResult:
    started = time.monotonic()
    orch = swarm.SwarmOrchestrator(rag=rag, scenario=scenario, parallel_n=2)
    verdict = orch.run()
    return RunResult(
        estimate=verdict.final_estimate,
        confidence=verdict.confidence_pct,
        seconds=time.monotonic() - started,
        calls=swarm.expected_call_counts(),
    )


_DEFAULT_AGENT_MAX_TOKENS = 750
_TYPICAL_PROMPT_TOKENS = 650      # measured against a populated RAG index


def _estimated_tokens(variant: Variant) -> int:
    """Fast-tier token spend for one run of this variant.

    Only the fast tier is modelled: it carries 32 of the 39 calls and is where
    the tighter tokens-per-minute ceiling bites.
    """
    counts = swarm.expected_call_counts()
    max_tokens = variant.max_tokens or _DEFAULT_AGENT_MAX_TOKENS
    return counts['fast'] * (_TYPICAL_PROMPT_TOKENS + max_tokens)


def _overlaps(a: VariantResult, b: VariantResult) -> bool:
    """True when the two variants' observed ranges intersect.

    Ranges, not means: with run-to-run variance this wide, two means can sit
    close while the underlying distributions clearly disagree.
    """
    if not a.estimates or not b.estimates:
        return False
    return not (max(a.estimates) < min(b.estimates) or max(b.estimates) < min(a.estimates))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=3,
                        help='runs per variant (>=3 to see the spread at all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the quota cost and exit without calling anything')
    args = parser.parse_args()

    plan = []
    for variant in VARIANTS:
        with _applied(variant):
            plan.append((variant.name, swarm.expected_call_counts(),
                         _estimated_tokens(variant)))

    total_calls = sum(c['total'] for _, c, _ in plan) * args.repeats
    total_tokens = sum(t for _, _, t in plan) * args.repeats
    print(f'Plan — {args.repeats} repeats per variant:')
    for name, counts, tokens in plan:
        print(f'  {name:20s} {counts["total"]:3d} calls/run '
              f'({counts["fast"]} fast, {counts["deep"]} deep)  '
              f'~{tokens:6,d} fast-tier tokens/run')
    print(f'  {"TOTAL":20s} {total_calls:3d} calls, ~{total_tokens:,} tokens')
    print('\nTokens, not calls, are what free tiers throttle on — a variant '
          'that leaves the call count alone can still be a large saving.')

    if args.dry_run:
        return 0

    if not llm.is_configured():
        print('\nNo provider configured. Set GROQ_API_KEY or GEMINI_API_KEY.')
        return 1

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()
    scenario = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}

    results: list[VariantResult] = []
    for variant in VARIANTS:
        vr = VariantResult(name=variant.name, rationale=variant.rationale)
        print(f'\n=== {variant.name} ===')
        for i in range(args.repeats):
            with _applied(variant):
                try:
                    run = _run_once(rag, scenario, variant)
                except Exception as exc:
                    print(f'  run {i + 1}: FAILED — {exc}')
                    continue
            vr.runs.append(run)
            print(f'  run {i + 1}: estimate={run.estimate} '
                  f'confidence={run.confidence}% ({run.seconds:.0f}s)')
        results.append(vr)

    baseline = next((r for r in results if r.name == 'full'), None)
    summaries = [r.summary() for r in results]
    for summary, result in zip(summaries, results):
        if baseline is not None and result is not baseline:
            summary['overlaps_full'] = _overlaps(result, baseline)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'repeats': args.repeats,
        'provider': llm.active_provider(),
        'variants': summaries,
    }, indent=2), encoding='utf-8')

    print(f'\nWrote {ARTIFACT}')
    print('\nA variant is a safe cut only if overlaps_full is true AND its '
          'spread is no wider than the baseline. Record the outcome either way '
          '— a cut that changes the verdict is itself a finding.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
