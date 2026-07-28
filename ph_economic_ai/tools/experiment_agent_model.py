"""Does a stronger agent model raise agreement, and is the rise earned?

`experiment_sign_contradiction` found that run 29's 46 percent was not analysts
disagreeing. The agents mostly read the shock the same way; what varied was
whether they could EMIT the answer in the form the parser and the metric
require. 61 percent of usable estimates carried no parseable causal chain and 26
percent of responses produced no estimate at all.

That points at agent capability rather than analysis, which is testable. This
runs the same scenario twice, changing exactly one thing: the model behind the 32
bulk agents.

    arm A   agents qwen2.5:3b (local)          the shipped configuration
    arm B   agents llama-3.1-8b-instant (Groq)

Only `STRATA_LLM_FAST_PROVIDER` moves. The base provider stays local in both
arms, so embeddings stay on `nomic-embed-text` and the retriever is held
constant; changing the base would have made this an A/B on the retriever too.
The judges stay on the deep tier in both arms for the same reason. Seeds are
keyed on the vintage (ADR-006), so both arms request identical seeds on the same
day, which makes this a paired comparison rather than two independent samples.

Four measures, and the last two are what stop a rise being taken at face value:

- **parse rate** — responses yielding a usable estimate
- **chain rate** — responses whose CAUSAL CHAIN parses, the format contract
- **agreement** — the number on the card
- **echo rate** — estimates equal to the physical anchor to two decimals

Echo is the control. A stronger model that simply repeats the anchor it was
handed will show higher agreement while adding nothing, and that is a rise to
reject rather than to ship. The prior shared-anchor experiment recorded five of
seven agents doing exactly that.

    python -m ph_economic_ai.tools.experiment_agent_model --dry-run
    python -m ph_economic_ai.tools.experiment_agent_model

Costs roughly 78 calls: 39 per arm, plus retries.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine import anchoring, llm, swarm, vintage
from ph_economic_ai.engine.rag import RagEngine
from ph_economic_ai.tools.experiment_sign_contradiction import (
    _ARROW, _CHAIN_RE, conclusion_direction)

ARTIFACT = Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts' / 'agent_model_ab.json'

SCENARIO = {'oil_pct': -8.0, 'usd_pct': -0.1, 'bsp_rate': 6.5, 'demand_index': 75.0}
_ECHO_TOL = 0.005

ARMS = [
    ('A_local_3b', None),
    ('B_groq_8b', 'groq'),
]


@contextlib.contextmanager
def _fast_provider(provider: Optional[str]):
    """Point the bulk agents at one provider, leaving everything else alone."""
    var = 'STRATA_LLM_FAST_PROVIDER'
    prior = os.environ.get(var)
    if provider is None:
        os.environ.pop(var, None)
    else:
        os.environ[var] = provider
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = prior


def _has_chain(statement: str) -> bool:
    text = ' '.join((statement or '').split())
    m = _CHAIN_RE.search(text)
    if not m:
        return False
    return len([s for s in _ARROW.split(m.group(1)) if s.strip()]) >= 2


def _measure(verdict, seconds: float) -> dict:
    responses = list(getattr(verdict, 'all_responses', []) or [])
    estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
    anchor = verdict.physical_anchor
    if anchor is None:
        anchor = anchoring.fuel_passthrough_anchor(
            SCENARIO['oil_pct'], SCENARIO['usd_pct'])

    echoes = [e for e in estimates if abs(e - anchor) <= _ECHO_TOL]
    counts = Counter(round(e, 2) for e in estimates)
    modal_value, modal_n = counts.most_common(1)[0] if counts else (None, 0)
    signs = [1 if e > 0 else -1 for e in estimates if e != 0]
    lead = max(set(signs), key=signs.count) if signs else 0
    contradictions = sum(
        1 for r in responses
        if r.price_estimate is not None
        and conclusion_direction(r.statement) is not None
        and (r.price_estimate > 0) != (conclusion_direction(r.statement) > 0))

    return {
        'model': llm.describe_model(llm.FAST),
        'provider': llm.provider_for(llm.FAST),
        'responses': len(responses),
        'parsed': len(estimates),
        'parse_rate_pct': round(100 * len(estimates) / max(len(responses), 1), 1),
        'chain_rate_pct': round(
            100 * sum(1 for r in responses if _has_chain(r.statement))
            / max(len(responses), 1), 1),
        'agreement_pct': verdict.confidence_pct,
        'agreement_n': getattr(verdict, 'agreement_n', 0),
        'agreement_regions': list(getattr(verdict, 'agreement_regions', (0, 0))),
        'direction_pct': round(100 * signs.count(lead) / len(signs)) if signs else 0,
        'self_contradictions': contradictions,
        'echo_exact_n': len(echoes),
        'echo_exact_pct': round(100 * len(echoes) / max(len(estimates), 1), 1),
        'modal_value': modal_value,
        'modal_is_anchor': (modal_value is not None
                            and abs(modal_value - anchor) <= _ECHO_TOL),
        'anchor': round(anchor, 3),
        'estimate': verdict.final_estimate,
        'estimate_source': verdict.estimate_source,
        'spread': round(max(estimates) - min(estimates), 2) if len(estimates) > 1 else 0.0,
        'seconds': round(seconds, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    counts = swarm.expected_call_counts()
    print(f'Paired A/B on the agent model. {counts["total"]} calls per arm, '
          f'{counts["total"] * len(ARMS)} total, plus retries.')
    print(f'  vintage: {vintage.describe_vintage()}  (both arms share seeds)')
    for name, prov in ARMS:
        with _fast_provider(prov):
            print(f'  {name:<12} agents -> {llm.provider_for(llm.FAST)}:'
                  f'{llm.describe_model(llm.FAST)}')
    print(f'  judges  -> {llm.provider_for(llm.DEEP)}:{llm.describe_model(llm.DEEP)} (held constant)')
    print(f'  embeddings stay local: {llm.is_local()}')
    if args.dry_run:
        return 0

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()

    results = {}
    for name, prov in ARMS:
        print(f'\n=== {name} ===')
        with _fast_provider(prov):
            llm.reset_provenance()
            started = time.monotonic()
            try:
                verdict = swarm.SwarmOrchestrator(
                    rag=rag, scenario=dict(SCENARIO), parallel_n=2).run()
            except Exception as exc:
                print(f'  FAILED — {type(exc).__name__}: {str(exc)[:200]}')
                results[name] = {'failed': str(exc)[:300]}
                continue
            results[name] = _measure(verdict, time.monotonic() - started)
        r = results[name]
        print(f'  {r["model"]}: parse {r["parse_rate_pct"]}%  chain {r["chain_rate_pct"]}%  '
              f'agreement {r["agreement_pct"]}% (n={r["agreement_n"]})  '
              f'direction {r["direction_pct"]}%  echo {r["echo_exact_pct"]}%  '
              f'({r["seconds"]:.0f}s)')

    print('\n' + '=' * 74)
    a, b = results.get('A_local_3b', {}), results.get('B_groq_8b', {})
    if 'failed' in a or 'failed' in b or not a or not b:
        print('One arm did not complete, so there is nothing to compare.')
    else:
        rows = [('parse rate %', 'parse_rate_pct'), ('chain rate %', 'chain_rate_pct'),
                ('agreement %', 'agreement_pct'), ('direction %', 'direction_pct'),
                ('estimates (n)', 'agreement_n'), ('self-contradictions', 'self_contradictions'),
                ('echo anchor %', 'echo_exact_pct'), ('spread PHP', 'spread'),
                ('seconds', 'seconds')]
        print(f'{"":<22}{"A local 3b":>14}{"B groq 8b":>14}   delta')
        for label, key in rows:
            av, bv = a.get(key, 0), b.get(key, 0)
            print(f'{label:<22}{av:>14}{bv:>14}   {bv - av:+g}')
        print()
        gain = b['agreement_pct'] - a['agreement_pct']
        if b.get('modal_is_anchor'):
            print(f'REJECT the gain: arm B\'s MODAL estimate is the anchor '
                  f'({b["modal_value"]:+.2f}). Higher agreement from agents '
                  f'repeating the number they were handed is suggestion, not '
                  f'consensus.')
        elif gain >= 10 and b['parse_rate_pct'] > a['parse_rate_pct']:
            print(f'Agreement rose {gain} points alongside a '
                  f'{b["parse_rate_pct"] - a["parse_rate_pct"]:+.0f} point parse rate, '
                  f'and the modal estimate is not the anchor. Consistent with the '
                  f'format-compliance reading rather than with echoing.')
        elif gain >= 10:
            print(f'Agreement rose {gain} points WITHOUT a parse-rate rise, so the '
                  f'format-compliance explanation does not account for it. Do not '
                  f'ship this until the cause is identified.')
        else:
            print(f'Agreement moved {gain:+d} points. The stronger agent model does '
                  f'not buy enough to justify the cost, and the honest move is to '
                  f'stay local.')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'measured_at': vintage.describe_vintage(),
        'scenario': SCENARIO, 'arms': results}, indent=2), encoding='utf-8')
    print(f'\nWrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
