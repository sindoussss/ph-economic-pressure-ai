"""Does the agreement survive a change of model?

A single-model run on 2026-07-29 scored **100 percent agreement over ONE distinct
estimate**. Pulling the raw statements showed the agents were blinded,
independent, and separately reasoned, with different prose and different causal
chains. They converged because they are the same model: twenty agents running
`qwen2.5:3b` are one model asked twenty times. Blinding removes peer
contamination and cannot remove model identity, so on a single-model roster the
agreement percentage substantially measures that model's determinism.

`DEC-029` therefore stopped treating the number as a quantity to raise, and
`Q-ENG-006` asked the only remaining question: does agreement hold when the model
changes? Agreement across models is evidence. Agreement within one model is not.

This runs the same scenario twice, shared seeds:

    A  homogeneous    every agent on the tier default, as shipped
    B  heterogeneous  the roster spans --models, crossed with region and role

Read `models_agree`: the model medians either sit inside the same
`_AGREEMENT_BAND` used to decide whether two AGENTS agree, or they do not. One
standard covers both questions and no new threshold is invented.

The first version of this compared `between_spread` against the mean within-model
RANGE and got the 2026-07-30 run backwards, calling it "agreement survives the
model change" when llama3.2's median was +2.50 and qwen2.5:3b's was +1.20, a
factor of 2.08. The range was 3.4 times that arm's own standard deviation, so it
was measuring a single outlier, and the comparison amounted to "is the gap smaller
than the within-model worst case", which nearly anything passes. Median absolute
deviation is now reported alongside the range, and the verdict uses the band.

## What this experiment cannot settle

**The models must be genuinely different for a null to mean anything.** Two sizes
of one family share a training lineage, so medians landing together across them
bounds how much heterogeneity was tested rather than showing agreement would
survive a different family. This prints a warning when every `--models` entry
shares a family prefix.

`DEC-022` is also worth remembering before reading B as a free improvement: a
stronger hosted agent model left agreement unchanged, dropped causal-chain
compliance from 100 to 37.5 percent, and cost three times the wall time. Mixing
model sizes can degrade the roster's output quality even when it widens the
spread, so the compliance columns matter as much as the agreement ones.

One run per arm. Enough to see a collapse, not enough to size one.

    python -m ph_economic_ai.tools.experiment_heterogeneous_roster --dry-run
    python -m ph_economic_ai.tools.experiment_heterogeneous_roster
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from ph_economic_ai.engine import anchoring, llm, swarm, vintage
from ph_economic_ai.engine.debate import unfilled_scaffold
from ph_economic_ai.engine.rag import RagEngine

ARTIFACT = (Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts'
            / 'heterogeneous_roster.json')

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}

#: Two sizes of one family, because they are what is installed. Deliberately
#: overridable: the point of the experiment is stronger with different families.
DEFAULT_MODELS = ['qwen2.5:3b', 'qwen2.5:7b']


def _measure(verdict, models_by_agent: dict, seconds: float) -> dict:
    responses = list(getattr(verdict, 'all_responses', []) or [])
    estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
    anchor = verdict.physical_anchor
    if anchor is None:
        anchor = anchoring.fuel_passthrough_anchor(
            SCENARIO['oil_pct'], SCENARIO['usd_pct'])
    counts = Counter(round(e, 2) for e in estimates)
    scaffolds = sum(1 for r in responses if unfilled_scaffold(r.statement or ''))
    return {
        'agreement_pct': verdict.confidence_pct,
        'agreement_n': getattr(verdict, 'agreement_n', 0),
        'distinct_estimates': len(counts),
        'spread': round(max(estimates) - min(estimates), 2) if len(estimates) > 1 else 0.0,
        'stdev': round(statistics.pstdev(estimates), 3) if len(estimates) > 1 else 0.0,
        'opening_diversity': getattr(verdict, 'agreement_diversity', 0.0),
        'echo_pct': round(100 * getattr(verdict, 'agreement_echo_n', 0)
                          / max(len(estimates), 1), 1),
        'parse_rate_pct': round(100 * len(estimates) / max(len(responses), 1), 1),
        # DEC-022: a mixed roster can widen the spread by degrading output
        # quality rather than by adding judgement. This is the control for that.
        'scaffold_echoes': scaffolds,
        'estimate': verdict.final_estimate,
        'estimate_source': verdict.estimate_source,
        'models': sorted(set(models_by_agent.values())) or ['<tier default>'],
        'across_models': getattr(verdict, 'agreement_models', {}) or {},
        'provenance': llm.provenance_id(),
        'seconds': round(seconds, 1),
    }


def _run_arm(rag: RagEngine, models: list[str]) -> dict:
    # The roster is read from the environment by `build_swarm_agents`, so the arm
    # sets it and restores it rather than leaving the process configured.
    previous = os.environ.get('STRATA_SWARM_AGENT_MODELS')
    os.environ['STRATA_SWARM_AGENT_MODELS'] = ','.join(models)
    llm.reset_provenance()
    try:
        agents = swarm.build_swarm_agents()
        models_by_agent = {a.name: a.model for a in agents if a.model}
        started = time.monotonic()
        verdict = swarm.SwarmOrchestrator(
            rag=rag, scenario=dict(SCENARIO), parallel_n=2).run()
        return _measure(verdict, models_by_agent, time.monotonic() - started)
    finally:
        if previous is None:
            os.environ.pop('STRATA_SWARM_AGENT_MODELS', None)
        else:
            os.environ['STRATA_SWARM_AGENT_MODELS'] = previous


def _interpret(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    x = b.get('across_models') or {}
    if not x.get('measurable'):
        out.append('NOT MEASURABLE: arm B resolved to a single model, so the '
                   'question this experiment exists to ask was not asked. Check '
                   'that every --models entry is actually pulled.')
        return out

    between, band = x['between_spread'], x.get('band', 0.50)
    mad, rng = x.get('within_mad', 0.0), x.get('within_range', 0.0)
    ratio = x.get('between_over_within')
    out.append(f"MEDIANS: {x['median_by_model']}  between {between} PHP/L; "
               f"within-model MAD {mad}, range {rng}"
               + (f"; ratio {ratio}" if ratio is not None else ''))

    # A BAND test, not a ratio. The first version compared `between` against the
    # mean RANGE and reported a run backwards: medians +2.50 and +1.20, a factor
    # of 2.08, called "agreement survives" because 1.30 fell under a 3.25 range
    # that one outlier had set.
    if x.get('models_agree'):
        out.append(
            f'AGREEMENT SURVIVES THE MODEL CHANGE: the medians land {between} '
            f'PHP/L apart, inside the {band} band that decides whether two AGENTS '
            f'agree. One standard for both questions. Still bounded by which '
            f'models were compared.')
    else:
        out.append(
            f"THE MODELS DISAGREE: medians {between} PHP/L apart against a {band} "
            f"band, so arm A's percentage was this split hidden by one model's "
            f"consistency rather than a consensus.")
    surv = x.get('survivors_by_model') or {}
    if surv:
        on_roster = set((x.get('n_by_model') or {}).keys())
        shut_out = sorted(on_roster - set(surv))
        out.append(
            f'BRACKET: survivors reaching the judges {surv}'
            + (f'. {", ".join(shut_out)} produced NONE, so it debated and reached '
               f'the published number through no one. The Critic and '
               f'ConfidenceScorer score peers in prose, so a shut-out is a '
               f'plausible style preference rather than a numbers one.'
               if shut_out else
               '. Four survivors over two models is too few to read as bias.'))
        # Only when the models actually differ. Run 2 had the medians 0.005 PHP/L
        # apart and this printed "landed nearest llama", which is naming noise.
        near = x.get('nearest_model') if not x.get('models_agree') else None
        if near:
            out.append(f'WINNER: the published estimate landed nearest {near}, '
                       f'while the population was {x.get("n_by_model")}. Population '
                       f'weight does not decide it; the deep-tier synthesis does.')

    if rng > 3 * max(mad, 0.01):
        out.append(
            f'  Note the within-model RANGE of {rng} against a MAD of {mad}: '
            f'outliers are present, which is why the range is reported and not '
            f'used as the test.')

    out.append(
        f"DISTINCT ESTIMATES: {a['distinct_estimates']} homogeneous, "
        f"{b['distinct_estimates']} heterogeneous, over agreement "
        f"{a['agreement_pct']} and {b['agreement_pct']} percent. A percentage "
        f"that barely moves while the distinct count changes is the metric "
        f"failing to respond to the thing it claims to measure.")

    # DEC-022's warning, checked rather than assumed.
    if b['scaffold_echoes'] > a['scaffold_echoes'] or b['parse_rate_pct'] < a['parse_rate_pct']:
        out.append(
            f"QUALITY COST: the mixed roster echoed the prompt template "
            f"{b['scaffold_echoes']} times against {a['scaffold_echoes']}, and "
            f"parsed {b['parse_rate_pct']} against {a['parse_rate_pct']} percent. "
            f"Some of B's extra spread is degraded output, not added judgement.")
    else:
        out.append(
            f"No quality cost: template echoes {a['scaffold_echoes']} to "
            f"{b['scaffold_echoes']}, parse rate {a['parse_rate_pct']} to "
            f"{b['parse_rate_pct']} percent. B's spread is not weaker compliance.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default=','.join(DEFAULT_MODELS),
                        help='comma separated models for arm B')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    counts = swarm.expected_call_counts()
    print(f'Heterogeneous-roster experiment. {counts["total"]} calls per arm, '
          f'{counts["total"] * 2} total.')
    print(f'  A homogeneous    tier default')
    print(f'  B heterogeneous  {models}')
    print(f'  vintage: {vintage.describe_vintage()}  (both arms share seeds)')
    if len({m.split(':')[0] for m in models}) < 2:
        print('  NOTE: every model is from one family, so a null result bounds '
              'how much heterogeneity was tested rather than proving agreement.')
    if args.dry_run:
        return 0

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()

    results: dict[str, dict] = {}
    for name, arm_models in (('A_homogeneous', []), ('B_heterogeneous', models)):
        print(f'\n=== {name} ===')
        try:
            results[name] = _run_arm(rag, arm_models)
        except Exception as exc:
            print(f'  FAILED — {type(exc).__name__}: {str(exc)[:200]}')
            results[name] = {'failed': str(exc)[:300]}
            continue
        r = results[name]
        print(f"  estimate {r['estimate']:+.2f} ({r['estimate_source']})  "
              f"agreement {r['agreement_pct']}% (n={r['agreement_n']})  "
              f"{r['distinct_estimates']} distinct  spread {r['spread']}  "
              f"diversity {r['opening_diversity']}  ({r['seconds']:.0f}s)")
        print(f"  models: {r['models']}")

    print('\n' + '=' * 78)
    ok = {k: v for k, v in results.items() if 'failed' not in v}
    if len(ok) == 2:
        rows = [('agreement %', 'agreement_pct'), ('estimates (n)', 'agreement_n'),
                ('distinct estimates', 'distinct_estimates'),
                ('spread PHP', 'spread'), ('stdev', 'stdev'),
                ('opening diversity', 'opening_diversity'),
                ('echo %', 'echo_pct'), ('parse rate %', 'parse_rate_pct'),
                ('template echoes', 'scaffold_echoes'), ('seconds', 'seconds')]
        names = ['A_homogeneous', 'B_heterogeneous']
        print(f'{"":<20}' + ''.join(f'{n:>18}' for n in names))
        for label, key in rows:
            print(f'{label:<20}' + ''.join(f'{ok[n].get(key, 0):>18}' for n in names))
        print()
        for line in _interpret(ok[names[0]], ok[names[1]]):
            print(f'{line}\n')
    else:
        print('Both arms are needed to compare.')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'measured_at': vintage.describe_vintage(),
        'scenario': SCENARIO,
        'requested_models': models,
        'arms': results,
        'interpretation': (_interpret(*(results[k] for k in
                                        ('A_homogeneous', 'B_heterogeneous')))
                           if len(ok) == 2 else []),
    }, indent=2), encoding='utf-8')
    print(f'Wrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
