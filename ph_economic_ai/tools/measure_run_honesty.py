"""The three things today's fixes predicted but did not measure.

Four defects were fixed on 2026-07-28 (ADR-004, ADR-005). Two of the fixes make
claims that stored runs cannot settle, and one older claim has never been tested
at all. Each is a way the work could be wrong, so each gets a number rather than
a paragraph.

1. **Echo rate.** ADR-004 gave every agent the mechanical pass-through anchor,
   expecting real agreement to rise. The shared-anchor experiment already found
   agents ECHOING a plausible anchor: five of seven returned it to two decimals,
   which `_robust_confidence_pct` scores as five independent agreements. If the
   agents are repeating the number they were handed, the agreement gain is
   suggestion and must be reported that way. This is the measurement that can
   invalidate the fix, which is why it runs first.

2. **Roster size against agreement.** ADR-005 restores the swarm from 13 agents
   to 20. A bigger room is not a better one: agreement could rise purely because
   there are more estimates in each average. Only a same-scenario, same-seed
   comparison separates the two, and this harness records the roster alongside
   the number so the comparison is possible at all.

3. **Reproducibility.** ADR-002 claimed a re-run on identical inputs reproduces
   and never demonstrated it; ADR-006 removed the two reasons it could not
   (a minute-resolution clock in every prompt, and seeds keyed on live market
   floats). With `--repeats 2` this checks whether it now actually does.

   Provider matters here and the result is only as strong as the provider.
   Ollama honours seeds; Groq's is best-effort upstream; Gemini discards them
   silently. A negative result on Groq or Gemini says nothing about the fix. Run
   this one locally.

Deliberately outside `benchmark/`: that is the validated half and must stay
reproducible with no API key, a boundary `tests/test_benchmark_isolation.py`
enforces. This is exploratory tooling and needs a provider.

Usage
-----
    python -m ph_economic_ai.tools.measure_run_honesty --dry-run
    python -m ph_economic_ai.tools.measure_run_honesty --repeats 2

Costs real quota: about 39 calls per repeat.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine import anchoring, llm, swarm, vintage
from ph_economic_ai.engine.rag import RagEngine

ARTIFACT = Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts' / 'run_honesty.json'

#: A verdict counts as echoing the anchor when it lands this close to it. Two
#: decimal places is the resolution the models actually quote at, so equality to
#: 2dp is repetition rather than coincidence.
_ECHO_TOLERANCE = 0.005
#: Wider band, for "clustered on the anchor without quoting it exactly".
_NEAR_ANCHOR = 0.25

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}


def _echo_stats(estimates: list[float], anchor: float) -> dict:
    """How much of the room is repeating the anchor rather than reasoning to it.

    `modal_is_anchor` is the decisive field. A handful of agents landing on the
    anchor is agreement with physics, which is the point of giving it to them.
    The MODE being the anchor, exactly, means the room's centre is the prompt.
    """
    if not estimates:
        return {'n': 0}
    echoes = [e for e in estimates if abs(e - anchor) <= _ECHO_TOLERANCE]
    near = [e for e in estimates if abs(e - anchor) <= _NEAR_ANCHOR]
    counts = Counter(round(e, 2) for e in estimates)
    modal_value, modal_n = counts.most_common(1)[0]
    return {
        'n': len(estimates),
        'anchor': round(anchor, 3),
        'echo_exact_n': len(echoes),
        'echo_exact_pct': round(100 * len(echoes) / len(estimates), 1),
        'within_0.25_n': len(near),
        'within_0.25_pct': round(100 * len(near) / len(estimates), 1),
        'modal_value': modal_value,
        'modal_n': modal_n,
        'modal_is_anchor': abs(modal_value - anchor) <= _ECHO_TOLERANCE,
        'mean_distance_from_anchor': round(
            statistics.fmean(abs(e - anchor) for e in estimates), 3),
        'spread': round(max(estimates) - min(estimates), 3),
    }


def _opening_collisions(responses: list) -> dict:
    """Is a low diversity score herding, or is it the prompt template?

    The corrected metric returned 6 distinct openings from 20 agents, and the
    swarm happens to seat 4 regions x 5 ROLES. Six is suspiciously close to five.
    If every Critic opens the same way regardless of region, the metric is
    reading prompt structure and calling it herding — the same error as the
    circular regex and the unreachable ceiling, a third time.

    The discriminator is which axis the collisions follow. Openings that collide
    ACROSS regions but never across roles are templated. Openings that collide
    across roles too are agents converging on the same first sentence, which is
    what the metric is supposed to catch.
    """
    first: dict = {}
    for r in responses:
        name = getattr(r, 'agent_name', None)
        rnd = getattr(r, 'round_num', 0) or 0
        if name is None:
            continue
        prior = first.get(name)
        if prior is None or rnd < (getattr(prior, 'round_num', 0) or 0):
            first[name] = r

    groups: dict[str, list[str]] = {}
    for name, r in first.items():
        key = ' '.join((getattr(r, 'statement', '') or '').split())[:80]
        if key:
            groups.setdefault(key, []).append(name)

    shared = {k: v for k, v in groups.items() if len(v) > 1}
    roles_per_group = [len({n.split()[-1] for n in v}) for v in shared.values()]
    return {
        'agents': len(first),
        'distinct_openings': len(groups),
        'colliding_groups': len(shared),
        # 1 means every collision is one role repeating itself across regions.
        'max_roles_in_a_collision': max(roles_per_group, default=0),
        'single_role_collisions': sum(1 for c in roles_per_group if c == 1),
        'cross_role_collisions': sum(1 for c in roles_per_group if c > 1),
        'examples': [{'opening': k[:70], 'agents': sorted(v)}
                     for k, v in sorted(shared.items(), key=lambda kv: -len(kv[1]))[:4]],
    }


def _roster_stats(responses: list) -> dict:
    """Who actually turned up, which ADR-005 is supposed to have restored."""
    names = {r.agent_name for r in responses}
    per_round = Counter(getattr(r, 'round_num', 0) for r in responses)
    parsed = [r for r in responses if r.price_estimate is not None]
    return {
        'agents': len(names),
        'responses': len(responses),
        'responses_per_round': dict(sorted(per_round.items())),
        'parse_rate_pct': round(100 * len(parsed) / len(responses), 1) if responses else 0.0,
        'roles_present': sorted({n.split()[-1] for n in names}),
    }


def _run_once(rag: RagEngine, scenario: dict) -> dict:
    started = time.monotonic()
    orch = swarm.SwarmOrchestrator(rag=rag, scenario=scenario, parallel_n=2)
    verdict = orch.run()
    seconds = time.monotonic() - started

    responses = list(getattr(verdict, 'all_responses', []) or [])
    estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
    anchor = verdict.physical_anchor
    if anchor is None:
        anchor = anchoring.fuel_passthrough_anchor(
            scenario.get('oil_pct', 0.0), scenario.get('usd_pct', 0.0))

    return {
        'estimate': verdict.final_estimate,
        'estimate_source': verdict.estimate_source,
        'agreement_pct': verdict.confidence_pct,
        'agreement_n': getattr(verdict, 'agreement_n', 0),
        'agreement_regions': list(getattr(verdict, 'agreement_regions', (0, 0))),
        # A percentage cannot separate agents who agreed from agents who copied.
        # These two can, and they are the reason round 1 was blinded.
        'agreement_distinct': getattr(verdict, 'agreement_distinct', 0),
        'agreement_diversity': getattr(verdict, 'agreement_diversity', 0.0),
        'regional': [
            {'pair': ' & '.join(v.region_pair), 'estimate': v.estimate,
             'agreement_pct': round((v.confidence or 0) * 100)}
            for v in verdict.regional_verdicts
        ],
        'echo': _echo_stats(estimates, anchor),
        'openings': _opening_collisions(responses),
        'roster': _roster_stats(responses),
        'seconds': round(seconds, 1),
    }


def _verdict_line(run: dict) -> str:
    e = run['echo']
    reg = run['agreement_regions']
    return (f"  estimate {run['estimate']:+.2f} ({run['estimate_source']})  "
            f"agreement {run['agreement_pct']}% over n={run['agreement_n']} "
            f"in {reg[0]}/{reg[1]} regions, {run['agreement_distinct']} distinct  "
            f"diversity {run['agreement_diversity']}  "
            f"roster {run['roster']['agents']} agents, "
            f"{run['roster']['parse_rate_pct']}% parsed  ({run['seconds']:.0f}s)")


def _interpret(runs: list[dict]) -> list[str]:
    """State what the numbers mean, including when they mean the fix is wrong."""
    out: list[str] = []
    if not runs:
        return ['No completed runs, so nothing is measured.']

    echoes = [r['echo'] for r in runs if r['echo'].get('n')]
    if echoes:
        modal_hits = sum(1 for e in echoes if e['modal_is_anchor'])
        worst = max(e['echo_exact_pct'] for e in echoes)
        if modal_hits:
            out.append(
                f'ECHO: the modal agent estimate IS the anchor in {modal_hits} of '
                f'{len(echoes)} runs. The agreement gain from the shared anchor is '
                f'suggestion, not consensus, and must be reported that way.')
        elif worst >= 40:
            out.append(
                f'ECHO: up to {worst}% of agents quoted the anchor exactly. Not '
                f'the mode, but high enough that the agreement figure is partly '
                f'measuring repetition.')
        else:
            out.append(
                f'ECHO: at most {worst}% of agents quoted the anchor exactly and '
                f'the mode is elsewhere. The room is reasoning around the anchor '
                f'rather than repeating it.')

    rosters = {r['roster']['agents'] for r in runs}
    out.append(f'ROSTER: {sorted(rosters)} agents. Compare agreement against a '
               f'stored 13-agent run before crediting the fix with the change; a '
               f'larger room raises n as well as agreement.')

    if len(runs) > 1:
        estimates = [r['estimate'] for r in runs if r['estimate'] is not None]
        identical = len(set(round(e, 2) for e in estimates)) == 1
        provider = llm.active_provider()
        note = '' if provider == 'ollama' else (
            f' Provider is {provider}, whose seed is not guaranteed, so a '
            f'negative result here does not indict the fix.')
        out.append(
            f'REPRODUCIBILITY: {len(estimates)} runs on identical inputs gave '
            f'{sorted(estimates)} — {"identical" if identical else "DIFFERENT"}.{note}')
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats', type=int, default=1,
                        help='2 or more also tests reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the quota cost and exit without calling anything')
    args = parser.parse_args()

    counts = swarm.expected_call_counts()
    print(f'Plan — {args.repeats} run(s) of the full swarm')
    print(f'  {counts["total"]} calls each ({counts["fast"]} fast, {counts["deep"]} deep), '
          f'{counts["total"] * args.repeats} total')
    print('  retries on unparsed estimates cost extra and are not in that number')
    print(f'  vintage: {vintage.describe_vintage()}')
    if args.dry_run:
        return 0

    if not llm.is_configured():
        print('\nNo provider configured. Set GROQ_API_KEY or GEMINI_API_KEY, '
              'or start Ollama.')
        return 1
    print(f'  provider: {llm.active_provider()} '
          f'({llm.describe_model(llm.FAST)} / {llm.describe_model(llm.DEEP)})')

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()

    runs: list[dict] = []
    for i in range(args.repeats):
        print(f'\n=== run {i + 1} of {args.repeats} ===')
        try:
            run = _run_once(rag, SCENARIO)
        except Exception as exc:
            print(f'  FAILED — {type(exc).__name__}: {exc}')
            continue
        runs.append(run)
        print(_verdict_line(run))
        e = run['echo']
        print(f"  anchor {e['anchor']:+.2f}  "
              f"quoted exactly by {e['echo_exact_n']}/{e['n']} agents "
              f"({e['echo_exact_pct']}%)  "
              f"mode {e['modal_value']:+.2f} x{e['modal_n']}"
              f"{'  <-- MODE IS THE ANCHOR' if e['modal_is_anchor'] else ''}")
        o = run['openings']
        print(f"  openings {o['distinct_openings']} distinct from {o['agents']} "
              f"agents; {o['colliding_groups']} collide "
              f"({o['single_role_collisions']} within one role, "
              f"{o['cross_role_collisions']} across roles)")
        for ex in o['examples']:
            print(f"    x{len(ex['agents'])}  {ex['agents']}")
            print(f"        {ex['opening']}")

    print('\n' + '=' * 70)
    for line in _interpret(runs):
        print(f'\n{line}')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'measured_at': vintage.describe_vintage(),
        'provider': llm.active_provider(),
        'models': {'fast': llm.describe_model(llm.FAST),
                   'deep': llm.describe_model(llm.DEEP)},
        'scenario': SCENARIO,
        'runs': runs,
        'interpretation': _interpret(runs),
    }, indent=2), encoding='utf-8')
    print(f'\nWrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
