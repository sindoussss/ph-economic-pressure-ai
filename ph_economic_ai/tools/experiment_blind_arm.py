"""Is the agent agreement real, or is the design manufacturing it?

The report shows a percentage labelled "agent agreement". Three mechanisms in
the swarm could produce that number without any agents independently agreeing.
All three were active when this was written; the first was turned off on
2026-07-29 on the strength of what this measured, so a re-run now compares the
current default against the old one rather than against itself.

1. **Round 1 was not blind.** It runs sequentially and every agent's prompt
   carried the statements of the agents before it. Now off by default for the
   estimating roles. The Forum was deliberately
   fixed to stop doing this, on the grounds recorded in the vault: agreement
   between agents that have read each other is herding rather than
   corroboration, and the confidence number is computed from exactly that
   agreement.

2. **Round 2 instructs agents to converge.** The reconciliation rule tells them
   to revise toward the group median unless they can cite a reason not to, and
   to prefer consensus over an outlier.

3. **The threshold in that instruction WAS the threshold the metric measures.**
   Agents are told to land within 1.00 PHP/L of the median, and `agreement`
   scored the fraction within 1.00 PHP/L of the medoid, so a compliant agent was
   scored as an agreeing agent by construction. The metric band moved to 0.50 on
   2026-07-29, so the two no longer coincide; the instruction itself is
   unchanged and still nudges toward consensus.

This runs the same scenario three times to separate them:

    A  peers visible  round 1 as it was before 2026-07-29
    B  blind          estimating roles cannot see same-round peers (now default)
    C  blind + free   also without the reconciliation rule

Only the estimating roles go blind. The Critic and ConfidenceScorer score other
agents by name and cannot work without reading them; their estimates are a
minority of the population and the alternative is breaking the elimination
bracket to protect a measurement.

Seeing EARLIER rounds is never removed. Round 2 exists so agents can respond to
what the room said, and taking that away would test a different system rather
than this one.

Read it this way: if agreement holds across all three arms, it is real and the
mechanisms were not doing the work. If it collapses in B, the number was
herding. If it holds in B and collapses in C, agents were following an
instruction to agree.

    python -m ph_economic_ai.tools.experiment_blind_arm --dry-run
    python -m ph_economic_ai.tools.experiment_blind_arm
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from ph_economic_ai.engine import anchoring, llm, swarm, vintage
from ph_economic_ai.engine.rag import RagEngine

ARTIFACT = Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts' / 'blind_arm.json'

SCENARIO = {'oil_pct': -8.0, 'usd_pct': -0.1, 'bsp_rate': 6.5, 'demand_index': 75.0}
_ECHO_TOL = 0.005

# Named by CONFIGURATION, not by which one happens to ship. Blinding became the
# default on 2026-07-29, so "A_shipped" would now point at the arm the experiment
# argued against, and a re-run would read backwards.
ARMS = [
    ('A_peers_visible', dict(blind_round_one=False, reconcile=True)),
    ('B_blind',         dict(blind_round_one=True,  reconcile=True)),
    ('C_blind_free',    dict(blind_round_one=True,  reconcile=False)),
]


def _measure(verdict, seconds: float) -> dict:
    responses = list(getattr(verdict, 'all_responses', []) or [])
    estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
    anchor = verdict.physical_anchor
    if anchor is None:
        anchor = anchoring.fuel_passthrough_anchor(
            SCENARIO['oil_pct'], SCENARIO['usd_pct'])
    signs = [1 if e > 0 else -1 for e in estimates if e != 0]
    lead = max(set(signs), key=signs.count) if signs else 0
    counts = Counter(round(e, 2) for e in estimates)
    return {
        'agreement_pct': verdict.confidence_pct,
        'agreement_n': getattr(verdict, 'agreement_n', 0),
        'agreement_regions': list(getattr(verdict, 'agreement_regions', (0, 0))),
        'echo_n': getattr(verdict, 'agreement_echo_n', 0),
        'echo_pct': round(100 * getattr(verdict, 'agreement_echo_n', 0)
                          / max(len(estimates), 1), 1),
        'direction_pct': round(100 * signs.count(lead) / len(signs)) if signs else 0,
        'parse_rate_pct': round(100 * len(estimates) / max(len(responses), 1), 1),
        # Shared with the engine so the report and the experiment cannot drift.
        # This tool carried its own copy until 2026-07-29, and that copy had the
        # engine's original bug: distinct openings over RESPONSES, when an agent
        # speaks in both rounds. The figures in the 2026-07-29 blind-arm artifact
        # (0.25 / 0.25 / 0.125) were computed over 32 responses from 20 agents, so
        # their ceiling was 0.625 and they are not comparable to anything measured
        # after this date. The comparison BETWEEN those arms survives — all three
        # shared one roster and one denominator — so "diversity did not rise under
        # blinding" still stands. Only the scale was wrong.
        'opening_diversity': swarm.opening_diversity(responses),
        'distinct_estimates': len(counts),
        'spread': round(max(estimates) - min(estimates), 2) if len(estimates) > 1 else 0.0,
        'stdev': round(statistics.pstdev(estimates), 3) if len(estimates) > 1 else 0.0,
        'estimate': verdict.final_estimate,
        'estimate_source': verdict.estimate_source,
        'seconds': round(seconds, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    counts = swarm.expected_call_counts()
    print(f'Blind-arm experiment. {counts["total"]} calls per arm, '
          f'{counts["total"] * len(ARMS)} total.')
    print(f'  agents on {llm.provider_for(llm.FAST)}, '
          f'judges on {llm.provider_for(llm.DEEP)}')
    print(f'  vintage: {vintage.describe_vintage()}  (all arms share seeds)')
    for name, cfg in ARMS:
        print(f'  {name:<14} blind_round_one={cfg["blind_round_one"]!s:<5} '
              f'reconcile={cfg["reconcile"]}')
    if args.dry_run:
        return 0

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()

    results: dict[str, dict] = {}
    for name, cfg in ARMS:
        print(f'\n=== {name} ===')
        started = time.monotonic()
        try:
            verdict = swarm.SwarmOrchestrator(
                rag=rag, scenario=dict(SCENARIO), parallel_n=2, **cfg).run()
        except Exception as exc:
            print(f'  FAILED — {type(exc).__name__}: {str(exc)[:200]}')
            results[name] = {'failed': str(exc)[:300]}
            continue
        results[name] = _measure(verdict, time.monotonic() - started)
        r = results[name]
        print(f'  agreement {r["agreement_pct"]}% (n={r["agreement_n"]})  '
              f'direction {r["direction_pct"]}%  echo {r["echo_pct"]}%  '
              f'openings {r["opening_diversity"]}  spread {r["spread"]}  '
              f'({r["seconds"]:.0f}s)')

    print('\n' + '=' * 78)
    ok = {k: v for k, v in results.items() if 'failed' not in v}
    if len(ok) < 2:
        print('Too few arms completed to compare.')
    else:
        rows = [('agreement %', 'agreement_pct'), ('direction %', 'direction_pct'),
                ('estimates (n)', 'agreement_n'), ('echo %', 'echo_pct'),
                ('opening diversity', 'opening_diversity'),
                ('distinct estimates', 'distinct_estimates'),
                ('spread PHP', 'spread'), ('stdev', 'stdev'),
                ('parse rate %', 'parse_rate_pct'), ('seconds', 'seconds')]
        names = [n for n, _ in ARMS if n in ok]
        print(f'{"":<20}' + ''.join(f'{n:>16}' for n in names))
        for label, key in rows:
            print(f'{label:<20}' + ''.join(f'{ok[n].get(key, 0):>16}' for n in names))
        print()

        a = ok.get('A_peers_visible')
        b = ok.get('B_blind')
        c = ok.get('C_blind_free')
        if a and b:
            drop = a['agreement_pct'] - b['agreement_pct']
            if drop >= 15:
                print(f'HERDING: blinding round 1 cost {drop} points of agreement. '
                      f'A large share of the shipped number was agents reading '
                      f'each other, not agreeing with each other.')
            elif drop <= 5:
                print(f'NOT HERDING: blinding round 1 moved agreement by {drop} '
                      f'points. Agents reach similar numbers without seeing each '
                      f'other, which is what the label claims.')
            else:
                print(f'PARTIAL: blinding cost {drop} points. Some of the number '
                      f'was peer visibility; most was not.')
            if b['opening_diversity'] > a['opening_diversity'] + 0.05:
                print(f'  Corroborated by opening diversity rising '
                      f'{a["opening_diversity"]} -> {b["opening_diversity"]}: the '
                      f'blind agents genuinely wrote different things.')
        if b and c:
            drop2 = b['agreement_pct'] - c['agreement_pct']
            if drop2 >= 15:
                print(f'INSTRUCTION-FOLLOWING: removing the reconciliation rule cost '
                      f'a further {drop2} points. Agents were converging because '
                      f'they were told to, and the metric scores that at the same '
                      f'1.00 PHP/L threshold the instruction names.')
            else:
                print(f'The reconciliation rule accounts for {drop2} points, so it '
                      f'is not the main driver.')
        print()
        print('One run per arm. Enough to see a collapse; not enough to size one.')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'measured_at': vintage.describe_vintage(),
        'scenario': SCENARIO,
        'provider_fast': llm.provider_for(llm.FAST),
        'provider_deep': llm.provider_for(llm.DEEP),
        'arms': results}, indent=2), encoding='utf-8')
    print(f'Wrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
