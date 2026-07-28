"""How much of the low agreement is agents contradicting themselves?

Run 29 reported 46 percent agent agreement over 23 estimates. The question this
answers is whether that number describes analysts who genuinely disagree, or an
artifact, because the two call for opposite responses: real disagreement should
be reported honestly and left alone, while an artifact should be removed.

The hypothesis came from the estimates themselves. Sorted, they were:

    -3.0 -3.0 -2.42 -2.42 -2.42 -2.42 -2.42 -2.0 -2.0 -2.0 -1.9 -1.42
     0.2  0.2   0.5   0.5   1.0   1.0   1.5   2.0  2.42  2.42  4.5

`-2.42` is exactly the physical anchor for that scenario, and `+2.42` is the same
magnitude with the sign flipped. Reading the statements behind the positives
settled it:

    NCR Forecaster:  "consumer sees a reduction in retail gasoline prices"
                     ESTIMATE: +PHP4.50/L

The agent reasons downward, states the reduction in its own causal chain, and
then writes a positive sign. `parse_fuel_estimate` already refuses an UNSIGNED
number contradicted by the prose; these are explicitly signed, so they pass.

This is deliberately a counterfactual on STORED data, not a new run. It costs no
quota, it is reproducible by anyone with the database, and the comparison is
paired: the same agents, the same scenario, the same seeds.

    python -m ph_economic_ai.tools.experiment_sign_contradiction
    python -m ph_economic_ai.tools.experiment_sign_contradiction --run 29
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine.swarm import _robust_confidence_pct, measure_agreement

DB = Path(__file__).resolve().parents[1] / 'cache' / 'trust.db'
ARTIFACT = Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts' / 'sign_contradiction.json'

# The detector must read the agent's CONCLUSION, not its premises.
#
# The first version searched the whole statement for fall language and was
# circular: every prompt states a -8.0% oil shock, so every agent restates it,
# and the detector fired on 100 percent of positive estimates AND 100 percent of
# negative ones. "Contradicts" had quietly become "is positive", so removing them
# from a bimodal set produced 96 percent agreement out of nothing.
#
# The prompt asks for `CAUSAL CHAIN: [shock] -> [market effect] -> [retail
# mechanism] -> [consumer impact]`. The LAST segment is the agent's own statement
# of where the price ends up, which is the only part that can contradict a sign.
_CHAIN_RE = re.compile(r'CAUSAL\s*CHAIN\s*:?\s*(.+?)(?:\n|ESTIMATE|$)',
                       re.IGNORECASE | re.DOTALL)
_ARROW = re.compile(r'->|→|=>')

_FALL_WORDS = re.compile(
    r'\b(reduction|reduced?|reduces|decrease[sd]?|declin\w+|drop\w*|fall\w*|'
    r'lower\w*|cheaper|relief|ease[sd]?|easing|savings?)\b', re.IGNORECASE)
_RISE_WORDS = re.compile(
    r'\b(increase[sd]?|rise[sn]?|rising|higher|hike[sd]?|surge\w*|climb\w*|'
    r'costlier|expensive|burden|upward)\b', re.IGNORECASE)


def conclusion_direction(statement: str) -> Optional[int]:
    """-1, +1, or None from the causal chain's final segment.

    None when the agent gave no chain, or when its conclusion names both
    directions or neither. Unknown is a real answer here: guessing would put the
    circularity straight back in.
    """
    text = ' '.join((statement or '').split())
    m = _CHAIN_RE.search(text)
    if not m:
        return None
    segments = [s.strip() for s in _ARROW.split(m.group(1)) if s.strip()]
    if len(segments) < 2:
        return None
    tail = segments[-1]
    down, up = bool(_FALL_WORDS.search(tail)), bool(_RISE_WORDS.search(tail))
    if down == up:                      # both or neither: not a usable signal
        return None
    return -1 if down else 1


def _classify(estimate: Optional[float], statement: str) -> str:
    """'contradicts', 'consistent', or 'unusable'."""
    if estimate is None:
        return 'unusable'
    stated = conclusion_direction(statement)
    if stated is None:
        return 'consistent'             # no claim to contradict
    if (estimate > 0) != (stated > 0):
        return 'contradicts'
    return 'consistent'


def _direction_agreement(values: list[float]) -> int:
    """Share of estimates sharing the majority sign.

    Reported beside magnitude agreement because they answer different questions,
    and a household's actual question is "will it go up", which is this one.
    """
    signs = [1 if v > 0 else -1 for v in values if v != 0]
    if not signs:
        return 0
    lead = max(set(signs), key=signs.count)
    return int(round(100 * signs.count(lead) / len(signs)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=int, default=None, help='run_id (default: latest)')
    parser.add_argument('--db', default=str(DB))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    run_id = args.run or list(conn.execute('SELECT MAX(run_id) FROM runs'))[0][0]
    run = dict(list(conn.execute('SELECT * FROM runs WHERE run_id=?', (run_id,)))[0])
    rows = [dict(r) for r in conn.execute(
        'SELECT agent_name, round_num, estimate, statement '
        'FROM agent_responses WHERE run_id=?', (run_id,))]
    conn.close()

    def region(name: str) -> str:
        for g in ('NCR', 'Central Luzon', 'Western Visayas', 'Davao Region'):
            if name.startswith(g):
                return g
        return '?'

    for r in rows:
        r['verdict'] = _classify(r['estimate'], r['statement'])

    kept = [r for r in rows if r['verdict'] == 'consistent' and r['estimate'] is not None]
    bad = [r for r in rows if r['verdict'] == 'contradicts']
    unusable = [r for r in rows if r['verdict'] == 'unusable']
    all_valid = [r for r in rows if r['estimate'] is not None]

    print(f'Run {run_id}: reported {run["confidence_pct"]}% over '
          f'{len(all_valid)} estimates, final {run["final_estimate"]:+.2f}')
    print(f'  responses {len(rows)}, unparseable {len(unusable)} '
          f'({100 * len(unusable) / max(len(rows), 1):.0f}%)')
    print()

    print('Agents whose ESTIMATE sign contradicts their own reasoning:')
    for r in sorted(bad, key=lambda r: -r['estimate']):
        text = ' '.join((r['statement'] or '').split())
        m = _CHAIN_RE.search(text)
        tail = ([s.strip() for s in _ARROW.split(m.group(1)) if s.strip()] or [''])[-1] if m else ''
        print(f'  {r["agent_name"][:30]:<30} {r["estimate"]:+.2f}   concluded: "{tail[:70]}"')
    if not bad:
        print('  (none)')
    print()
    print(f'  {len(bad)} of {len(all_valid)} usable estimates '
          f'({100 * len(bad) / max(len(all_valid), 1):.0f}%)')
    # How much of the room the detector can even judge. A conclusion-only test is
    # blind to an agent that gave no causal chain, and saying so is the
    # difference between a measurement and an overclaim.
    judged = sum(1 for r in all_valid if conclusion_direction(r['statement']) is not None)
    print(f'  judged {judged} of {len(all_valid)}; the rest gave no parseable '
          f'causal chain, so nothing can be said about their sign')
    print()

    # Per region, as the master verdict measures it.
    def per_region(items):
        groups: dict[str, list[float]] = {}
        for r in items:
            groups.setdefault(region(r['agent_name']), []).append(r['estimate'])
        scored = [(_robust_confidence_pct(v, None), len(v))
                  for v in groups.values() if len(v) >= 2]
        if not scored:
            return 0, 0, 0
        return (int(round(statistics.mean(s for s, _ in scored))),
                sum(n for _, n in scored), len(scored))

    as_measured, n_all, reg_all = per_region(all_valid)
    corrected, n_kept, reg_kept = per_region(kept)

    print('AGREEMENT, per region and averaged (the metric the card shows):')
    print(f'  as measured                    {as_measured}%  over {n_all} estimates, {reg_all} regions')
    print(f'  refusing self-contradictions   {corrected}%  over {n_kept} estimates, {reg_kept} regions')
    print()
    print('DIRECTION agreement (does the room agree it goes up or down?):')
    print(f'  as measured                    {_direction_agreement([r["estimate"] for r in all_valid])}%')
    print(f'  refusing self-contradictions   {_direction_agreement([r["estimate"] for r in kept])}%')
    print()

    unjudged = len(all_valid) - judged
    no_chain_pct = 100 * unjudged / max(len(all_valid), 1)
    print('READING')
    print(f'  Self-contradiction is real but small: {len(bad)} estimates, worth '
          f'{corrected - as_measured} points. It does not explain a {as_measured}% figure.')
    print(f'  The larger finding is format compliance. {unjudged} of '
          f'{len(all_valid)} usable estimates ({no_chain_pct:.0f}%) carried no '
          f'parseable causal chain, and {len(unusable)} of {len(rows)} responses '
          f'produced no estimate at all.')
    print('  The agents mostly READ the shock the same way. What varies is whether '
          'they emit the answer in the form the parser and the metric require, '
          'which is a capability and prompt-contract problem rather than an '
          'analytical disagreement.')
    print()
    print('  NOT a proposal to flip signs. An agent that concludes "a reduction" '
          'and writes "+4.50" has made one claim too many; which it meant is '
          'unknowable from outside, so the honest options are to re-ask it or to '
          'refuse the number, never to guess.')
    print('  NOT a licence to drop estimates that merely disagree. Only a '
          'contradiction with the agent\'s OWN stated conclusion qualifies, and '
          'an agent that gave no conclusion is left alone.')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'run_id': run_id,
        'reported_pct': run['confidence_pct'],
        'responses': len(rows),
        'unparseable': len(unusable),
        'usable': len(all_valid),
        'contradicting': len(bad),
        'agreement_as_measured': as_measured,
        'agreement_refusing_contradictions': corrected,
        'direction_as_measured': _direction_agreement([r['estimate'] for r in all_valid]),
        'direction_refusing_contradictions': _direction_agreement([r['estimate'] for r in kept]),
        'contradicting_agents': [
            {'agent': r['agent_name'], 'estimate': r['estimate']} for r in bad],
    }, indent=2), encoding='utf-8')
    print(f'\nWrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
