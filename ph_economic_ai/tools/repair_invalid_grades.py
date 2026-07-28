"""Clear grades that were scored against a baseline that never existed.

Every graded run in the store recorded an actual price change of exactly
-14.44 PHP/L: the observed 84.38 minus `_FALLBACK_RETAIL_PRICE_PHP` of 98.82,
which the stored scenario carried instead of the price the run reasoned from.
No week saw a 14 peso move. See ADR-008.

Those grades then set agent trust. `compute_accuracy_score` floors at a 3 PHP/L
error, so every agent scored 0.0 on accuracy; at 60 percent weight in the trust
update that drags a 0.60-internal agent to 0.24, under the 0.30 demotion line.
Seven of twenty agents were benched by an outcome that never happened.

The code no longer produces such grades (`DEC-020`), but the rows and the trust
scores they poisoned remain. This clears them.

What is cleared:

- `actual_price_change`, `accuracy_error`, `graded_at`, `graded_against` on any
  run whose implied outcome is outside the plausibility bound. The runs
  themselves, their estimates and their agent responses are untouched, so they
  become ungraded rather than deleted and can be graded honestly later.
- `trust_score` back to the neutral prior, `avg_accuracy_error` to NULL, and the
  tier back to default, for every agent.

What is KEPT, deliberately:

- `runs_participated` and `avg_internal_score`. Those come from response quality
  (citations, causal chain), never from the grading, so they are real evidence
  and discarding them would throw away good data to fix bad.
- Every price observation. Those are real DOE readings.

    python -m ph_economic_ai.tools.repair_invalid_grades            # dry run
    python -m ph_economic_ai.tools.repair_invalid_grades --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / 'cache' / 'trust.db'

# Same bound the estimate parser and the grader use. A weekly DOE adjustment
# beyond this is a broken baseline, not a market event.
from ph_economic_ai.engine.debate import _MAX_REALISTIC_FUEL_PHP_L as _MAX_CHANGE
from ph_economic_ai.engine.store import _TRUST_INIT
from ph_economic_ai.engine.swarm import _FALLBACK_RETAIL_PRICE_PHP


def _invalid_runs(conn) -> list[dict]:
    """Grades that cannot be trusted, and the rule that caught each.

    Two rules, because there are two failure modes and a magnitude bound alone
    only finds one of them:

    **stale baseline** — the stored `current_price` is the fallback constant, so
    the run never recorded the price it reasoned from. This is the real test.
    Every graded run in the store failed it.

    **implausible outcome** — the implied change exceeds the same bound the
    estimate parser applies. Catches a stale baseline once the observation is
    real, producing the -14.44 that made the defect visible.

    The magnitude rule was written first and missed ten rows recording an actual
    change of exactly +0.00: at that point the price scrape was ALSO returning
    the fallback, so the grade was the constant minus itself. A spurious zero is
    more dangerous than a spurious 14, because it looks like a quiet week and
    scores every estimate as wrong by its own magnitude.
    """
    rows = [dict(r) for r in conn.execute(
        'SELECT run_id, scenario_json, final_estimate, actual_price_change, '
        'accuracy_error FROM runs WHERE actual_price_change IS NOT NULL')]
    bad = []
    for r in rows:
        change = r['actual_price_change']
        try:
            baseline = json.loads(r['scenario_json']).get('current_price')
        except (ValueError, TypeError):
            baseline = None
        r['baseline'] = baseline

        reasons = []
        if baseline is not None and abs(baseline - _FALLBACK_RETAIL_PRICE_PHP) < 0.005:
            reasons.append('baseline is the fallback constant')
        if change is not None and abs(change) > _MAX_CHANGE:
            reasons.append(f'implied change beyond +/-{_MAX_CHANGE:.0f}')
        if reasons:
            r['reasons'] = reasons
            bad.append(r)
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true',
                        help='write the changes (default is a dry run)')
    parser.add_argument('--db', default=str(DB))
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    graded = list(conn.execute(
        'SELECT COUNT(*) FROM runs WHERE actual_price_change IS NOT NULL'))[0][0]
    bad = _invalid_runs(conn)
    agents = [dict(r) for r in conn.execute(
        'SELECT agent_name, trust_score, current_model_tier, runs_participated, '
        'avg_internal_score FROM agent_trust ORDER BY trust_score')]

    print(f'Database: {args.db}')
    print(f'  graded runs: {graded}, of which invalid: {len(bad)}')
    if bad:
        by_change: dict[float, int] = {}
        for r in bad:
            by_change[round(r['actual_price_change'], 2)] = \
                by_change.get(round(r['actual_price_change'], 2), 0) + 1
        for change, count in sorted(by_change.items()):
            note = ('the fallback minus itself: the price scrape was also '
                    'failing' if abs(change) < 0.005 else
                    'a real observation against a stale baseline')
            print(f'  {count:>3} runs recorded an actual change of {change:+.2f}  '
                  f'({note})')
        print(f'  every one stored a baseline of '
              f'{_FALLBACK_RETAIL_PRICE_PHP}, the fallback constant, so none '
              f'recorded the price it actually reasoned from')
    print()

    demoted = [a for a in agents if a['current_model_tier'] == 'demoted']
    print(f'  agents tracked: {len(agents)}, currently demoted: {len(demoted)}')
    for a in demoted:
        print(f'    {a["agent_name"]:<34} trust {a["trust_score"]:.2f}  '
              f'internal {a["avg_internal_score"]:.2f}  '
              f'(internal is real evidence and is KEPT)')
    print()

    if not args.apply:
        print('DRY RUN. Nothing written. Re-run with --apply to make these changes:')
        print(f'  - clear the grade columns on {len(bad)} runs (rows and estimates kept)')
        print(f'  - reset {len(agents)} trust scores to the {_TRUST_INIT} neutral prior')
        print('  - keep runs_participated, avg_internal_score, and all price observations')
        conn.close()
        return 0

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = Path(args.db).with_suffix(f'.db.bak_{stamp}')
    conn.close()
    shutil.copy2(args.db, backup)
    print(f'Backed up to {backup}')

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    ids = [r['run_id'] for r in bad]
    if ids:
        conn.executemany(
            'UPDATE runs SET actual_price_change=NULL, accuracy_error=NULL, '
            'graded_at=NULL, graded_against=NULL WHERE run_id=?',
            [(i,) for i in ids])
    conn.execute(
        'UPDATE agent_trust SET trust_score=?, avg_accuracy_error=NULL, '
        'current_model_tier=?, last_updated=?',
        (_TRUST_INIT, 'default', datetime.now().isoformat()))
    conn.commit()

    still_graded = list(conn.execute(
        'SELECT COUNT(*) FROM runs WHERE actual_price_change IS NOT NULL'))[0][0]
    tiers = {r[0]: r[1] for r in conn.execute(
        'SELECT current_model_tier, COUNT(*) FROM agent_trust GROUP BY 1')}
    kept = list(conn.execute(
        'SELECT COUNT(*), SUM(runs_participated) FROM agent_trust'))[0]
    obs = list(conn.execute('SELECT COUNT(*) FROM price_observations'))[0][0]
    responses = list(conn.execute('SELECT COUNT(*) FROM agent_responses'))[0][0]
    conn.close()

    print()
    print('After:')
    print(f'  graded runs remaining: {still_graded}  (was {graded})')
    print(f'  trust tiers: {tiers}')
    print(f'  agents kept with their participation history: {kept[0]} '
          f'({kept[1]} run-participations retained)')
    print(f'  agent responses untouched: {responses}')
    print(f'  price observations untouched: {obs}')
    print()
    print('The runs are ungraded, not deleted. Once real observations accumulate '
          'against a correct baseline they will grade honestly.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
