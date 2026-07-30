"""Is the published number the room's answer, or the judge's reading of it?

`Q-ENG-011`. A heterogeneous agent roster diversifies the DEBATE and not the
CONCLUSION. The survivors feed the regional judges and the master judge, all on
the deep tier, which `assign_models` does not reach, so however many families sit
in the room the final figure is one model's synthesis. `ui.honesty.synthesis_note`
says so on the card. Saying so is not the same as knowing how much it matters.

The measurable question is narrow and answerable: run the SAME agent population
through two DIFFERENT masters and see whether the published number moves.

    A  judge on the tier default
    B  judge on --judge-model

Read it against the agent spread. If the two judges land within the agreement
band of each other, the synthesis is reading the room rather than imposing a
view, and the single-model synthesis is a limitation on paper more than in
practice. If they land further apart than the models they are summarising, the
published number is substantially the judge's opinion and the roster work below
it is decorative.

## Why this is not "make the master heterogeneous"

A single synthesis call can only be one model, so there is no ensemble master to
build here. Varying the judge per REGION would confound model with region, the
exact confound `assign_models` exists to avoid, and would make the regional cards
uninterpretable in order to answer a question about the national one. Comparing
two whole runs is the version of the question that has an answer.

## What it cannot settle

The agents are resampled between arms, so the two masters do not see an identical
room: some of any difference is the debate moving, not the judge. Holding the room
fixed would mean replaying stored responses through a fresh master, which the
engine has no path for today. Treat a small difference as evidence and a large one
as a prompt to build that path.

One run per arm. `DEC-035` is explicit that a single-run cross-model verdict is
not publishable, and that rule was written after exactly this mistake, so repeat
before concluding.

    python -m ph_economic_ai.tools.experiment_judge_model --dry-run
    python -m ph_economic_ai.tools.experiment_judge_model --judge-model llama3.2
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from ph_economic_ai.engine import llm, swarm, vintage
from ph_economic_ai.engine.rag import RagEngine

ARTIFACT = (Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts'
            / 'judge_model.json')

SCENARIO = {'oil_pct': 5.0, 'usd_pct': 2.0, 'bsp_rate': 6.5, 'demand_index': 72.0}


#: Beyond this size ratio the two judges are not comparable and the run measures
#: capability rather than family. 3.2B against 7.6B is 2.4x, which is how the
#: first attempt at this experiment ended up answering a question it was not
#: asking.
_SIZE_RATIO_LIMIT = 1.8


def _size_mismatch_warning(a_name: str, b_name: str) -> str:
    """Warn when the two judges differ enough in size to confound the result."""
    try:
        import requests
        tags = requests.get(f'{llm.ollama_host()}/api/tags', timeout=5).json()
        size = {m['name']: m['size'] for m in tags.get('models', [])}
    except Exception:
        return ''

    def _of(name):
        return size.get(name) or size.get(f'{name}:latest')

    a_sz, b_sz = _of(a_name), _of(b_name)
    if not a_sz or not b_sz:
        return ''
    if max(a_sz, b_sz) <= _SIZE_RATIO_LIMIT * min(a_sz, b_sz):
        return ''
    return (f'WARNING: {a_name} is {a_sz / 1e9:.1f} GB and {b_name} is '
            f'{b_sz / 1e9:.1f} GB, a {max(a_sz, b_sz) / min(a_sz, b_sz):.1f}x gap. '
            f'That confounds model FAMILY with model CAPABILITY and the result '
            f'will mostly measure the latter. Prefer judges of comparable size.')


def _measure(verdict, seconds: float) -> dict:
    responses = list(getattr(verdict, 'all_responses', []) or [])
    estimates = [r.price_estimate for r in responses if r.price_estimate is not None]
    across = getattr(verdict, 'agreement_models', {}) or {}
    return {
        'estimate': verdict.final_estimate,
        'estimate_source': verdict.estimate_source,
        'agreement_pct': verdict.confidence_pct,
        'agent_median': round(statistics.median(estimates), 3) if estimates else None,
        'agent_spread': round(max(estimates) - min(estimates), 2) if len(estimates) > 1 else 0.0,
        'median_by_model': across.get('median_by_model'),
        'synthesis_model': across.get('synthesis_model'),
        'regional': [round(v.estimate, 2) for v in verdict.regional_verdicts
                     if v.estimate is not None],
        'provenance': llm.provenance_id(),
        'seconds': round(seconds, 1),
    }


def _run_arm(rag: RagEngine, judge: str | None) -> dict:
    previous = os.environ.get('STRATA_SWARM_JUDGE_MODEL')
    if judge:
        os.environ['STRATA_SWARM_JUDGE_MODEL'] = judge
    else:
        os.environ.pop('STRATA_SWARM_JUDGE_MODEL', None)
    llm.reset_provenance()
    try:
        started = time.monotonic()
        verdict = swarm.SwarmOrchestrator(
            rag=rag, scenario=dict(SCENARIO), parallel_n=2).run()
        return _measure(verdict, time.monotonic() - started)
    finally:
        if previous is None:
            os.environ.pop('STRATA_SWARM_JUDGE_MODEL', None)
        else:
            os.environ['STRATA_SWARM_JUDGE_MODEL'] = previous


def _interpret(a: dict, b: dict) -> list[str]:
    out: list[str] = []
    if a['estimate'] is None or b['estimate'] is None:
        return ['An arm produced no estimate, so the two cannot be compared.']

    # A judge whose estimate was CLAMPED did not publish a judgement. The anchor
    # rejected its raw output as implausible and substituted its own boundary, so
    # comparing the arms would compare one judge against the physics guard.
    #
    # This check exists because the first version of this interpretation missed
    # it. Measured 2026-07-31: llama3.2 as master came back clamped in all three
    # runs at an identical +0.21, which is the CLAMP reproducing rather than a
    # judge reproducing, with regional verdicts identical within each run and
    # negative while the agents' median was +1.2. A judge failing the task was
    # being reported as "PARTIAL: the judges differ".
    bad = [nm for nm, arm in (('A', a), ('B', b))
           if arm.get('estimate_source') != 'agent']
    if bad:
        sources = ', '.join(f"{nm}={arm['estimate_source']}"
                            for nm, arm in (('A', a), ('B', b)))
        return [
            f"NOT A COMPARISON: arm {' and '.join(bad)} did not publish its own "
            f"estimate ({sources}); the anchor overrode it, so comparing the two "
            f"numbers compares a judge against the physics guard.",
            f"A {a['estimate']:+.2f} regional {a['regional']}    "
            f"B {b['estimate']:+.2f} regional {b['regional']}",
            "A judge that cannot produce a plausible synthesis is a CAPABILITY "
            "result, not an opinion result. Check the judges are comparable in "
            "size before reading anything into the gap.",
        ]

    gap = abs(a['estimate'] - b['estimate'])
    band = swarm._AGREEMENT_BAND
    out.append(f"JUDGES: {a['synthesis_model']} published {a['estimate']:+.2f}, "
               f"{b['synthesis_model']} published {b['estimate']:+.2f}, "
               f"{gap:.2f} PHP/L apart against a {band} band.")

    spread = max(a['agent_spread'], b['agent_spread'])
    out.append(f"AGENTS: medians {a['agent_median']} and {b['agent_median']}, "
               f"widest within-arm spread {spread} PHP/L.")

    if gap <= band:
        out.append(
            'THE SYNTHESIS IS READING THE ROOM: the two judges land inside the '
            'band used to call two agents agreeing, so the single-model synthesis '
            'is a limitation on paper more than in practice. Bounded by one run '
            'per arm and by the agents being resampled between them.')
    elif gap > spread:
        out.append(
            f'THE JUDGE IS THE ANSWER: the two masters differ by {gap:.2f} PHP/L, '
            f'more than the {spread} the agents themselves span. The published '
            f'number is substantially the judge\'s opinion, and the roster work '
            f'beneath it is decorative until the synthesis is addressed.')
    else:
        out.append(
            f'PARTIAL: the judges differ by {gap:.2f} PHP/L, wider than the {band} '
            f'band but inside the {spread} the agents span. One run cannot '
            f'separate that from the debate having moved between arms.')

    out.append('DEC-035: one run per arm is not a publishable cross-model verdict. '
               'Repeat before concluding.')
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge-model', default='llama3.2',
                        help='the model arm B synthesises with')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    counts = swarm.expected_call_counts()
    print(f'Judge-model experiment. {counts["total"]} calls per arm, '
          f'{counts["total"] * 2} total.')
    print(f'  A judge {llm.describe_model(llm.DEEP)} (tier default)')
    print(f'  B judge {args.judge_model}')
    print(f'  agents unchanged in both arms: {swarm.roster_models() or "tier default"}')
    print(f'  vintage: {vintage.describe_vintage()}')
    # Family and CAPABILITY are different variables, and the first run of this
    # experiment confounded them: llama3.2 is 3.2B against qwen2.5:7b's 7.6B, so
    # arm B swapped a 7B judge for a 3B one and the result measured size.
    warning = _size_mismatch_warning(llm.describe_model(llm.DEEP), args.judge_model)
    if warning:
        print(f'\n  {warning}')

    if args.dry_run:
        return 0

    have = llm.installed_models()
    if have and args.judge_model not in have:
        print(f'\n{args.judge_model} is not installed. Pull it first.')
        return 1

    rag = RagEngine()
    print('\nFetching RAG sources...')
    rag.fetch_all()

    results: dict[str, dict] = {}
    for name, judge in (('A_default_judge', None), ('B_other_judge', args.judge_model)):
        print(f'\n=== {name} ===')
        try:
            results[name] = _run_arm(rag, judge)
        except Exception as exc:
            print(f'  FAILED — {type(exc).__name__}: {str(exc)[:200]}')
            results[name] = {'failed': str(exc)[:300]}
            continue
        r = results[name]
        print(f"  estimate {r['estimate']:+.2f} ({r['estimate_source']})  "
              f"agents median {r['agent_median']} spread {r['agent_spread']}  "
              f"regional {r['regional']}  ({r['seconds']:.0f}s)")

    print('\n' + '=' * 78)
    ok = {k: v for k, v in results.items() if 'failed' not in v}
    lines = (_interpret(ok['A_default_judge'], ok['B_other_judge'])
             if len(ok) == 2 else ['Both arms are needed to compare.'])
    for line in lines:
        print(f'\n{line}')

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps({
        'question': 'Q-ENG-011: is the published number the room or the judge?',
        'measured_at': vintage.describe_vintage(),
        'scenario': SCENARIO,
        'judge_model_b': args.judge_model,
        'arms': results,
        'interpretation': lines,
    }, indent=2), encoding='utf-8')
    print(f'\nWrote {ARTIFACT}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
