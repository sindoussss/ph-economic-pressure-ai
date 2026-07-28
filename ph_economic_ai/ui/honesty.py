"""Shared honesty vocabulary so 'exploratory' vs 'validated' reads consistently
across the app. Strings only — the tested seam; panels build the QLabels."""

EXPLORATORY = 'exploratory'
VALIDATED = 'validated'
AGREEMENT_NOTE = 'agent agreement varies per run, not a calibrated probability'


def consensus_note() -> str:
    """Italic note under the swarm consensus number."""
    return f'{EXPLORATORY} · {AGREEMENT_NOTE}'


#: Below this many distinct values, a high percentage is describing a collapsed
#: room rather than a converged one. Measured: 32 agents producing 2 distinct
#: estimates scored 100 percent, and blinding them tripled the distinct values
#: while moving the percentage 8 points.
COLLAPSE_DISTINCT = 3
#: Below this share of agents opening differently, the room is writing alike.
#: A genuine halfway point: `opening_diversity` counts one statement per agent,
#: so 1.0 is attainable. It did not used to be — the metric divided by RESPONSES
#: while agents spoke twice, capping a 20-agent run at 0.625 and putting this
#: threshold at four fifths of an unreachable maximum. The earlier citation here
#: (0.25, "8 distinct openings from 32 statements") was on that broken scale and
#: is not comparable to anything measured after 2026-07-29.
COLLAPSE_DIVERSITY = 0.5


def agreement_caveat(n: int, distinct: int = 0, diversity: float = 0.0) -> str:
    """The sentence that stops a high percentage being read as strong consensus.

    A percentage cannot distinguish agents who independently reached the same
    number from agents who copied it: the estimate vector is identical either
    way. These two counts are the signal that can, so when they say the room has
    collapsed, the card says so next to the number rather than leaving a reader
    to infer it from a figure that looks reassuring.

    It reports the narrowing and stops there. It used to end "one view held
    widely, not many views converging", which names a CAUSE the number cannot
    support: a live run traced eleven identical openings not to agents copying
    each other but to a small model returning the prompt's causal-chain template
    verbatim. Same measurement, opposite diagnosis. The retry in
    `swarm.unfilled_scaffold` handles that case now, and the wording no longer
    asserts which of the two a reader is looking at.
    """
    if n < 2:
        return ''
    problems = []
    if 0 < distinct < COLLAPSE_DISTINCT:
        problems.append(f'only {distinct} distinct value'
                        f'{"s" if distinct != 1 else ""} among {n} estimates')
    if 0 < diversity < COLLAPSE_DIVERSITY:
        problems.append(f'{diversity:.0%} of the statements open differently')
    if not problems:
        return ''
    return ('this reads as consensus but the room has narrowed: '
            + ' and '.join(problems)
            + ' — read the number as weaker than it looks, and check the '
              'agents’ own words before relying on it')


def agreement_basis(n: int, regions: tuple[int, int] = (0, 0),
                    echo_n: int = 0, distinct: int = 0,
                    diversity: float = 0.0) -> str:
    """How the agreement percentage was measured, in the reader's words.

    The percentage alone cannot distinguish a room of twenty from a pair of
    survivors, and the app shipped a headline measured over two agents directly
    above regional cards measured over nine. Whatever the number is, it now
    arrives with the population it came from.

    `echo_n` is the part of that population which restated the physical anchor
    verbatim. Every agent is handed the mechanical pass-through, and an agent
    repeating it is scored identically to one independently arriving at it, so
    without this the percentage silently overstates corroboration. Measured at
    25.0 percent of estimates locally and 19.4 percent hosted, which is about a
    quarter of the figure the card reports.
    """
    if n < 2:
        return 'not measurable — too few agents produced a usable estimate'
    counted, total = regions
    where = f' across {counted} of {total} regions' if total else ''
    basis = f'measured over {n} agent estimates{where}'
    if distinct:
        basis += f', taking {distinct} distinct value{"s" if distinct != 1 else ""}'
    if echo_n > 0:
        share = round(100 * echo_n / n)
        basis += (f'; {echo_n} of them ({share}%) restate the physical anchor '
                  f'rather than reaching it independently')
    return basis


RECALL_NOTE = (
    'this run was not recomputed — the inputs have not moved since, so the '
    'stored answer is the current answer'
)


def recall_note(detail: str = '') -> str:
    """Label for a report rebuilt from a stored run rather than a fresh one.

    A recalled number must say so. Showing an hour-old answer as though it were
    just computed is exactly the kind of quiet claim this project refuses to
    make elsewhere, and it would be the easiest one to miss: the report looks
    identical either way.
    """
    return f'{detail} {RECALL_NOTE}'.strip() if detail else RECALL_NOTE


def interact_caption() -> str:
    """One-line caption for the Ask-an-Agent tab."""
    return 'Answers are exploratory — generated by local agents, not validated.'
