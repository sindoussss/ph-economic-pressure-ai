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


def cross_model_note(across: dict) -> str:
    """Whether the agreement survived a change of model, in the reader's words.

    This is the only line on the card that can distinguish agreement from one
    model's determinism. A measured run scored 100 percent over ONE distinct
    estimate with agents that were blinded, independent and separately reasoned:
    twenty agents on one model are one model asked twenty times, so on a
    single-model roster the percentage is substantially that model's consistency.

    Silent on a single-model roster. It must not imply a comparison it did not
    make, and an absent line is honest where "1 model" dressed up as a finding
    would not be.
    """
    if not across or not across.get('measurable'):
        return ''
    n = across.get('models', 0)
    between = across.get('between_spread', 0.0)
    within = across.get('within_spread', 0.0)
    lead = f'checked across {n} different models'
    if within <= 0.005:
        return (f'{lead}: each was internally identical, so the agents are not '
                f'sampling opinions, they are reciting one')
    if between <= within:
        return (f'{lead}: they land ₱{between:.2f}/L apart, within the ₱{within:.2f} '
                f'each spans on its own — the agreement is not one model repeating '
                f'itself')
    return (f'{lead}: they land ₱{between:.2f}/L apart while each spans only '
            f'₱{within:.2f} on its own — the models disagree, and this percentage '
            f'averages over that')


#: The four regions a swarm group actually debates. Every other region's figure is
#: scaled from one of these, so the two are not the same kind of number and the
#: card should not present them as one.
DEBATED_REGIONS = ('NCR', 'Central Luzon', 'Western Visayas', 'Davao Region')


def regional_basis() -> str:
    """How the 17 regional figures were produced, under the map.

    They are not seventeen forecasts. Four region groups debate, and the other
    thirteen figures are those four scaled by a fixed freight premium, so a
    reader comparing Zamboanga to NCR is comparing an estimate to arithmetic
    performed on a different region's estimate.

    Worth stating because nothing here has ever been checked. There is no
    regional price series in the project: every stored file is national or
    national CPI, and `engine.ground_truth` has no notion of region, so grading
    has only ever scored the national number. The multipliers were not fitted
    either, having arrived in a rebrand commit as assumed constants.

    Scoped in the vault as the regional multiplier backtest. Until a DOE regional
    series exists, "derived, not forecast, and never validated" is the accurate
    description and this is where it belongs.
    """
    return ('4 region groups debated; the other 13 figures are scaled from them '
            'by a fixed freight premium. Only the national figure is graded '
            'against a real price, so treat the per-region numbers as derived '
            'rather than forecast.')


def regional_tooltip_note(region: str) -> str:
    """One line inside a region's tooltip, naming which kind of number it is."""
    if region in DEBATED_REGIONS:
        return 'debated by this region\'s agents'
    return 'scaled from a debated region, not separately forecast'


RECALL_NOTE = (
    'this run was not recomputed — nothing the run depends on has moved, so the '
    'stored answer is the current answer'
)

#: Used instead of RECALL_NOTE when something DID move but stayed inside its
#: tolerance, which is the ordinary case and used to be described as "the inputs
#: have not moved since".
RECALL_NOTE_WITHIN_TOLERANCE = (
    'this run was not recomputed — the market did move, but too little to be '
    'worth re-answering, so a fresh run would differ slightly'
)


def recall_note(detail: str = '', drift: str = '') -> str:
    """Label for a report rebuilt from a stored run rather than a fresh one.

    A recalled number must say so. Showing an hour-old answer as though it were
    just computed is exactly the kind of quiet claim this project refuses to
    make elsewhere, and it would be the easiest one to miss: the report looks
    identical either way.

    `drift` is the honest half. This note read "the inputs have not moved since",
    and the recall gate it describes is a TOLERANCE, not an equality test, so the
    sentence was false on any run where a field moved inside its band. Brent
    moving 74.20 to 74.69 passes the gate and changes three lines of the data
    brief that prefixes every prompt in the run, and the local model reproduces a
    call exactly given the same seed, so the stored answer is demonstrably not the
    answer a fresh run would give. Reusing it is still right. Saying nothing moved
    was not.
    """
    body = RECALL_NOTE_WITHIN_TOLERANCE if drift else RECALL_NOTE
    parts = [p for p in (detail, body) if p]
    text = ' '.join(parts)
    return f'{text} ({drift})' if drift else text


def interact_caption() -> str:
    """One-line caption for the Ask-an-Agent tab."""
    return 'Answers are exploratory — generated by local agents, not validated.'
