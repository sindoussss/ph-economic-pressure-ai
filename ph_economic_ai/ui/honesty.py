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
    band = across.get('band', 0.50)
    lead = f'checked across {n} different models'
    # The band test, not a ratio against within-model spread. The first version
    # compared `between` to the mean RANGE and would have told a reader "the
    # agreement is not one model repeating itself" on a live run where the two
    # models' medians were +2.50 and +1.20, a factor of 2.08 apart, because one
    # outlier had inflated the range to 3.25. A false reassurance in the honesty
    # layer is the worst place for one.
    if across.get('models_agree'):
        return (f'{lead}: their medians land ₱{between:.2f}/L apart, inside the '
                f'₱{band:.2f} band used to call two agents agreeing, so this is '
                f'not one model repeating itself')
    # Naming the winner. A reader told only that the models disagree learns the
    # disagreement and none of its consequence: on the 2026-07-30 run the two
    # medians were +2.50 and +1.20 and the published number was +2.39, so one
    # model's answer was published and the other's was not.
    won = across.get('nearest_model')
    whose = f', and the published number is {won}’s' if won else ''
    return (f'{lead}: their medians are ₱{between:.2f}/L apart, wider than the '
            f'₱{band:.2f} band used to call two agents agreeing — the models '
            f'disagree{whose}, and this percentage averages over that')


def unscored_survivor_note(regions: int = 0) -> str:
    """That a regional card rests on a tie-break rather than on a tournament.

    The elimination bracket is supposed to select the group's best agent, and the
    survivor is the only thing the regional judge reads. When the Critic and the
    ConfidenceScorer produce no parseable scores, every combined score comes out
    identical and the round removes agents without measuring anything, so the
    survivor is whoever the reproducible tie-break happened to reach.

    `scores_are_degenerate` has detected this since the bracket was written, and
    it logged a warning and emitted an event and stopped there. The card built
    from that survivor looked exactly like one chosen on merit. Seen live on
    2026-07-31 in one group of one run.

    Silent when every group scored, which is the normal case.
    """
    if regions < 1:
        return ''
    which = 'a region' if regions == 1 else f'{regions} regions'
    return (f'{which} had no usable agent scores, so the agent representing it was '
            f'chosen by tie-break rather than by the tournament — that region\'s '
            f'figure rests on an arbitrary pick, not a ranked one')


def bracket_note(across: dict) -> str:
    """Whether the elimination bracket let every model reach the synthesis.

    The survivors are the only agents the regional judges read, so a model with no
    survivors contributed nothing to the published number however many agents it
    fielded. That is checkable and worth saying. It is also a live possibility
    rather than a hypothetical: the Critic and the ConfidenceScorer score peers by
    name and in prose, so a model whose writing they prefer wins the tournament
    regardless of its numbers. Five paired runs found no such bias: survivors were
    2 and 2 every time. The 19-to-13 split that prompted this appeared once and did
    not reproduce.

    Only the shut-out case gets a verdict. With four survivors and two models a
    3-to-1 split is well within chance, and calling that bias would be the kind of
    unearned conclusion this project keeps having to retract.
    """
    counts = (across or {}).get('survivors_by_model')
    if not counts:
        return ''
    listed = ', '.join(f'{m} {n}' for m, n in sorted(counts.items()))
    on_roster = set((across.get('n_by_model') or {}).keys())
    shut_out = sorted(on_roster - set(counts))
    if shut_out:
        return (f'the elimination bracket sent {listed} through to the judges, and '
                f'none from {", ".join(shut_out)} — that model debated but reached '
                f'the final number through no one')
    return (f'survivors reaching the judges: {listed} (too few to read as bias '
            f'either way)')


def synthesis_note(across: dict) -> str:
    """That the headline comes from ONE model however varied the roster is.

    A heterogeneous roster diversifies the debate and not the synthesis. The
    survivors feed the regional judges and the master judge, all on the deep tier,
    which the agent roster does not touch, so the published number is one model's
    reading of the room. That holds regardless of whether the models agree: over
    five paired runs their medians sat a median 0.250 PHP/L apart, and the
    published figure still came from the deep tier rather than from any tally of
    the agents.

    Silent on a single-model roster, where there is no contrast to draw.
    """
    if not across or not across.get('measurable'):
        return ''
    model = across.get('synthesis_model')
    if not model:
        return ''
    return (f'the agents ran on several models but the final number is one '
            f'model’s synthesis ({model}), so it is a judgement about the debate '
            f'rather than a vote across it')


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

    **Updated 2026-08-01, and the update is narrower than it looks.** The premiums
    were assumed constants that arrived in a rebrand commit; eleven of them have
    now been measured against DOE's published regional prices, 2023 to 2026, and
    nine were wrong and were corrected. So "unfitted" is no longer true of the
    premium.

    What has NOT changed is the thing this note exists for. Only the national
    figure is graded against a real price, `engine.ground_truth` still has no
    notion of region, and whether a level premium should scale a price CHANGE at
    all remains unsettled: `Q-ENG-009` closed as infeasible, needing centuries of
    weekly data to resolve. A measured premium applied to a change is still
    arithmetic performed on another region's estimate.

    So the wording keeps "derived rather than forecast" and stops calling the
    premium unfitted, because overstating the correction would be the same
    failure in the opposite direction.

    The count is read from `swarm.MEASURED_MULTIPLIERS` rather than written in,
    because a number in prose drifts from the table it describes and this note
    exists to stop exactly that kind of drift.
    """
    from ph_economic_ai.engine.swarm import ALL_REGIONS, MEASURED_MULTIPLIERS
    return (f'4 region groups debated; the other 13 figures are scaled from them '
            f'by a regional price premium measured against DOE prices for '
            f'{len(MEASURED_MULTIPLIERS)} of {len(ALL_REGIONS)} regions. Only the '
            f'national figure is graded against a real price, so treat the '
            f'per-region numbers as derived rather than forecast.')


#: Regions for which DOE publishes no retail price series, so their figures cannot
#: be checked against an observed price at all.
#:
#: Established 2026-07-31 by enumerating DOE's price-monitoring archive, 2713
#: documents spanning 2017 to 2026. DOE publishes nothing north of NCR: zero files
#: for Region III, Region I, Region II or CAR, against controls of 1243 South
#: Luzon, 470 NCR, 348 Visayas and 333 Mindanao.
#:
#: **Central Luzon is the consequential one.** It is one of the four groups the
#: swarm actually DEBATES, not one of the thirteen scaled from them, so a quarter
#: of the debating capacity produces a number no source can contradict. Phase 0
#: labelled the derived figures and left this case unlabelled, because at the time
#: nobody knew it existed.
UNVALIDATABLE_REGIONS = frozenset({
    'Central Luzon', 'Ilocos Region', 'Cagayan Valley', 'CAR',
})


def regions_resting_on_an_assumption() -> frozenset:
    """Regions whose displayed figure depends on a multiplier nobody measured.

    Wider than `UNVALIDATABLE_REGIONS`, and the difference is the point.

    `derive_regional_estimates` scales a region by its premium **relative to its
    anchor**, which is correct (`DEC-030`): the anchor's survivor estimate already
    carries the anchor's own freight, so only the increment is outstanding. But it
    means a figure inherits its anchor's provenance as well as its own. A measured
    region divided by an assumed anchor is not a measured figure.

    Two things made this visible only after nine multipliers were corrected:

    * **CALABARZON, MIMAROPA and Bicol are measured and anchored to Central
      Luzon**, whose 1.02 is a guess DOE can never check. CALABARZON measures
      0.979 against NCR and displays 0.98/1.02 = 0.961.
    * **BARMM is assumed at 1.10 and anchored to Davao**, which was corrected
      downward to 0.96. Lowering an anchor RAISES everything scaled off it, so
      BARMM rose 0.24 to become the largest figure on the map without anything
      about BARMM being measured.

    So the count here is 8 of 17, not 4. Saying 4 understates it.
    """
    from ph_economic_ai.engine.swarm import ALL_REGIONS, ASSUMED_MULTIPLIERS

    anchors = {g['anchor']: g['name'] for g in ALL_REGIONS
               if g['name'] == g.get('name') and g['anchor'] is not None
               and g['name'] in ('NCR', 'Central Luzon', 'Western Visayas',
                                 'Davao Region')}
    return frozenset(
        g['name'] for g in ALL_REGIONS
        if g['name'] in ASSUMED_MULTIPLIERS
        or anchors.get(g['anchor']) in ASSUMED_MULTIPLIERS)


def regional_tooltip_note(region: str) -> str:
    """One line inside a region's tooltip, naming which kind of number it is."""
    debated = region in DEBATED_REGIONS
    if region in UNVALIDATABLE_REGIONS:
        # Worth saying even for a debated region, and especially for one: being
        # argued over by agents is not the same as being checkable.
        how = ('debated by this region\'s agents' if debated
               else 'scaled from a debated region')
        return f'{how}, and DOE publishes no price series here, so it cannot be checked'
    if debated:
        return 'debated by this region\'s agents'
    return 'scaled from a debated region, not separately forecast'


def unvalidatable_note() -> str:
    """The line under the map naming what no source can check.

    Separate from `regional_basis`, which explains how the figures were PRODUCED.
    This is about whether they can ever be graded, which is a different claim and
    a stronger one: derived is a statement about method, unfalsifiable is a
    statement about evidence.

    **Two counts, because they are two facts.** Four regions have no DOE series
    at all. EIGHT displayed figures rest on a multiplier nobody measured, because
    a figure inherits its anchor's provenance: CALABARZON, MIMAROPA and Bicol are
    measured but divide by Central Luzon's unmeasurable 1.02, and BARMM is itself
    assumed. Reporting only the four understates it, which is why the wider count
    is stated first and the narrower one qualifies it.
    """
    resting = regions_resting_on_an_assumption()
    debated = sorted(UNVALIDATABLE_REGIONS & set(DEBATED_REGIONS))
    n = len(UNVALIDATABLE_REGIONS)
    base = (f'{len(resting)} of the 17 rest on a freight premium nobody has '
            f'measured, {n} of them because DOE publishes no price series for '
            f'the region at all and the rest because they are scaled from one '
            f'that has none, so nothing published can confirm or contradict them')
    if not debated:
        return base
    return (f'{base} — including {", ".join(debated)}, which the agents actually '
            f'debate rather than derive, and which anchors the figures scaled '
            f'from it')


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
