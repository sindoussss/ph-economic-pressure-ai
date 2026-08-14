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


def tile_narrow_marker(n: int, distinct: int) -> str:
    """A three-word flag for a space-constrained tile, not the full caveat.

    `agreement_caveat`'s sentence does not fit a 180px history tile without
    overflowing it, and the tile already wraps a longer obstacle phrase once
    (`grade_status_label`) rather than clip mid-word. This is the same
    signal at the size the tile actually has: a reader who wants the reason
    can still open the run, which shows the fully-guarded report card.

    Empty string when there is nothing to flag, same convention as
    `agreement_caveat`.
    """
    if 0 < distinct < COLLAPSE_DISTINCT:
        return 'narrow room'
    return ''


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


def spread_line(estimates, unit: str, pct: int = 0) -> str:
    """What the agents actually said, in place of a percentage.

    A reader asked "how much do the agents agree" and shown "100%" learns
    nothing checkable. The same run said either two numbers or twenty, and the
    percentage is identical: measured on 2026-07-29, 32 agents produced TWO
    distinct estimates spanning 0.26 PHP/L and scored 100 percent.

    So the count of distinct values and the span between them lead, and the
    percentage follows in brackets as a summary of them rather than as the
    finding. Neither is a probability, and the distinct count is the one a
    panelist can check against the agent cards on the same screen.
    """
    values = sorted({round(float(e), 2) for e in (estimates or []) if e is not None})
    n = len([e for e in (estimates or []) if e is not None])
    if n == 0:
        return 'no agent produced a usable estimate'
    if n == 1:
        return f'1 agent estimate, so there is nothing to agree with'
    if len(values) == 1:
        # The case the percentage flatters most: every agent said one number, and
        # a range from a value to itself would read as a measurement.
        body = (f'{n} agent estimates, all of them {values[0]:+.2f} {unit} — '
                f'one value, not a converging range')
    else:
        body = (f'{n} agent estimates taking {len(values)} distinct values, '
                f'{values[0]:+.2f} to {values[-1]:+.2f} {unit} '
                f'(span {values[-1] - values[0]:.2f})')
    return f'{body} · {pct}% agreement' if pct else body


def band_provenance(band: dict) -> str:
    """Where the displayed range comes from, stated whether or not it is good news.

    This used to appear only when the band was UNCALIBRATED, which is the wrong
    way round: a reader who never sees the line cannot tell a calibrated band
    from a screen that forgot to warn them. The line is always present and only
    its content changes, so its absence is never the message.
    """
    # What one sample IS, per sector. Gas grades a run against a pricing week;
    # food grades a calendar MONTH, and many runs inside a month are one sample.
    # Calling those "runs" would overstate the evidence by exactly the factor the
    # month rule removes, so the noun follows the sector. Gas still reads 'runs'
    # and its wording is unchanged.
    unit = band.get('sample_unit', 'runs')
    if band.get('calibrated'):
        return (f'range calibrated on this app’s own {band.get("n_graded", 0)} '
                f'graded {unit} (split conformal)')
    if not band.get('gradable', True):
        # "0 of 12" would promise a threshold this sector can never reach.
        # Electricity has no outcome series in a comparable unit to calibrate on
        # at all, which is a different sentence.
        return ('range NOT calibrated — it is a stated prior, and this sector has '
                'no graded outcome series at all, so it cannot become calibrated '
                'by waiting')
    n = int(band.get('n_graded', 0) or 0)
    return (f'range NOT calibrated — it is a stated prior, not this app’s measured '
            f'error ({n} of {_MIN_GRADED()} graded {unit} available)')


def band_provenance_tooltip(band: dict) -> str:
    """Plain-language companion to `band_provenance`, for a hover tooltip.

    Same underlying fact, aimed at a reader who does not already know what
    'calibrated' means in a forecasting context -- the precise wording in
    `band_provenance` itself stays exactly as-is."""
    if band.get('calibrated'):
        return (f'This range has been checked against {band.get("n_graded", 0)} '
                f'of this app’s own past forecasts.')
    return ("This isn't a proven range — it's the app's best guess at how "
            "wide the swing could be, not something checked against real "
            "outcomes yet.")


#: Plain-language tooltip for the agent-estimate spread line (`spread_line`).
SPREAD_LINE_TOOLTIP = (
    "This app runs several AI agents that each independently guess the price "
    "change. This shows how many gave different answers and how far apart "
    "they were."
)

#: Plain-language tooltip for the narrowed-room caveat (`agreement_caveat`).
AGREEMENT_CAVEAT_TOOLTIP = (
    "A high agreement number can be misleading if most agents just repeated "
    "the same guess instead of agreeing independently. This flags when "
    "that's happening."
)

#: Plain-language tooltip for the "direction agreement N% (not a probability)" line.
DIRECTION_AGREEMENT_TOOLTIP = (
    "This is NOT the odds of the price actually rising or falling — it's "
    "just how many agents agreed on which direction, which is a different "
    "thing."
)

#: Plain-language tooltip for the anchor-record / Diebold-Mariano line (gas only).
ANCHOR_RECORD_TOOLTIP = (
    "This checks whether the app's method reliably beats just assuming 'no "
    "change.' The test says the improvement isn't big enough to be sure it "
    "isn't luck."
)


def _MIN_GRADED() -> int:
    from ph_economic_ai.engine.interval import MIN_GRADED_FOR_CALIBRATION
    return MIN_GRADED_FOR_CALIBRATION


def chain_integrity_line(track_record) -> str:
    """Whether this app's own prediction log can prove it has not been edited.

    `track_record_line` reports how many forecasts have been GRADED, from
    `trust.db` -- the mutable store, corrected when a grade turns out to
    compare the wrong week (`RSK-023`). This is a different and stronger
    claim: `track_record.jsonl` is append-only and hash-chained, so a reader
    does not have to trust that no row was quietly edited after the fact --
    `verify_chain()` checks it. `False` here would mean the file on disk does
    not match its own hashes, which is worth surfacing loudly rather than
    silently trusting the mutable numbers next to it.

    Silent when nothing has been filed yet, matching `track_record_line`'s own
    "none yet" phrasing rather than asserting integrity over an empty log.
    """
    try:
        rows = track_record.all_rows()
    except Exception:
        return ''
    if not rows:
        return ''
    n_pred = sum(1 for r in rows if r.get('kind') == 'prediction')
    n_withdrawn = sum(1 for r in rows if r.get('kind') == 'withdrawal')
    intact = track_record.verify_chain()
    if not intact:
        return (f'tamper check FAILED on this app’s own {n_pred}-prediction '
                f'hash-chained log — do not trust any number derived from it '
                f'until this is investigated')
    sc = track_record.scorecard()
    withdrawn = f', {n_withdrawn} withdrawn' if n_withdrawn else ''
    if sc['n_matured']:
        return (f'hash-chain verified across {n_pred} locked-in predictions{withdrawn} '
                f'— {sc["n_matured"]} matured at {sc["mae"]:.2f} PHP/L mean '
                f'absolute error, {sc["coverage_90"]:.0%} landed inside their stated band')
    return (f'hash-chain verified across {n_pred} locked-in predictions{withdrawn}, '
           f'none matured yet — each is locked in before its outcome is known, '
           f'so this log cannot be edited with hindsight')


def track_record_line(n_graded: int, n_runs: int = 0) -> str:
    """How many of the app's own forecasts have been checked against a price.

    The honest answer has been zero since 2026-08-05, when every grade on record
    turned out to compare a forecast for one week against a different week's
    price and all three were withdrawn (`RSK-023`). A screen that shows a
    forecast without showing this invites a reader to assume the number has a
    track record behind it.
    """
    if n_graded > 0:
        stored = f' of {n_runs}' if n_runs else ''
        return (f'{n_graded}{stored} of this app’s own forecasts have been graded '
                f'against the price of the week they forecast')
    stored = f'{n_runs} runs are stored and ' if n_runs else ''
    return (f'{stored}none of this app’s own forecasts has been graded yet — a '
            f'grade needs the forecast week’s price to settle, and the grades that '
            f'compared the wrong week were withdrawn')


#: The physical pass-through anchor's backtest, from
#: `benchmark/artifacts/anchor_validation.json`. Read at call time rather than
#: written into prose, because a figure typed into a sentence drifts from the
#: artifact it describes.
def anchor_record_line() -> str:
    """The one measured accuracy result the app can point at, with its limit.

    The physical anchor beats "assume no change" on mean absolute error over 78
    months, 2.21 against 2.643 PHP/L. That is the strongest honest sentence on
    this screen, and it comes with a caveat that must travel with it: the
    Diebold-Mariano test on the same comparison returns p 0.065, so the margin
    does not reach the conventional threshold. Reporting the MAE win alone would
    be exactly the overstatement this project keeps retracting.

    Returns '' when the artifact is missing rather than inventing a fallback.
    """
    try:
        import json
        from pathlib import Path
        p = (Path(__file__).resolve().parent.parent / 'benchmark' / 'artifacts'
             / 'anchor_validation.json')
        bt = json.loads(p.read_text(encoding='utf-8'))['backtest']
        mae, naive = float(bt['mae_anchor_php_l']), float(bt['mae_naive_php_l'])
        n, dm = int(bt['n_months']), bt.get('dm_vs_naive') or {}
        direction = float(bt.get('directional_accuracy', 0))
    except Exception:
        return ''
    line = (f'the physical pass-through this forecast is anchored to is off by '
            f'₱{mae:.2f}/L on average against ₱{naive:.2f} for assuming no change, '
            f'over {n} months, and calls the direction right {direction:.0%} of the time')
    p_value = dm.get('p_value')
    if p_value is not None and not dm.get('significant'):
        line += (f' — but the difference does not reach significance '
                 f'(Diebold-Mariano p {p_value:.3f}), so read it as the better bet '
                 f'and not as a proven edge')
    return line


#: Short label per grading obstacle, for a tile that has room for three words.
#: Keyed on `ground_truth.UNGRADED_REASONS`. The old label was "pending DOE" for
#: every ungraded run, which reads as "the price has not arrived yet" and is
#: false for four runs in five: most are blocked on a week whose observations
#: disagree, and two can never be graded because they stored no baseline.
_GRADE_LABELS = {
    None: 'graded',
    'not_due': 'still forecasting',
    'no_price_yet': 'awaiting the week’s price',
    'target_week_ambiguous': 'that week never settled',
    'baseline_week_ambiguous': 'its own week never settled',
    'baseline_contradicted': 'read a different price series',
    'implausible_change': 'baseline was stale — not gradable',
    'no_baseline': 'no baseline stored — not gradable',
}


def grade_status_label(obstacle) -> str:
    """Three words for why a run carries no grade. Never guesses."""
    return _GRADE_LABELS.get(obstacle, 'not graded')


def grade_backlog_line(counts: dict) -> str:
    """One sentence accounting for every ungraded run, by reason.

    Written for the question a panelist asks after seeing a zero: "so is it
    broken?". The answer is that the outcomes have not settled, and the shape of
    the backlog says which kind of not-settled, which is checkable and is not
    the same as a system that never ran.
    """
    total = sum(int(v) for v in (counts or {}).values())
    if not total:
        return ''
    # `not_due` leads because it is the only benign one, and because leaving it
    # out made the line fail its own arithmetic: the strip said 34 runs stored
    # and then accounted for 32, and a reader who subtracts finds two runs the
    # screen declined to explain.
    order = ('not_due', 'no_price_yet', 'target_week_ambiguous',
             'baseline_week_ambiguous', 'baseline_contradicted',
             'implausible_change', 'no_baseline')
    parts = [f'{counts[k]} {_GRADE_LABELS[k]}' for k in order if counts.get(k)]
    extra = [f'{v} {k}' for k, v in sorted((counts or {}).items())
             if k not in order and k is not None and v]
    return f'{total} ungraded: ' + ', '.join(parts + extra)


def trust_basis(provenance: dict) -> str:
    """What the trust leaderboard's numbers rest on.

    The leaderboard showed a score per agent and nothing about where it came
    from, the same gap the forecast card had. Since 2026-08-05 every score is a
    replay of recorded events, so the honest line is a count of them.

    An empty log is the current state and reads as such: the scores were reset
    to the neutral prior because the movements behind them could not be
    reproduced from any surviving evidence, and nothing has moved them since.
    That is a stronger statement than a leaderboard of identical numbers with no
    explanation, which reads as a bug.
    """
    graded = int((provenance or {}).get('grade', 0) or 0)
    responses = int((provenance or {}).get('response', 0) or 0)
    if not (graded or responses):
        return ('every score is at the neutral prior — the movements behind the '
                'old scores could not be reproduced from surviving evidence, so '
                'they were cleared rather than shown as measurements')
    # Only the non-zero halves. "0 from response quality" reads as a bug rather
    # than as the state after a reset, which is when it happens.
    parts = []
    if responses:
        parts.append(f'{responses} from response quality')
    if graded:
        parts.append(f'{graded} from graded outcomes')
    else:
        parts.append('none from a graded outcome yet')
    since = (provenance or {}).get('since')
    when = f' since {str(since)[:10]}' if since else ''
    return (f'every score replays from {responses + graded} recorded events'
            f'{when}: ' + ', '.join(parts))


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

    **Extended 2026-08-01 by the pre-registered accuracy result.** Graded on
    DOE's own regional changes over 294 paired weeks, the derivation's MAE is
    1.311 against 1.217 for assuming no regional change: the corrected interval
    on that difference is [-0.110, +0.309] and contains zero, so the declared
    reading is NO MEASURABLE DIFFERENCE rather than a win for either.

    Against DELTA-EQUAL, which removes the premium and propagates the national
    change unscaled, the interval is [+0.014, +0.024] and excludes zero:
    **applying the premium is measurably worse than not applying it.** Small in
    magnitude, clean in sign, and the one result that survived a rebuild of the
    inputs unchanged.

    The first run of this test used 217 weeks and put the gap against
    zero-change at +0.248. It was computed on a level series that neither
    deduplicated city-weeks nor recovered the South Luzon regions from their
    filenames, so CALABARZON, MIMAROPA and Bicol were each missing about 183
    weeks. Correcting that added 77 paired weeks and shrank the gap to +0.095.
    The verdict did not move; the margin did.

    The decision rule fixed before the run said labelling, not deletion, and that
    is what this is. `DEC-021`'s spirit: the number stays, the claim around it
    stops overreaching.
    """
    from ph_economic_ai.engine.swarm import ALL_REGIONS, MEASURED_MULTIPLIERS
    debated = len(DEBATED_REGIONS)
    weeks = _regional_accuracy().get('n_weeks')
    # Every figure here is read, none typed. The docstring above states that
    # principle for the multiplier count and the same sentence then hardcoded
    # "4", "13" and "294 weeks" -- the last of which is the accuracy result a
    # panelist is told, and it sits in `regional_accuracy.json` two directories
    # away. A note written to stop drift is the worst place to leave a figure
    # that can drift.
    over = f'over {weeks} weeks' if weeks else 'in a pre-registered test'
    # Whether that result predates the price-basis guard. The regional series
    # merged DOE's `common` price with the midpoint of its published range, two
    # quantities differing by a mean 0.758 PHP/L, and a change measured across a
    # switch between them is now refused -- which removes 9.2 percent of the
    # region-week pairs the published run used.
    #
    # Derived, not typed: an artifact from a re-run carries `basis_guard` and
    # this clause disappears on its own. Re-running is a research action under
    # `DEC-010`, so it is not done here and the result is labelled instead.
    if weeks and not _regional_accuracy().get('basis_guard'):
        over += ' (measured before the price-basis guard, on a pool 9% larger)'
    # The multiplier VALUES have the same defect as the accuracy result they sit
    # beside, found the same week (`RSK-034` follow-up): `regional_level_premiums.json`
    # (Phase 2b, `tools/regional_level_premiums.py`, the nine corrected medians) also
    # predates the basis guard and was never re-derived on the corrected panel, for the
    # same DEC-010 reason. Distinct from `regional_multiplier_backtest.json`, which is
    # the withdrawn Phase 2 CHANGE-vs-LEVEL regression and permanently infeasible for an
    # unrelated reason -- confusing the two here would point this caveat at the wrong
    # artifact. This one carries no `basis_guard` field either, so the caveat is derived
    # the same way and disappears the same way once a re-run adds one.
    measured_caveat = ('; that measurement also predates the price-basis guard'
                        if not _regional_level_premiums().get('basis_guard') else '')
    return (f'{debated} region groups debated; the other '
            f'{len(ALL_REGIONS) - debated} figures are scaled from them '
            f'by a regional price premium measured against DOE prices for '
            f'{len(MEASURED_MULTIPLIERS)} of {len(ALL_REGIONS)} regions'
            f'{measured_caveat}. '
            f'**Graded against DOE’s own regional prices {over}, '
            f'this derivation is no more accurate than assuming no regional '
            f'change at all**, and applying the premium is measurably worse '
            f'than not applying it. Only the national figure is graded against a '
            f'real price, so treat the per-region numbers as derived rather than '
            f'forecast.')


def _regional_accuracy() -> dict:
    """The pre-registered regional accuracy result, read rather than restated.

    Returns `{}` when the artifact is missing, so the caller degrades to naming
    the test instead of inventing a sample size.
    """
    try:
        import json
        from pathlib import Path
        import ph_economic_ai
        return json.loads(
            (Path(ph_economic_ai.__file__).parent / 'benchmark' / 'artifacts'
             / 'regional_accuracy.json').read_text(encoding='utf-8'))
    except Exception:
        return {}


def _regional_level_premiums() -> dict:
    """The pre-registered freight-multiplier correction, read rather than restated.

    This is `regional_level_premiums.json` (Phase 2b), which backs the nine
    corrected medians in `swarm.MEASURED_MULTIPLIERS`. Not to be confused with
    `regional_multiplier_backtest.json` (Phase 2), the withdrawn CHANGE-vs-LEVEL
    regression that closed as permanently infeasible for an unrelated reason.

    Returns `{}` when the artifact is missing, so the caller degrades to no
    caveat rather than inventing a claim about it.
    """
    try:
        import json
        from pathlib import Path
        import ph_economic_ai
        return json.loads(
            (Path(ph_economic_ai.__file__).parent / 'benchmark' / 'artifacts'
             / 'regional_level_premiums.json').read_text(encoding='utf-8'))
    except Exception:
        return {}


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

    # Which region anchors each group. Three things were wrong with how this was
    # built, none of which changed today's answer:
    #
    # * `g['name'] == g.get('name')` is a tautology, the `n == n` shape `DEC-060`
    #   already recorded once as a defect class.
    # * `DEBATED_REGIONS` was typed out again as a literal tuple, five lines
    #   below the module constant holding the same four names. A literal copy
    #   does not follow the constant.
    # * `anchors.get(...)` returning None dropped a region from the result
    #   SILENTLY. That understates a figure the map prints as "N of the 17 rest
    #   on a freight premium nobody has measured", and understating is the
    #   dangerous direction for that sentence.
    anchors = {g['anchor']: g['name'] for g in ALL_REGIONS
               if g['name'] in DEBATED_REGIONS and g['anchor'] is not None}

    def rests(group: dict) -> bool:
        if group['name'] in ASSUMED_MULTIPLIERS:
            return True
        anchoring = anchors.get(group['anchor'])
        if anchoring is None:
            # No identifiable anchor means its provenance cannot be established,
            # which is exactly what this set is counting. Fail toward the larger
            # count rather than out of it.
            return True
        return anchoring in ASSUMED_MULTIPLIERS

    return frozenset(g['name'] for g in ALL_REGIONS if rests(g))


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
    from ph_economic_ai.engine.swarm import ALL_REGIONS

    resting = regions_resting_on_an_assumption()
    debated = sorted(UNVALIDATABLE_REGIONS & set(DEBATED_REGIONS))
    n = len(UNVALIDATABLE_REGIONS)
    # `len(ALL_REGIONS)`, not a typed 17. Half this sentence derived its numbers
    # and half typed them, which is the arrangement that drifts silently: the
    # derived half moves with the table and the typed half does not.
    base = (f'{len(resting)} of the {len(ALL_REGIONS)} rest on a freight '
            f'premium nobody has '
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
