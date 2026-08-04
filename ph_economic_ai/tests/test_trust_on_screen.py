"""What the screen must say before a reader can trust the number on it.

Four questions a panelist asks, and the line that answers each:

1. "How much do the agents agree?" -- a percentage cannot answer it. 32 agents
   producing TWO distinct estimates scored 100 percent. `spread_line` reports
   the distinct values and their span, which is checkable against the agent
   cards on the same screen.
2. "Where does that range come from?" -- `band_provenance`, on every card,
   whether or not the news is good. It used to appear only when the band was
   uncalibrated, so its absence carried the message.
3. "Has the app ever been right?" -- `track_record_line` and
   `grade_backlog_line`. The honest answer since 2026-08-05 is that nothing has
   been graded, and the reasons are enumerable.
4. "So is there any measured accuracy at all?" -- `anchor_record_line`, which
   must carry its own caveat: the MAE win does not reach significance.
"""
import pytest

from ph_economic_ai.engine import interval as _interval
from ph_economic_ai.ui import honesty


# -- 1. distinct values, not a percentage -------------------------------------

def test_a_collapsed_room_is_not_reported_as_a_range():
    """The measured case: many agents, one number. A 'range' from a value to
    itself would read as a measurement."""
    line = honesty.spread_line([2.5] * 32, '₱/L', 100)
    assert 'one value, not a converging range' in line
    assert '+2.50 to +2.50' not in line
    assert '32 agent estimates' in line


def test_the_distinct_count_leads_and_the_percentage_follows():
    line = honesty.spread_line([1.0, 1.0, 1.5, -0.5], '₱/L', 60)
    assert line.index('distinct values') < line.index('60% agreement')
    assert '3 distinct values' in line
    assert 'span 2.00' in line


def test_two_rooms_a_percentage_cannot_tell_apart_read_differently():
    """The whole reason this line exists. Both score 100 percent."""
    collapsed = honesty.spread_line([2.5] * 8, '₱/L', 100)
    spread = honesty.spread_line([2.3, 2.4, 2.5, 2.6, 2.7, 2.4, 2.5, 2.6],
                                 '₱/L', 100)
    assert collapsed != spread
    assert '5 distinct values' in spread


def test_a_lone_estimate_is_not_agreement():
    assert 'nothing to agree with' in honesty.spread_line([1.0], '₱/L', 100)


def test_no_estimates_says_so_rather_than_showing_zero():
    assert honesty.spread_line([], '₱/L') == 'no agent produced a usable estimate'


# -- 2. provenance is always present ------------------------------------------

def test_an_uncalibrated_band_says_it_is_a_prior():
    line = honesty.band_provenance(_interval.band(-1.28, [], sector='gas'))
    assert 'NOT calibrated' in line
    assert 'stated prior' in line
    assert f'0 of {_interval.MIN_GRADED_FOR_CALIBRATION}' in line


def test_a_calibrated_band_also_gets_a_line():
    """The absence of a warning must never be the message. A reader who sees no
    provenance cannot tell a calibrated band from a screen that forgot."""
    errors = [0.2] * _interval.MIN_GRADED_FOR_CALIBRATION
    line = honesty.band_provenance(_interval.band(-1.28, errors, sector='gas'))
    assert line
    assert 'calibrated' in line and 'NOT calibrated' not in line
    assert str(_interval.MIN_GRADED_FOR_CALIBRATION) in line


# -- 3. the track record, including why it is empty ---------------------------

def test_an_empty_record_does_not_read_as_a_dead_app():
    """A bare zero invites the harsher reading. The stored-run count is the
    other half of the fact."""
    line = honesty.track_record_line(0, 32)
    assert '32 runs are stored' in line
    assert 'none of this app’s own forecasts has been graded yet' in line
    assert 'withdrawn' in line


def test_a_real_record_is_reported_plainly():
    assert '14' in honesty.track_record_line(14, 40)


def test_the_backlog_names_the_obstacle_rather_than_pending():
    """"pending DOE" was the old label on every ungraded run, and it reads as
    "the price has not arrived yet". That is false for four runs in five."""
    line = honesty.grade_backlog_line({
        'no_price_yet': 10, 'target_week_ambiguous': 10,
        'baseline_week_ambiguous': 10, 'no_baseline': 2})
    assert line.startswith('32 ungraded:')
    assert 'awaiting the week’s price' in line
    assert 'never settled' in line
    assert 'pending' not in line


def test_an_empty_backlog_produces_no_line():
    assert honesty.grade_backlog_line({}) == ''


def test_a_failed_lookup_never_labels_a_run_graded():
    """`grade_status_label` is fed an obstacle that may be a fallback sentinel.
    None means gradable, so a fallback of None would print 'graded' on a run
    that is not."""
    assert honesty.grade_status_label(None) == 'graded'
    assert honesty.grade_status_label('unknown') == 'not graded'
    assert honesty.grade_status_label('no_price_yet') != 'graded'


def test_every_grading_obstacle_has_a_label():
    """A new refusal reason must not reach the screen as 'not graded'."""
    from ph_economic_ai.engine.ground_truth import UNGRADED_REASONS
    for reason in UNGRADED_REASONS:
        assert honesty.grade_status_label(reason) != 'not graded', reason


# -- 4. the one measured result, with its limit -------------------------------

def test_the_anchor_result_carries_its_own_caveat():
    """The MAE win is real and does not reach significance. Reporting the first
    without the second is the overstatement this project keeps retracting."""
    line = honesty.anchor_record_line()
    assert '2.21' in line and '2.64' in line
    assert 'Diebold-Mariano' in line
    assert 'not as a proven edge' in line
    assert line.index('2.21') < line.index('Diebold-Mariano')


def test_the_anchor_line_matches_the_artifact_it_describes():
    """A figure typed into prose drifts from the artifact. This one is read."""
    import json
    from pathlib import Path
    import ph_economic_ai
    p = (Path(ph_economic_ai.__file__).parent / 'benchmark' / 'artifacts'
         / 'anchor_validation.json')
    bt = json.loads(p.read_text(encoding='utf-8'))['backtest']
    line = honesty.anchor_record_line()
    assert f'{bt["mae_anchor_php_l"]:.2f}' in line
    assert f'{bt["n_months"]} months' in line


# -- the grader and the screen give one answer --------------------------------

def test_the_screen_asks_the_grader_rather_than_restating_the_rule(due_run):
    """The reason shown on a tile comes from `grade_verdict`, the same function
    that decides whether to grade. "pending DOE" survived three revisions of
    what actually blocks a grade because it was written independently."""
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=84.38, estimate=-1.28)
    verdict = grade_verdict(store, dict(store.get_run(run_id)))
    assert verdict['obstacle'] == 'no_price_yet'
    assert 'no price has been observed' in verdict['reason']
    assert honesty.grade_status_label(verdict['obstacle']) == 'awaiting the week’s price'


def test_an_ambiguous_baseline_week_is_named_as_such(due_run):
    from datetime import timedelta
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=84.38, estimate=-1.28, price=89.51)
    made = dict(store.get_run(run_id))['timestamp']
    own = store.cycle_prices(made)[1]
    store.record_price_observation(84.38, observed_at=own + timedelta(days=1))
    store.record_price_observation(89.51, observed_at=own + timedelta(days=3))

    verdict = grade_verdict(store, dict(store.get_run(run_id)))
    assert verdict['obstacle'] == 'baseline_week_ambiguous'
    assert '84.38' in verdict['reason'] and '89.51' in verdict['reason']


def test_a_gradable_run_reports_no_obstacle(due_run):
    from ph_economic_ai.engine.ground_truth import grade_verdict

    store, run_id = due_run(baseline=85.00, estimate=-0.5, price=84.38)
    verdict = grade_verdict(store, dict(store.get_run(run_id)))
    assert verdict['obstacle'] is None
    assert verdict['actual_change'] == pytest.approx(-0.62)
    assert verdict['reason'].startswith('pricing week ')


def test_the_estimates_reach_the_card_not_just_their_percentage():
    """`spread_line` needs the values. A SectorReading that carries only the
    percentage cannot answer the question the percentage fails at."""
    from ph_economic_ai.engine.pressure_brief import SectorReading
    r = SectorReading(sector='gas', direction='rising', estimate=1.4, unit='₱/L',
                      confidence=100, estimates=[1.4, 1.4, 1.5])
    assert r.to_dict()['estimates'] == [1.4, 1.4, 1.5]
    assert '3 agent estimates' in honesty.spread_line(r.estimates, r.unit, r.confidence)


def test_an_ungradable_sector_is_not_promised_a_threshold():
    """Food and electricity are never graded against an observed price. "0 of
    12 graded runs" implies 12 are reachable by waiting; they are not."""
    food = honesty.band_provenance(_interval.band(-0.3, [], sector='food'))
    assert 'cannot become calibrated by waiting' in food
    assert '12' not in food

    fuel = honesty.band_provenance(_interval.band(-1.28, [], sector='gas'))
    assert f'0 of {_interval.MIN_GRADED_FOR_CALIBRATION}' in fuel
    assert 'cannot become calibrated' not in fuel


def test_the_band_says_which_sectors_can_ever_be_graded():
    assert _interval.band(0.0, [], sector='gas')['gradable'] is True
    assert _interval.band(0.0, [], sector='food')['gradable'] is False


def test_the_estimates_field_cannot_be_hit_positionally():
    """`SectorReading` is built positionally in places, and a field added in the
    middle silently reassigns every argument after it. That is what happened
    when `direction_agreement` was added: three call sites passed `drivers` into
    it, and nothing failed because nothing did arithmetic on the value until
    `estimates` arrived and tried to average a source name."""
    from ph_economic_ai.engine.pressure_brief import SectorReading
    with pytest.raises(TypeError):
        SectorReading('gas', 'rising', 1.0, '₱/L', 100, 0, [], [], [1.0])
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100, 90, ['d'], ['s'])
    assert r.direction_agreement == 90 and r.estimates == []


def test_the_backlog_accounts_for_every_stored_run():
    """The strip said 34 runs stored and then explained 32. A reader who
    subtracts finds two runs the screen declined to account for; those two were
    simply still inside their forecast week."""
    line = honesty.grade_backlog_line({'not_due': 2, 'no_price_yet': 10,
                                       'target_week_ambiguous': 10,
                                       'baseline_week_ambiguous': 10,
                                       'no_baseline': 2})
    assert line.startswith('34 ungraded:')
    assert line.index('2 still forecasting') < line.index('10 awaiting')
