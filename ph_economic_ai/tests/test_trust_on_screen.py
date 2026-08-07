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


# ── a census is not a calibration window ─────────────────────────────────────

def _graded_store(tmp_path, n):
    """`n` graded runs, built directly so the test is about counting, not grading."""
    import sqlite3
    from ph_economic_ai.engine.store import AgentTrustStore
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    con = sqlite3.connect(s._path)
    for i in range(n):
        con.execute(
            "INSERT INTO runs (timestamp, scenario_json, final_estimate, "
            "confidence_pct, actual_price_change, accuracy_error, graded_at, "
            "target_date, horizon_days) VALUES (?, '{\"current_price\": 85.0}', "
            "-0.5, 70, 0.1, 0.4, '2026-08-01T00:00:00+00:00', "
            "'2026-08-01T00:00:00+00:00', 7.0)",
            (f'2026-07-{(i % 28) + 1:02d}T00:00:00+00:00',))
    con.commit()
    con.close()
    return s


def test_the_graded_count_is_not_capped_by_the_calibration_window(tmp_path):
    """`len(get_graded_errors())` was used as the number of graded runs in three
    places and is capped at 200. Past that the screen would report "200 of 5000
    graded" and the backlog arithmetic would silently drop runs.

    Same defect as "34 stored, 32 explained": a total taken from a windowed
    query. That symptom was fixed for the data of the day; this is the cause.
    """
    s = _graded_store(tmp_path, 250)
    assert len(s.get_graded_errors()) == 200, 'the window is deliberate'
    assert s.count_graded_runs() == 250, 'the census is not'
    s.close()


def test_the_backlog_still_accounts_for_every_run_past_the_window(tmp_path):
    """The arithmetic `stored - due - graded` is what a reader checks by
    subtracting, so it must not use the capped figure."""
    s = _graded_store(tmp_path, 250)
    stored, due, graded = s.count_runs(), len(s.get_due_runs()), s.count_graded_runs()
    assert stored == 250 and due == 0
    assert stored - due - graded == 0, 'nothing unexplained'
    # With the capped figure this would have claimed 50 runs were still forecasting.
    assert stored - due - len(s.get_graded_errors()) == 50
    s.close()


def test_the_track_record_line_reports_the_census(tmp_path):
    s = _graded_store(tmp_path, 250)
    line = honesty.track_record_line(s.count_graded_runs(), s.count_runs())
    assert '250 of 250' in line
    s.close()


# ── a sign that lies ─────────────────────────────────────────────────────────

def test_the_master_prompt_signs_a_falling_verdict_correctly():
    """A literal '+' in front of a signed number produced "+₱-1.20/L" in the
    MASTER JUDGE's prompt, which is the model that produces the headline.

    Not cosmetic here: the causal-chain prompt was once handed a raw dataclass
    repr and the model answered +₱13.70/L against a +₱2.54/L consensus. The
    correct idiom sits two lines above, on the scenario shocks."""
    import inspect
    from ph_economic_ai.engine import swarm
    src = inspect.getsource(swarm)
    assert "f'+₱{v.estimate:.2f}/L'" not in src
    assert "{v.estimate:+.2f} ₱/L" in src


def test_the_run_table_signs_a_falling_forecast_correctly(tmp_path):
    """Every falling forecast rendered as "+₱-3.53/L" on the Agent Performance
    table, and the leading plus is the character a reader scans first."""
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    from ph_economic_ai.engine.store import AgentTrustStore
    from ph_economic_ai.ui.agent_performance import AgentPerformancePanel

    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    s.save_run(scenario={'current_price': 85.0}, final_estimate=-3.53,
               confidence_pct=62, horizon_days=7.0)
    panel = AgentPerformancePanel(s)
    panel.refresh()
    cells = [panel._run_table.item(0, c).text()
             for c in range(panel._run_table.columnCount())]
    assert '-3.53 ₱/L' in cells
    assert not any('+₱-' in c for c in cells)
    s.close()


def test_the_run_table_names_the_obstacle_and_does_not_tick(tmp_path):
    """'⏳ Pending DOE' and 'Graded ✓' were replaced on the landing tiles and
    survived here, on the sibling screen nobody swept."""
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)

    from ph_economic_ai.engine.store import AgentTrustStore
    from ph_economic_ai.ui.agent_performance import AgentPerformancePanel

    s = AgentTrustStore(db_path=str(tmp_path / 'trust2.db'))
    s.save_run(scenario={'current_price': 85.0}, final_estimate=-1.0,
               confidence_pct=70, horizon_days=7.0)
    panel = AgentPerformancePanel(s)
    panel.refresh()
    text = ' | '.join(panel._run_table.item(0, c).text()
                      for c in range(panel._run_table.columnCount()))
    assert 'Pending DOE' not in text
    assert '✓' not in text
    s.close()


def test_the_causal_chain_prompt_shows_how_to_write_a_fall():
    """Its worked example is an entirely rising scenario, so a model describing
    an easing week has no template for a negative magnitude."""
    from ph_economic_ai.engine import live_data
    import inspect
    src = inspect.getsource(live_data)
    assert 'the SIGN must match the direction' in src
    assert 'never "+₱-1.42/L"' in src


# ── figures in prose must be read, not typed ─────────────────────────────────

def test_the_regional_note_reads_its_week_count_from_the_artifact():
    """`regional_basis` states the anti-drift principle in its own docstring --
    the multiplier count is read "because a number in prose drifts from the table
    it describes" -- and the same sentence then hardcoded "294 weeks", which is
    the accuracy result a panelist is told, sitting in `regional_accuracy.json`
    two directories away."""
    import json
    from pathlib import Path
    import ph_economic_ai
    art = json.loads((Path(ph_economic_ai.__file__).parent / 'benchmark'
                      / 'artifacts' / 'regional_accuracy.json').read_text(encoding='utf-8'))
    assert f"over {art['n_weeks']} weeks" in honesty.regional_basis()


def test_the_regional_note_reads_its_region_counts_from_the_roster():
    from ph_economic_ai.engine.swarm import ALL_REGIONS, MEASURED_MULTIPLIERS
    line = honesty.regional_basis()
    debated = len(honesty.DEBATED_REGIONS)
    assert f'{debated} region groups debated' in line
    assert f'the other {len(ALL_REGIONS) - debated} figures' in line
    assert f'{len(MEASURED_MULTIPLIERS)} of {len(ALL_REGIONS)} regions' in line


def test_the_unvalidatable_note_reads_its_denominator_from_the_roster():
    """Half the sentence derived its numbers and half typed them, which is the
    arrangement that drifts silently: the derived half moves with the table and
    the typed half does not."""
    from ph_economic_ai.engine.swarm import ALL_REGIONS
    assert f'of the {len(ALL_REGIONS)} rest' in honesty.unvalidatable_note()


def test_the_regional_note_degrades_without_inventing_a_sample_size(monkeypatch):
    """A missing artifact must not produce a confident week count."""
    monkeypatch.setattr(honesty, '_regional_accuracy', dict)
    line = honesty.regional_basis()
    assert 'weeks' not in line
    assert 'in a pre-registered test' in line


# ── the assumption count must not shrink by accident ─────────────────────────

def _resting_ast():
    """The function's parsed body, comments excluded.

    Grepping the source text cannot tell code from prose: the first version of
    these two tests matched the comment that EXPLAINS the tautology and failed
    on a correct function. The AST carries no comments, so it answers the
    question actually being asked.
    """
    import ast
    import inspect
    import textwrap
    src = textwrap.dedent(
        inspect.getsource(honesty.regions_resting_on_an_assumption))
    return ast.parse(src)


def test_the_resting_set_uses_the_debated_constant_not_a_copy():
    """`DEBATED_REGIONS` was typed out again as a literal tuple five lines below
    the module constant holding the same four names. A literal copy does not
    follow the constant."""
    import ast
    tree = _resting_ast()
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (set(honesty.DEBATED_REGIONS) & literals), (
        'a debated region name is written into the code rather than read')
    assert any(isinstance(n, ast.Name) and n.id == 'DEBATED_REGIONS'
               for n in ast.walk(tree))


def test_the_resting_set_has_no_self_comparison():
    """`g['name'] == g.get('name')` is always True. `DEC-060` recorded that exact
    shape once already, from `attempts == _TRENDS_ATTEMPTS`."""
    import ast
    for node in ast.walk(_resting_ast()):
        if isinstance(node, ast.Compare) and len(node.comparators) == 1:
            left = ast.dump(node.left)
            right = ast.dump(node.comparators[0])
            assert left != right, f'comparison of a value with itself: {left}'


def test_an_unresolvable_anchor_counts_as_resting(monkeypatch):
    """`anchors.get(...)` returning None used to drop the region SILENTLY, which
    understates a figure the map prints as "N of the 17 rest on a freight
    premium nobody has measured". Understating is the dangerous direction for
    that sentence, so an unidentifiable anchor now counts as resting."""
    from ph_economic_ai.engine import swarm
    roster = [{'name': 'NCR', 'anchor': 0},
              {'name': 'Orphan', 'anchor': 99}]     # anchor group with no debated region
    monkeypatch.setattr(swarm, 'ALL_REGIONS', roster)
    monkeypatch.setattr(swarm, 'ASSUMED_MULTIPLIERS', frozenset())
    assert 'Orphan' in honesty.regions_resting_on_an_assumption()


def test_the_live_roster_still_reports_eight():
    """The rewrite must not move today's answer; it removes ways it could move
    without anyone noticing."""
    resting = honesty.regions_resting_on_an_assumption()
    assert len(resting) == 8
    assert {'BARMM', 'CALABARZON', 'MIMAROPA', 'Bicol Region'} <= resting


def test_every_declared_unvalidatable_region_has_no_measured_series():
    """The four are a claim about DOE's archive. `regional_accuracy.json` lists
    every region with a measured per-region MAE, so the claim is checkable and
    a data refresh that gave one of them a series would contradict it."""
    import json
    from pathlib import Path
    import ph_economic_ai
    art = json.loads((Path(ph_economic_ai.__file__).parent / 'benchmark'
                      / 'artifacts' / 'regional_accuracy.json').read_text(encoding='utf-8'))
    measured = set(art['per_region_mae'])
    assert not (honesty.UNVALIDATABLE_REGIONS & measured), (
        'a region declared unvalidatable now has a measured series')


def test_a_result_that_predates_the_basis_guard_is_labelled(monkeypatch):
    """Both regional_accuracy.json and regional_level_premiums.json carried this
    caveat until the pre-registered re-runs of 2026-08-07 (`RSK-034`): DOE's
    `common` price merged with the midpoint of its range with no record of
    which, and a computation from before the fix is not the computation the
    code now performs. The real artifacts now carry `basis_guard: true` and no
    longer show it (`test_the_real_regional_artifacts_have_been_rerun` pins
    that), so the labelling behaviour itself is tested against a simulated
    stale artifact rather than against current reality."""
    monkeypatch.setattr(honesty, '_regional_accuracy',
                        lambda: {'n_weeks': 271})
    line = honesty.regional_basis()
    assert 'over 271 weeks' in line
    assert 'measured before the price-basis guard' in line


def test_the_multiplier_result_is_labelled_independently(monkeypatch):
    """`regional_level_premiums.json` (Phase 2b, the nine corrected medians in
    `swarm.MEASURED_MULTIPLIERS`) carries its own caveat, distinct from the
    accuracy result's. Distinct too from `regional_multiplier_backtest.json`,
    the withdrawn and permanently infeasible Phase 2 test, which this caveat
    is not about."""
    monkeypatch.setattr(honesty, '_regional_level_premiums', lambda: {})
    line = honesty.regional_basis()
    assert 'that measurement also predates the price-basis guard' in line


def test_the_label_disappears_when_the_test_is_rerun(monkeypatch):
    """Self-removing, so nobody has to remember to delete it. An artifact from a
    re-run carries `basis_guard` and the clause goes."""
    monkeypatch.setattr(honesty, '_regional_accuracy',
                        lambda: {'n_weeks': 271, 'basis_guard': True})
    monkeypatch.setattr(honesty, '_regional_level_premiums',
                        lambda: {'basis_guard': True})
    line = honesty.regional_basis()
    assert 'over 271 weeks' in line
    assert 'price-basis guard' not in line


def test_the_two_basis_guard_labels_are_independent(monkeypatch):
    """A re-run of one pre-registered test must not silently clear the other's
    caveat. Only the accuracy artifact is re-run here; the multiplier backtest's
    caveat must survive on its own."""
    monkeypatch.setattr(honesty, '_regional_accuracy',
                        lambda: {'n_weeks': 271, 'basis_guard': True})
    monkeypatch.setattr(honesty, '_regional_level_premiums', lambda: {})
    line = honesty.regional_basis()
    assert 'over 271 weeks' in line
    assert 'measured before the price-basis guard' not in line
    assert 'that measurement also predates the price-basis guard' in line


def test_the_real_regional_artifacts_have_been_rerun():
    """`RSK-034` closed 2026-08-07: both pre-registered tests were re-run on the
    basis-guarded panel per the owner's decision, and the committed artifacts
    are from those runs, not the originals. If this ever fails, either a stale
    artifact was committed by mistake or the field name drifted from what
    `honesty.py` reads."""
    assert honesty._regional_accuracy().get('basis_guard') is True
    assert honesty._regional_level_premiums().get('basis_guard') is True
