"""Running the same question twice should not produce two answers.

The user's case: run Strata on a Monday afternoon and get gas, food and
electricity numbers. Run it again an hour later with the DOE adjustment still
days out and every source unmoved, and expect the same three numbers.

The app returned eight different gas estimates on 2026-07-27 (-0.52, -0.10,
-1.03, -2.94, -0.97, -2.14, -1.12, -0.60) for a market that had not moved. Two
separate defects: nothing recorded that a question had already been answered,
and the seeds that were supposed to make a re-run reproduce were keyed on live
values that never held still.
"""
import datetime as dt

import pytest

from ph_economic_ai.engine import recall, vintage
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict

PHT = dt.timezone(dt.timedelta(hours=8))
TUESDAY_NOON = dt.datetime(2026, 7, 28, 12, 0, tzinfo=PHT)

# Real scenarios from cache/trust.db. Runs 27 and 28 are the pair that matters:
# 0.07 percentage points apart on oil, which is the same market read.
RUN_28 = {'oil_pct': -3.31, 'usd_pct': -0.06, 'bsp_rate': 6.5,
          'demand_index': 74.2, 'current_price': 98.82}
RUN_27 = {'oil_pct': -3.38, 'usd_pct': -0.05, 'bsp_rate': 6.5,
          'demand_index': 74.3, 'current_price': 98.83}
MOVED = {'oil_pct': -1.10, 'usd_pct': -0.05, 'bsp_rate': 6.5,
         'demand_index': 74.2, 'current_price': 98.82}


class _Brief:
    def __init__(self, brent=68.4, wti=64.1, usd_php=58.2):
        self.brent, self.wti, self.usd_php = brent, wti, usd_php
        self.fetched_at = '2026-07-28 04:11 UTC'


def _snap(scenario, brief=None):
    return vintage.input_snapshot(scenario, brief)


@pytest.fixture
def store(tmp_path):
    s = AgentTrustStore(db_path=str(tmp_path / 'trust.db'))
    yield s
    s.close()


def _verdict(estimate=-0.52):
    return MasterVerdict(
        final_estimate=estimate, confidence_pct=61,
        dissenting_regions=['Western Visayas & Davao Region'],
        reasoning='ESTIMATE: -₱0.52/L',
        regional_verdicts=[RegionalVerdict(
            0, ('NCR', 'Central Luzon'), -0.50, 0.74, 'x', ('a', 'b'))],
        agreement_n=12, agreement_regions=(3, 4),
        regional_estimates={'NCR': -0.52}, physical_anchor=-1.02,
        estimate_source='agent',
    )


def _save(store, scenario, brief=None, estimate=-0.52, key=None):
    """Persist a completed run the way main_window does."""
    inputs = _snap(scenario, brief)
    run_key = key or vintage.vintage_key('ollama:a:b', TUESDAY_NOON)
    run_id = store.save_run(scenario=scenario, final_estimate=estimate,
                            confidence_pct=61, run_key=run_key)
    store.attach_verdict(run_id, recall.build_snapshot(
        master_verdict=_verdict(estimate), food_estimate=-2.0,
        electricity_estimate=1.0, scenario=scenario, inputs=inputs))
    return run_id, run_key


# ── The vintage bucket ────────────────────────────────────────────────────────

def test_the_same_day_is_the_same_bucket():
    morning = dt.datetime(2026, 7, 28, 8, 0, tzinfo=PHT)
    evening = dt.datetime(2026, 7, 28, 21, 30, tzinfo=PHT)
    assert vintage.vintage_key('m', morning) == vintage.vintage_key('m', evening)


def test_a_different_day_is_a_different_bucket():
    today = dt.datetime(2026, 7, 28, 12, 0, tzinfo=PHT)
    tomorrow = dt.datetime(2026, 7, 29, 12, 0, tzinfo=PHT)
    assert vintage.vintage_key('m', today) != vintage.vintage_key('m', tomorrow)


def test_a_different_model_is_a_different_bucket():
    """A 3b local answer and a hosted answer are not the same answer, and the
    provider comes from the environment rather than from anything in the app."""
    assert (vintage.vintage_key('ollama:qwen3b', TUESDAY_NOON)
            != vintage.vintage_key('groq:llama70b', TUESDAY_NOON))


def test_the_fuel_cycle_is_the_week_already_begun():
    """Prices step on Tuesday 06:00 PHT, so the cycle a run belongs to is the
    boundary already crossed, not the next one."""
    tuesday_7am = dt.datetime(2026, 7, 28, 7, 0, tzinfo=PHT)
    assert vintage.vintage(tuesday_7am)['fuel_cycle'] == '2026-07-28'
    tuesday_5am = dt.datetime(2026, 7, 28, 5, 0, tzinfo=PHT)
    assert vintage.vintage(tuesday_5am)['fuel_cycle'] == '2026-07-21'


def test_the_prompt_header_names_a_window_not_a_minute():
    """`fetched_at` was minute resolution and printed into every prompt, which
    made a seeded re-run impossible in principle."""
    label = vintage.describe_vintage(TUESDAY_NOON)
    assert '2026-07-28' in label
    assert ':' not in label.split('(')[0]      # no clock time


# ── Tolerance, not a grid ─────────────────────────────────────────────────────

def test_a_market_that_has_not_moved_counts_as_unchanged():
    """Runs 27 and 28: 0.07pp apart on oil. An earlier version quantised onto a
    fixed grid and put these two in different buckets, because they straddle
    -3.375. A tolerance has no boundaries to straddle."""
    assert vintage.inputs_unchanged(_snap(RUN_28), _snap(RUN_27))


def test_a_market_that_has_moved_does_not():
    assert not vintage.inputs_unchanged(_snap(RUN_28), _snap(MOVED))


def test_the_brief_numbers_are_compared_too():
    """`brent` and `usd_php` reach the agents directly through the DATA BRIEF, so
    two runs can share a scenario and still have shown different numbers."""
    assert vintage.inputs_unchanged(
        _snap(RUN_28, _Brief(brent=68.4)), _snap(RUN_28, _Brief(brent=68.7)))
    assert not vintage.inputs_unchanged(
        _snap(RUN_28, _Brief(brent=68.4)), _snap(RUN_28, _Brief(brent=72.0)))


def test_a_briefed_run_never_recalls_a_timed_out_one():
    """The run path has a nine second timeout that proceeds with no brief at all.
    Those two runs saw different worlds."""
    assert not vintage.inputs_unchanged(_snap(RUN_28, _Brief()), _snap(RUN_28))


def test_a_missing_value_counts_as_changed():
    """Absence is not evidence that nothing moved. Being wrong in this direction
    costs one honest re-run."""
    partial = dict(_snap(RUN_28))
    partial['oil_pct'] = None
    assert not vintage.inputs_unchanged(_snap(RUN_28), partial)


def test_garbage_never_matches():
    assert not vintage.inputs_unchanged(None, _snap(RUN_28))
    assert not vintage.inputs_unchanged(_snap(RUN_28), 'not a dict')


# ── Storing and recalling ─────────────────────────────────────────────────────

def test_a_second_run_the_same_day_recalls_the_first(store):
    run_id, key = _save(store, RUN_28)
    hit = recall.find_recall(store, key, _snap(RUN_27))
    assert hit is not None
    assert hit.run_id == run_id
    assert hit.master_verdict.final_estimate == pytest.approx(-0.52)
    assert hit.master_verdict.confidence_pct == 61
    assert hit.food_estimate == pytest.approx(-2.0)
    assert hit.electricity_estimate == pytest.approx(1.0)


def test_the_regional_breakdown_survives_the_round_trip(store):
    _, key = _save(store, RUN_28)
    hit = recall.find_recall(store, key, _snap(RUN_28))
    assert hit.master_verdict.agreement_n == 12
    assert hit.master_verdict.agreement_regions == (3, 4)
    assert hit.master_verdict.physical_anchor == pytest.approx(-1.02)
    assert hit.master_verdict.dissenting_regions == ['Western Visayas & Davao Region']
    rv = hit.master_verdict.regional_verdicts[0]
    assert rv.region_pair == ('NCR', 'Central Luzon')
    assert rv.estimate == pytest.approx(-0.50)


def test_a_moved_market_is_not_recalled(store):
    _, key = _save(store, RUN_28)
    assert recall.find_recall(store, key, _snap(MOVED)) is None


def test_yesterdays_run_is_not_recalled(store):
    _save(store, RUN_28, key=vintage.vintage_key('m', TUESDAY_NOON))
    tomorrow = TUESDAY_NOON + dt.timedelta(days=1)
    assert recall.find_recall(
        store, vintage.vintage_key('m', tomorrow), _snap(RUN_28)) is None


def test_an_unfinished_run_is_never_recalled(store):
    """A run that crashed before a verdict is not an answer. Recalling one would
    turn a transient failure into a permanent one for the rest of the day."""
    key = vintage.vintage_key('m', TUESDAY_NOON)
    store.save_run(scenario=RUN_28, final_estimate=None, confidence_pct=0,
                   run_key=key)
    assert recall.find_recall(store, key, _snap(RUN_28)) is None


def test_a_run_without_a_snapshot_is_not_recalled(store):
    """Rows that predate this feature have the headline numbers but not enough to
    rebuild the report."""
    key = vintage.vintage_key('m', TUESDAY_NOON)
    store.save_run(scenario=RUN_28, final_estimate=-0.52, confidence_pct=61,
                   run_key=key)
    assert recall.find_recall(store, key, _snap(RUN_28)) is None


def test_the_newest_matching_run_wins(store):
    _save(store, RUN_28, estimate=-0.52)
    newer, key = _save(store, RUN_27, estimate=-0.44)
    hit = recall.find_recall(store, key, _snap(RUN_28))
    assert hit.run_id == newer
    assert hit.master_verdict.final_estimate == pytest.approx(-0.44)


def test_a_moved_candidate_does_not_block_a_matching_older_one(store):
    """Newest first, but a candidate whose inputs moved is skipped rather than
    ending the search."""
    older, key = _save(store, RUN_28, estimate=-0.52)
    _save(store, MOVED, estimate=-3.0)
    hit = recall.find_recall(store, key, _snap(RUN_27))
    assert hit is not None and hit.run_id == older


def test_a_snapshot_from_another_version_is_ignored(store):
    key = vintage.vintage_key('m', TUESDAY_NOON)
    run_id = store.save_run(scenario=RUN_28, final_estimate=-0.52,
                            confidence_pct=61, run_key=key)
    snap = recall.build_snapshot(_verdict(), -2.0, 1.0, RUN_28, _snap(RUN_28))
    snap['version'] = recall.SNAPSHOT_VERSION + 99
    store.attach_verdict(run_id, snap)
    assert recall.find_recall(store, key, _snap(RUN_28)) is None


def test_a_store_that_raises_falls_back_to_running_fresh():
    class Broken:
        def find_runs_by_key(self, key, limit=8):
            raise RuntimeError('database is locked')
    assert recall.find_recall(Broken(), 'k', _snap(RUN_28)) is None


def test_no_key_means_no_recall(store):
    assert recall.find_recall(store, '', _snap(RUN_28)) is None


# ── The label ─────────────────────────────────────────────────────────────────

def test_a_recalled_run_says_so(store):
    run_id, key = _save(store, RUN_28)
    hit = recall.find_recall(store, key, _snap(RUN_28))
    note = hit.describe()
    assert 'Recalled' in note
    assert f'#{run_id}' in note


def test_the_label_states_the_age(store):
    _, key = _save(store, RUN_28)
    hit = recall.find_recall(store, key, _snap(RUN_28))
    then = dt.datetime.fromisoformat(hit.timestamp)
    assert 'hours ago' in hit.describe(then + dt.timedelta(hours=3))
    assert 'day' in hit.describe(then + dt.timedelta(days=2))


def test_the_honesty_vocabulary_carries_the_recall_wording():
    from ph_economic_ai.ui import honesty
    assert 'not recomputed' in honesty.recall_note()
    assert 'run #7' in honesty.recall_note('Recalled from run #7.')


# ── Snapshot handling ─────────────────────────────────────────────────────────

def test_a_snapshot_without_a_verdict_is_still_valid_json():
    """The gas run creates the row; the sectors finish later. A snapshot taken
    between those two points must not raise."""
    snap = recall.build_snapshot(None, None, None, RUN_28, _snap(RUN_28))
    assert snap['gas'] is None
    assert recall.restore_master_verdict(snap) is None


def test_restoring_garbage_returns_none_rather_than_raising():
    assert recall.restore_master_verdict({'version': recall.SNAPSHOT_VERSION,
                                          'gas': 'not a dict'}) is None
    assert recall.restore_master_verdict(None) is None
