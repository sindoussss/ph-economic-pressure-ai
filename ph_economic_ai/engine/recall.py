"""Reopening a run instead of paying for it again.

The user's case, stated plainly: run Strata on a Monday afternoon, get gas +1,
food -2, electricity +1. Run it again an hour later, with the DOE adjustment
still days away and every source unmoved, and get the same three numbers back.

The app could not do that. It had no notion of two runs being the same run, so
it re-ran the full 39-call swarm and two sector debates every time and returned
whatever it sampled: eight runs on 2026-07-27 gave eight different answers to an
unchanged market. See engine/vintage.py for why the seeds did not save it.

Two things make recall safe rather than merely cheap:

1. **It is labelled.** A recalled answer says so, with the date and run id it
   came from. Silently presenting an hour-old number as a fresh read is the
   defect this project exists to avoid, not a feature.
2. **It is refusable.** `find_recall` is a lookup, not a policy. The caller
   decides, and the user can always force a fresh run.

The snapshot is deliberately a plain dict rather than a pickled `MasterVerdict`.
A pickle of an engine dataclass is a schema that changes whenever the dataclass
does, and unpickling one written by an older version of the app is exactly the
kind of silent breakage that ends with a wrong number on screen.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Optional

#: Bump when the snapshot's shape changes in a way older readers cannot handle.
#: A snapshot from a different version is ignored rather than guessed at, which
#: costs one re-run and cannot put a misread number on screen.
SNAPSHOT_VERSION = 1


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def build_snapshot(
    master_verdict: Any = None,
    food_estimate: Optional[float] = None,
    electricity_estimate: Optional[float] = None,
    scenario: Optional[dict] = None,
    inputs: Optional[dict] = None,
) -> dict:
    """Freeze a completed run into something the report can be rebuilt from.

    `inputs` is the `vintage.input_snapshot` this run was answered on. It is what
    a later run compares against to decide whether the market has moved, so a
    snapshot without it can never be recalled.

    Everything is read defensively through `getattr`. This is called from the UI
    thread at the end of a run, and a snapshot that raises would take a finished
    run down with it after all the model calls have already been paid for.
    """
    snap: dict[str, Any] = {
        'version': SNAPSHOT_VERSION,
        'scenario': dict(scenario or {}),
        'inputs': dict(inputs or {}),
        'food_estimate': _f(food_estimate),
        'electricity_estimate': _f(electricity_estimate),
    }
    if master_verdict is None:
        snap['gas'] = None
        return snap

    verdicts = []
    for v in getattr(master_verdict, 'regional_verdicts', None) or []:
        verdicts.append({
            'judge_id': getattr(v, 'judge_id', None),
            'region_pair': list(getattr(v, 'region_pair', ()) or ()),
            'estimate': _f(getattr(v, 'estimate', None)),
            'confidence': _f(getattr(v, 'confidence', None)),
            'reasoning': getattr(v, 'reasoning', '') or '',
            'survivor_names': list(getattr(v, 'survivor_names', ()) or ()),
            'rejected_estimate': _f(getattr(v, 'rejected_estimate', None)),
        })
    snap['gas'] = {
        'final_estimate': _f(getattr(master_verdict, 'final_estimate', None)),
        'confidence_pct': int(getattr(master_verdict, 'confidence_pct', 0) or 0),
        'agreement_n': int(getattr(master_verdict, 'agreement_n', 0) or 0),
        'agreement_regions': list(
            getattr(master_verdict, 'agreement_regions', (0, 0)) or (0, 0)),
        'dissenting_regions': list(
            getattr(master_verdict, 'dissenting_regions', None) or []),
        'reasoning': getattr(master_verdict, 'reasoning', '') or '',
        'regional_verdicts': verdicts,
        'regional_estimates': getattr(master_verdict, 'regional_estimates', None),
        'physical_anchor': _f(getattr(master_verdict, 'physical_anchor', None)),
        'estimate_source': getattr(master_verdict, 'estimate_source', 'agent'),
    }
    return snap


def restore_master_verdict(snapshot: dict) -> Optional[Any]:
    """Rebuild a MasterVerdict from a snapshot, or None if it cannot be trusted.

    `all_responses` is deliberately left empty. The per-agent transcripts live in
    the `agent_responses` table and are not part of the snapshot, so a recalled
    run shows the verdict and the regional breakdown but does not pretend to
    replay the debate. The report handles an empty response list already.
    """
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get('version') != SNAPSHOT_VERSION:
        logging.info('recall: ignoring snapshot version %r, expected %r',
                     snapshot.get('version'), SNAPSHOT_VERSION)
        return None
    gas = snapshot.get('gas')
    if not isinstance(gas, dict):
        return None

    from ph_economic_ai.engine.swarm import MasterVerdict, RegionalVerdict
    try:
        verdicts = [
            RegionalVerdict(
                judge_id=v.get('judge_id') or 0,
                region_pair=tuple(v.get('region_pair') or ()),
                estimate=v.get('estimate'),
                confidence=v.get('confidence') or 0.0,
                reasoning=v.get('reasoning') or '',
                survivor_names=tuple(v.get('survivor_names') or ()),
                rejected_estimate=v.get('rejected_estimate'),
            )
            for v in gas.get('regional_verdicts') or []
        ]
        regions = gas.get('agreement_regions') or (0, 0)
        return MasterVerdict(
            final_estimate=gas.get('final_estimate'),
            confidence_pct=int(gas.get('confidence_pct') or 0),
            dissenting_regions=list(gas.get('dissenting_regions') or []),
            reasoning=gas.get('reasoning') or '',
            regional_verdicts=verdicts,
            agreement_n=int(gas.get('agreement_n') or 0),
            agreement_regions=(int(regions[0]), int(regions[1])),
            regional_estimates=gas.get('regional_estimates'),
            physical_anchor=gas.get('physical_anchor'),
            estimate_source=gas.get('estimate_source') or 'agent',
        )
    except Exception:
        # A malformed snapshot must cost a re-run, never a wrong number.
        logging.exception('recall: could not restore snapshot, running fresh')
        return None


class RecalledRun:
    """A previous run that answers the question being asked now."""

    def __init__(self, row: dict, snapshot: dict, master_verdict: Any):
        self.row = row
        self.snapshot = snapshot
        self.master_verdict = master_verdict

    @property
    def run_id(self) -> int:
        return int(self.row.get('run_id') or 0)

    @property
    def timestamp(self) -> str:
        return str(self.row.get('timestamp') or '')

    @property
    def food_estimate(self) -> Optional[float]:
        return _f(self.snapshot.get('food_estimate'))

    @property
    def electricity_estimate(self) -> Optional[float]:
        return _f(self.snapshot.get('electricity_estimate'))

    def age_hours(self, now: Optional[dt.datetime] = None) -> Optional[float]:
        try:
            then = dt.datetime.fromisoformat(self.timestamp)
        except (TypeError, ValueError):
            return None
        if then.tzinfo is None:
            then = then.replace(tzinfo=dt.timezone.utc)
        current = now or dt.datetime.now(dt.timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=dt.timezone.utc)
        return (current - then).total_seconds() / 3600.0

    def describe(self, now: Optional[dt.datetime] = None) -> str:
        """The label the report shows. Never silent about being a recall."""
        hours = self.age_hours(now)
        when = self.timestamp[:10] or 'an earlier run'
        if hours is None:
            age = ''
        elif hours < 1:
            age = ', a few minutes ago'
        elif hours < 24:
            age = f', {int(hours)} hour{"s" if int(hours) != 1 else ""} ago'
        else:
            age = f', {int(hours // 24)} day{"s" if int(hours // 24) != 1 else ""} ago'
        return (f'Recalled from run #{self.run_id} on {when}{age}. '
                f'Inputs unchanged since, so the answer is unchanged.')


def find_recall(store: Any, run_key: str, inputs: dict) -> Optional[RecalledRun]:
    """The stored run that already answers this question, or None to run fresh.

    Two gates, and both must pass. `run_key` narrows to today's runs on this
    pricing week with these models; `vintage.inputs_unchanged` then checks that
    the market has not actually moved since one of them. The key alone would
    recall an answer from before a ₱3 oil swing, and the tolerance alone cannot
    be looked up in an index.

    Every failure path returns None. A store that raises, a snapshot from another
    version, inputs that have moved: all of them mean "run it properly", which
    costs time and is never wrong.
    """
    from ph_economic_ai.engine import vintage

    if store is None or not run_key:
        return None
    try:
        rows = store.find_runs_by_key(run_key)
    except Exception:
        logging.exception('recall: lookup failed, running fresh')
        return None

    for row in rows or []:
        try:
            snapshot = store.get_run_verdict(int(row['run_id']))
        except Exception:
            logging.exception('recall: could not read snapshot, running fresh')
            return None
        if not snapshot:
            continue
        if not vintage.inputs_unchanged(snapshot.get('inputs'), inputs):
            # Same day, different market. Newest first, so once one candidate is
            # too old to match there is no point checking older ones.
            logging.info('recall: run #%s is in the bucket but its inputs moved',
                         row.get('run_id'))
            continue
        verdict = restore_master_verdict(snapshot)
        if verdict is None:
            continue
        return RecalledRun(row, snapshot, verdict)
    return None
