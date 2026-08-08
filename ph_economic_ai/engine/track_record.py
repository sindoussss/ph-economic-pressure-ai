"""Append-only, hash-chained, two-phase prediction log.

A prediction is locked when made (phase A). Its outcome is written as a separate
row once the real price is known (phase B) -> no hindsight. Each row hashes the
previous row's hash, so editing any past row breaks chain verification.
"""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

GENESIS = '0' * 64


def _hash_row(payload: dict, prev_hash: str) -> str:
    blob = json.dumps(payload, sort_keys=True) + prev_hash
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


class TrackRecord:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        rows = self.all_rows()
        return rows[-1]['row_hash'] if rows else GENESIS

    def _append(self, payload: dict) -> dict:
        prev = self._last_hash()
        payload = dict(payload)
        payload['prev_hash'] = prev
        payload['row_hash'] = _hash_row(payload, prev)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')
        return payload

    def record_prediction(self, target_month, predicted, low, high, model_version,
                          run_id=None) -> str:
        """Lock in a prediction (phase A). Returns the run_id it was filed under.

        `run_id` lets a caller correlate this row with its own record of the same
        run (e.g. the SQLite `runs.run_id` this prediction came from) instead of
        tracking two independent identifiers for one event. A generated uuid is
        used only when the caller has no id of its own to offer.
        """
        run_id = str(run_id) if run_id is not None else uuid.uuid4().hex[:12]
        self._append({
            'kind': 'prediction',
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'target_month': target_month,
            'predicted': float(predicted),
            'low': float(low),
            'high': float(high),
            'model_version': model_version,
        })
        return run_id

    def record_outcome(self, run_id, actual) -> None:
        # Every stored `run_id` is a string (`record_prediction` coerces it, so
        # a caller's own generated id and a caller-supplied one, e.g. SQLite's
        # integer primary key, compare the same way). Coercing here too means
        # `record_outcome(1, ...)` and `record_outcome('1', ...)` find the same
        # row instead of the int silently matching nothing.
        run_id = str(run_id)
        pred = next((r for r in self.all_rows()
                     if r.get('kind') == 'prediction' and r['run_id'] == run_id), None)
        if pred is None:
            raise KeyError(f'no prediction with run_id={run_id}')
        self._append({
            'kind': 'outcome',
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'actual': float(actual),
            'error': float(actual) - pred['predicted'],
            'inside_band': bool(pred['low'] <= float(actual) <= pred['high']),
        })

    def record_withdrawal(self, run_id, reason: str) -> None:
        """Mark a prior outcome invalid (phase B, undone) without erasing it.

        The chain is append-only, so an invalid grade cannot be deleted or
        rewritten without breaking every hash after it — the same property that
        makes the log trustworthy forbids quietly correcting it. A withdrawal is
        instead its own row: the original prediction and outcome stay in the
        file exactly as filed, and `scorecard` excludes any outcome a later
        withdrawal names, so the visible record and the honest one agree.
        """
        run_id = str(run_id)
        outcome = next((r for r in self.all_rows()
                        if r.get('kind') == 'outcome' and r['run_id'] == run_id), None)
        if outcome is None:
            raise KeyError(f'no outcome with run_id={run_id} to withdraw')
        self._append({
            'kind': 'withdrawal',
            'run_id': run_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': reason,
        })

    def all_rows(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def verify_chain(self) -> bool:
        prev = GENESIS
        for row in self.all_rows():
            stored = row.get('row_hash')
            payload = {k: v for k, v in row.items() if k != 'row_hash'}
            if payload.get('prev_hash') != prev:
                return False
            if _hash_row({k: v for k, v in payload.items() if k != 'prev_hash'} | {'prev_hash': prev}, prev) != stored:
                return False
            prev = stored
        return True

    def scorecard(self) -> dict:
        """MAE and band coverage over matured, still-valid outcomes.

        A withdrawn outcome is excluded rather than treated as ordinary
        evidence: it is on record as measuring the wrong question (`RSK-023`),
        and folding it into an accuracy average would let a known-bad grade
        quietly move a number a reader takes as this app's track record.

        Excluded by ROW POSITION relative to the run's most recent withdrawal,
        not by run_id alone. `store.withdraw_cross_cycle_grades` documents and
        supports a withdrawn run becoming "eligible again" and grading
        correctly once its own week settles -- `record_outcome` has no guard
        against a run_id it has already written an outcome for, so that regrade
        appends a SECOND outcome row. Keying the exclusion on run_id alone
        dropped that valid regrade too, forever: a run withdrawn once could
        never appear in the scorecard again even after being correctly
        regraded. An outcome recorded AFTER its run_id's latest withdrawal is
        the regrade, not the thing that was withdrawn, and survives.
        """
        rows = self.all_rows()
        last_withdrawal_idx: dict = {}
        for i, r in enumerate(rows):
            if r.get('kind') == 'withdrawal':
                last_withdrawal_idx[r['run_id']] = i
        outcomes = [r for i, r in enumerate(rows)
                   if r.get('kind') == 'outcome'
                   and i > last_withdrawal_idx.get(r['run_id'], -1)]
        n = len(outcomes)
        if n == 0:
            return {'n_matured': 0, 'mae': None, 'coverage_90': None}
        mae = sum(abs(o['error']) for o in outcomes) / n
        cov = sum(1 for o in outcomes if o['inside_band']) / n
        return {'n_matured': n, 'mae': mae, 'coverage_90': cov}
