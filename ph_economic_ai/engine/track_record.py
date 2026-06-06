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

    def record_prediction(self, target_month, predicted, low, high, model_version) -> str:
        run_id = uuid.uuid4().hex[:12]
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
        rows = self.all_rows()
        outcomes = [r for r in rows if r.get('kind') == 'outcome']
        n = len(outcomes)
        if n == 0:
            return {'n_matured': 0, 'mae': None, 'coverage_90': None}
        mae = sum(abs(o['error']) for o in outcomes) / n
        cov = sum(1 for o in outcomes if o['inside_band']) / n
        return {'n_matured': n, 'mae': mae, 'coverage_90': cov}
