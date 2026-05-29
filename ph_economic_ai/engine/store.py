from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine.ground_truth import compute_accuracy_score

_DEFAULT_DB = Path(__file__).parent.parent / 'cache' / 'trust.db'
_TRUST_INIT = 0.5
_EMA_ALPHA  = 0.3
_TRUST_MIN  = 0.05
_TRUST_MAX  = 0.95


class AgentTrustStore:
    def __init__(self, db_path: str | None = None):
        self._path = db_path or str(_DEFAULT_DB)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._migrate()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _migrate(self) -> None:
        cur = self._conn.cursor()
        cur.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                scenario_json     TEXT    NOT NULL,
                final_estimate    REAL,
                confidence_pct    INTEGER,
                internal_quality  REAL,
                actual_price_change REAL,
                accuracy_error    REAL,
                graded_at         TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_responses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id          INTEGER NOT NULL REFERENCES runs(run_id),
                agent_name      TEXT    NOT NULL,
                round_num       INTEGER NOT NULL,
                estimate        REAL,
                statement       TEXT,
                citation_count  INTEGER DEFAULT 0,
                has_causal_chain INTEGER DEFAULT 0,
                internal_score  REAL    DEFAULT 0.5,
                model_used      TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_trust (
                agent_name          TEXT PRIMARY KEY,
                trust_score         REAL    NOT NULL DEFAULT 0.5,
                runs_participated   INTEGER NOT NULL DEFAULT 0,
                avg_internal_score  REAL    NOT NULL DEFAULT 0.5,
                avg_accuracy_error  REAL,
                current_model_tier  TEXT    NOT NULL DEFAULT 'default',
                last_updated        TEXT    NOT NULL
            );
        ''')
        self._conn.commit()

    # ── Run persistence ───────────────────────────────────────────────────────

    def save_run(self, scenario: dict, final_estimate: Optional[float],
                 confidence_pct: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                'INSERT INTO runs (timestamp, scenario_json, final_estimate, confidence_pct) '
                'VALUES (?, ?, ?, ?)',
                (datetime.now(timezone.utc).isoformat(),
                 json.dumps(scenario), final_estimate, confidence_pct),
            )
            self._conn.commit()
            return cur.lastrowid

    def update_run_quality(self, run_id: int, internal_quality: float) -> None:
        with self._lock:
            self._conn.execute(
                'UPDATE runs SET internal_quality=? WHERE run_id=?',
                (internal_quality, run_id),
            )
            self._conn.commit()

    def save_agent_responses(self, run_id: int, responses: list[dict]) -> None:
        with self._lock:
            self._conn.executemany(
                'INSERT INTO agent_responses '
                '(run_id, agent_name, round_num, estimate, statement, '
                ' citation_count, has_causal_chain, internal_score, model_used) '
                'VALUES (:run_id, :agent_name, :round_num, :estimate, :statement, '
                '        :citation_count, :has_causal_chain, :internal_score, :model_used)',
                [{'run_id': run_id, **r} for r in responses],
            )
            self._conn.commit()

    def get_agent_responses(self, run_id: int) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                'SELECT * FROM agent_responses WHERE run_id=?', (run_id,)
            )
            return [dict(row) for row in cur.fetchall()]

    def get_ungraded_runs(self, min_age_days: float = 5.0) -> list[dict]:
        """Return runs not yet graded and older than min_age_days."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM runs WHERE actual_price_change IS NULL "
                "AND (julianday('now') - julianday(timestamp)) >= ?",
                (min_age_days,),
            )
            return [dict(row) for row in cur.fetchall()]

    def apply_ground_truth_grade(self, run_id: int, actual_change: float) -> None:
        """Grade a run against actual DOE price change, update agent trust."""
        with self._lock:
            row = self._conn.execute(
                'SELECT * FROM runs WHERE run_id=?', (run_id,)
            ).fetchone()
            if row is None:
                return
            # Idempotency guard — already graded
            if row['actual_price_change'] is not None:
                return
            final_est = row['final_estimate']
            error = abs(final_est - actual_change) if final_est is not None else None
            self._conn.execute(
                'UPDATE runs SET actual_price_change=?, accuracy_error=?, graded_at=? '
                'WHERE run_id=?',
                (actual_change, error, datetime.now(timezone.utc).isoformat(), run_id),
            )
            # Grade each agent response — use no-commit helper for atomicity
            cur = self._conn.execute(
                'SELECT * FROM agent_responses WHERE run_id=?', (run_id,)
            )
            responses = [dict(r) for r in cur.fetchall()]
            for resp in responses:
                est = resp['estimate']
                if est is None:
                    continue
                accuracy_score = compute_accuracy_score(est, actual_change)
                self._update_trust_no_commit(
                    resp['agent_name'],
                    internal_score=resp['internal_score'],
                    accuracy_score=accuracy_score,
                )
            # Single atomic commit covers both the run update and all trust updates
            self._conn.commit()

    def get_recent_runs(self, limit: int = 20) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                'SELECT * FROM runs ORDER BY run_id DESC LIMIT ?', (limit,)
            )
            return [dict(row) for row in cur.fetchall()]

    def total_runs(self) -> int:
        with self._lock:
            return self._conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]

    # ── Trust management ──────────────────────────────────────────────────────

    def get_trust(self, agent_name: str) -> float:
        with self._lock:
            row = self._conn.execute(
                'SELECT trust_score FROM agent_trust WHERE agent_name=?', (agent_name,)
            ).fetchone()
            return float(row['trust_score']) if row else _TRUST_INIT

    def get_all_trust(self) -> dict[str, float]:
        with self._lock:
            cur = self._conn.execute('SELECT agent_name, trust_score FROM agent_trust')
            return {row['agent_name']: float(row['trust_score']) for row in cur.fetchall()}

    def get_all_trust_rows(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                'SELECT * FROM agent_trust ORDER BY trust_score DESC'
            )
            return [dict(row) for row in cur.fetchall()]

    def _update_trust_no_commit(self, agent_name: str, internal_score: float,
                                accuracy_score: Optional[float] = None) -> None:
        """Insert/update trust without committing — caller must commit."""
        old_row = self._conn.execute(
            'SELECT trust_score FROM agent_trust WHERE agent_name=?', (agent_name,)
        ).fetchone()
        old_trust = float(old_row['trust_score']) if old_row else _TRUST_INIT
        if accuracy_score is not None:
            raw = 0.4 * internal_score + 0.6 * accuracy_score
        else:
            raw = internal_score
        new_trust = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * old_trust
        new_trust = max(_TRUST_MIN, min(_TRUST_MAX, new_trust))
        tier = trust_tier(new_trust)
        self._conn.execute(
            '''INSERT INTO agent_trust (agent_name, trust_score, runs_participated,
               avg_internal_score, avg_accuracy_error, current_model_tier, last_updated)
               VALUES (?, ?, 1, ?, ?, ?, ?)
               ON CONFLICT(agent_name) DO UPDATE SET
                 trust_score        = excluded.trust_score,
                 runs_participated  = runs_participated + 1,
                 avg_internal_score = (avg_internal_score + excluded.avg_internal_score) / 2,
                 avg_accuracy_error = COALESCE(
                     (avg_accuracy_error + excluded.avg_accuracy_error) / 2,
                     excluded.avg_accuracy_error
                 ),
                 current_model_tier = excluded.current_model_tier,
                 last_updated       = excluded.last_updated''',
            (agent_name, new_trust, internal_score, accuracy_score, tier,
             datetime.now(timezone.utc).isoformat()),
        )

    def update_trust(self, agent_name: str, internal_score: float,
                     accuracy_score: Optional[float] = None) -> None:
        with self._lock:
            self._update_trust_no_commit(agent_name, internal_score, accuracy_score)
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def trust_tier(trust: float) -> str:
    """Return 'promoted', 'demoted', or 'default' for a given trust score."""
    if trust > 0.70:
        return 'promoted'
    if trust < 0.30:
        return 'demoted'
    return 'default'
