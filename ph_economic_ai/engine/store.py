from __future__ import annotations

import json
import math
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ph_economic_ai.engine.ground_truth import compute_accuracy_score

_DEFAULT_DB = Path(__file__).parent.parent / 'cache' / 'trust.db'

#: The product this app forecasts and grades against. An observation of anything
#: else is not evidence about this series, however plausible its number looks.
FORECAST_GRADE = 'RON 95'
#: When the scraper started selecting the grade BY NAME instead of taking the
#: median across every fuel on the page (`swarm._PRICE_GRADE_PREFERENCE`, commit
#: `f050c53`). Observations recorded before this are of an unidentified product.
_GRADE_SELECTION_FIXED_AT = '2026-07-31T01:29:00+00:00'

_TRUST_INIT = 0.5
_EMA_ALPHA  = 0.3
_TRUST_MIN  = 0.05
_TRUST_MAX  = 0.95
# How fast a benched agent's trust returns to neutral, per run it sits out. Same
# rate as the EMA that lowered it, so a bench lasts about as long as the run of
# poor scores that earned it. See `recover_benched`.
_BENCH_RECOVERY_ALPHA = 0.3

# How far ahead a run is a forecast for, when the caller does not say. The app's
# fuel question is "what happens to the price in the next week", so a run is graded
# against a price observed around seven days out, not against whatever the price is
# on the day the grader happens to run (RSK-018).
DEFAULT_HORIZON_DAYS = 7.0
# How far from `target_date` an observation may sit and still be considered a fair
# grade for that run. DOE prices move weekly, so a few days either side is the
# matching period; beyond that the observation is measuring a different week.
GRADE_TOLERANCE_DAYS = 3.5


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
            -- Every movement of a trust score, in the order it happened.
            --
            -- Trust was a running EMA with no history, so it recorded a
            -- CONCLUSION and destroyed the evidence for it. Withdrawing a grade
            -- therefore could not undo the trust it had moved, and three
            -- withdrawn grades (`RSK-023`) left permanent residue in the roster.
            -- Inverting the EMA is not a fix: `old = (new - a*raw)/(1-a)` holds
            -- only if nothing clamped and nothing has happened since, and both
            -- had.
            --
            -- With the log, trust is a REPLAY of its events rather than a
            -- number that accumulated. Removing evidence removes its effect
            -- exactly, and a trust score can be audited back to the runs that
            -- produced it.
            CREATE TABLE IF NOT EXISTS trust_events (
                event_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at  TEXT    NOT NULL,
                agent_name   TEXT    NOT NULL,
                kind         TEXT    NOT NULL,   -- response | grade | recovery
                raw          REAL,               -- the EMA target; NULL for recovery
                run_id       INTEGER,            -- the evidence this rests on
                internal_score  REAL,
                accuracy_score  REAL,
                abs_error       REAL   -- PHP/L, so the per-agent average error
                                       -- can be replayed. Not recoverable from
                                       -- accuracy_score, which floors at zero.
            );
            CREATE INDEX IF NOT EXISTS trust_events_order
                ON trust_events (occurred_at, event_id);
            CREATE INDEX IF NOT EXISTS trust_events_run
                ON trust_events (run_id, kind);
        ''')
        self._conn.commit()

        # Sector estimates added after initial release — add to existing DBs.
        existing = {r['name'] for r in cur.execute('PRAGMA table_info(runs)').fetchall()}
        for col in ('food_estimate', 'electricity_estimate'):
            if col not in existing:
                cur.execute(f'ALTER TABLE runs ADD COLUMN {col} REAL')

        # Reproducibility columns. A run used to record only its scenario and its
        # answer, so a stored run could not be re-derived even in principle: the
        # retrieved evidence was gone and the sampler was unseeded. `evidence_json`
        # keeps what the agents actually read; `run_seed` and `temperature` keep the
        # sampling settings that produced the numbers.
        for col, decl in (('evidence_json', 'TEXT'),
                          ('run_seed', 'INTEGER'),
                          ('temperature', 'REAL')):
            if col not in existing:
                cur.execute(f'ALTER TABLE runs ADD COLUMN {col} {decl}')

        # Horizon matching (RSK-018). Grading used to score every ungraded run
        # against whatever the price happened to be on the day the grader ran, so a
        # five-day-old forecast and a sixty-day-old one were judged against the same
        # number. `target_date` records the period a run is actually a forecast FOR,
        # and `graded_against` records which observation it was finally scored on,
        # so a grade can be audited rather than trusted.
        for col, decl in (('target_date', 'TEXT'),
                          ('horizon_days', 'REAL'),
                          ('graded_against', 'TEXT')):
            if col not in existing:
                cur.execute(f'ALTER TABLE runs ADD COLUMN {col} {decl}')

        # Recall. `run_key` is the vintage fingerprint from engine/vintage.py: the
        # pricing week, the day, the quantised scenario and market inputs, and the
        # model identity. It answers "have I already answered this exact question
        # today", which the app previously could not ask at all -- eight runs on
        # 2026-07-27 returned eight different numbers for an unchanged market.
        #
        # `verdict_json` is the answer itself, stored well enough to put the report
        # back on screen without calling a model. The run row already had the three
        # headline numbers, but not the regional verdicts, the dissent or the
        # agreement basis, so a stored run could be summarised and not reopened.
        for col, decl in (('run_key', 'TEXT'),
                          ('verdict_json', 'TEXT')):
            if col not in existing:
                cur.execute(f'ALTER TABLE runs ADD COLUMN {col} {decl}')
        # Recall reads by key, newest first, on every run. Without this it is a
        # full scan of a table that grows by one row per run forever.
        cur.execute(
            'CREATE INDEX IF NOT EXISTS idx_runs_run_key ON runs(run_key)')

        # Observed prices over time. Without a history there is nothing to grade a
        # past run against except the present, which is the defect itself.
        cur.executescript('''
            CREATE TABLE IF NOT EXISTS price_observations (
                observed_at TEXT PRIMARY KEY,
                price       REAL NOT NULL,
                grade       TEXT
            );
        ''')

        # WHICH PRODUCT an observation is of. Without it the table is not a price
        # series, because two fuel grades can sit in it looking like one price
        # that moved.
        #
        # That is not hypothetical: the cycle opened 2026-07-28 held 84.38 on the
        # Thursday and 89.51 on the Friday, on days no adjustment happens, and it
        # blocked ten runs from grading as an "unsettled week". They are two
        # different products. Until `f050c53` on 2026-07-31 the scraper took the
        # MEDIAN across every fuel on the page, and the page that day listed
        # Diesel 81.13, Diesel Plus 83.94, Unleaded 91 84.38, Premium 95 89.51
        # and Kerosene 111.43 -- median 84.38, which is Unleaded 91. The app
        # forecasts RON 95. The changeover in the stored data falls within
        # fifteen minutes of that commit.
        ev_cols = {r['name'] for r in
                   cur.execute('PRAGMA table_info(trust_events)').fetchall()}
        if ev_cols and 'abs_error' not in ev_cols:
            cur.execute('ALTER TABLE trust_events ADD COLUMN abs_error REAL')

        obs_cols = {r['name'] for r in
                    cur.execute('PRAGMA table_info(price_observations)').fetchall()}
        if obs_cols and 'grade' not in obs_cols:
            cur.execute("ALTER TABLE price_observations ADD COLUMN grade TEXT")
            # Rows written before the selection fix are labelled `unknown`, not
            # `RON 91`. The evidence that they ARE Unleaded 91 is strong -- the
            # value matches that product exactly on the one day the page was
            # recorded -- but the rule in force did not select a product at all,
            # it took the middle of a list. `unknown` is what is actually known,
            # and it is enough: whatever they are, they are not RON 95.
            cur.execute("UPDATE price_observations SET grade='unknown' "
                        "WHERE grade IS NULL AND observed_at < ?",
                        (_GRADE_SELECTION_FIXED_AT,))
            cur.execute(f"UPDATE price_observations SET grade='{FORECAST_GRADE}' "
                        "WHERE grade IS NULL")
        self._conn.commit()

    # ── Run persistence ───────────────────────────────────────────────────────

    def save_run(self, scenario: dict, final_estimate: Optional[float],
                 confidence_pct: int, evidence: Optional[dict] = None,
                 run_seed: Optional[int] = None,
                 temperature: Optional[float] = None,
                 horizon_days: float = DEFAULT_HORIZON_DAYS,
                 run_key: Optional[str] = None,
                 verdict: Optional[dict] = None) -> int:
        """Persist a run.

        `evidence` is what the agents actually retrieved, and `run_seed` /
        `temperature` are the sampling settings. Storing all three is what makes a
        run reproducible: the scenario alone is not enough, because the same
        scenario against different retrieved text, or against an unseeded sampler,
        legitimately produces a different answer. All three are optional so older
        callers and existing rows stay valid.

        `horizon_days` fixes the period the run is a forecast for, stored as
        `target_date`. Grading later scores the run against a price observed near
        that date rather than against whatever the price is when the grader runs.

        `run_key` is the vintage fingerprint (engine/vintage.py) and `verdict` is
        a snapshot of the answer complete enough to reopen the report without
        calling a model. Together they are what `find_run_by_key` recalls.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            target = now + timedelta(days=float(horizon_days))
            cur = self._conn.execute(
                'INSERT INTO runs (timestamp, scenario_json, final_estimate, '
                'confidence_pct, evidence_json, run_seed, temperature, '
                'target_date, horizon_days, run_key, verdict_json) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (now.isoformat(),
                 json.dumps(scenario), final_estimate, confidence_pct,
                 json.dumps(evidence, default=str) if evidence is not None else None,
                 run_seed, temperature, target.isoformat(), float(horizon_days),
                 run_key,
                 json.dumps(verdict, default=str) if verdict is not None else None),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_run(self, run_id: int) -> Optional[dict]:
        """One run row by id, or None.

        The store had no single-row read at all: callers wanting one run pulled
        `get_recent_runs(200)` and filtered in Python.
        """
        with self._lock:
            row = self._conn.execute(
                'SELECT * FROM runs WHERE run_id=?', (run_id,)).fetchone()
        return dict(row) if row else None

    def find_runs_by_key(self, run_key: str, limit: int = 8) -> list[dict]:
        """Completed runs in one vintage bucket, newest first.

        Returns candidates rather than a single answer because the key is
        deliberately coarse: it narrows to "today, this pricing week, these
        models", and the caller then checks whether the inputs actually match
        within tolerance (see `vintage.inputs_unchanged`). A hash cannot express
        "close enough" and a tolerance cannot be indexed, so the work is split.

        "Completed" means the run reached a verdict worth showing again. A run
        that crashed before producing an estimate is not an answer, and recalling
        one would turn a transient failure into a permanent one for the rest of
        the day: every later run would match the same key and be handed the same
        blank. `verdict_json` is required for the same reason — a row from before
        this feature has the headline numbers but not enough to rebuild the
        report, and half a report presented as a recall is worse than re-running.
        """
        if not run_key:
            return []
        with self._lock:
            rows = self._conn.execute(
                'SELECT * FROM runs WHERE run_key=? AND final_estimate IS NOT NULL '
                'AND verdict_json IS NOT NULL ORDER BY run_id DESC LIMIT ?',
                (run_key, int(limit))).fetchall()
        return [dict(r) for r in rows]

    def get_run_verdict(self, run_id: int) -> Optional[dict]:
        """The stored verdict snapshot for a run, or None."""
        with self._lock:
            row = self._conn.execute(
                'SELECT verdict_json FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None or row['verdict_json'] is None:
            return None
        try:
            return json.loads(row['verdict_json'])
        except (ValueError, TypeError):
            return None

    def attach_verdict(self, run_id: int, verdict: dict) -> None:
        """Attach a verdict snapshot after the fact.

        The gas run creates the row, but food and electricity finish later and on
        their own threads, so the snapshot cannot be complete at insert time.
        Until it is attached the row has no `verdict_json` and `find_run_by_key`
        will not recall it, which is the correct behaviour for a run still in
        flight.
        """
        with self._lock:
            self._conn.execute(
                'UPDATE runs SET verdict_json=? WHERE run_id=?',
                (json.dumps(verdict, default=str), run_id))
            self._conn.commit()

    def get_run_evidence(self, run_id: int) -> Optional[dict]:
        """The evidence a stored run saw, or None if it predates the column."""
        with self._lock:
            row = self._conn.execute(
                'SELECT evidence_json FROM runs WHERE run_id=?', (run_id,)).fetchone()
        if row is None or row['evidence_json'] is None:
            return None
        try:
            return json.loads(row['evidence_json'])
        except (ValueError, TypeError):
            return None

    def update_run_quality(self, run_id: int, internal_quality: float) -> None:
        with self._lock:
            self._conn.execute(
                'UPDATE runs SET internal_quality=? WHERE run_id=?',
                (internal_quality, run_id),
            )
            self._conn.commit()

    def update_run_sectors(self, run_id: int, food_estimate: Optional[float],
                           electricity_estimate: Optional[float]) -> None:
        with self._lock:
            self._conn.execute(
                'UPDATE runs SET food_estimate=?, electricity_estimate=? WHERE run_id=?',
                (food_estimate, electricity_estimate, run_id),
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
        """Return runs not yet graded and older than min_age_days.

        A non-positive `min_age_days` means "no age filter" and skips the clock
        comparison entirely. That is both the obvious reading of the argument and a
        fix for a real race: a run saved and queried in the same instant differs
        from `julianday('now')` by around a millisecond (~1.2e-08 days), so on a
        loaded machine the two can resolve equal or inverted and a just-saved run
        vanishes from its own query. That flake was rare enough to survive many
        green suites and only appeared when the CPU was saturated.
        """
        with self._lock:
            if min_age_days <= 0:
                cur = self._conn.execute(
                    'SELECT * FROM runs WHERE actual_price_change IS NULL')
            else:
                cur = self._conn.execute(
                    "SELECT * FROM runs WHERE actual_price_change IS NULL "
                    "AND (julianday('now') - julianday(timestamp)) >= ?",
                    (min_age_days,),
                )
            return [dict(row) for row in cur.fetchall()]

    # ── Observed prices, for horizon-matched grading ──────────────────────────

    def record_price_observation(self, price: float, observed_at=None,
                                 grade: str = FORECAST_GRADE) -> None:
        """Store an observed price so past runs can be graded against their own
        period instead of against the present.

        **One row per (day, price).** `INSERT OR REPLACE` keys on `observed_at`,
        which carries microseconds, so every call wrote a new row: the grading
        poll runs six-hourly and every run records too, and the table reached 568
        rows holding EIGHT observations, one price repeated 157 times in a day.

        That is the same failure shape as an agreement percentage: a number that
        looks like evidence of density and is not. "568 price observations" is a
        sentence someone would say to a panel, and it would be false.

        A price that genuinely moves within a day still records, because the key
        is the pair and not the day alone. Philippine retail prices are a weekly
        step function, so sub-day resolution carries no information a grade could
        use -- `price_near` has a 3.5 day tolerance.

        `grade` names WHICH PRODUCT was observed, and without it this table is
        not a price series at all. The cycle opened 2026-07-28 held Unleaded 91
        at 84.38 and Premium 95 at 89.51 and read as one price that moved twice
        inside a week, on days no adjustment happens. It blocked ten runs from
        grading as an "unsettled week" when nothing about the week was unsettled.
        """
        when = (observed_at or datetime.now(timezone.utc))
        stamp = when.isoformat() if hasattr(when, 'isoformat') else str(when)
        value = float(price)
        with self._lock:
            seen = self._conn.execute(
                'SELECT 1 FROM price_observations '
                'WHERE substr(observed_at, 1, 10) = ? AND price = ? '
                'AND COALESCE(grade, ?) = ? LIMIT 1',
                (stamp[:10], value, grade, grade)).fetchone()
            if seen is not None:
                return
            self._conn.execute(
                'INSERT OR REPLACE INTO price_observations '
                '(observed_at, price, grade) VALUES (?, ?, ?)',
                (stamp, value, grade))
            self._conn.commit()

    def deduplicate_price_observations(self) -> int:
        """Collapse repeated (day, price, GRADE) rows, keeping the earliest of each.

        Returns the number removed.

        **The grade is part of the key, and leaving it out silently deleted real
        observations.** Two fuels can cost the same on the same day. Grouping on
        (day, price) alone put them in one group and `MIN(rowid)` kept whichever
        was inserted first, so a RON 95 reading could be dropped and a RON 91
        reading left standing in its place. The week then reports no RON 95 price
        at all and stops grading, and nothing anywhere says why.

        This method has claimed to be "information-preserving" before and it was
        false then too: collapsing 568 rows moved a graded outcome from 84.38 to
        89.51. It preserves information WITHIN a (day, price, grade) group, which
        is the claim that can actually be checked, and that is all it claims.
        """
        with self._lock:
            before = self._conn.execute(
                'SELECT COUNT(*) FROM price_observations').fetchone()[0]
            self._conn.execute(
                'DELETE FROM price_observations WHERE rowid NOT IN ('
                '  SELECT MIN(rowid) FROM price_observations '
                '  GROUP BY substr(observed_at, 1, 10), price, COALESCE(grade, ?))',
                (FORECAST_GRADE,))
            self._conn.commit()
            after = self._conn.execute(
                'SELECT COUNT(*) FROM price_observations').fetchone()[0]
        return before - after

    def price_near(self, target_date, tolerance_days: float = GRADE_TOLERANCE_DAYS,
                   grade: str = FORECAST_GRADE):
        """The observation closest to `target_date`, or None if none is close enough.

        Returning None is the point. A run whose period has no matching observation
        stays ungraded rather than being scored against a price from a different
        week, which is what produced misleading trust and accuracy views.

        Restricted to one PRODUCT, like `cycle_prices`. Without that it would
        happily answer a RON 95 question with a RON 91 reading, which is the
        defect that made one pricing week look unsettled and blocked ten runs.
        Grading no longer calls this -- it uses `cycle_price` -- but a method that
        returns the wrong series when asked is a defect whether or not the app
        currently asks.
        """
        stamp = target_date.isoformat() if hasattr(target_date, 'isoformat') else str(target_date)
        with self._lock:
            # Ranked by CALENDAR-DAY distance first, then by timestamp. The
            # observations are a weekly step function sampled by a six-hourly
            # poll, so the hour carries no information -- but ranking on the raw
            # timestamp made the match depend on how many times a day the poll
            # happened to run. Collapsing 568 duplicate rows to their 8 real
            # observations moved one run's graded outcome from 84.38 to 89.51,
            # because the nearest surviving row fell on the NEXT day. Same
            # information, different grade, which is not a property a grader may
            # have.
            #
            # Day distance makes a same-day observation always win, and the
            # timestamp only breaks ties within equally distant days.
            row = self._conn.execute(
                'SELECT observed_at, price, '
                '  ABS(julianday(observed_at) - julianday(?)) AS gap, '
                '  ABS(julianday(date(observed_at)) - julianday(date(?))) AS daygap '
                'FROM price_observations WHERE COALESCE(grade, ?) = ? '
                'ORDER BY daygap ASC, gap ASC LIMIT 1',
                (stamp, stamp, grade, grade)
            ).fetchone()
        if row is None or row['gap'] is None or row['gap'] > tolerance_days:
            return None
        return {'observed_at': row['observed_at'], 'price': float(row['price']),
                'gap_days': float(row['gap'])}

    def cycle_price(self, target_date):
        """The price for the PRICING WEEK containing `target_date`, or None.

        Philippine retail fuel is a step function with a Tuesday 06:00 boundary.
        A weekly forecast is a claim about the step, so the outcome it is graded
        against has to be the step -- not whichever scrape happened to land
        nearest the target timestamp.

        Two defects this replaces, both live in the stored record:

        * **The graded price came from the wrong week.** Runs targeting
          2026-08-03, inside the cycle opened 07-28, were graded against an
          observation from 08-04, which is the NEXT cycle. A forecast for one
          week scored against another week's price.
        * **The source moved inside a cycle.** It read 84.38 on Thursday 30 July
          and 89.51 on Friday 31 July, both inside the cycle opened 07-28, on
          days no adjustment happens. That +5.13 became the "actual change" for
          two runs and produced the only two zero scores in the track record.

        **An ambiguous week yields None rather than a guess.** When observations
        inside one cycle disagree, the week has no single price and nothing here
        can say which was real. Refusing is the same rule `price_near` already
        applies across weeks, and `DEC-045`'s: a missing grade shrinks the record,
        a wrong grade corrupts it and is permanent.

        Returns `{'price', 'cycle', 'n_observations'}` or None.
        """
        from ph_economic_ai.engine import price_calendar, vintage

        stamp = (target_date.isoformat() if hasattr(target_date, 'isoformat')
                 else str(target_date))
        prices, start, n = self.cycle_prices(target_date)
        if not prices:
            return None
        if len(prices) > 1:
            # The week has no single price. Grading against any one of them is a
            # coin toss recorded as an outcome.
            return None
        return {'price': prices.pop(), 'cycle': start.date().isoformat(),
                'n_observations': n}

    def cycle_prices(self, when):
        """Every distinct price observed in the pricing week containing `when`.

        Returns `(prices, cycle_start, n_observations)`. Split out of
        `cycle_price` because a caller needs to tell "this week has no price"
        from "this week has two and cannot say which" -- absence is normal for
        an old run, disagreement means the week is unusable as either end of a
        measured change.
        """
        from ph_economic_ai.engine import price_calendar, vintage

        stamp = (when.isoformat() if hasattr(when, 'isoformat') else str(when))
        moment = datetime.fromisoformat(stamp)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=price_calendar.PH_TZ)
        start = vintage.fuel_cycle_start(moment)
        end = start + timedelta(days=7)

        with self._lock:
            rows = self._conn.execute(
                # Only the product this app forecasts. An observation of another
                # grade is not evidence about this series however plausible its
                # number is, and mixing them is what made one week look like it
                # held two prices.
                'SELECT price FROM price_observations '
                'WHERE julianday(observed_at) >= julianday(?) '
                '  AND julianday(observed_at) <  julianday(?) '
                '  AND COALESCE(grade, ?) = ?',
                (start.isoformat(), end.isoformat(),
                 FORECAST_GRADE, FORECAST_GRADE)).fetchall()
        return {round(float(r['price']), 2) for r in rows}, start, len(rows)

    def target_cycle(self, run: dict):
        """The pricing WEEK a run is a forecast for, as a cycle-start datetime.

        Not `timestamp + horizon_days`. That instant is derived, and bucketing it
        to the second split one batch of ten runs across two different weeks:
        runs 23 to 32 were all launched to forecast the adjustment of
        2026-08-04, and their stored targets land between 40 seconds before
        06:00:00 PHT and 36 seconds after it, because each was computed as
        "now plus the time remaining" at a slightly different `now`. Three
        landed in the week they were forecasting and seven in the week they
        started from, on a margin of under a minute.

        The unit of the forecast is the week, so the week is what is derived:
        the run was made inside some cycle and is a claim about a LATER one.
        `horizon_days` says how many steps ahead, and every horizon the app has
        ever stored is "until the next adjustment", so that count is one.
        """
        from ph_economic_ai.engine import price_calendar, vintage

        made = datetime.fromisoformat(run['timestamp'])
        if made.tzinfo is None:
            made = made.replace(tzinfo=price_calendar.PH_TZ)
        horizon = float(run.get('horizon_days') or DEFAULT_HORIZON_DAYS)
        # A forecast is always about a week that has not happened yet, so at
        # least one step; `ceil` covers a horizon that spans several.
        steps = max(1, math.ceil(horizon / 7.0))
        return vintage.fuel_cycle_start(made) + timedelta(days=7 * steps)

    def withdraw_cross_cycle_grades(self) -> list[dict]:
        """Un-grade runs scored against a different pricing week than they forecast.

        Every grade in the stored record was one: three runs forecasting the week
        opened 2026-07-28, each scored against an observation from the week opened
        08-04. A forecast for one week compared to another week's price is not a
        loose grade, it is a grade of a different question -- the same defect
        `RSK-018` was raised for, one level up from the stale baseline.

        They become eligible again rather than deleted. Once their own week has
        an unambiguous price they will grade correctly; if it never does, they
        stay honestly ungraded.

        **The trust those grades moved is reversed too**, by deleting their
        events and replaying what is left. It used not to be: trust was a running
        EMA that kept a conclusion and destroyed its evidence, so the roster
        carried movement from grades that no longer existed. The withdrawn
        entry's `trust_moved` reports which agents changed and by how much,
        because a silent correction to a trust score is the same class of problem
        as the residue it fixes.

        Returns the withdrawn rows, so a caller can report what changed.
        """
        from ph_economic_ai.engine import price_calendar, vintage

        withdrawn = []
        with self._lock:
            rows = self._conn.execute(
                'SELECT * FROM runs WHERE graded_at IS NOT NULL').fetchall()
        for row in rows:
            run = dict(row)
            # `target_cycle`, the same rule the grader uses. Deriving it here
            # from `timestamp + horizon` instead would let the withdrawal and
            # the grader disagree about which week a run forecasts, and a
            # withdrawal that disagrees with the grader is just a second bug.
            target_cycle = self.target_cycle(run).date().isoformat()

            against = (run.get('graded_against') or '').split(' ')[0]
            if against.startswith('pricing'):          # already cycle-aligned
                continue
            try:
                observed = datetime.fromisoformat(against)
            except ValueError:
                continue
            if vintage.fuel_cycle_start(observed).date().isoformat() == target_cycle:
                continue

            withdrawn.append({'run_id': run['run_id'], 'target_cycle': target_cycle,
                              'was_error': run.get('accuracy_error')})
            with self._lock:
                self._conn.execute(
                    'UPDATE runs SET actual_price_change=NULL, accuracy_error=NULL, '
                    'graded_at=NULL, graded_against=NULL WHERE run_id=?',
                    (run['run_id'],))
                # The evidence goes with the grade. Leaving these behind is what
                # made a withdrawal cosmetic: the run showed as ungraded while
                # the roster still carried the score it had been given.
                self._conn.execute(
                    "DELETE FROM trust_events WHERE run_id=? AND kind='grade'",
                    (run['run_id'],))
                self._conn.commit()

        if withdrawn:
            moved = self.replay_trust()
            for entry in withdrawn:
                entry['trust_moved'] = moved
        return withdrawn

    def get_due_runs(self) -> list[dict]:
        """Ungraded runs whose forecast period has actually elapsed.

        Rows written before `target_date` existed fall back to
        `timestamp + horizon`, so legacy runs are still gradable but on the same
        horizon-matched terms.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM runs WHERE actual_price_change IS NULL AND "
                "julianday('now') >= julianday("
                "  COALESCE(target_date, datetime(timestamp, '+' || ? || ' days')))",
                (DEFAULT_HORIZON_DAYS,),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_graded_errors(self, limit: int = 200) -> list[float]:
        """Absolute errors of graded runs, newest first.

        These feed the conformal band in `engine/interval.py`. They are only
        meaningful because grading is cycle-aligned (`RSK-023`): each error
        compares a run against the price of the pricing WEEK it forecast. Under
        the earlier horizon match (`RSK-018`) the tolerance was 3.5 days either
        side, which spans a week boundary, and every error this returned came
        from a comparison against a different week.
        """
        with self._lock:
            rows = self._conn.execute(
                'SELECT accuracy_error FROM runs '
                'WHERE accuracy_error IS NOT NULL '
                'ORDER BY graded_at DESC LIMIT ?', (int(limit),)
            ).fetchall()
        return [float(r['accuracy_error']) for r in rows]

    def count_runs(self) -> int:
        """How many runs are stored, graded or not.

        The screen shows this beside the graded count. Zero graded runs and zero
        runs are different facts: the first says the outcomes have not settled
        yet, the second says the app has never been used, and showing only the
        graded count lets a reader take the harsher reading.
        """
        with self._lock:
            return int(self._conn.execute(
                'SELECT COUNT(*) FROM runs').fetchone()[0])

    def effective_target_date(self, run: dict) -> str:
        """The period a run is a forecast for, inferring it for legacy rows."""
        if run.get('target_date'):
            return run['target_date']
        stamp = run['timestamp']
        base = datetime.fromisoformat(stamp)
        horizon = run.get('horizon_days') or DEFAULT_HORIZON_DAYS
        return (base + timedelta(days=float(horizon))).isoformat()

    def _latest_responses_no_commit(self, run_id: int):
        """One response per agent: its final word on this run.

        The same rule `forum._latest_per_agent` applies to consensus, confidence
        and the judge. Grading was the one place that ignored it, and it iterated
        every response row instead.

        Two defects came out of that, both live on the app's first graded run:

        * **One forecast moved an agent's trust twice.** Run 32 has 20 agents and
          32 responses, so twelve agents took two EMA updates from a single
          outcome and moved further than the eight who spoke once. That weights
          trust by how much an agent talks.
        * **A withdrawn estimate was still graded.** Central Luzon DataExtractor
          said 1.20 in round one and revised to 1.35 in round two, and both were
          scored. An agent's answer is the one it ends on; scoring a position it
          already retracted measures the debate, not the forecast.
        """
        return self._conn.execute(
            'SELECT * FROM agent_responses a WHERE a.run_id = ? AND a.id = ('
            '  SELECT b.id FROM agent_responses b'
            '  WHERE b.run_id = a.run_id AND b.agent_name = a.agent_name'
            '  ORDER BY b.round_num DESC, b.id DESC LIMIT 1)',
            (run_id,)).fetchall()

    def rebuild_grade_events(self, run_id: int) -> dict:
        """Rewrite an already-graded run's trust events under the current rule.

        Needed because the grading rule has changed twice under a stored grade,
        and a trust score is only auditable if it reflects the rule in force
        rather than the one that happened to be running the day it was written.

        Deletes the run's `grade` events, rebuilds them from its final-word
        responses, and replays. The run's own grade is untouched: the outcome it
        was scored against has not changed, only how many times that outcome was
        allowed to move each agent.
        """
        with self._lock:
            row = self._conn.execute(
                'SELECT actual_price_change, graded_at FROM runs WHERE run_id=?',
                (run_id,)).fetchone()
            if row is None or row['actual_price_change'] is None:
                return {'rebuilt': 0, 'skipped': 'run is not graded'}
            actual = float(row['actual_price_change'])
            before = self._conn.execute(
                "SELECT COUNT(*) FROM trust_events WHERE run_id=? AND kind='grade'",
                (run_id,)).fetchone()[0]
            self._conn.execute(
                "DELETE FROM trust_events WHERE run_id=? AND kind='grade'", (run_id,))
            written = 0
            for resp in self._latest_responses_no_commit(run_id):
                if resp['estimate'] is None:
                    continue
                internal = float(resp['internal_score'] or _TRUST_INIT)
                acc = compute_accuracy_score(float(resp['estimate']), actual)
                self._conn.execute(
                    'INSERT INTO trust_events (occurred_at, agent_name, kind, raw, '
                    'run_id, internal_score, accuracy_score, abs_error) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (row['graded_at'], resp['agent_name'], 'grade',
                     0.4 * internal + 0.6 * acc, run_id, internal, acc,
                     abs(float(resp['estimate']) - actual)))
                written += 1
            self._conn.commit()
        moved = self.replay_trust()
        return {'rebuilt': written, 'was': before, 'trust_moved': moved}

    def apply_ground_truth_grade(self, run_id: int, actual_change: float,
                                 graded_against: Optional[str] = None) -> None:
        """Grade a run against actual DOE price change, update agent trust.

        `graded_against` records WHICH observation was used, so a grade can be
        audited later rather than taken on trust.

        Each agent is scored ONCE, on its final word. See
        `_latest_responses_no_commit` for the two defects that came from grading
        every response row.
        """
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
                'UPDATE runs SET actual_price_change=?, accuracy_error=?, graded_at=?, '
                'graded_against=? WHERE run_id=?',
                (actual_change, error, datetime.now(timezone.utc).isoformat(),
                 graded_against, run_id),
            )
            responses = [dict(r) for r in
                         self._latest_responses_no_commit(run_id)]
            for resp in responses:
                est = resp['estimate']
                if est is None:
                    continue
                accuracy_score = compute_accuracy_score(est, actual_change)
                self._update_trust_no_commit(
                    resp['agent_name'],
                    internal_score=resp['internal_score'],
                    accuracy_score=accuracy_score,
                    # The miss in PHP/L. `accuracy_score` floors at zero beyond a
                    # 3 peso error, so the error cannot be recovered from it.
                    abs_error=abs(float(est) - actual_change),
                    # Tied to the run, so withdrawing that run's grade removes
                    # exactly the trust movement the grade caused.
                    run_id=run_id,
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
                                accuracy_score: Optional[float] = None,
                                run_id: Optional[int] = None,
                                occurred_at: Optional[str] = None,
                                abs_error: Optional[float] = None) -> None:
        """Insert/update trust without committing — caller must commit.

        Also appends to `trust_events`, so the score stays reproducible from the
        evidence rather than being an accumulated number nobody can take apart.
        """
        old_row = self._conn.execute(
            'SELECT trust_score FROM agent_trust WHERE agent_name=?', (agent_name,)
        ).fetchone()
        old_trust = float(old_row['trust_score']) if old_row else _TRUST_INIT
        if accuracy_score is not None:
            raw = 0.4 * internal_score + 0.6 * accuracy_score
        else:
            raw = internal_score
        self._conn.execute(
            'INSERT INTO trust_events (occurred_at, agent_name, kind, raw, run_id, '
            'internal_score, accuracy_score, abs_error) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (occurred_at or datetime.now(timezone.utc).isoformat(), agent_name,
             'grade' if accuracy_score is not None else 'response',
             raw, run_id, internal_score, accuracy_score, abs_error))
        new_trust = _EMA_ALPHA * raw + (1 - _EMA_ALPHA) * old_trust
        new_trust = max(_TRUST_MIN, min(_TRUST_MAX, new_trust))
        tier = trust_tier(new_trust)
        self._conn.execute(
            '''INSERT INTO agent_trust (agent_name, trust_score, runs_participated,
               avg_internal_score, avg_accuracy_error, current_model_tier, last_updated)
               VALUES (?, ?, 0, 0.5, NULL, ?, ?)
               ON CONFLICT(agent_name) DO UPDATE SET
                 trust_score        = excluded.trust_score,
                 current_model_tier = excluded.current_model_tier,
                 last_updated       = excluded.last_updated''',
            (agent_name, new_trust, tier,
             datetime.now(timezone.utc).isoformat()),
        )
        self._refresh_agent_aggregates_no_commit(agent_name)

    def _refresh_agent_aggregates_no_commit(self, agent_name: str) -> None:
        """Recompute the three summary columns from the tables that hold the facts.

        All three used to be maintained incrementally, and all three were lies:

        * **`runs_participated` counted trust UPDATES, not runs.** It incremented
          once per `_update_trust_no_commit` call, and an agent takes one at run
          time and another when the run is graded. Live: an agent showed 72 in a
          store holding 34 runs. `ADR-008` quotes "991 run participations" in the
          research record, and that number is of the same kind. Same failure
          shape as "568 price observations" -- a count that looks like evidence
          of volume and is counting something else.
        * **`avg_internal_score` was `(old + new) / 2`**, which is an EMA with
          alpha 0.5, not a mean. A column named `avg_` holding a recency-weighted
          blend is a claim in a name.
        * **`avg_accuracy_error` held an accuracy SCORE**, bound from
          `compute_accuracy_score`, so a value of 0.55 in a column named "error"
          reads as an average miss of 0.55 PHP/L when it is a 0-to-1 quality
          score. Now the mean absolute error in PHP/L, matching `runs.accuracy_error`.

        Derived rather than accumulated, so they cannot drift from their sources,
        and `replay_trust` recomputes them the same way.
        """
        self._conn.execute(
            '''UPDATE agent_trust SET
                 runs_participated = (
                   SELECT COUNT(DISTINCT run_id) FROM agent_responses
                   WHERE agent_name = ?),
                 avg_internal_score = COALESCE((
                   SELECT AVG(internal_score) FROM agent_responses
                   WHERE agent_name = ?), 0.5),
                 avg_accuracy_error = (
                   SELECT AVG(abs_error) FROM trust_events
                   WHERE agent_name = ? AND kind = 'grade' AND abs_error IS NOT NULL)
               WHERE agent_name = ?''',
            (agent_name, agent_name, agent_name, agent_name))

    def update_trust(self, agent_name: str, internal_score: float,
                     accuracy_score: Optional[float] = None) -> None:
        with self._lock:
            self._update_trust_no_commit(agent_name, internal_score, accuracy_score)
            self._conn.commit()

    def replay_trust(self) -> dict:
        """Recompute every trust score from `trust_events`, in order.

        The fix for the residue `RSK-023` left behind. Withdrawing a grade used
        to clear the grade and leave the trust it had moved, because trust was a
        running EMA that kept a conclusion and destroyed its evidence. Inverting
        the EMA is not available: `old = (new - a*raw)/(1-a)` holds only when
        nothing clamped and nothing has happened since, and by the time a grade
        is withdrawn both are usually false.

        Replay sidesteps that. Trust is defined as the EMA over the surviving
        events, so removing evidence removes its effect exactly rather than
        approximately, and any score can be traced back to the runs behind it.

        Returns `{agent: {'before', 'after', 'events'}}` for agents that moved,
        so a caller can report the correction instead of applying it silently.
        """
        with self._lock:
            before = {r['agent_name']: float(r['trust_score']) for r in
                      self._conn.execute('SELECT agent_name, trust_score '
                                         'FROM agent_trust').fetchall()}
            events = self._conn.execute(
                'SELECT * FROM trust_events ORDER BY occurred_at, event_id'
            ).fetchall()

            trust: dict = {}
            counts: dict = {}
            for e in events:
                agent = e['agent_name']
                old = trust.get(agent, _TRUST_INIT)
                if e['kind'] == 'recovery':
                    # Decay toward the neutral prior, not an EMA on evidence.
                    new = old + _BENCH_RECOVERY_ALPHA * (_TRUST_INIT - old)
                else:
                    new = _EMA_ALPHA * float(e['raw']) + (1 - _EMA_ALPHA) * old
                trust[agent] = max(_TRUST_MIN, min(_TRUST_MAX, new))
                counts[agent] = counts.get(agent, 0) + 1

            # Every agent the store knows about, not only those with surviving
            # events. An agent whose entire evidence was withdrawn has no events
            # left, so iterating the replay alone SKIPPED it and left it wearing
            # the score that evidence gave it -- the same residue this method
            # exists to remove, in the one corner where it is total.
            for agent in before:
                trust.setdefault(agent, _TRUST_INIT)
                counts.setdefault(agent, 0)

            now = datetime.now(timezone.utc).isoformat()
            moved = {}
            for agent, score in trust.items():
                self._conn.execute(
                    '''INSERT INTO agent_trust (agent_name, trust_score,
                       runs_participated, avg_internal_score, avg_accuracy_error,
                       current_model_tier, last_updated)
                       VALUES (?, ?, 0, 0.5, NULL, ?, ?)
                       ON CONFLICT(agent_name) DO UPDATE SET
                         trust_score        = excluded.trust_score,
                         current_model_tier = excluded.current_model_tier,
                         last_updated       = excluded.last_updated''',
                    (agent, score, trust_tier(score), now))
                # The summary columns are recomputed from their sources, not
                # replayed: they are facts about the responses table and the
                # event log, so deriving them is the only way they cannot drift.
                # `runs_participated` and `avg_internal_score` do not depend on
                # grading at all, which is the separation ADR-008's reset kept;
                # `avg_accuracy_error` does, so a withdrawal correctly changes it.
                self._refresh_agent_aggregates_no_commit(agent)
                if abs(before.get(agent, _TRUST_INIT) - score) > 1e-9:
                    moved[agent] = {'before': before.get(agent, _TRUST_INIT),
                                    'after': score, 'events': counts[agent]}
            self._conn.commit()
        return moved

    def reconstruct_trust_events(self, since: Optional[str] = None) -> dict:
        """Rebuild the event log for history recorded before the log existed.

        Idempotent and refuses to run over an existing log, because rebuilding
        on top of real events would double every movement.

        Builds one `response` event per agent per run it answered in, at the
        run's timestamp and carrying the internal score on its response, plus
        one `grade` event per graded response at the run's `graded_at`. `since`
        restricts it to runs after a known-state anchor.

        **Do not use this to repair a store whose history predates the log. It
        was tried on the live store and it does not work.** Replay starts every
        agent at the neutral prior, so a reconstruction is faithful only when the
        store was genuinely at that prior at `since` and every movement after it
        is reconstructible. Neither held:

        * 1217 trust updates have happened; 623 are reconstructible. Most of the
          difference rests on grades `ADR-008` deleted as fiction, so the
          evidence is gone by design.
        * Reconstructing the whole history moved all twenty agents UP, by up to
          +0.18, pushing three across the 0.70 promotion threshold. Promotions on
          evidence that does not exist is the failure this project keeps
          retracting.
        * Anchoring at `ADR-008`'s repair, which did set every agent to exactly
          the prior, still moved all twenty by up to +0.14 and created two
          promotions. Adding back the withdrawn grades' movement to test the
          residual left a 0.28 gap, so the anchored version could not be
          validated against the live scores either.

        The finding that matters is the one underneath: **the stored trust
        scores cannot be reproduced from any surviving evidence.** That is why
        the log exists. It makes replay exact from the moment it starts, and it
        cannot reach backwards.

        **Also not recoverable, permanently:** bench recoveries from before the
        log. `recover_benched` wrote no history, so a replayed score for an agent
        that was ever benched sits further from neutral than the truth. No agent
        is currently below the 0.30 demotion threshold, which is not the same as
        knowing none ever was. `exact` is therefore never True here.
        """
        with self._lock:
            if self._conn.execute(
                    'SELECT 1 FROM trust_events LIMIT 1').fetchone() is not None:
                return {'reconstructed': 0, 'skipped': 'log already exists',
                        'exact': True}

            rows = self._conn.execute(
                '''SELECT r.run_id, r.timestamp, r.graded_at, r.actual_price_change,
                          a.agent_name, a.internal_score, a.estimate, a.id
                   FROM runs r JOIN agent_responses a ON a.run_id = r.run_id
                   WHERE (? IS NULL OR r.timestamp > ?)
                   ORDER BY r.run_id, a.id''', (since, since)).fetchall()

            final_word = {
                (r['run_id'], r['agent_name']): r['id'] for r in
                self._conn.execute(
                    'SELECT a.run_id, a.agent_name, a.id FROM agent_responses a '
                    'WHERE a.id = (SELECT b.id FROM agent_responses b '
                    '  WHERE b.run_id = a.run_id AND b.agent_name = a.agent_name '
                    '  ORDER BY b.round_num DESC, b.id DESC LIMIT 1)').fetchall()}
            events, seen_response = [], set()
            for row in rows:
                key = (row['run_id'], row['agent_name'])
                if key not in seen_response:
                    # One per agent per run: `update_trust` is called once per
                    # agent from the scores dict, not once per response row.
                    seen_response.add(key)
                    events.append((row['timestamp'], row['agent_name'], 'response',
                                   float(row['internal_score'] or _TRUST_INIT),
                                   row['run_id'],
                                   float(row['internal_score'] or _TRUST_INIT), None))
                if (row['graded_at'] and row['actual_price_change'] is not None
                        and row['estimate'] is not None
                        and row['id'] == final_word.get(
                            (row['run_id'], row['agent_name']))):
                    # Once per agent, on its FINAL WORD, matching
                    # `apply_ground_truth_grade`. It used to be once per response
                    # row, which moved a talkative agent's trust twice on one
                    # outcome and scored estimates the agent had already revised.
                    acc = compute_accuracy_score(float(row['estimate']),
                                                 float(row['actual_price_change']))
                    internal = float(row['internal_score'] or _TRUST_INIT)
                    events.append((row['graded_at'], row['agent_name'], 'grade',
                                   0.4 * internal + 0.6 * acc, row['run_id'],
                                   internal, acc))

            events.sort(key=lambda e: (e[0], e[4] or 0))
            self._conn.executemany(
                'INSERT INTO trust_events (occurred_at, agent_name, kind, raw, '
                'run_id, internal_score, accuracy_score) VALUES (?, ?, ?, ?, ?, ?, ?)',
                events)
            self._conn.commit()
        return {'reconstructed': len(events),
                'agents': len({e[1] for e in events}),
                'since': since,
                # Never claimed exact. `since` makes the START trustworthy; it
                # cannot conjure the bench decays that were never written down.
                'exact': False,
                'cannot_recover': 'bench recoveries, which wrote no history'}

    def trust_provenance(self) -> dict:
        """What the displayed trust scores rest on: counts by kind, and since when.

        The leaderboard shows a number per agent and nothing about where it came
        from, which is the same gap the forecast card had. `{'response': n,
        'grade': n, 'recovery': n, 'since': iso or None}`.
        """
        with self._lock:
            kinds = {r['kind']: r['n'] for r in self._conn.execute(
                'SELECT kind, COUNT(*) n FROM trust_events GROUP BY kind')}
            since = self._conn.execute(
                'SELECT MIN(occurred_at) FROM trust_events').fetchone()[0]
        return {'response': kinds.get('response', 0),
                'grade': kinds.get('grade', 0),
                'recovery': kinds.get('recovery', 0),
                'since': since}

    def reset_trust_to_prior(self) -> dict:
        """Return every agent to the neutral prior and start the event log clean.

        For a store whose trust predates `trust_events`. Those scores were built
        by movements that no longer have evidence behind them -- 1217 updates on
        the live store, 623 reconstructible -- and `reconstruct_trust_events`
        documents why rebuilding them produces promotions nobody earned. A score
        that cannot be derived from anything is not a measurement, and this
        project's answer to that has been consistent since `ADR-008`, which reset
        the same column for the same reason.

        What it costs is real and worth stating: the post-`ADR-008` history, four
        runs of internal-score movement, goes with it. What it buys is that every
        score from here is reproducible from the runs behind it, so "where does
        this 0.63 come from" has a complete answer.

        `runs_participated` and the quality averages are deliberately kept. They
        describe how much an agent ran and how well it wrote, which no grading
        defect touched -- the same line `ADR-008` drew.

        Returns the scores it cleared, so the correction is reported rather than
        applied silently.
        """
        with self._lock:
            before = {r['agent_name']: float(r['trust_score']) for r in
                      self._conn.execute('SELECT agent_name, trust_score '
                                         'FROM agent_trust').fetchall()}
            self._conn.execute(
                'UPDATE agent_trust SET trust_score=?, current_model_tier=?, '
                'last_updated=?',
                (_TRUST_INIT, trust_tier(_TRUST_INIT),
                 datetime.now(timezone.utc).isoformat()))
            self._conn.execute('DELETE FROM trust_events')
            self._conn.commit()
        return {'reset': before, 'to': _TRUST_INIT}

    def recover_benched(self, agent_name: str) -> float:
        """Move a benched agent's trust back toward the neutral prior.

        Trust is an EMA that only moves when an agent produces a response, so a
        benched agent's score never moves again: it is benched because its trust
        is low, and its trust stays low because it is benched. That ratchet is
        not hypothetical. Seven of twenty swarm agents were frozen by it, all
        carrying `last_updated` of 2026-07-27T14:28 while the thirteen still
        running carried the current run's timestamp, and the groups they belonged
        to shrank from five agents to three.

        Absence is not evidence of being wrong, so trust decays toward neutral
        rather than toward either extreme: the agent returns on probation, not
        vindicated. `runs_participated` is deliberately not incremented and the
        quality averages are untouched, because the agent did not run.

        Returns the new trust score.
        """
        with self._lock:
            row = self._conn.execute(
                'SELECT trust_score FROM agent_trust WHERE agent_name=?', (agent_name,)
            ).fetchone()
            if row is None:
                # Never scored, so nothing to recover from. It is already neutral.
                return _TRUST_INIT
            old = float(row['trust_score'])
            new = old + _BENCH_RECOVERY_ALPHA * (_TRUST_INIT - old)
            new = max(_TRUST_MIN, min(_TRUST_MAX, new))
            # Logged like any other movement. A decay that is not in the log
            # cannot be replayed, and a replay that silently drops it would
            # return an agent to a score it had already recovered from.
            self._conn.execute(
                'INSERT INTO trust_events (occurred_at, agent_name, kind, raw, '
                'run_id, internal_score, accuracy_score) '
                'VALUES (?, ?, ?, NULL, NULL, NULL, NULL)',
                (datetime.now(timezone.utc).isoformat(), agent_name, 'recovery'))
            self._conn.execute(
                '''UPDATE agent_trust
                   SET trust_score = ?, current_model_tier = ?, last_updated = ?
                   WHERE agent_name = ?''',
                (new, trust_tier(new), datetime.now(timezone.utc).isoformat(),
                 agent_name),
            )
            self._conn.commit()
            return new

    def close(self) -> None:
        self._conn.close()


def trust_tier(trust: float) -> str:
    """Return 'promoted', 'demoted', or 'default' for a given trust score."""
    if trust > 0.70:
        return 'promoted'
    if trust < 0.30:
        return 'demoted'
    return 'default'
