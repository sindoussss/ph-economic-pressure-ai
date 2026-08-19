"""Where a scheduled refresh records what happened, so a failure leaves a trace.

The weekly `DOE price notice` task returned exit code 1 on 2026-08-19 and left
nothing behind: no message, no timestamp, no reason. It was caught only because
someone happened to be looking. Task Scheduler keeps a last result code that
nobody reads, and the app's staleness banner reports that the feed is behind
without ever saying why.

This is deliberately not the `logging` module. A weekly job wants a short,
durable, machine-readable history it can be asked questions about -- when did it
last succeed, what did it say when it broke -- not a stream of formatted lines.
One JSON object per line, newest last, capped.

Two properties are load-bearing and both are tested:

  * the wrapper re-raises. Recording a failure must never resolve it, or exit 1
    becomes exit 0 and the scheduler reports success over a rotting feed.
  * a broken log cannot break the job. If the log cannot be written the refresh
    still runs and the original error still propagates, because a diagnostic
    that takes down the thing it observes is worse than no diagnostic.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import pathlib
import time
import traceback
from typing import Iterator, Optional

#: Repository-root `logs/`. Machine state, gitignored: these records describe one
#: installation's runs and mean nothing in another checkout.
LOG_DIR = pathlib.Path(__file__).resolve().parents[2] / 'logs'

#: Records kept per tool. A weekly job reaches this in about four years, and the
#: interesting questions ("when did it last work") are answered from the tail.
MAX_RECORDS = 200

#: Characters of traceback retained. Enough to identify the failing call without
#: letting one bad run dominate the file.
_MAX_TRACEBACK = 2000


def log_path(tool: str) -> pathlib.Path:
    return LOG_DIR / f'{tool}.jsonl'


def _append(tool: str, record: dict) -> None:
    """Append one record, trimming to MAX_RECORDS.

    Separate from `logged_run` so a test can make writing fail and assert the
    job survives it.
    """
    path = log_path(tool)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, default=str) + '\n')

    rows = read_records(tool)
    if len(rows) > MAX_RECORDS:
        keep = rows[-MAX_RECORDS:]
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r, default=str) + '\n' for r in keep),
                       encoding='utf-8')
        tmp.replace(path)


def read_records(tool: str) -> list:
    """Every intact record, oldest first.

    Malformed lines are skipped rather than raised on. A run killed mid-write
    leaves a partial line, and the moment this file is read is precisely the
    moment after something went wrong -- refusing to parse then would withhold
    the history exactly when it is wanted.
    """
    path = log_path(tool)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def last_record(tool: str) -> Optional[dict]:
    rows = read_records(tool)
    return rows[-1] if rows else None


def last_failure(tool: str) -> Optional[dict]:
    """The most recent failed run, or None if none has failed."""
    for row in reversed(read_records(tool)):
        if not row.get('ok'):
            return row
    return None


def describe(tool: str) -> str:
    """A human-readable line about the last run, for `--check` output."""
    row = last_record(tool)
    if row is None:
        return f'{tool}: no run recorded yet'
    when = row.get('started_at', '?')
    if row.get('ok'):
        note = f'{tool}: last run {when} succeeded'
        failure = last_failure(tool)
        if failure:
            note += f' (last failure {failure.get("started_at")}: ' \
                    f'{failure.get("error_type")})'
        return note
    return (f'{tool}: last run {when} FAILED with '
            f'{row.get("error_type")}: {row.get("error")}')


@contextlib.contextmanager
def logged_run(tool: str, **context) -> Iterator[dict]:
    """Record the outcome of the block, then let it behave as though unwrapped.

    `context` is merged into the record, so a caller can note what the run was
    asked to do (`weeks=60`) alongside what happened.

    Catches BaseException, not Exception: KeyboardInterrupt and SystemExit do
    not inherit from Exception, and a cancelled run recorded as though it never
    happened is the same blind spot as a cancelled CI job reading as neither
    pass nor fail.
    """
    started = dt.datetime.now(dt.timezone.utc)
    clock = time.monotonic()
    record = {'tool': tool, 'started_at': started.isoformat(), **context}

    def write(extra: dict) -> None:
        record.update(extra)
        record['duration_s'] = round(time.monotonic() - clock, 3)
        # A logging failure must never mask the outcome being logged. This is
        # the only place a bare except is correct here.
        with contextlib.suppress(Exception):
            _append(tool, record)

    try:
        yield record
    except BaseException as exc:
        write({'ok': False,
               'error_type': type(exc).__name__,
               'error': str(exc) or type(exc).__name__,
               'traceback': ''.join(traceback.format_exc())[-_MAX_TRACEBACK:]})
        raise
    else:
        write({'ok': True})
