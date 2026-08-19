"""A scheduled job needs somewhere its failures land.

`DOE price notice` runs weekly on the maintainer machine. On 2026-08-19 it
returned exit code 1 and left nothing behind: no message, no timestamp, no
reason. That run happened to be under active investigation, so the cause was
found within the hour. A failure nobody was watching would have passed unnoticed
until the staleness banner appeared a week later, and the banner says the feed is
behind, never why.

The exit code is not the problem. Nobody reads exit codes from Task Scheduler.

Two properties matter more than the format:

  * **Logging must never swallow the failure.** A wrapper that records an
    exception and then returns cleanly converts a loud failure into a silent
    one, which is worse than no logging at all. The exception is re-raised and
    the exit code preserved.
  * **A broken log must not break the job.** If the log cannot be written, the
    refresh still runs and the original error still propagates. The log is a
    diagnostic, and a diagnostic that can take down the thing it observes is a
    liability.
"""
import json

import pytest

from ph_economic_ai.tools import run_log


@pytest.fixture(autouse=True)
def _isolated_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(run_log, 'LOG_DIR', tmp_path / 'logs')


def test_a_successful_run_is_recorded():
    with run_log.logged_run('demo', weeks=6):
        pass
    rows = run_log.read_records('demo')
    assert len(rows) == 1
    assert rows[0]['ok'] is True
    assert rows[0]['weeks'] == 6
    assert rows[0]['tool'] == 'demo'
    assert 'started_at' in rows[0] and 'duration_s' in rows[0]


def test_a_failing_run_is_recorded_with_the_reason():
    with pytest.raises(RuntimeError, match='no notice'):
        with run_log.logged_run('demo'):
            raise RuntimeError('no notice for this week')

    row, = run_log.read_records('demo')
    assert row['ok'] is False
    assert row['error_type'] == 'RuntimeError'
    assert 'no notice for this week' in row['error']
    assert 'raise RuntimeError' in row['traceback'], (
        'the traceback is the part that actually shortens a diagnosis')


def test_the_failure_is_re_raised_rather_than_swallowed():
    """The property that makes this safe to wrap a scheduled job in.

    A logger that catches an exception and returns turns exit 1 into exit 0, so
    the scheduler reports success and the feed silently rots. Recording a failure
    must not resolve it.
    """
    with pytest.raises(ZeroDivisionError):
        with run_log.logged_run('demo'):
            1 / 0
    assert run_log.read_records('demo')[0]['ok'] is False


def test_a_keyboard_interrupt_is_recorded_and_propagates():
    """KeyboardInterrupt and SystemExit do not inherit from Exception.

    Catching only Exception would record a cancelled run as though it never
    happened, which is the same blind spot as a cancelled CI job reading as
    neither pass nor fail.
    """
    with pytest.raises(KeyboardInterrupt):
        with run_log.logged_run('demo'):
            raise KeyboardInterrupt
    row, = run_log.read_records('demo')
    assert row['ok'] is False
    assert row['error_type'] == 'KeyboardInterrupt'


def test_an_unwritable_log_does_not_break_the_job(monkeypatch):
    """The log is a diagnostic. A diagnostic must not take down what it observes."""
    def boom(*a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(run_log, '_append', boom)

    with run_log.logged_run('demo'):        # success path survives
        pass

    with pytest.raises(RuntimeError, match='real failure'):
        with run_log.logged_run('demo'):
            raise RuntimeError('real failure')


def test_the_original_error_survives_a_logging_error(monkeypatch):
    """If both fail, the caller must see the REAL failure, not the log's."""
    monkeypatch.setattr(run_log, '_append',
                        lambda *a, **k: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(RuntimeError) as excinfo:
        with run_log.logged_run('demo'):
            raise RuntimeError('the thing that actually went wrong')
    assert 'actually went wrong' in str(excinfo.value)


def test_last_failure_finds_the_most_recent_one():
    with pytest.raises(RuntimeError):
        with run_log.logged_run('demo'):
            raise RuntimeError('first')
    with run_log.logged_run('demo'):
        pass
    with pytest.raises(RuntimeError):
        with run_log.logged_run('demo'):
            raise RuntimeError('second')
    with run_log.logged_run('demo'):
        pass

    assert run_log.last_failure('demo')['error'].startswith('second')
    assert run_log.last_record('demo')['ok'] is True


def test_no_failure_yet_reads_as_none():
    with run_log.logged_run('demo'):
        pass
    assert run_log.last_failure('demo') is None


def test_the_log_is_bounded():
    """A weekly job outlives the machine. An unbounded log is a slow leak."""
    for _ in range(run_log.MAX_RECORDS + 25):
        with run_log.logged_run('demo'):
            pass
    rows = run_log.read_records('demo')
    assert len(rows) == run_log.MAX_RECORDS


def test_a_corrupt_line_does_not_break_reading():
    """A run killed mid-write leaves a partial line. Reading must survive it,
    because the moment the log matters is right after something went wrong."""
    with run_log.logged_run('demo'):
        pass
    path = run_log.log_path('demo')
    with path.open('a', encoding='utf-8') as fh:
        fh.write('{"ok": true, "trunc\n')
    with run_log.logged_run('demo'):
        pass

    rows = run_log.read_records('demo')
    assert len(rows) == 2, 'the two intact records survive the corrupt one'
    assert all(isinstance(r, dict) for r in rows)


def test_records_are_one_json_object_per_line():
    with run_log.logged_run('demo'):
        pass
    text = run_log.log_path('demo').read_text(encoding='utf-8')
    assert text.endswith('\n')
    assert json.loads(text.strip())['tool'] == 'demo'


def test_the_log_directory_is_not_committed():
    """Machine state, not repository content. `logs/` must stay ignored."""
    import pathlib
    root = pathlib.Path(run_log.__file__).resolve().parents[2]
    ignore = (root / '.gitignore').read_text(encoding='utf-8')
    assert any(line.strip() in ('logs/', '/logs/', 'logs')
               for line in ignore.splitlines()), (
        'logs/ must be gitignored: run records are machine state')


# ── The half that makes the log worth writing ────────────────────────────────

def test_check_fails_while_the_last_run_is_broken(monkeypatch, capsys):
    """`--check` must answer "is this job healthy", not only "is the feed old".

    The two come apart for a full week. If the refresh starts crashing on a
    Tuesday, the committed feed stays inside its seven-day staleness window
    until the following Tuesday, so a staleness-only check reports success over
    a job that has already stopped working. That week of silence is the exact
    gap this logging exists to close.
    """
    from ph_economic_ai.tools import refresh_doe_adjustment as tool

    with pytest.raises(RuntimeError):
        with run_log.logged_run(tool.TOOL):
            raise RuntimeError('503 from the document server')

    assert tool.check() == 1, (
        'a broken refresh must fail the check even while the feed is current')
    assert 'FAILED' in capsys.readouterr().out


def test_check_passes_once_a_later_run_succeeds(capsys):
    """Only the MOST RECENT run decides health. A failure that has since been
    fixed is history, not an outstanding alarm."""
    from ph_economic_ai.tools import refresh_doe_adjustment as tool

    with pytest.raises(RuntimeError):
        with run_log.logged_run(tool.TOOL):
            raise RuntimeError('transient')
    with run_log.logged_run(tool.TOOL):
        pass

    assert tool.check() == 0
    capsys.readouterr()
