@echo off
REM Weekly DOE notice refresh, as run by the `DOE price notice` scheduled task.
REM
REM This exists because the task's command line could not live in Task Scheduler
REM directly. Expressed there as a cmd.exe /c one-liner with redirection, it ran,
REM reported exit code 0, and executed nothing at all -- the precise failure this
REM whole change is meant to make impossible. A script is testable by running it,
REM and its contents are in version control rather than in a registry blob.
REM
REM Two logs, for two different failure classes:
REM   logs\refresh_doe_adjustment.jsonl        structured, written by run_log.py,
REM                                            covers anything that fails once
REM                                            Python is running
REM   logs\refresh_doe_adjustment.console.log  raw stdout/stderr, the only record
REM                                            when the interpreter itself cannot
REM                                            start (a missing dependency, a
REM                                            deleted venv)
REM
REM Register it with:
REM   schtasks /Create /TN "DOE price notice" /SC WEEKLY /D TUE /ST 08:00 ^
REM     /TR "<repo>\scripts\refresh_doe_weekly.cmd"

setlocal
cd /d "%~dp0.." || exit /b 1

if not exist "logs" mkdir "logs"
set "LOG=logs\refresh_doe_adjustment.console.log"

echo. >> "%LOG%"
echo ===== [%DATE% %TIME%] refresh starting >> "%LOG%"

if not exist ".venv\Scripts\python.exe" (
    echo FATAL: .venv\Scripts\python.exe not found. The pinned interpreter is >> "%LOG%"
    echo required; see the note on never using the unpinned system Python.  >> "%LOG%"
    exit /b 2
)

".venv\Scripts\python.exe" -m ph_economic_ai.tools.refresh_doe_adjustment >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo ===== [%DATE% %TIME%] exit %RC% >> "%LOG%"
exit /b %RC%
