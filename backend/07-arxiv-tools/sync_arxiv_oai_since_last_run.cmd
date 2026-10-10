@echo off
rem NOTE: keep this file ASCII-only. cmd.exe reads batch files in the console code page
rem (GBK on Chinese Windows), and UTF-8 Chinese text shifts its line offsets, so commands
rem after a Chinese comment get cut into fragments. tests/unit/test_windows_launchers.py
rem enforces this.

rem Enable delayed expansion so that variables assigned inside an if/for block can be
rem read back in the same block with the !VAR! syntax (percent syntax is fixed when
rem the whole block is parsed).
setlocal enabledelayedexpansion

rem ---------------------------------------------------------------------
rem Double-click launcher for incremental arXiv OAI-PMH sync (local Windows).
rem The Linux production equivalent is sync_arxiv_oai_since_last_run.sh in this folder.
rem
rem How it works:
rem   1. Read the cursor file: the last full day that was synced successfully.
rem   2. Sync range = [cursor + 1 day, yesterday]. Only complete days are synced,
rem      today is never included. If several days were missed they are caught up at once.
rem   3. Activate the conda env and run sync_arxiv_oai.py through the local proxy.
rem   4. Advance the cursor to yesterday only when the sync exits with code 0;
rem      on failure the cursor stays put and the same range is retried next time.
rem
rem Requirements: conda with an env named new_rag, and a local HTTP proxy on
rem 127.0.0.1:7897. The script waits for Enter before closing, so it is not meant
rem to run unattended from Task Scheduler.
rem ---------------------------------------------------------------------

rem Folder of this script, with a trailing backslash (backend\07-arxiv-tools\).
set "PROJECT_ROOT=%~dp0"
rem Conda env to activate; change it here if your env has a different name.
set "CONDA_ENV_NAME=new_rag"
rem Cursor and summary live in backend\data\arxiv-oai-sync. In production releases
rem backend/data is a link to shared/backend/data, so they survive redeploys.
set "STATE_DIR=%PROJECT_ROOT%..\data\arxiv-oai-sync"
rem Cursor file: a single line YYYY-MM-DD, the last day synced successfully (inclusive).
set "STATE_FILE=%STATE_DIR%\sync_arxiv_oai_since_last_run.state"
rem Run summary JSON, read by the backend /sync-status endpoint for the home page.
set "META_FILE=%STATE_DIR%\sync_arxiv_oai_since_last_run.meta.json"
rem The options below are passed straight to sync_arxiv_oai.py (see its --help).
rem Seconds between requests; the service enforces a minimum of 3.5.
set "INTERVAL_SECONDS=5"
rem HTTP timeout per request, in seconds.
set "TIMEOUT_SECONDS=60"
rem Maximum attempts per page request.
set "MAX_RETRIES=5"
rem Log level: DEBUG / INFO / WARNING / ERROR.
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%sync_arxiv_oai.py" (
    echo sync_arxiv_oai.py was not found in "%PROJECT_ROOT%".
    exit /b 1
)
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%"

rem Batch has no date arithmetic, so PowerShell computes the dates as yyyy-MM-dd.
rem Yesterday is the end of the sync range; two days ago is the default cursor.
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.AddDays(-1).ToString('yyyy-MM-dd')"') do set "YESTERDAY=%%I"
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.AddDays(-2).ToString('yyyy-MM-dd')"') do set "TWO_DAYS_AGO=%%I"
set "UNTIL_DATE=%YESTERDAY%"

rem Read the first line of the cursor file, if it exists.
set "LAST_UNTIL_DATE="
if exist "%STATE_FILE%" (
    set /p LAST_UNTIL_DATE=<"%STATE_FILE%"
)

rem Validate the cursor as a real yyyy-MM-dd date (2026-02-30 is rejected, for example).
rem On failure PowerShell prints an empty line, and for /f skips empty lines without
rem running the set, so the result goes into a cleared temp variable first and is then
rem copied back. A valid cursor comes back normalized; an invalid one leaves the temp
rem variable empty, and setting an empty value deletes LAST_UNTIL_DATE so that the
rem "not defined" fallback below kicks in.
if defined LAST_UNTIL_DATE (
    set "VALID_UNTIL_DATE="
    for /f %%I in ('powershell -NoProfile -Command "try { [datetime]::ParseExact('%LAST_UNTIL_DATE%', 'yyyy-MM-dd', $null).ToString('yyyy-MM-dd') } catch { '' }"') do set "VALID_UNTIL_DATE=%%I"
    set "LAST_UNTIL_DATE=!VALID_UNTIL_DATE!"
)

rem Missing, empty or invalid cursor: treat it as "synced up to two days ago", so the
rem first run only syncs yesterday instead of crawling a large history. To backfill
rem older data, run sync_arxiv_oai.py by hand with explicit --from / --until.
if not defined LAST_UNTIL_DATE (
    set "LAST_UNTIL_DATE=%TWO_DAYS_AGO%"
)

rem Start of the range = the day after the cursor.
for /f %%I in ('powershell -NoProfile -Command "(Get-Date '%LAST_UNTIL_DATE%').AddDays(1).ToString('yyyy-MM-dd')"') do set "FROM_DATE=%%I"

rem yyyy-MM-dd strings compare the same way as the dates they represent. A start date
rem after yesterday means everything is already synced.
if "!FROM_DATE!" GTR "!UNTIL_DATE!" (
    echo [%DATE% %TIME%] No new full day to sync.
    echo [%DATE% %TIME%] Last successful until: !LAST_UNTIL_DATE!
    echo [%DATE% %TIME%] Latest complete day: !UNTIL_DATE!
    echo.
    echo Press Enter to exit...
    set /p "USER_INPUT="
    exit /b 0
)

rem Locate conda: prefer the CONDA_EXE variable that conda sets itself,
rem otherwise take the first hit of "where conda" on PATH.
set "CONDA_EXE_PATH="
if defined CONDA_EXE (
    set "CONDA_EXE_PATH=%CONDA_EXE%"
) else (
    for /f "delims=" %%I in ('where conda 2^>nul') do (
        set "CONDA_EXE_PATH=%%I"
        goto :conda_found
    )
)
:conda_found
if not defined CONDA_EXE_PATH (
    echo conda was not found on PATH, and CONDA_EXE is not set.
    exit /b 1
)

rem "call" is required: without it control would not return to this script
rem after the conda batch file finishes.
call "%CONDA_EXE_PATH%" activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment: %CONDA_ENV_NAME%
    exit /b 1
)

rem After activation, "python" on PATH is the interpreter of the conda env.
set "PYTHON_EXE=python"

rem Force traffic through the local proxy so arXiv stays reachable from CN networks.
rem Python requests picks these variables up automatically. They only affect this
rem window and its child processes; update the port here if your proxy changes.
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"

echo [%DATE% %TIME%] Running incremental arXiv OAI-PMH sync from !FROM_DATE! to !UNTIL_DATE!
echo [%DATE% %TIME%] Last successful until: !LAST_UNTIL_DATE!
echo [%DATE% %TIME%] State file: %STATE_FILE%
echo [%DATE% %TIME%] Metadata file: %META_FILE%
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Python: %PYTHON_EXE%
echo [%DATE% %TIME%] Interval: %INTERVAL_SECONDS% seconds, Timeout: %TIMEOUT_SECONDS% seconds, MaxRetries: %MAX_RETRIES%, LogLevel: %LOG_LEVEL%

rem Run the sync: papers go into the local OAI database, the run summary into META_FILE.
"%PYTHON_EXE%" "%PROJECT_ROOT%sync_arxiv_oai.py" --from "!FROM_DATE!" --until "!UNTIL_DATE!" --meta-file "%META_FILE%" --interval %INTERVAL_SECONDS% --timeout %TIMEOUT_SECONDS% --max-retries %MAX_RETRIES% --log-level %LOG_LEVEL%

rem Save the exit code right away; later commands overwrite ERRORLEVEL.
set "EXIT_CODE=%ERRORLEVEL%"
rem Advance the cursor only on success; on failure the same range is retried next run.
if "%EXIT_CODE%"=="0" (
    >"%STATE_FILE%" echo !UNTIL_DATE!
    echo [%DATE% %TIME%] Updated state file with last successful full day: !UNTIL_DATE!
)

echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
echo.
rem A double-clicked window closes as soon as the script ends; wait so the output can be read.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
