@echo off
rem NOTE: keep this file ASCII-only. cmd.exe reads batch files in the console code page
rem (GBK on Chinese Windows), and UTF-8 Chinese text shifts its line offsets, so commands
rem after a Chinese comment get cut into fragments. tests/unit/test_windows_launchers.py
rem enforces this.
setlocal

rem ---------------------------------------------------------------------
rem Double-click launcher for incremental arXiv OAI-PMH sync (local Windows).
rem The Linux production equivalent is sync_arxiv_oai_since_last_run.sh in this folder.
rem
rem All date math, cursor handling and resume logic live in sync_arxiv_oai.py:
rem   1. Sync range = [cursor + 1 day, yesterday]; today is never included.
rem   2. Days are synced one at a time and the cursor advances after each finished day.
rem      If the run is interrupted or a day fails, the cursor stays at the last finished
rem      day and the next run resumes from the failed day.
rem   3. Missing cursor: start from the latest OAI datestamp in the local database.
rem   4. Days without records (OAI noRecordsMatch, e.g. weekends) count as success.
rem   5. Stored papers are also embedded into Milvus (arxiv_paper_embeddings). If the
rem      embedding API or Milvus fails, the day still counts as success and the cursor
rem      moves on; run backfill_arxiv_oai_embeddings.cmd later to fill the gaps.
rem      Only papers first submitted in the last ARXIV_OAI_EMBEDDING_RETENTION_MONTHS
rem      months (default 4) are embedded; after a successful sync, vectors of older
rem      papers are deleted, except papers some user liked or disliked.
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
rem Run progress and summary JSON, read by the backend /sync-status endpoint for the home page.
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
    goto :finish_with_error
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
    goto :finish_with_error
)

rem "call" is required: without it control would not return to this script
rem after the conda batch file finishes.
call "%CONDA_EXE_PATH%" activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment: %CONDA_ENV_NAME%
    goto :finish_with_error
)

rem Force traffic through the local proxy so arXiv stays reachable from CN networks.
rem Python requests picks these variables up automatically. They only affect this
rem window and its child processes; update the port here if your proxy changes.
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"

echo [%DATE% %TIME%] Running incremental arXiv OAI-PMH sync
echo [%DATE% %TIME%] State file: %STATE_FILE%
echo [%DATE% %TIME%] Metadata file: %META_FILE%
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Interval: %INTERVAL_SECONDS% seconds, Timeout: %TIMEOUT_SECONDS% seconds, MaxRetries: %MAX_RETRIES%, LogLevel: %LOG_LEVEL%

rem After activation, "python" on PATH is the interpreter of the conda env.
python "%PROJECT_ROOT%sync_arxiv_oai.py" --state-file "%STATE_FILE%" --meta-file "%META_FILE%" --interval %INTERVAL_SECONDS% --timeout %TIMEOUT_SECONDS% --max-retries %MAX_RETRIES% --log-level %LOG_LEVEL%

rem Save the exit code right away; later commands overwrite ERRORLEVEL.
rem 0 = success, 1 = a day failed or another sync is running, 2 = bad arguments, 130 = interrupted.
set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
goto :finish

:finish_with_error
set "EXIT_CODE=1"

:finish
echo.
rem A double-clicked window closes as soon as the script ends; wait so the output can be read.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
