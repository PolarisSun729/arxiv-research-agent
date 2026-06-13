@echo off
setlocal enabledelayedexpansion

rem Double-click launcher for incremental arXiv OAI-PMH sync.
rem It remembers the last successfully synced full day in a local state file.
set "PROJECT_ROOT=%~dp0"
set "CONDA_ENV_NAME=new_rag"
set "STATE_FILE=%PROJECT_ROOT%sync_arxiv_oai_since_last_run.state"
set "META_FILE=%PROJECT_ROOT%sync_arxiv_oai_since_last_run.meta.json"
set "INTERVAL_SECONDS=5"
set "TIMEOUT_SECONDS=60"
set "MAX_RETRIES=5"
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%sync_arxiv_oai.py" (
    echo sync_arxiv_oai.py was not found in "%PROJECT_ROOT%".
    exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.AddDays(-1).ToString('yyyy-MM-dd')"') do set "YESTERDAY=%%I"
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.AddDays(-2).ToString('yyyy-MM-dd')"') do set "TWO_DAYS_AGO=%%I"
set "UNTIL_DATE=%YESTERDAY%"

set "LAST_UNTIL_DATE="
if exist "%STATE_FILE%" (
    set /p LAST_UNTIL_DATE=<"%STATE_FILE%"
)

if defined LAST_UNTIL_DATE (
    for /f %%I in ('powershell -NoProfile -Command "try { [datetime]::ParseExact('%LAST_UNTIL_DATE%', 'yyyy-MM-dd', $null).ToString('yyyy-MM-dd') } catch { '' }"') do set "LAST_UNTIL_DATE=%%I"
)

if not defined LAST_UNTIL_DATE (
    set "LAST_UNTIL_DATE=%TWO_DAYS_AGO%"
)

for /f %%I in ('powershell -NoProfile -Command "(Get-Date '%LAST_UNTIL_DATE%').AddDays(1).ToString('yyyy-MM-dd')"') do set "FROM_DATE=%%I"

if "!FROM_DATE!" GTR "!UNTIL_DATE!" (
    echo [%DATE% %TIME%] No new full day to sync.
    echo [%DATE% %TIME%] Last successful until: !LAST_UNTIL_DATE!
    echo [%DATE% %TIME%] Latest complete day: !UNTIL_DATE!
    echo.
    echo Press Enter to exit...
    set /p "USER_INPUT="
    exit /b 0
)

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

call "%CONDA_EXE_PATH%" activate %CONDA_ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda environment: %CONDA_ENV_NAME%
    exit /b 1
)

set "PYTHON_EXE=python"

rem Force traffic through the local proxy so arXiv stays reachable from CN networks.
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"

echo [%DATE% %TIME%] Running incremental arXiv OAI-PMH sync from !FROM_DATE! to !UNTIL_DATE!
echo [%DATE% %TIME%] Last successful until: !LAST_UNTIL_DATE!
echo [%DATE% %TIME%] State file: %STATE_FILE%
echo [%DATE% %TIME%] Metadata file: %META_FILE%
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Python: %PYTHON_EXE%
echo [%DATE% %TIME%] Interval: %INTERVAL_SECONDS% seconds, Timeout: %TIMEOUT_SECONDS% seconds, MaxRetries: %MAX_RETRIES%, LogLevel: %LOG_LEVEL%

"%PYTHON_EXE%" "%PROJECT_ROOT%sync_arxiv_oai.py" --from "!FROM_DATE!" --until "!UNTIL_DATE!" --meta-file "%META_FILE%" --interval %INTERVAL_SECONDS% --timeout %TIMEOUT_SECONDS% --max-retries %MAX_RETRIES% --log-level %LOG_LEVEL%

set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" (
    >"%STATE_FILE%" echo !UNTIL_DATE!
    echo [%DATE% %TIME%] Updated state file with last successful full day: !UNTIL_DATE!
)

echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
echo.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
