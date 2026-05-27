@echo off
setlocal enabledelayedexpansion

rem Double-click launcher for the last 6 months arXiv OAI-PMH sync.
rem Edit these values if you want to change the default behavior.
set "PROJECT_ROOT=%~dp0"
set "CONDA_ENV_NAME=new_rag"
set "MONTHS_BACK=6"
set "DRY_RUN=0"
set "INTERVAL_SECONDS=5"
set "TIMEOUT_SECONDS=60"
set "MAX_RETRIES=5"
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%sync_arxiv_oai.py" (
    echo sync_arxiv_oai.py was not found in "%PROJECT_ROOT%".
    exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.AddMonths(-%MONTHS_BACK%).ToString('yyyy-MM-dd')"') do set "FROM_DATE=%%I"
for /f %%I in ('powershell -NoProfile -Command "(Get-Date).Date.ToString('yyyy-MM-dd')"') do set "UNTIL_DATE=%%I"

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
echo [%DATE% %TIME%] Running arXiv OAI-PMH sync from !FROM_DATE! to !UNTIL_DATE!
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Python: %PYTHON_EXE%
echo [%DATE% %TIME%] Interval: %INTERVAL_SECONDS% seconds, Timeout: %TIMEOUT_SECONDS% seconds, MaxRetries: %MAX_RETRIES%, LogLevel: %LOG_LEVEL%

if "%DRY_RUN%"=="1" (
    echo [%DATE% %TIME%] Dry run is enabled.
    "%PYTHON_EXE%" "%PROJECT_ROOT%sync_arxiv_oai.py" --from "!FROM_DATE!" --until "!UNTIL_DATE!" --dry-run --interval %INTERVAL_SECONDS% --timeout %TIMEOUT_SECONDS% --max-retries %MAX_RETRIES% --log-level %LOG_LEVEL%
) else (
    "%PYTHON_EXE%" "%PROJECT_ROOT%sync_arxiv_oai.py" --from "!FROM_DATE!" --until "!UNTIL_DATE!" --interval %INTERVAL_SECONDS% --timeout %TIMEOUT_SECONDS% --max-retries %MAX_RETRIES% --log-level %LOG_LEVEL%
)

set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
echo.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
