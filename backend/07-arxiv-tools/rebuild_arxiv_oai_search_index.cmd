@echo off
setlocal enabledelayedexpansion

rem Double-click launcher for rebuilding local arXiv OAI search indexes.
set "PROJECT_ROOT=%~dp0"
set "CONDA_ENV_NAME=new_rag"
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%rebuild_arxiv_oai_search_index.py" (
    echo rebuild_arxiv_oai_search_index.py was not found in "%PROJECT_ROOT%".
    exit /b 1
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

echo [%DATE% %TIME%] Rebuilding local arXiv OAI search indexes...
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Log level: %LOG_LEVEL%

python "%PROJECT_ROOT%rebuild_arxiv_oai_search_index.py" --log-level %LOG_LEVEL%

set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
echo.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
