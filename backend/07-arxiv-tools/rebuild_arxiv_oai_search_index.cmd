@echo off
rem NOTE: keep this file ASCII-only. cmd.exe reads batch files in the console code page
rem (GBK on Chinese Windows), and UTF-8 Chinese text shifts its line offsets, so commands
rem after a Chinese comment get cut into fragments. tests/unit/test_windows_launchers.py
rem enforces this.

rem Enable delayed expansion, consistent with the other launchers in this folder.
setlocal enabledelayedexpansion

rem ---------------------------------------------------------------------
rem Double-click launcher for rebuilding local arXiv OAI search indexes (local Windows).
rem
rem Activates the conda env and runs rebuild_arxiv_oai_search_index.py, which rebuilds the
rem FTS5 full-text index and the category index from the arxiv_oai_papers table.
rem The daily incremental sync keeps these indexes up to date, so this is only needed
rem after a first bulk import, after an index format upgrade, or when local search
rem reports that the index is missing or a rebuild failed.
rem
rem Default behavior: resume from the checkpoint if the previous run failed or was
rem interrupted, otherwise clear the indexes and rebuild from scratch.
rem This launcher passes no extra options. For --status (show progress) or --reset
rem (force a full rebuild), run from the backend folder, for example:
rem   python 07-arxiv-tools\rebuild_arxiv_oai_search_index.py --status
rem Stop the backend first, otherwise SQLite may be locked (error: database_locked).
rem Requirements: conda with an env named new_rag.
rem ---------------------------------------------------------------------

rem Folder of this script, with a trailing backslash (backend\07-arxiv-tools\).
set "PROJECT_ROOT=%~dp0"
rem Conda env to activate; change it here if your env has a different name.
set "CONDA_ENV_NAME=new_rag"
rem Log level: DEBUG / INFO / WARNING / ERROR.
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%rebuild_arxiv_oai_search_index.py" (
    echo rebuild_arxiv_oai_search_index.py was not found in "%PROJECT_ROOT%".
    exit /b 1
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

echo [%DATE% %TIME%] Rebuilding local arXiv OAI search indexes...
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Log level: %LOG_LEVEL%

rem The rebuild only touches the local SQLite database, so no proxy is needed.
rem The result is printed as JSON with status "success" or "failed".
python "%PROJECT_ROOT%rebuild_arxiv_oai_search_index.py" --log-level %LOG_LEVEL%

rem Save the exit code right away; later commands overwrite ERRORLEVEL.
set "EXIT_CODE=%ERRORLEVEL%"
echo [%DATE% %TIME%] Finished with exit code %EXIT_CODE%
echo.
rem A double-clicked window closes as soon as the script ends; wait so the output can be read.
echo Press Enter to exit...
set /p "USER_INPUT="
exit /b %EXIT_CODE%
