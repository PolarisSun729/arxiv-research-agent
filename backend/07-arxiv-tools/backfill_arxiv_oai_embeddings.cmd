@echo off
rem NOTE: keep this file ASCII-only. cmd.exe reads batch files in the console code page
rem (GBK on Chinese Windows), and UTF-8 Chinese text shifts its line offsets, so commands
rem after a Chinese comment get cut into fragments. tests/unit/test_windows_launchers.py
rem enforces this.
setlocal

rem ---------------------------------------------------------------------
rem Double-click launcher for backfilling arXiv OAI paper embeddings (local Windows).
rem
rem Runs backfill_arxiv_oai_embeddings.py, which scans arxiv_oai_papers from newest to
rem oldest and embeds every paper that has no vector in arxiv_paper_embeddings yet.
rem Only papers first submitted inside the retention window are embedded
rem (ARXIV_OAI_EMBEDDING_RETENTION_MONTHS, default 6 months of 30 days). After that,
rem vectors of older papers are deleted, except papers some user liked or disliked.
rem Papers that already have a vector are skipped, so it is safe to stop and run again.
rem This launcher passes no extra options and embeds ALL missing papers (this costs
rem embedding API money). To only count what is missing and what would be deleted,
rem or to cap one run, use the backend folder, for example:
rem   python 07-arxiv-tools\backfill_arxiv_oai_embeddings.py --dry-run
rem   python 07-arxiv-tools\backfill_arxiv_oai_embeddings.py --limit 2000
rem It shares the lock file with the OAI sync, so it exits if a sync is running.
rem Requirements: conda with an env named new_rag, and Milvus running.
rem ---------------------------------------------------------------------

rem Folder of this script, with a trailing backslash (backend\07-arxiv-tools\).
set "PROJECT_ROOT=%~dp0"
rem Conda env to activate; change it here if your env has a different name.
set "CONDA_ENV_NAME=new_rag"
rem Log level: DEBUG / INFO / WARNING / ERROR.
set "LOG_LEVEL=INFO"

if not exist "%PROJECT_ROOT%backfill_arxiv_oai_embeddings.py" (
    echo backfill_arxiv_oai_embeddings.py was not found in "%PROJECT_ROOT%".
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

echo [%DATE% %TIME%] Backfilling arXiv OAI paper embeddings...
echo [%DATE% %TIME%] Conda environment: %CONDA_ENV_NAME%
echo [%DATE% %TIME%] Log level: %LOG_LEVEL%

rem Only the embedding API and the local Milvus are called, so no arXiv proxy is set.
rem The result is printed as JSON with status success / partial / interrupted / failed.
python "%PROJECT_ROOT%backfill_arxiv_oai_embeddings.py" --log-level %LOG_LEVEL%

rem Save the exit code right away; later commands overwrite ERRORLEVEL.
rem 0 = success, 1 = some papers failed or a sync is running, 130 = interrupted.
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
