param(
    [string]$ProjectRoot = $PSScriptRoot,
    [string]$PythonExe = "python",
    [int]$DaysBack = 1,
    [switch]$DryRun,
    [double]$IntervalSeconds = 5.0,
    [double]$TimeoutSeconds = 60.0,
    [int]$MaxRetries = 5,
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"
$CondaEnvName = "new_rag"

if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "ProjectRoot not found: $ProjectRoot"
}

$scriptPath = Join-Path $ProjectRoot "sync_arxiv_oai.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "sync_arxiv_oai.py not found at: $scriptPath"
}

$condaCommand = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCommand) {
    throw "conda was not found on PATH."
}

& $condaCommand.Source "shell.powershell" "hook" | Out-String | Invoke-Expression
conda activate $CondaEnvName

# OAI-PMH datestamps are date-based, so this uses a calendar-day window:
# from yesterday 00:00 to today 00:00 by default.
$untilDate = (Get-Date).Date
$fromDate = $untilDate.AddDays(-1 * [Math]::Abs($DaysBack))

$from = $fromDate.ToString("yyyy-MM-dd")
$until = $untilDate.ToString("yyyy-MM-dd")

$arguments = @(
    $scriptPath,
    "--from", $from,
    "--until", $until,
    "--interval", $IntervalSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--timeout", $TimeoutSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--max-retries", $MaxRetries.ToString(),
    "--log-level", $LogLevel
)

if ($DryRun.IsPresent) {
    $arguments += "--dry-run"
}

Write-Host ("[{0}] Running arXiv OAI-PMH sync for {1} -> {2}" -f (Get-Date), $from, $until)
Write-Host ("[{0}] Conda environment: {1}" -f (Get-Date), $CondaEnvName)
Write-Host ("[{0}] Command: {1} {2}" -f (Get-Date), $PythonExe, ($arguments -join " "))

& $PythonExe @arguments
exit $LASTEXITCODE
