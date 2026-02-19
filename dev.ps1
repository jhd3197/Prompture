# Prompture dev launcher.
#
# Usage:
#   .\dev.ps1              # run all checks once (default)
#   .\dev.ps1 watch-lint   # watch Python files and lint on changes
#   .\dev.ps1 check        # run all checks once (alias)

param(
    [ValidateSet("check", "watch-lint")]
    [string]$Mode = "check"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyPath = "$Root\prompture"

function Invoke-AllChecks {
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] " -ForegroundColor DarkGray -NoNewline
    Write-Host "Running checks..." -ForegroundColor Cyan
    Write-Host ""

    # --- ruff check ---
    Write-Host "  ruff check      " -ForegroundColor Cyan -NoNewline
    $out = & ruff check $pyPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ok" -ForegroundColor Green
    } else {
        Write-Host "fail" -ForegroundColor Red
        $out | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }

    # --- ruff format (auto-fix) ---
    Write-Host "  ruff format     " -ForegroundColor Cyan -NoNewline
    $out = & ruff format --check $pyPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ok" -ForegroundColor Green
    } else {
        & ruff format $pyPath 2>&1 | Out-Null
        $fixed = ($out | Select-String "^Would reformat:").Count
        Write-Host "fixed $fixed files" -ForegroundColor Yellow
    }

    # --- mypy type check ---
    Write-Host "  mypy            " -ForegroundColor Cyan -NoNewline
    $out = & mypy $pyPath 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "ok" -ForegroundColor Green
    } else {
        $errCount = ($out | Select-String "^Found \d+ error").Count
        if ($errCount -gt 0) {
            $summary = ($out | Select-String "^Found \d+ error").Line
            Write-Host "fail ($summary)" -ForegroundColor Red
        } else {
            Write-Host "fail" -ForegroundColor Red
        }
        $out | Where-Object { $_ -match "error:" } | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }

    # --- pytest ---
    Write-Host "  pytest          " -ForegroundColor Cyan -NoNewline
    $out = & pytest --tb=short -q 2>&1
    if ($LASTEXITCODE -eq 0) {
        $passLine = ($out | Select-String "passed").Line
        if ($passLine) {
            Write-Host "ok ($passLine)" -ForegroundColor Green
        } else {
            Write-Host "ok" -ForegroundColor Green
        }
    } else {
        $failLine = ($out | Select-String "failed|error").Line | Select-Object -First 1
        if ($failLine) {
            Write-Host "fail ($failLine)" -ForegroundColor Red
        } else {
            Write-Host "fail" -ForegroundColor Red
        }
        $out | Where-Object { $_ -match "FAILED|ERROR" } | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    }

    Write-Host ""
}

# --- Single run for "check" mode ---
if ($Mode -eq "check") {
    Write-Host "[dev] Running all checks for Prompture" -ForegroundColor Cyan
    Write-Host "[dev]   Source: $pyPath" -ForegroundColor DarkGray
    Write-Host ""
    Invoke-AllChecks
    return
}

# --- watch-lint mode ---
Write-Host "[dev] Watching for lint + type + test errors" -ForegroundColor Cyan
Write-Host "[dev]   Source: $pyPath" -ForegroundColor DarkGray
Write-Host "[dev] Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

Invoke-AllChecks

$pyWatcher = [System.IO.FileSystemWatcher]::new($pyPath, "*.py")
$pyWatcher.IncludeSubdirectories = $true
$pyWatcher.NotifyFilter = [System.IO.NotifyFilters]::LastWrite -bor
                          [System.IO.NotifyFilters]::FileName -bor
                          [System.IO.NotifyFilters]::CreationTime
$pyWatcher.EnableRaisingEvents = $true

$lastRun = [datetime]::MinValue

$handler = {
    $now = Get-Date
    $script:lastChange = $now
}

$script:lastChange = [datetime]::MinValue

Register-ObjectEvent $pyWatcher Changed -Action $handler | Out-Null
Register-ObjectEvent $pyWatcher Created -Action $handler | Out-Null
Register-ObjectEvent $pyWatcher Renamed -Action $handler | Out-Null

try {
    while ($true) {
        Start-Sleep -Milliseconds 500
        if ($script:lastChange -ne [datetime]::MinValue -and
            ((Get-Date) - $script:lastChange).TotalMilliseconds -gt 800 -and
            $script:lastChange -ne $lastRun) {
            $lastRun = $script:lastChange
            Invoke-AllChecks
        }
    }
} finally {
    $pyWatcher.Dispose()
    Get-EventSubscriber | Unregister-Event
    Write-Host "[dev] Watcher stopped." -ForegroundColor Green
}
