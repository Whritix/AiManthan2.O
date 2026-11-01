<#
.SYNOPSIS
    Inspectra - One-Command Demo Launcher (PowerShell)
.DESCRIPTION
    Launches the real-time surface inspection pipeline and web UI.
    Defaults to profile=steel, source=dataset, detector=auto.
#>

[CmdletBinding()]
param (
    [string]$Profile = "steel",
    [string]$Source = "dataset",
    [string]$Detector = "auto",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Host ""
Write-Host " ===============================================================" -ForegroundColor Cyan
Write-Host "           INSPECTRA - Real-Time Defect Detection           " -ForegroundColor Cyan
Write-Host "     YOLO11s + PatchCore-lite + Novelty Gate + Fusion Policy     " -ForegroundColor Cyan
Write-Host " ===============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Kill stale process on port
Write-Host "[1/5] Checking port $Port..." -ForegroundColor Yellow
$proc = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($proc) {
    Write-Host "      Freeing port $Port (PID $proc)..." -ForegroundColor Gray
    Stop-Process -Id $proc -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 2. Check Python modules
Write-Host "[2/5] Verifying Python environment..." -ForegroundColor Yellow
$pyCheck = python -c "import fastapi, uvicorn, cv2, numpy, pydantic, torch; print('OK')" 2>$null
if ($LASTEXITCODE -ne 0 -or $pyCheck -notmatch "OK") {
    Write-Error "Missing required Python packages. Run: pip install -r requirements.txt"
}
Write-Host "      Python dependencies verified." -ForegroundColor Green

# 3. Check Web distribution
Write-Host "[3/5] Verifying Web frontend build..." -ForegroundColor Yellow
$distIndex = Join-Path $ProjectRoot "web\dist\index.html"
if (-not (Test-Path $distIndex)) {
    Write-Host "      Web dist missing. Building now..." -ForegroundColor Yellow
    Push-Location (Join-Path $ProjectRoot "web")
    npm run build
    Pop-Location
    if (-not (Test-Path $distIndex)) {
        Write-Error "Failed to build web UI. Please check web build logs."
    }
}
Write-Host "      Web assets ready." -ForegroundColor Green

# 4. Configuration
Write-Host "[4/5] Pipeline Configuration:" -ForegroundColor Yellow
Write-Host "      Profile  : $Profile" -ForegroundColor Gray
Write-Host "      Source   : $Source" -ForegroundColor Gray
Write-Host "      Detector : $Detector" -ForegroundColor Gray
Write-Host "      URL      : http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "      API Docs : http://127.0.0.1:$Port/docs" -ForegroundColor Cyan

$env:INDORE_PROFILE = $Profile
$env:INDORE_SOURCE = $Source
$env:INDORE_DETECTOR = $Detector

# 5. Launch
Write-Host "[5/5] Starting Uvicorn server..." -ForegroundColor Green
Write-Host ""
Write-Host "  =============================================================" -ForegroundColor Green
Write-Host "  >> Cell active at http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "  >> Press Ctrl+C to terminate." -ForegroundColor Green
Write-Host "  =============================================================" -ForegroundColor Green
Write-Host ""

python -m uvicorn server.app.main:app --host 127.0.0.1 --port $Port --no-access-log
