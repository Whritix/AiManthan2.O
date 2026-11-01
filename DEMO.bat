@echo off
setlocal enabledelayedexpansion
title INSPECTRA — DEMO LAUNCHER

echo.
echo  ╔═══════════════════════════════════════════════════════════════╗
echo  ║         INSPECTRA — Real-Time Defect Detection          ║
echo  ║    YOLO11s + PatchCore-lite + Novelty Gate + Fusion          ║
echo  ╚═══════════════════════════════════════════════════════════════╝
echo.

:: ── locate project root ───────────────────────────────────────────────────
set "INDORE_ROOT=%~dp0"
if "%INDORE_ROOT:~-1%"=="\" set "INDORE_ROOT=%INDORE_ROOT:~0,-1%"
cd /d "%INDORE_ROOT%"

:: ── default to the 21-class model used by the five-product console ────────
if not defined INDORE_DETECTOR set "INDORE_DETECTOR=unified"

:: ── kill any stale server on port 8000 ───────────────────────────────────
echo [1/5] Clearing port 8000...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: ── verify Python deps ────────────────────────────────────────────────────
echo [2/5] Verifying Python dependencies...
python -c "import fastapi, uvicorn, cv2, numpy, pydantic, torch; print('  OK')" 2>nul
if errorlevel 1 (
    echo   ERROR: Missing Python packages.
    echo   Run:  pip install fastapi uvicorn opencv-python numpy pydantic torch
    pause
    exit /b 1
)

:: ── check web dist ────────────────────────────────────────────────────────
echo [3/5] Checking web build...
if not exist "%INDORE_ROOT%\web\dist\index.html" (
    echo   Web build missing — building now...
    cd /d "%INDORE_ROOT%\web"
    call npm run build >nul 2>&1
    if errorlevel 1 (
        echo   ERROR: npm run build failed.
        echo   Run:  cd web ^&^& npm install ^&^& npm run build
        pause
        exit /b 1
    )
    cd /d "%INDORE_ROOT%"
    echo   OK — built.
) else (
    echo   OK — pre-built.
)

:: ── print config ──────────────────────────────────────────────────────────
echo [4/5] Configuration:
echo   Profile  : %INDORE_PROFILE%  (default: steel)
echo   Source   : %INDORE_SOURCE%   (default: dataset)
echo   Detector : %INDORE_DETECTOR% (default: unified)
echo   API      : http://127.0.0.1:8000
echo   UI       : http://127.0.0.1:8000  (served from /dist)
echo.

:: ── launch ────────────────────────────────────────────────────────────────
echo [5/5] Starting Inspectra server...
echo.
echo  ┌─────────────────────────────────────────────────────────────┐
echo  │  READY:  http://127.0.0.1:8000                             │
echo  │  API:    http://127.0.0.1:8000/docs                        │
echo  │  Press Ctrl+C to stop.                                     │
echo  └─────────────────────────────────────────────────────────────╘
echo.

python -m uvicorn server.app.main:app --host 127.0.0.1 --port 8000 --no-access-log

pause
