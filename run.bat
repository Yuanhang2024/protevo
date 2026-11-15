@echo off
rem protevo one-click launcher (Windows): auto venv + deps + interactive TUI
rem Usage: double-click this file, or run  run.bat
setlocal
cd /d "%~dp0"

set "PY="
if defined PROTEVO_PY set "PY=%PROTEVO_PY%"
python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY py -3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PY=py -3"
if not defined PY python3 -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" >nul 2>&1 && set "PY=python3"
if not defined PY (
  echo [ERROR] Python 3.10+ not found. Install from https://www.python.org/downloads/
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtualenv .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo [ERROR] venv creation failed & exit /b 1 )
)
set "VPY=.venv\Scripts\python.exe"

%VPY% -c "import protevo, pandas, sklearn, scipy" >nul 2>&1
if errorlevel 1 (
  echo [2/3] Installing dependencies (first run only, may take minutes^)...
  %VPY% -m pip install -q -r requirements.txt
  if errorlevel 1 ( echo [ERROR] pip install failed, check network & exit /b 1 )
) else (
  echo [2/3] Dependencies ready, skipping install
)

echo [3/3] Starting protevo TUI (type quit to exit; help for commands^)...
%VPY% -m protevo.tui
