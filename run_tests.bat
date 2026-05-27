@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Run install.bat first.
    pause
    exit /b 1
)

uv run --no-sync afd qa tests --output test_outputs --save-debug
if errorlevel 1 (
    echo Tests failed.
    pause
    exit /b 1
)

echo Tests complete. Check test_outputs\ and test_outputs\debug\.
pause
