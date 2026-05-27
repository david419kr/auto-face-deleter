@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo uv not found. Installing uv with the official Windows installer...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%APPDATA%\uv\bin;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo Failed to find uv after installation. Open a new terminal and run install.bat again.
    pause
    exit /b 1
)

echo Syncing Python environment...
uv sync
if errorlevel 1 (
    echo uv sync failed.
    pause
    exit /b 1
)

echo Downloading models and checking CUDA...
uv run afd preflight --download-models --device cuda
if errorlevel 1 (
    echo Preflight failed.
    pause
    exit /b 1
)

echo Install complete.
pause
