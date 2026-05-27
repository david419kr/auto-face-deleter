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

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

echo Creating single Python 3.10 environment...
uv venv --python 3.10 --clear "%VENV%"
if errorlevel 1 (
    echo Failed to create .venv.
    pause
    exit /b 1
)

echo Installing CUDA PyTorch and hysts detector stack...
uv pip install --python "%PY%" pip wheel setuptools==68.2.2 numpy==1.23.5
if errorlevel 1 goto failed

uv pip install --python "%PY%" torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
if errorlevel 1 goto failed

uv pip install --python "%PY%" mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
if errorlevel 1 goto failed

uv pip install --python "%PY%" --no-build-isolation chumpy==0.70
if errorlevel 1 goto failed

uv pip install --python "%PY%" mmdet==2.28.2 mmpose==0.29.0 anime-face-detector==0.0.9
if errorlevel 1 goto failed

uv pip install --python "%PY%" --force-reinstall --no-build-isolation xtcocotools==1.14.3 numpy==1.23.5 setuptools==68.2.2
if errorlevel 1 goto failed

echo Installing Auto Face Deleter...
uv pip install --python "%PY%" -e .
if errorlevel 1 goto failed

echo Checking CUDA and warming up detector...
uv run --no-sync afd preflight --device cuda --warmup-detector
if errorlevel 1 goto failed

echo Install complete.
pause
exit /b 0

:failed
echo Install failed.
pause
exit /b 1
