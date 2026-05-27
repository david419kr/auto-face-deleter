@echo off
setlocal

cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Run install.bat first.
  exit /b 1
)

set "VENV=.hysts-venv"
set "PY=%VENV%\Scripts\python.exe"

uv venv --python 3.10 "%VENV%" || exit /b 1

uv pip install --python "%PY%" pip wheel setuptools==68.2.2 numpy==1.23.5 || exit /b 1
uv pip install --python "%PY%" torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117 || exit /b 1
uv pip install --python "%PY%" mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html || exit /b 1
uv pip install --python "%PY%" --no-build-isolation chumpy==0.70 || exit /b 1
uv pip install --python "%PY%" mmdet==2.28.2 mmpose==0.29.0 anime-face-detector==0.0.9 || exit /b 1
uv pip install --python "%PY%" --force-reinstall --no-build-isolation xtcocotools==1.14.3 numpy==1.23.5 setuptools==68.2.2 || exit /b 1

"%PY%" -c "import torch, numpy, mmcv, mmdet, mmpose, anime_face_detector; print('hysts probe env ok', torch.__version__, torch.cuda.is_available(), numpy.__version__, mmcv.__version__)" || exit /b 1
"%PY%" -c "import anime_face_detector; anime_face_detector.create_detector('yolov3', device='cuda:0'); print('hysts detector ready')" || exit /b 1

echo.
echo Hysts probe environment is ready.
echo Run:
echo   uv run afd qa examples --output example_outputs --save-debug
