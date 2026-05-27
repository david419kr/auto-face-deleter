# Auto Face Deleter

Windows + NVIDIA CUDA 환경에서 anime풍 일러스트의 얼굴 특징을 로컬에서 자동 감지하고 지우는 Python 도구입니다. 목표 출력은 `examples/*_faceless.png`처럼 눈, 코, 입, 홍조, 눈물 같은 얼굴 특징을 지우고 머리카락, 귀, 옷, 배경은 최대한 보존하는 것입니다.

## 설치

1. NVIDIA 드라이버가 설치되어 있고 `nvidia-smi`가 동작하는지 확인합니다.
2. 이 폴더에서 `install.bat`를 실행합니다.
3. 설치 스크립트는 `uv`가 없으면 공식 Windows installer로 설치하고, `.venv`를 만든 뒤 CUDA PyTorch와 필요한 패키지를 설치합니다.
4. 마지막에 detector와 Anime-LaMa 모델을 `models/`에 다운로드하고 `torch.cuda.is_available()`을 확인합니다.

설치 중 다운로드되는 모델:

- `models/yolov8x6_animeface.pt`: `Fuyucchi/yolov8_animeface`
- `models/anime-manga-big-lama.pt`: IOPaint/Sanster Anime-Manga LaMa 계열 모델

## 실행

이미지를 `input/` 폴더에 넣고 `run.bat`를 실행하면 `output/`에 결과가 저장됩니다.

파일이나 폴더를 `run.bat` 위로 드래그앤드롭해도 됩니다. 여러 파일을 동시에 드래그하면 순서대로 처리합니다.

CLI 직접 실행:

```bat
uv run afd process tests --output test_outputs --recursive --backend hybrid --device cuda --save-debug
```

단일 파일:

```bat
uv run afd process tests\01_reika_original.png --output output\01_reika_faceless.png --backend hybrid --device cuda --save-debug
```

빠른 로컬 검증:

```bat
run_tests.bat
```

## 주요 옵션

- `--backend hybrid`: 기본값. Anime-LaMa inpaint 후 피부색 기반 smoothing을 적용합니다.
- `--backend skinfill`: Anime-LaMa 없이 OpenCV inpaint와 피부색 smoothing만 사용합니다.
- `--device cuda`: 기본값. CUDA가 없으면 실패합니다. 조용히 CPU로 폴백하지 않습니다.
- `--device cpu`: CPU 실행을 명시적으로 허용할 때만 사용합니다.
- `--aggression low|normal|high`: 얼굴 특징 삭제 강도입니다.
- `--mask-dilate N`: 마스크를 추가로 넓힙니다.
- `--feather N`: 얼굴 smoothing 경계 feather 크기입니다.
- `--max-faces N`: 이미지당 처리할 최대 얼굴 수입니다.
- `--save-debug`: `output/debug/`에 bbox, raw mask, refined mask, erase mask, crop before/after를 저장합니다.

## 모델과 로컬-only 범위

설치 및 `afd models download` 단계에서는 Hugging Face와 GitHub에서 오픈소스 모델 파일을 다운로드합니다. 실제 `afd process`, `run.bat`, `run_tests.bat` 실행 중 얼굴 감지와 얼굴 삭제는 외부 AI API 없이 로컬 Python/PyTorch/OpenCV에서만 수행합니다.

모델을 다시 받으려면:

```bat
uv run afd models download --force
```

사전 점검:

```bat
uv run afd preflight --device cuda
```

## 테스트와 품질 확인

`examples/`에는 원본과 목표 faceless 결과 쌍이 있습니다. 회귀 비교용으로 다음을 실행할 수 있습니다.

```bat
uv run afd qa examples --output example_outputs --backend hybrid --device cuda --save-debug
```

`tests/`에는 원본 입력 이미지가 들어 있습니다.

```bat
uv run afd qa tests --output test_outputs --backend hybrid --device cuda --save-debug
```

결과가 과하게 지워지면 `--aggression low --mask-dilate 0 --feather 8`을 사용하고, 눈/입 흔적이 남으면 `--aggression high --mask-dilate 1`을 사용합니다.

## 라이선스 주의

기본 detector 모델 `Fuyucchi/yolov8_animeface`는 AGPL-3.0 라이선스로 배포됩니다. 개인/내부 사용을 기준으로 선택한 기본값이며, 배포 제품에 포함하려면 AGPL 의무를 검토해야 합니다. 이 저장소는 모델 가중치를 커밋하지 않고 설치 시 다운로드합니다.

## 문제 해결

CUDA 확인:

```bat
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`CUDA was requested, but torch.cuda.is_available() is false`가 나오면 CUDA wheel이 제대로 설치되지 않았거나 현재 터미널이 다른 Python 환경을 보고 있는 것입니다. `install.bat`를 다시 실행하고, 그래도 실패하면 `.venv`를 삭제한 뒤 다시 설치하세요.

모델 파일이 없다는 오류가 나오면:

```bat
uv run afd models download
```
