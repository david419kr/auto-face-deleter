# Auto Face Deleter

Windows + NVIDIA CUDA 환경에서 anime풍 일러스트의 얼굴 특징을 로컬에서 자동 감지하고 지우는 Python 도구입니다. 현재 기본 파이프라인은 `hysts/anime-face-detector` 랜드마크 검출 후 OpenCV 기반 non-LaMa prefill을 적용합니다.

목표는 `examples/*_faceless.png`처럼 얼굴형은 유지하면서 눈, 눈썹, 코, 입, 홍조, 눈물 같은 얼굴 특징을 지우는 것입니다. 최악의 경우 머리카락 일부가 바뀌는 것보다 얼굴 특징이 남는 것을 더 큰 실패로 봅니다.

## 설치

1. NVIDIA 드라이버가 설치되어 있고 `nvidia-smi`가 동작하는지 확인합니다.
2. 이 폴더에서 `install.bat`를 실행합니다.
3. 설치 스크립트는 `uv`가 없으면 공식 Windows installer로 설치합니다.
4. 메인 `.venv`와 별도 `.hysts-venv`를 만들고, `.hysts-venv` 안에 `anime-face-detector`와 OpenMMLab 계열 의존성을 설치합니다.
5. 마지막에 hysts detector를 한 번 생성해 랜드마크 모델 가중치를 다운로드하고 CUDA 사용 가능 여부를 확인합니다.

현재 처리에 YOLOv8 `Fuyucchi/yolov8_animeface` 모델은 사용하지 않습니다. 기존 YOLO 경로와 `ultralytics` 의존성은 제거했습니다.

## 실행

이미지를 `input/` 폴더에 넣고 `run.bat`를 실행하면 `output/`에 결과가 저장됩니다.

파일이나 폴더를 `run.bat` 위로 드래그앤드롭해도 됩니다. 여러 파일을 동시에 드래그하면 순서대로 처리합니다.

기본 모드: 피부색 추정 non-LaMa prefill

```bat
run.bat
uv run afd process tests --output test_outputs --save-debug
```

흰색 prefill 모드:

```bat
run.bat --white
uv run afd process tests --output test_outputs_white --white --save-debug
```

단일 파일:

```bat
uv run afd process tests\01_reika_original.png --output output\01_reika_faceless.png --save-debug
uv run afd process tests\01_reika_original.png --output output\01_reika_white.png --white --save-debug
```

빠른 로컬 검증:

```bat
run_tests.bat
```

## CLI 모드

- 기본값: 피부색을 추정해서 얼굴 영역을 채웁니다.
- `--white`, `-w`: 피부색을 추정하지 않고 pure white로 얼굴 영역을 채웁니다.
- `--lama`, `-l`: 예약된 옵션입니다. 다음 단계에서 LaMa/IOPaint prefill 모드로 연결할 예정이며, 현재는 명확한 에러로 중단합니다.
- `-l -w`를 같이 넘기면 LaMa 모드가 우선입니다. 현재 구현에서는 LaMa 미구현 에러가 납니다.
- `--max-faces N`: 이미지당 처리할 최대 얼굴 수입니다.
- `--save-debug`: `output/debug/`에 landmarks, face mask, feature mask, hair protect mask, result 이미지를 저장합니다.
- `--device cuda`: 기본값. CUDA가 없으면 실패합니다. 조용히 CPU로 폴백하지 않습니다.
- `--device cpu`: CPU 실행을 명시적으로 허용할 때만 사용합니다.
- `--hysts-python PATH`: 별도 hysts Python 경로를 직접 지정합니다.
- `--hysts-device cuda:0`: hysts detector가 사용할 장치를 지정합니다.

## 모델과 로컬-only 범위

설치 단계에서는 오픈소스 패키지와 모델 가중치를 다운로드합니다. 실제 `afd process`, `run.bat`, `run_tests.bat` 실행 중 얼굴 감지와 얼굴 삭제는 외부 AI API 없이 로컬 Python/PyTorch/OpenCV에서만 수행합니다.

현재 non-LaMa 모드는 Anime-LaMa 모델을 사용하지 않습니다. 다음 단계 LaMa 모드용 모델을 미리 받고 싶으면:

```bat
uv run afd models download
```

사전 점검:

```bat
uv run afd preflight --device cuda --hysts-device cuda:0
```

모델 다운로드까지 포함해 점검하려면:

```bat
uv run afd preflight --download-models --device cuda --hysts-device cuda:0
```

## 테스트와 품질 확인

`examples/`에는 원본과 목표 faceless 결과 쌍이 있습니다. 회귀 비교용으로 다음을 실행할 수 있습니다.

```bat
uv run afd qa examples --output example_outputs --save-debug
uv run afd qa examples --output example_outputs_white --white --save-debug
```

`tests/`에는 원본 입력 이미지가 들어 있습니다.

```bat
uv run afd qa tests --output test_outputs --save-debug
uv run afd qa tests --output test_outputs_white --white --save-debug
```

## 라이선스 주의

`hysts/anime-face-detector`는 MIT License입니다. 단, 이 도구는 `torch`, `mmcv-full`, `mmdet`, `mmpose` 같은 별도 의존성을 함께 사용하므로 배포 목적이면 각 의존성의 라이선스도 확인해야 합니다.

## 문제 해결

CUDA 확인:

```bat
nvidia-smi
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`Hysts python not found`가 나오면:

```bat
install_hysts_probe.bat
```

`CUDA was requested, but torch.cuda.is_available() is false`가 나오면 CUDA wheel이 제대로 설치되지 않았거나 현재 터미널이 다른 Python 환경을 보고 있는 것입니다. `install.bat`를 다시 실행하고, 그래도 실패하면 `.venv`와 `.hysts-venv`를 삭제한 뒤 다시 설치하세요.
