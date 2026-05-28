# Auto Face Deleter

Local Windows CUDA tool for simple anime face cleanup. It detects anime face landmarks locally and can either remove facial features or crop away the whole head.

Supported inputs: `png`, `jpg`, `jpeg`, `webp`, `avif`, `bmp`, `tif`, `tiff`.

Face removal outputs are PNG:

- Default: `name_faceless.png`
- White mode: `name_faceless-white.png`
- Keep mouth and lower face: `name_faceless-exclude-mouth.png`
- White keep mouth and lower face: `name_faceless-white-exclude-mouth.png`
- Eye-only: `name_eye.png`
- White eye-only: `name_eye-white.png`

Crop mode outputs:

- One detected face: `name_crop.png`
- Zero or multiple detected faces: original file copied as `name_skipped.ext`

`--exclude-mouth` and `--eye-only` are ignored in crop mode.

## Install

Run:

```bat
install.bat
```

The installer creates a Python 3.10 `.venv`, installs the local CUDA detector stack, installs this tool, and checks CUDA.

## Run

Put images in `input\`, then run:

```bat
run.bat
```

Choose a mode with the arrow keys and Enter. Results are saved to `output\`.

You can also drag and drop image files or folders onto `run.bat`.
If command-line arguments are passed, the menu is skipped.

## CLI

```bat
uv run --no-sync afd process input --output output
uv run --no-sync afd process input --output output --white
uv run --no-sync afd process input --output output --exclude-mouth
uv run --no-sync afd process input --output output --white --exclude-mouth
uv run --no-sync afd process input --output output --eye-only
uv run --no-sync afd process input --output output --white --eye-only
uv run --no-sync afd process input --output output --crop
```

Useful options:

```bat
--max-faces N
--save-debug
--device cuda
```
