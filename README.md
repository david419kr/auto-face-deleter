# Auto Face Deleter

Local Windows CUDA tool for simple anime face cleanup. It detects anime face landmarks locally and can either remove facial features or crop away the whole head.

Supported inputs: `png`, `jpg`, `jpeg`, `webp`, `avif`, `bmp`, `tif`, `tiff`.

Face removal outputs are PNG:

- Default: `name_faceless.png`
- White mode: `name_faceless-white.png`

Crop mode outputs:

- One detected face: `name_crop.png`
- Zero or multiple detected faces: original file copied as `name_skipped.ext`

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

Results are saved to `output\`.

White fill mode:

```bat
run-white.bat
```

Head crop mode:

```bat
run-crop.bat
```

You can also drag and drop image files or folders onto `run.bat`, `run-white.bat`, or `run-crop.bat`.

## CLI

```bat
uv run --no-sync afd process input --output output
uv run --no-sync afd process input --output output --white
uv run --no-sync afd process input --output output --crop
```

Useful options:

```bat
--max-faces N
--save-debug
--device cuda
```
