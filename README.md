# Auto Face Deleter

Local Windows CUDA tool for removing anime-style facial features from images. It detects anime face landmarks locally and fills the face area with either estimated skin color or pure white.

Supported inputs: `png`, `jpg`, `jpeg`, `webp`, `avif`, `bmp`, `tif`, `tiff`.

Outputs are always PNG:

- Default: `name_faceless.png`
- White mode: `name_faceless-white.png`

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

You can also drag and drop image files or folders onto `run.bat` or `run-white.bat`.

## CLI

```bat
uv run --no-sync afd process input --output output
uv run --no-sync afd process input --output output --white
```

Useful options:

```bat
--max-faces N
--save-debug
--device cuda
```
