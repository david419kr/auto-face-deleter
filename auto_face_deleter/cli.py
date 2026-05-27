from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .constants import PROJECT_ROOT
from .models import preflight as run_preflight
from .pipeline import process_path
from .qa import run_qa
from .types import ProcessOptions

console = Console()
app = typer.Typer(no_args_is_help=True, help="Local anime face feature remover.")


def _exit_runtime_error(error: RuntimeError) -> None:
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(1)


def _options(
    recursive: bool,
    save_debug: bool,
    max_faces: int | None,
    white: bool,
    crop: bool,
    device: str,
) -> ProcessOptions:
    return ProcessOptions(
        recursive=recursive,
        save_debug=save_debug,
        max_faces=max_faces,
        white=False if crop else white,
        crop=crop,
        device=device,
    )


@app.command()
def process(
    input_path: Annotated[Path, typer.Argument(help="Input image file or directory.")] = Path("input"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file or directory.")] = Path("output"),
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
    white: Annotated[
        bool,
        typer.Option("--white", "-w", help="Use pure white prefill instead of estimated skin color."),
    ] = False,
    crop: Annotated[
        bool,
        typer.Option("--crop", "-c", help="Crop away the detected head instead of filling the face."),
    ] = False,
    device: Annotated[str, typer.Option("--device", help="cuda, cuda:0, or cpu.")] = "cuda",
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = False,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
) -> None:
    opts = _options(recursive, save_debug, max_faces, white, crop, device)
    try:
        processed, faces = process_path(input_path, output, opts)
    except RuntimeError as error:
        _exit_runtime_error(error)
    mode = "crop" if crop else "white" if white else "skin"
    console.print(f"[green]Done[/green] mode={mode} images={processed} faces={faces} output={output}")


@app.command()
def preflight(
    device: Annotated[str, typer.Option("--device")] = "cuda",
    warmup_detector: Annotated[bool, typer.Option("--warmup-detector/--no-warmup-detector")] = False,
) -> None:
    try:
        run_preflight(device=device, warmup_detector=warmup_detector)
    except RuntimeError as error:
        _exit_runtime_error(error)


@app.command()
def qa(
    target: Annotated[str, typer.Argument(help="'examples' or 'tests'.")] = "tests",
    output: Annotated[Path, typer.Option("--output", "-o")] = PROJECT_ROOT / "test_outputs",
    white: Annotated[bool, typer.Option("--white", "-w")] = False,
    crop: Annotated[bool, typer.Option("--crop", "-c")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = True,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
) -> None:
    opts = _options(True, save_debug, max_faces, white, crop, device)
    try:
        run_qa(target, output, opts)
    except RuntimeError as error:
        _exit_runtime_error(error)
