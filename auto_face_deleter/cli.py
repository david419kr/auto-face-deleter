from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .constants import DEFAULT_MODEL_DIR, PROJECT_ROOT
from .models import download_lama, preflight as run_preflight
from .pipeline import process_path
from .qa import run_qa
from .types import ProcessOptions

console = Console()
app = typer.Typer(no_args_is_help=True, help="Local anime face feature remover.")
models_app = typer.Typer(help="Download or inspect local models.")
app.add_typer(models_app, name="models")


def _exit_runtime_error(error: RuntimeError) -> None:
    console.print(f"[red]Error:[/red] {error}")
    raise typer.Exit(1)


def _options(
    recursive: bool,
    save_debug: bool,
    max_faces: int | None,
    model_dir: Path,
    white: bool,
    lama: bool,
    device: str,
    hysts_python: Path | None,
    hysts_device: str | None,
) -> ProcessOptions:
    return ProcessOptions(
        recursive=recursive,
        save_debug=save_debug,
        max_faces=max_faces,
        model_dir=model_dir,
        white=False if lama else white,
        lama=lama,
        device=device,
        hysts_python=hysts_python,
        hysts_device=hysts_device,
    )


@app.command()
def process(
    input_path: Annotated[Path, typer.Argument(help="Input image file or directory.")] = Path("input"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file or directory.")] = Path("output"),
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
    white: Annotated[
        bool,
        typer.Option("--white", "-w", help="Use pure white non-LaMa prefill instead of estimated skin color."),
    ] = False,
    lama: Annotated[
        bool,
        typer.Option("--lama", "-l", help="Use LaMa mode. Reserved for the next implementation step."),
    ] = False,
    device: Annotated[str, typer.Option("--device", help="cuda, cuda:0, or cpu.")] = "cuda",
    hysts_device: Annotated[
        str | None,
        typer.Option("--hysts-device", help="Detector device for the hysts environment. Defaults from --device."),
    ] = None,
    hysts_python: Annotated[Path | None, typer.Option("--hysts-python", help="Path to .hysts-venv Python.")] = None,
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = False,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    opts = _options(recursive, save_debug, max_faces, model_dir, white, lama, device, hysts_python, hysts_device)
    try:
        processed, faces = process_path(input_path, output, opts)
    except RuntimeError as error:
        _exit_runtime_error(error)
    mode = "lama" if lama else "white" if white else "skin"
    console.print(f"[green]Done[/green] mode={mode} images={processed} faces={faces} output={output}")


@app.command()
def preflight(
    download_models: Annotated[bool, typer.Option("--download-models/--no-download-models")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    hysts_device: Annotated[str, typer.Option("--hysts-device")] = "cuda:0",
    hysts_python: Annotated[Path | None, typer.Option("--hysts-python")] = None,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    try:
        run_preflight(
            model_dir=model_dir,
            device=device,
            download_models=download_models,
            hysts_python=hysts_python,
            hysts_device=hysts_device,
        )
    except RuntimeError as error:
        _exit_runtime_error(error)


@models_app.command("download")
def models_download(
    lama: Annotated[bool, typer.Option("--lama", help="Download the future LaMa model. Currently same as default.")] = False,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    del lama
    try:
        path = download_lama(model_dir, force=force)
    except RuntimeError as error:
        _exit_runtime_error(error)
    console.print(f"[green]Anime-LaMa ready:[/green] {path}")


@app.command()
def qa(
    target: Annotated[str, typer.Argument(help="'examples' or 'tests'.")] = "tests",
    output: Annotated[Path, typer.Option("--output", "-o")] = PROJECT_ROOT / "test_outputs",
    white: Annotated[bool, typer.Option("--white", "-w")] = False,
    lama: Annotated[bool, typer.Option("--lama", "-l", help="Reserved for the next implementation step.")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    hysts_device: Annotated[str | None, typer.Option("--hysts-device")] = None,
    hysts_python: Annotated[Path | None, typer.Option("--hysts-python")] = None,
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = True,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    opts = _options(True, save_debug, max_faces, model_dir, white, lama, device, hysts_python, hysts_device)
    try:
        run_qa(target, output, opts)
    except RuntimeError as error:
        _exit_runtime_error(error)
