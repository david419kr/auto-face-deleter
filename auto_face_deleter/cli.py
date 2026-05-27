from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .constants import DEFAULT_MODEL_DIR, PROJECT_ROOT
from .models import download_all, download_lama, download_yolo, preflight as run_preflight
from .pipeline import process_path
from .qa import run_qa
from .types import Aggression, Backend, ProcessOptions

console = Console()
app = typer.Typer(no_args_is_help=True, help="Local anime face feature remover.")
models_app = typer.Typer(help="Download or inspect local models.")
app.add_typer(models_app, name="models")


def _options(
    backend: Backend,
    device: str,
    conf: float,
    recursive: bool,
    aggression: Aggression,
    save_debug: bool,
    mask_dilate: int,
    feather: int,
    max_faces: int | None,
    imgsz: int,
    model_dir: Path,
) -> ProcessOptions:
    return ProcessOptions(
        backend=backend,
        device=device,
        conf=conf,
        recursive=recursive,
        aggression=aggression,
        save_debug=save_debug,
        mask_dilate=mask_dilate,
        feather=feather,
        max_faces=max_faces,
        imgsz=imgsz,
        model_dir=model_dir,
    )


@app.command()
def process(
    input_path: Annotated[Path, typer.Argument(..., help="Input image file or directory.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output file or directory.")] = Path("output"),
    recursive: Annotated[bool, typer.Option("--recursive/--no-recursive")] = True,
    backend: Annotated[Backend, typer.Option("--backend", help="Face erase backend.")] = Backend.hybrid,
    device: Annotated[str, typer.Option("--device", help="cuda, cuda:0, or cpu.")] = "cuda",
    conf: Annotated[float, typer.Option("--conf", min=0.01, max=0.99)] = 0.25,
    aggression: Annotated[Aggression, typer.Option("--aggression")] = Aggression.normal,
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = False,
    mask_dilate: Annotated[int, typer.Option("--mask-dilate", min=0, max=24)] = 0,
    feather: Annotated[int, typer.Option("--feather", min=0, max=80)] = 12,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
    imgsz: Annotated[int, typer.Option("--imgsz", min=320, max=2048)] = 1024,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    opts = _options(backend, device, conf, recursive, aggression, save_debug, mask_dilate, feather, max_faces, imgsz, model_dir)
    processed, faces = process_path(input_path, output, opts)
    console.print(f"[green]Done[/green] images={processed} faces={faces} output={output}")


@app.command()
def preflight(
    download_models: Annotated[bool, typer.Option("--download-models/--no-download-models")] = False,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    run_preflight(model_dir=model_dir, device=device, download_models=download_models)


@models_app.command("download")
def models_download(
    detector_only: Annotated[bool, typer.Option("--detector-only/--all")] = False,
    lama_only: Annotated[bool, typer.Option("--lama-only/--all-models")] = False,
    force: Annotated[bool, typer.Option("--force/--no-force")] = False,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    if detector_only and lama_only:
        raise typer.BadParameter("--detector-only and --lama-only cannot be used together")
    if detector_only:
        path = download_yolo(model_dir, force=force)
        console.print(f"[green]Detector ready:[/green] {path}")
    elif lama_only:
        path = download_lama(model_dir, force=force)
        console.print(f"[green]Anime-LaMa ready:[/green] {path}")
    else:
        download_all(model_dir, force=force)
        console.print(f"[green]All models ready:[/green] {model_dir.resolve()}")


@app.command()
def qa(
    target: Annotated[str, typer.Argument(help="'examples' or 'tests'.")] = "tests",
    output: Annotated[Path, typer.Option("--output", "-o")] = PROJECT_ROOT / "test_outputs",
    backend: Annotated[Backend, typer.Option("--backend")] = Backend.hybrid,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    conf: Annotated[float, typer.Option("--conf", min=0.01, max=0.99)] = 0.25,
    aggression: Annotated[Aggression, typer.Option("--aggression")] = Aggression.normal,
    save_debug: Annotated[bool, typer.Option("--save-debug/--no-save-debug")] = True,
    mask_dilate: Annotated[int, typer.Option("--mask-dilate", min=0, max=24)] = 0,
    feather: Annotated[int, typer.Option("--feather", min=0, max=80)] = 12,
    max_faces: Annotated[int | None, typer.Option("--max-faces", min=1)] = None,
    imgsz: Annotated[int, typer.Option("--imgsz", min=320, max=2048)] = 1024,
    model_dir: Annotated[Path, typer.Option("--model-dir")] = DEFAULT_MODEL_DIR,
) -> None:
    opts = _options(backend, device, conf, True, aggression, save_debug, mask_dilate, feather, max_faces, imgsz, model_dir)
    run_qa(target, output, opts)
