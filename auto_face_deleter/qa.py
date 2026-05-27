from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .constants import PROJECT_ROOT
from .io import is_image_file
from .pipeline import process_path
from .types import ProcessOptions

console = Console()


def qa_inputs(kind: str) -> Path:
    normalized = kind.lower()
    if normalized not in {"examples", "tests"}:
        raise RuntimeError("qa target must be 'examples' or 'tests'")
    return PROJECT_ROOT / normalized


def run_qa(kind: str, output: Path, options: ProcessOptions) -> None:
    source = qa_inputs(kind)
    if kind == "examples":
        temp_input = output / "_examples_originals"
        temp_input.mkdir(parents=True, exist_ok=True)
        for path in source.iterdir():
            if is_image_file(path) and "_original" in path.stem:
                target = temp_input / path.name
                if not target.exists() or target.stat().st_mtime < path.stat().st_mtime:
                    target.write_bytes(path.read_bytes())
        source = temp_input

    processed, faces = process_path(source, output, options)
    console.print(f"[green]QA complete[/green] images={processed} faces={faces} output={output}")
    if processed == 0 or faces == 0:
        raise RuntimeError("QA failed: no processed images or no detected faces")
