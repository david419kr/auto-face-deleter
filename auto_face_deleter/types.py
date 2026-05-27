from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessOptions:
    recursive: bool
    save_debug: bool
    max_faces: int | None
    model_dir: Path
    white: bool = False
    lama: bool = False
    device: str = "cuda"
    hysts_python: Path | None = None
    hysts_device: str | None = None
