from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProcessOptions:
    recursive: bool
    save_debug: bool
    max_faces: int | None
    white: bool = False
    device: str = "cuda"
