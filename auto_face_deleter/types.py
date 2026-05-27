from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class Backend(str, Enum):
    hybrid = "hybrid"
    skinfill = "skinfill"


class Aggression(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


@dataclass(frozen=True)
class Detection:
    xyxy: tuple[int, int, int, int]
    confidence: float
    cls: int = 0


@dataclass
class FaceMask:
    index: int
    detection: Detection
    crop_box: tuple[int, int, int, int]
    skin_color: tuple[int, int, int]
    raw_mask: np.ndarray
    refined_mask: np.ndarray
    erase_mask: np.ndarray


@dataclass
class ProcessOptions:
    backend: Backend
    device: str
    conf: float
    aggression: Aggression
    recursive: bool
    save_debug: bool
    mask_dilate: int
    feather: int
    max_faces: int | None
    imgsz: int
    model_dir: Path
