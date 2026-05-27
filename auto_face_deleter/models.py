from __future__ import annotations

import sys
import warnings

from rich.console import Console

console = Console()

warnings.filterwarnings("ignore", message="On January 1, 2023, MMCV will release v2.0.0.*")


def detector_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def ensure_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "This tool does not silently fall back to CPU; use --device cpu explicitly."
        )


def preflight(device: str = "cuda", warmup_detector: bool = False) -> None:
    console.print("[cyan]Python[/cyan]", sys.version.split()[0])

    import anime_face_detector
    import cv2
    import mmcv
    import mmdet
    import mmpose
    import numpy as np
    import torch

    console.print("[cyan]numpy[/cyan]", np.__version__)
    console.print("[cyan]opencv[/cyan]", cv2.__version__)
    console.print("[cyan]torch[/cyan]", torch.__version__)
    console.print("[cyan]mmcv[/cyan]", mmcv.__version__)
    console.print("[cyan]mmdet[/cyan]", mmdet.__version__)
    console.print("[cyan]mmpose[/cyan]", mmpose.__version__)
    console.print("[cyan]anime-face-detector[/cyan]", getattr(anime_face_detector, "__version__", "installed"))

    ensure_device(device)
    if device.startswith("cuda"):
        console.print("[green]CUDA available[/green]", torch.cuda.get_device_name(0))

    if warmup_detector:
        anime_face_detector.create_detector("yolov3", device=detector_device(device))
        console.print("[green]Detector ready.[/green]")
    else:
        console.print("[green]Preflight complete.[/green]")
