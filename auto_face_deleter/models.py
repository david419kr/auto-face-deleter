from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from .constants import (
    ANIME_LAMA_FILENAME,
    ANIME_LAMA_MD5,
    ANIME_LAMA_URL,
    DEFAULT_MODEL_DIR,
    PROJECT_ROOT,
)

console = Console()


def lama_model_path(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    return model_dir / ANIME_LAMA_FILENAME


def default_hysts_python() -> Path:
    return Path(os.environ.get("HYSTS_PYTHON", PROJECT_ROOT / ".hysts-venv" / "Scripts" / "python.exe"))


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path, expected_md5: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(f"Downloading {destination.name}", total=None)

        def hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0 and progress.tasks[task_id].total is None:
                progress.update(task_id, total=total_size)
            progress.update(task_id, completed=min(block_num * block_size, total_size))

        urllib.request.urlretrieve(url, tmp, hook)

    if expected_md5:
        actual = md5sum(tmp)
        if actual.lower() != expected_md5.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded {destination.name} failed MD5 check: {actual} != {expected_md5}")

    tmp.replace(destination)
    return destination


def download_lama(model_dir: Path = DEFAULT_MODEL_DIR, force: bool = False) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = lama_model_path(model_dir)
    if destination.exists() and not force:
        if md5sum(destination).lower() == ANIME_LAMA_MD5.lower():
            return destination
        console.print(f"[yellow]{destination.name} exists but MD5 differs; re-downloading.[/yellow]")
    return _download_url(ANIME_LAMA_URL, destination, ANIME_LAMA_MD5)


def download_all(model_dir: Path = DEFAULT_MODEL_DIR, force: bool = False) -> None:
    download_lama(model_dir, force=force)


def ensure_lama_model(model_dir: Path = DEFAULT_MODEL_DIR) -> None:
    if not lama_model_path(model_dir).exists():
        raise RuntimeError(f"Missing Anime-LaMa model: {lama_model_path(model_dir)}\nRun: afd models download --lama")


def ensure_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "This tool does not silently fall back to CPU; use --device cpu explicitly."
        )


def check_hysts_env(hysts_python: Path, device: str, warmup: bool = False) -> None:
    if not hysts_python.exists():
        raise RuntimeError(f"Hysts python not found: {hysts_python}. Run install_hysts_probe.bat first.")

    code = r'''
import sys

device = sys.argv[1]
warmup = sys.argv[2] == "1"

import torch
import numpy
import cv2
import mmcv
import mmdet
import mmpose
import anime_face_detector

print("hysts-python", sys.version.split()[0])
print("hysts-torch", torch.__version__, "cuda", torch.cuda.is_available())
print("hysts-numpy", numpy.__version__)
print("hysts-opencv", cv2.__version__)
print("hysts-mmcv", mmcv.__version__)

if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested for hysts detector, but torch.cuda.is_available() is false.")

if warmup:
    anime_face_detector.create_detector("yolov3", device=device)
    print("hysts-detector", "ready")
'''
    proc = subprocess.run(
        [str(hysts_python), "-c", code, device, "1" if warmup else "0"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.stdout.strip():
        console.print(proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(
            "Hysts detector environment check failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout={proc.stdout[-4000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )


def preflight(
    model_dir: Path = DEFAULT_MODEL_DIR,
    device: str = "cuda",
    download_models: bool = False,
    hysts_python: Path | None = None,
    hysts_device: str = "cuda:0",
) -> None:
    if download_models:
        download_all(model_dir)

    console.print("[cyan]Python[/cyan]", sys.version.split()[0])
    import cv2
    import numpy as np
    import torch

    console.print("[cyan]numpy[/cyan]", np.__version__)
    console.print("[cyan]opencv[/cyan]", cv2.__version__)
    console.print("[cyan]torch[/cyan]", torch.__version__)
    console.print("[cyan]model_dir[/cyan]", str(model_dir.resolve()))

    ensure_device(device)
    if device.startswith("cuda"):
        console.print("[green]CUDA available[/green]", torch.cuda.get_device_name(0))

    check_hysts_env(hysts_python or default_hysts_python(), hysts_device, warmup=download_models)
    console.print("[green]Preflight complete.[/green]")
