from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.progress import DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from .constants import (
    ANIME_LAMA_FILENAME,
    ANIME_LAMA_MD5,
    ANIME_LAMA_URL,
    DEFAULT_MODEL_DIR,
    YOLO_FILENAME,
    YOLO_REPO_ID,
)

console = Console()


def yolo_model_path(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    return model_dir / YOLO_FILENAME


def lama_model_path(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    return model_dir / ANIME_LAMA_FILENAME


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
            raise RuntimeError(
                f"Downloaded {destination.name} failed MD5 check: {actual} != {expected_md5}"
            )

    tmp.replace(destination)
    return destination


def download_yolo(model_dir: Path = DEFAULT_MODEL_DIR, force: bool = False) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = yolo_model_path(model_dir)
    if destination.exists() and not force:
        return destination

    console.print(f"[cyan]Downloading detector model[/cyan] {YOLO_REPO_ID}/{YOLO_FILENAME}")
    cached = Path(
        hf_hub_download(
            repo_id=YOLO_REPO_ID,
            filename=YOLO_FILENAME,
            local_dir=str(model_dir),
        )
    )
    if cached.resolve() != destination.resolve():
        shutil.copy2(cached, destination)
    return destination


def download_lama(model_dir: Path = DEFAULT_MODEL_DIR, force: bool = False) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = lama_model_path(model_dir)
    if destination.exists() and not force:
        if md5sum(destination).lower() != ANIME_LAMA_MD5.lower():
            console.print(f"[yellow]{destination.name} exists but MD5 differs; re-downloading.[/yellow]")
        else:
            return destination
    return _download_url(ANIME_LAMA_URL, destination, ANIME_LAMA_MD5)


def download_all(model_dir: Path = DEFAULT_MODEL_DIR, force: bool = False) -> None:
    download_yolo(model_dir, force=force)
    download_lama(model_dir, force=force)


def ensure_models(model_dir: Path = DEFAULT_MODEL_DIR, include_lama: bool = True) -> None:
    missing: list[str] = []
    if not yolo_model_path(model_dir).exists():
        missing.append(str(yolo_model_path(model_dir)))
    if include_lama and not lama_model_path(model_dir).exists():
        missing.append(str(lama_model_path(model_dir)))
    if missing:
        joined = "\n  - ".join(missing)
        raise RuntimeError(f"Missing model files:\n  - {joined}\nRun: afd models download")


def ensure_device(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false. "
            "This tool does not silently fall back to CPU; use --device cpu explicitly."
        )


def preflight(model_dir: Path = DEFAULT_MODEL_DIR, device: str = "cuda", download_models: bool = False) -> None:
    if download_models:
        download_all(model_dir)

    console.print("[cyan]Python[/cyan]", sys.version.split()[0])
    import cv2
    import numpy as np
    import torch
    import ultralytics

    console.print("[cyan]numpy[/cyan]", np.__version__)
    console.print("[cyan]opencv[/cyan]", cv2.__version__)
    console.print("[cyan]torch[/cyan]", torch.__version__)
    console.print("[cyan]ultralytics[/cyan]", ultralytics.__version__)
    console.print("[cyan]model_dir[/cyan]", str(model_dir.resolve()))

    ensure_device(device)
    if device.startswith("cuda"):
        console.print("[green]CUDA available[/green]", torch.cuda.get_device_name(0))

    ensure_models(model_dir, include_lama=True)
    console.print("[green]All required models are present.[/green]")
