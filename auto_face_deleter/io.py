from __future__ import annotations

from pathlib import Path

from PIL import Image

from .constants import IMAGE_EXTENSIONS


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def gather_images(input_path: Path, recursive: bool = False) -> list[Path]:
    if input_path.is_file():
        if not is_image_file(input_path):
            raise RuntimeError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise RuntimeError(f"Input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise RuntimeError(f"Input path is not a file or directory: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    images = sorted(path for path in iterator if is_image_file(path))
    if not images:
        raise RuntimeError(f"No images found in: {input_path}")
    return images


def resolve_output_path(input_root: Path, image_path: Path, output: Path, total: int) -> Path:
    if total == 1 and output.suffix.lower() in IMAGE_EXTENSIONS:
        return output

    if input_root.is_dir():
        relative = image_path.relative_to(input_root)
    else:
        relative = image_path.name
    return output / relative


def open_image_rgba(path: Path) -> tuple[Image.Image, bool]:
    image = Image.open(path)
    has_alpha = image.mode in {"RGBA", "LA"} or ("transparency" in image.info)
    return image.convert("RGBA"), has_alpha
