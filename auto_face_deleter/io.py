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


MODE_SUFFIXES = (
    "_faceless-white-exclude-mouth",
    "_faceless-exclude-mouth",
    "_faceless-white",
    "_faceless",
    "_eye-white",
    "_eye",
    "_crop",
    "_skipped",
)


def mode_output_suffix(
    white: bool = False,
    crop: bool = False,
    exclude_mouth: bool = False,
    eye_only: bool = False,
) -> str:
    if crop:
        return "_crop"
    if white and eye_only:
        return "_eye-white"
    if eye_only:
        return "_eye"
    if white and exclude_mouth:
        return "_faceless-white-exclude-mouth"
    if exclude_mouth:
        return "_faceless-exclude-mouth"
    if white:
        return "_faceless-white"
    return "_faceless"


def apply_mode_suffix(stem: str, suffix: str) -> str:
    for existing in MODE_SUFFIXES:
        if stem.endswith(existing):
            stem = stem[: -len(existing)]
            break
    return f"{stem}{suffix}"


def unique_output_path(path: Path, reserved: set[Path]) -> Path:
    candidate = path
    counter = 1
    while candidate.exists() or candidate.resolve() in reserved:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        counter += 1
    reserved.add(candidate.resolve())
    return candidate


def resolve_output_path(
    input_root: Path,
    image_path: Path,
    output: Path,
    suffix: str,
    reserved: set[Path],
    extension: str = ".png",
) -> Path:
    if output.suffix.lower() in IMAGE_EXTENSIONS:
        target_dir = output.parent
        stem = output.stem
    elif input_root.is_dir():
        relative = image_path.relative_to(input_root)
        target_dir = output / relative.parent
        stem = relative.stem
    else:
        target_dir = output
        stem = image_path.stem

    if not extension.startswith("."):
        extension = f".{extension}"
    target = target_dir / f"{apply_mode_suffix(stem, suffix)}{extension}"
    return unique_output_path(target, reserved)


def open_image_rgba(path: Path) -> tuple[Image.Image, bool]:
    image = Image.open(path)
    has_alpha = image.mode in {"RGBA", "LA"} or ("transparency" in image.info)
    return image.convert("RGBA"), has_alpha
