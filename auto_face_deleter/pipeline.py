from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .hysts_engine import HystsFaceDeleter
from .io import gather_images, mode_output_suffix, resolve_output_path
from .types import ProcessOptions

console = Console()


def process_path(input_path: Path, output: Path, options: ProcessOptions) -> tuple[int, int]:
    images = gather_images(input_path, recursive=options.recursive)
    deleter = HystsFaceDeleter(options)
    predictions = deleter.detect_images(images)
    processed = 0
    faces = 0
    debug_root = output / "debug" if output.suffix == "" else output.parent / "debug"
    suffix = mode_output_suffix(
        white=options.white,
        crop=options.crop,
        exclude_mouth=options.exclude_mouth,
        eye_only=options.eye_only,
    )
    reserved_outputs: set[Path] = set()

    for image_path in images:
        image_predictions = predictions.get(str(image_path), [])
        output_suffix = suffix
        output_extension = ".png"
        if options.crop and len(image_predictions) != 1:
            output_suffix = "_skipped"
            output_extension = image_path.suffix
        output_path = resolve_output_path(
            input_path,
            image_path,
            output,
            suffix=output_suffix,
            reserved=reserved_outputs,
            extension=output_extension,
        )
        console.print(f"[cyan]Processing[/cyan] {image_path} -> {output_path}")
        detected = deleter.process_image(
            image_path,
            output_path,
            image_predictions,
            debug_dir=debug_root,
        )
        processed += 1
        faces += detected
        if options.crop and detected != 1:
            console.print(f"[yellow]Skipped crop:[/yellow] {image_path} detected_faces={detected}")
        elif detected == 0:
            console.print(f"[yellow]No face detected:[/yellow] {image_path}")

    return processed, faces
