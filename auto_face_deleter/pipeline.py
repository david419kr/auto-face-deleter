from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console

from .debug import save_debug_images
from .detector import AnimeFaceDetector
from .inpaint import erase_faces
from .io import gather_images, open_image_rgba, resolve_output_path
from .masking import build_face_mask
from .types import Backend, FaceMask, ProcessOptions

console = Console()


class FaceDeleter:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self.detector = AnimeFaceDetector(
            model_dir=options.model_dir,
            device=options.device,
            conf=options.conf,
            imgsz=options.imgsz,
            max_faces=options.max_faces,
        )
        self.lama = None

    def process_image(self, image_path: Path, output_path: Path, debug_dir: Path | None = None) -> int:
        rgba_image, _ = open_image_rgba(image_path)
        rgba = np.array(rgba_image)
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy()

        detections = self.detector.detect(image_path)
        masks: list[FaceMask] = []
        for index, detection in enumerate(detections):
            mask = build_face_mask(
                rgb,
                detection,
                index=index,
                aggression=self.options.aggression,
                extra_dilate=self.options.mask_dilate,
            )
            if int(mask.refined_mask.max()) > 0:
                masks.append(mask)

        if masks:
            result_rgb = erase_faces(
                rgb,
                masks,
                backend=self.options.backend,
                aggression=self.options.aggression,
                feather=self.options.feather,
                lama=self.lama,
            )
        else:
            result_rgb = rgb

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_rgba = np.dstack([result_rgb, alpha])
        Image.fromarray(out_rgba, mode="RGBA").save(output_path)

        if self.options.save_debug and debug_dir is not None:
            save_debug_images(rgb, result_rgb, detections, masks, debug_dir, output_path.stem)

        return len(detections)


def process_path(input_path: Path, output: Path, options: ProcessOptions) -> tuple[int, int]:
    images = gather_images(input_path, recursive=options.recursive)
    deleter = FaceDeleter(options)
    processed = 0
    faces = 0
    debug_root = output / "debug" if output.suffix == "" else output.parent / "debug"

    for image_path in images:
        output_path = resolve_output_path(input_path, image_path, output, total=len(images))
        console.print(f"[cyan]Processing[/cyan] {image_path} -> {output_path}")
        detected = deleter.process_image(image_path, output_path, debug_dir=debug_root)
        processed += 1
        faces += detected
        if detected == 0:
            console.print(f"[yellow]No face detected:[/yellow] {image_path}")

    return processed, faces
