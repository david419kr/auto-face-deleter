from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .types import Detection, FaceMask


def save_debug_images(
    image_rgb: np.ndarray,
    result_rgb: np.ndarray,
    detections: list[Detection],
    masks: list[FaceMask],
    debug_dir: Path,
    stem: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)

    boxed = Image.fromarray(image_rgb).convert("RGB")
    draw = ImageDraw.Draw(boxed)
    for i, det in enumerate(detections):
        x0, y0, x1, y1 = det.xyxy
        draw.rectangle((x0, y0, x1, y1), outline=(255, 64, 64), width=3)
        draw.text((x0, max(0, y0 - 14)), f"{i}:{det.confidence:.2f}", fill=(255, 64, 64))
    boxed.save(debug_dir / f"{stem}_bbox.png")

    for face in masks:
        i = face.index
        x0, y0, x1, y1 = face.crop_box
        Image.fromarray(face.raw_mask).save(debug_dir / f"{stem}_face{i:02d}_raw_mask.png")
        Image.fromarray(face.refined_mask).save(debug_dir / f"{stem}_face{i:02d}_refined_mask.png")
        Image.fromarray(face.erase_mask).save(debug_dir / f"{stem}_face{i:02d}_erase_mask.png")
        Image.fromarray(image_rgb[y0:y1, x0:x1]).save(debug_dir / f"{stem}_face{i:02d}_before.png")
        Image.fromarray(result_rgb[y0:y1, x0:x1]).save(debug_dir / f"{stem}_face{i:02d}_after.png")

    Image.fromarray(result_rgb).save(debug_dir / f"{stem}_result.png")
