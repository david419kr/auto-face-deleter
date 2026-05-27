from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .models import ensure_device, lama_model_path
from .types import Aggression, Backend, FaceMask


FACE_ALPHA = {
    Aggression.low: 0.90,
    Aggression.normal: 0.98,
    Aggression.high: 1.0,
}


class AnimeLamaInpainter:
    def __init__(self, model_dir: Path, device: str = "cuda") -> None:
        ensure_device(device)
        path = lama_model_path(model_dir)
        if not path.exists():
            raise RuntimeError(f"Anime-LaMa model not found: {path}. Run: afd models download")
        self.device = torch.device(device)
        self.model = torch.jit.load(str(path), map_location=self.device).eval()

    @staticmethod
    def _pad_to_mod(image: np.ndarray, mask: np.ndarray, mod: int = 8) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        h, w = image.shape[:2]
        pad_h = (mod - h % mod) % mod
        pad_w = (mod - w % mod) % mod
        if pad_h == 0 and pad_w == 0:
            return image, mask, (h, w)
        image_padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return image_padded, mask_padded, (h, w)

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if int(mask.max()) == 0:
            return image_rgb

        padded_image, padded_mask, original_size = self._pad_to_mod(image_rgb, mask)
        image_tensor = (
            torch.from_numpy(padded_image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        mask_tensor = (
            torch.from_numpy((padded_mask > 0).astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            output = self.model(image_tensor, mask_tensor)
        output_np = output[0].permute(1, 2, 0).detach().float().cpu().numpy()
        output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)
        h, w = original_size
        return output_np[:h, :w]


def _crop_bounds(mask: np.ndarray, margin: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + margin), min(height, y1 + margin)


def _feather(mask: np.ndarray, amount: int) -> np.ndarray:
    binary = (mask > 0).astype(np.float32)
    if amount <= 0:
        return binary
    k = max(3, amount * 2 + 1)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(binary, (k, k), amount / 2.0)
    return np.clip(blurred, 0.0, 1.0)


def _plausible_skin(crop: np.ndarray, skin_color: tuple[int, int, int]) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    rgb = crop.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - skin_lab, axis=2)
    return (
        (dist < 42.0)
        & (val > 45)
        & (val < 252)
        & (sat < 145)
        & (r >= g - 32)
        & (r >= b - 12)
        & (g >= b - 52)
        & ~((val > 242) & (sat < 18))
    )


def _interpolate_rows(row_values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    h = row_values.shape[0]
    valid = ~np.isnan(row_values[:, 0])
    if int(valid.sum()) == 0:
        return np.repeat(fallback[None, :], h, axis=0)
    if int(valid.sum()) == 1:
        row_values[~valid] = row_values[valid][0]
        return row_values
    x = np.arange(h, dtype=np.float32)
    out = np.zeros((h, 3), dtype=np.float32)
    for c in range(3):
        out[:, c] = np.interp(x, x[valid], row_values[valid, c])
    out = cv2.GaussianBlur(out[:, None, :], (1, 31), 0)[:, 0, :]
    return out


def _skin_field(
    crop: np.ndarray,
    face_mask: np.ndarray,
    feature_mask: np.ndarray,
    skin_color: tuple[int, int, int],
) -> np.ndarray:
    h, w = crop.shape[:2]
    fallback = np.array(skin_color, dtype=np.float32)
    skin_samples = face_mask & _plausible_skin(crop, skin_color) & ~feature_mask
    if int(skin_samples.sum()) < max(20, int(face_mask.sum() * 0.03)):
        skin_samples = face_mask & ~feature_mask

    row_values = np.full((h, 3), np.nan, dtype=np.float32)
    for y in range(h):
        row_pixels = crop[y, skin_samples[y]]
        if row_pixels.shape[0] >= 3:
            row_values[y] = np.median(row_pixels.astype(np.float32), axis=0)
    rows = _interpolate_rows(row_values, fallback)
    field = np.repeat(rows[:, None, :], w, axis=1)

    yy, xx = np.ogrid[:h, :w]
    ys, xs = np.where(face_mask)
    if xs.size:
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        rx = max(1.0, (float(xs.max() - xs.min()) + 1.0) * 0.58)
        ry = max(1.0, (float(ys.max() - ys.min()) + 1.0) * 0.70)
        radial = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        highlight = np.clip(1.055 - radial * 0.075, 0.92, 1.06).astype(np.float32)
        field *= highlight[:, :, None]

    if int(skin_samples.sum()) > 0:
        low_freq = crop.astype(np.float32).copy()
        low_freq[~skin_samples] = field[~skin_samples]
        for sigma in (9, 17, 29):
            low_freq = cv2.GaussianBlur(low_freq, (0, 0), sigmaX=sigma, sigmaY=sigma)
            low_freq[skin_samples] = crop[skin_samples].astype(np.float32) * 0.35 + low_freq[skin_samples] * 0.65
        field = field * 0.72 + low_freq * 0.28

    return np.clip(field, 0, 255).astype(np.uint8)


def _skin_fill_face(
    image_rgb: np.ndarray,
    face: FaceMask,
    aggression: Aggression,
    feather: int,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    face_mask_full = face.erase_mask > 0
    bounds = _crop_bounds(face.erase_mask, margin=max(8, feather * 2), width=width, height=height)
    if bounds is None:
        return image_rgb

    x0, y0, x1, y1 = bounds
    crop = image_rgb[y0:y1, x0:x1]
    face_mask = face_mask_full[y0:y1, x0:x1]
    feature_mask = (face.raw_mask[y0:y1, x0:x1] > 0) & face_mask
    if int(face_mask.sum()) == 0:
        return image_rgb

    fill = _skin_field(crop, face_mask, feature_mask, face.skin_color)
    if aggression != Aggression.low:
        fill = cv2.bilateralFilter(fill, d=15, sigmaColor=26, sigmaSpace=18)
    if aggression == Aggression.high:
        fill = cv2.GaussianBlur(fill, (0, 0), sigmaX=1.2, sigmaY=1.2)

    alpha_amount = max(1, feather)
    alpha = _feather(face_mask.astype(np.uint8) * 255, alpha_amount)[:, :, None]
    alpha *= FACE_ALPHA[aggression]

    out = image_rgb.copy()
    out_crop = (crop.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha).clip(0, 255)
    out[y0:y1, x0:x1] = out_crop.astype(np.uint8)

    # A light second pass kills remaining feature ghosts without moving the silhouette.
    polish = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), sigmaX=0.55, sigmaY=0.55)
    core = cv2.erode(face_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    core_alpha = (core.astype(np.float32) * (0.18 if aggression == Aggression.low else 0.28))[:, :, None]
    region = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (region * (1.0 - core_alpha) + polish.astype(np.float32) * core_alpha).clip(0, 255).astype(np.uint8)
    return out


def erase_faces(
    image_rgb: np.ndarray,
    masks: list[FaceMask],
    backend: Backend,
    aggression: Aggression,
    feather: int,
    lama: AnimeLamaInpainter | None = None,
) -> np.ndarray:
    del backend, lama
    result = image_rgb.copy()
    for face in masks:
        if int(face.erase_mask.max()) == 0:
            continue
        result = _skin_fill_face(result, face, aggression, feather)
    return result
