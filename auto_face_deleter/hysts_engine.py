from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

import cv2
import numpy as np
from PIL import Image

from .io import open_image_rgba
from .types import ProcessOptions


warnings.filterwarnings("ignore", message="On January 1, 2023, MMCV will release v2.0.0.*")
warnings.filterwarnings("ignore", message='"ImageToTensor" pipeline is replaced by "DefaultFormatBundle".*')


@dataclass
class FaceDebug:
    index: int
    bbox: list[float]
    keypoints: np.ndarray
    face_mask: np.ndarray
    feature_mask: np.ndarray
    hair_mask: np.ndarray


def detector_device_from_options(options: ProcessOptions) -> str:
    if options.device == "cuda":
        return "cuda:0"
    return options.device


def load_detector_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
            else:
                rgb = image.convert("RGB")
    except OSError as error:
        raise RuntimeError(f"Failed to read image for detection: {path}\n{error}") from error
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)


def plausible_skin(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    arr = rgb.astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (
        (val > 55)
        & (val <= 255)
        & (sat > 2)
        & (sat < 145)
        & (r >= g - 34)
        & (r >= b - 8)
        & (g >= b - 54)
        & ((r - b) < 150)
        & ~((val > 248) & (sat < 10))
    )


def strict_skin(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    arr = rgb.astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (
        (val > 70)
        & (val <= 255)
        & (sat > 5)
        & (sat < 115)
        & (r >= g - 24)
        & (r >= b - 2)
        & (g >= b - 42)
        & ((r - b) < 125)
        & ~((val > 248) & (sat < 10))
    )


def normalize_skin_color(color: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(np.uint8([[np.clip(color, 0, 255).astype(np.uint8)]]), cv2.COLOR_RGB2HSV)[0, 0]
    hsv[1] = np.uint8(np.clip(int(hsv[1]), 20, 82))
    hsv[2] = np.uint8(np.clip(int(hsv[2]), 118, 250))
    return cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)


def estimate_skin_color(rgb: np.ndarray, face_mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(face_mask)
    if xs.size == 0:
        return np.array([224, 185, 165], dtype=np.float32)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    h = max(1, y1 - y0)
    yy = np.arange(rgb.shape[0])[:, None]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    line_art = cv2.dilate((cv2.Canny(gray, 36, 120) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    clean = (
        face_mask
        & strict_skin(rgb)
        & ~feature_mask
        & ~line_art
        & (yy >= y0 + h * 0.22)
        & (yy <= y0 + h * 0.92)
    )
    if int(clean.sum()) < 40:
        clean = face_mask & plausible_skin(rgb) & ~feature_mask & ~line_art
    if int(clean.sum()) < 20:
        clean = face_mask & ~feature_mask
    if int(clean.sum()) < 20:
        clean = face_mask

    pixels = rgb[clean].reshape(-1, 3).astype(np.float32)
    hsv_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    bright = hsv_pixels[:, 2] >= np.percentile(hsv_pixels[:, 2], 58)
    if int(bright.sum()) >= 12:
        pixels = pixels[bright]
    return normalize_skin_color(np.median(pixels, axis=0).astype(np.float32))


def landmark_masks(shape: tuple[int, int], keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    pts = keypoints[:, :2].astype(np.float32)

    face_poly = pts[[5, 6, 7, 8, 9, 10, 4, 3, 2, 1, 0]]
    face = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(face, [np.round(face_poly).astype(np.int32)], 255)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    feature = np.zeros((height, width), dtype=np.uint8)
    face_width = max(1.0, float(np.linalg.norm(pts[4] - pts[0])))
    for group, scale_x, scale_y in (
        ([11, 12, 13, 14, 15, 16], 0.38, 0.42),
        ([17, 18, 19, 20, 21, 22], 0.38, 0.42),
        ([24, 25, 26, 27], 0.48, 0.44),
    ):
        g = pts[group]
        x0, y0 = g.min(axis=0)
        x1, y1 = g.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        pad_x = max(5.0, bw * scale_x, face_width * 0.012)
        pad_y = max(5.0, bh * scale_y, face_width * 0.014)
        cv2.ellipse(
            feature,
            (int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))),
            (int(round((bw / 2) + pad_x)), int(round((bh / 2) + pad_y))),
            0,
            0,
            360,
            255,
            -1,
        )

    brow_thickness = max(8, int(round(face_width * 0.045)))
    brow_pad_y = max(4, int(round(face_width * 0.018)))
    for group in ([5, 6, 7], [8, 9, 10]):
        brow_pts = pts[group]
        brow = np.round(brow_pts).astype(np.int32)
        cv2.polylines(feature, [brow], False, 255, brow_thickness, cv2.LINE_AA)
        for x, y in brow_pts:
            cv2.ellipse(
                feature,
                (int(round(x)), int(round(y))),
                (brow_thickness, brow_pad_y + brow_thickness // 3),
                0,
                0,
                360,
                255,
                -1,
            )

    nose = pts[23]
    mouth_mid = pts[[24, 25, 26, 27]].mean(axis=0)
    eye_mid = pts[[14, 15, 16, 20, 21, 22]].mean(axis=0)
    nose_end = (eye_mid + mouth_mid) / 2.0
    radius = max(4, int(np.linalg.norm(pts[4] - pts[0]) * 0.018))
    cv2.line(feature, tuple(np.round(nose).astype(int)), tuple(np.round(nose_end).astype(int)), 255, radius * 2)
    feature = cv2.dilate(feature, np.ones((5, 5), np.uint8), iterations=1)
    feature = cv2.bitwise_and(feature, face)
    return face, feature


def hair_protect_mask(
    rgb: np.ndarray,
    geometry_mask: np.ndarray,
    feature_mask: np.ndarray,
    skin_color: np.ndarray,
) -> np.ndarray:
    ys, xs = np.where(geometry_mask)
    if xs.size == 0:
        return np.zeros_like(geometry_mask, dtype=bool)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height = max(1, y1 - y0)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[np.clip(skin_color, 0, 255).astype(np.uint8)]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - skin_lab, axis=2)

    hair_like = (
        ((val < 118) & (dist > 26))
        | ((sat > 52) & (val < 190) & (dist > 34))
        | ((dist > 56) & (val < 172))
    )
    candidate = geometry_mask & hair_like & ~feature_mask
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    labels, stats = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)[1:3]
    protected = np.zeros_like(candidate, dtype=bool)
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(8, int(geometry_mask.sum() * 0.00035)):
            continue
        y = int(stats[label, cv2.CC_STAT_TOP])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_top = y <= y0 + height * 0.18
        top_connected = touches_top and h > height * 0.025
        if top_connected:
            protected[labels == label] = True

    if int(protected.sum()) > 0:
        protected = cv2.dilate(protected.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    return protected & geometry_mask & ~feature_mask


def refine_face_mask(rgb: np.ndarray, geometry_mask: np.ndarray, feature_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skin_color = estimate_skin_color(rgb, geometry_mask, feature_mask)
    hair = hair_protect_mask(rgb, geometry_mask, feature_mask, skin_color)
    face = geometry_mask | feature_mask
    face = cv2.morphologyEx(face.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return face, hair


def make_skin_field(rgb: np.ndarray, face_mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    height, width = face_mask.shape
    base_color = estimate_skin_color(rgb, face_mask, feature_mask)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    line_art = cv2.dilate((cv2.Canny(gray, 36, 120) > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    samples = face_mask & strict_skin(rgb) & ~feature_mask & ~line_art
    if int(samples.sum()) < 40:
        samples = face_mask & plausible_skin(rgb) & ~feature_mask & ~line_art
    if int(samples.sum()) < 20:
        samples = face_mask

    field = np.full((height, width, 3), base_color, dtype=np.float32)
    ys, xs = np.where(face_mask)
    if xs.size:
        cx = xs.mean()
        cy = ys.mean()
        rx = max(1.0, (xs.max() - xs.min() + 1) * 0.62)
        ry = max(1.0, (ys.max() - ys.min() + 1) * 0.76)
        yy, xx = np.ogrid[:height, :width]
        radial = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        y_norm = np.clip((yy - ys.min()) / max(1.0, float(ys.max() - ys.min())), 0.0, 1.0)
        field *= np.clip(1.045 - radial * 0.052 - y_norm * 0.018, 0.93, 1.055)[:, :, None]

    if int(samples.sum()) >= 20:
        values = np.zeros((height, width, 3), dtype=np.float32)
        values[samples] = rgb[samples].astype(np.float32)
        weights = samples.astype(np.float32)
        smooth_values = cv2.GaussianBlur(values, (0, 0), sigmaX=33, sigmaY=33)
        smooth_weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=33, sigmaY=33)
        valid = smooth_weights > 0.002
        sample_field = field.copy()
        sample_field[valid] = smooth_values[valid] / smooth_weights[valid, None]
        field = field * 0.88 + sample_field * 0.12

    return cv2.bilateralFilter(field.clip(0, 255).astype(np.uint8), 13, 18, 16)


def make_white_field(rgb: np.ndarray) -> np.ndarray:
    return np.full_like(rgb, 255, dtype=np.uint8)


def blend_field(rgb: np.ndarray, field: np.ndarray, mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    alpha = composite_alpha(mask, feature_mask)[:, :, None]
    out = rgb.astype(np.float32) * (1.0 - alpha) + field.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def composite_alpha(mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.float32)
    alpha = cv2.GaussianBlur(binary, (0, 0), sigmaX=1.15, sigmaY=1.15)
    core = cv2.erode((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    alpha[core] = 1.0
    alpha[cv2.dilate(feature_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0] = 1.0
    alpha[alpha < 0.04] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def fill_face(rgb: np.ndarray, face_mask: np.ndarray, feature_mask: np.ndarray, white: bool = False) -> np.ndarray:
    field = make_white_field(rgb) if white else make_skin_field(rgb, face_mask, feature_mask)
    return blend_field(rgb, field, face_mask, feature_mask)


def valid_keypoint_xy(keypoints: np.ndarray, min_score: float = 0.1) -> np.ndarray:
    if keypoints.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if keypoints.shape[1] >= 3:
        valid = keypoints[:, 2] >= min_score
        pts = keypoints[valid, :2]
        if pts.shape[0] >= 4:
            return pts.astype(np.float32)
    return keypoints[:, :2].astype(np.float32)


def compute_head_crop_box(width: int, height: int, keypoints: np.ndarray) -> tuple[int, int, int, int, str]:
    contour = keypoints[:11] if keypoints.shape[0] >= 11 else keypoints
    pts = valid_keypoint_xy(contour)
    if pts.shape[0] == 0:
        raise RuntimeError("Cannot crop without face keypoints.")

    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    span = max(1.0, float(max(x1 - x0, y1 - y0)))
    margin = max(4.0, span * 0.08)
    cx, cy = pts.mean(axis=0)

    edge_distances = {
        "left": float(cx) / max(1, width),
        "right": float(width - cx) / max(1, width),
        "top": float(cy) / max(1, height),
        "bottom": float(height - cy) / max(1, height),
    }
    edge = min(edge_distances, key=edge_distances.get)

    if edge == "top":
        cut = int(np.ceil(y1 + margin))
        cut = int(np.clip(cut, 1, height - 1))
        return 0, cut, width, height, edge
    if edge == "bottom":
        cut = int(np.floor(y0 - margin))
        cut = int(np.clip(cut, 1, height - 1))
        return 0, 0, width, cut, edge
    if edge == "left":
        cut = int(np.ceil(x1 + margin))
        cut = int(np.clip(cut, 1, width - 1))
        return cut, 0, width, height, edge

    cut = int(np.floor(x0 - margin))
    cut = int(np.clip(cut, 1, width - 1))
    return 0, 0, cut, height, edge


def save_crop_debug_image(
    image_rgb: np.ndarray,
    pred: dict,
    crop_box: tuple[int, int, int, int],
    edge: str,
    debug_dir: Path,
    stem: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    bbox = np.round(np.asarray(pred["bbox"][:4])).astype(int)
    cv2.rectangle(overlay, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
    for idx, (x, y, score) in enumerate(np.asarray(pred["keypoints"], dtype=np.float32)):
        color = (0, 0, 255) if score >= 0.3 else (0, 255, 255)
        pt = (int(round(x)), int(round(y)))
        cv2.circle(overlay, pt, 4, color, -1)
        cv2.putText(overlay, str(idx), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    x0, y0, x1, y1 = crop_box
    if edge in {"left", "right"}:
        x = x0 if edge == "left" else x1
        cv2.line(overlay, (x, 0), (x, image_rgb.shape[0] - 1), (255, 0, 255), 3)
    else:
        y = y0 if edge == "top" else y1
        cv2.line(overlay, (0, y), (image_rgb.shape[1] - 1, y), (255, 0, 255), 3)
    cv2.imwrite(str(debug_dir / f"{stem}_crop_debug.png"), overlay)


def save_debug_images(
    image_rgb: np.ndarray,
    result_rgb: np.ndarray,
    faces: list[FaceDebug],
    debug_dir: Path,
    stem: str,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    overlay = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for face in faces:
        bbox = np.round(np.asarray(face.bbox[:4])).astype(int)
        cv2.rectangle(overlay, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
        for idx, (x, y, score) in enumerate(face.keypoints):
            color = (0, 0, 255) if score >= 0.3 else (0, 255, 255)
            pt = (int(round(x)), int(round(y)))
            cv2.circle(overlay, pt, 4, color, -1)
            cv2.putText(overlay, str(idx), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imwrite(str(debug_dir / f"{stem}_landmarks.png"), overlay)
    for face in faces:
        i = face.index
        Image.fromarray(face.face_mask.astype(np.uint8) * 255).save(debug_dir / f"{stem}_face{i:02d}_face_mask.png")
        Image.fromarray(face.feature_mask.astype(np.uint8) * 255).save(debug_dir / f"{stem}_face{i:02d}_feature_mask.png")
        Image.fromarray(face.hair_mask.astype(np.uint8) * 255).save(debug_dir / f"{stem}_face{i:02d}_hair_protect_mask.png")
    Image.fromarray(result_rgb).save(debug_dir / f"{stem}_result.png")


class HystsFaceDeleter:
    def __init__(self, options: ProcessOptions) -> None:
        self.options = options
        self.device = detector_device_from_options(options)
        try:
            import anime_face_detector
        except ImportError as error:
            raise RuntimeError("anime-face-detector is not installed. Run install.bat first.") from error
        self.detector = anime_face_detector.create_detector("yolov3", device=self.device)

    def detect_images(self, image_paths: list[Path]) -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = {}
        for path in image_paths:
            image = load_detector_image(path)
            preds = self.detector(image)
            result[str(path)] = [
                {
                    "bbox": pred["bbox"].astype(float).tolist(),
                    "keypoints": pred["keypoints"].astype(float).tolist(),
                }
                for pred in preds
            ]
        return result

    def process_image(
        self,
        image_path: Path,
        output_path: Path,
        predictions: list[dict],
        debug_dir: Path | None = None,
    ) -> int:
        rgba_image, _ = open_image_rgba(image_path)
        rgba = np.array(rgba_image)
        original_rgb = rgba[:, :, :3].copy()
        result_rgb = original_rgb.copy()
        alpha = rgba[:, :, 3].copy()

        predictions = sorted(predictions, key=lambda item: float(item["bbox"][4]), reverse=True)
        raw_count = len(predictions)
        if self.options.crop:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_count != 1:
                if image_path.resolve() != output_path.resolve():
                    copy2(image_path, output_path)
                return raw_count

            keypoints = np.asarray(predictions[0]["keypoints"], dtype=np.float32)
            crop_box_with_edge = compute_head_crop_box(rgba_image.width, rgba_image.height, keypoints)
            crop_box = crop_box_with_edge[:4]
            edge = crop_box_with_edge[4]
            rgba_image.crop(crop_box).save(output_path, format="PNG")
            if self.options.save_debug and debug_dir is not None:
                save_crop_debug_image(original_rgb, predictions[0], crop_box, edge, debug_dir, output_path.stem)
            return raw_count

        if self.options.max_faces is not None:
            predictions = predictions[: self.options.max_faces]

        faces: list[FaceDebug] = []
        for index, pred in enumerate(predictions):
            keypoints = np.asarray(pred["keypoints"], dtype=np.float32)
            geometry, feature = landmark_masks(result_rgb.shape[:2], keypoints)
            face, hair = refine_face_mask(result_rgb, geometry > 0, feature > 0)
            result_rgb = fill_face(result_rgb, face, feature > 0, white=self.options.white)
            faces.append(
                FaceDebug(
                    index=index,
                    bbox=list(pred["bbox"]),
                    keypoints=keypoints,
                    face_mask=face,
                    feature_mask=feature > 0,
                    hair_mask=hair,
                )
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.dstack([result_rgb, alpha]), mode="RGBA").save(output_path, format="PNG")
        if self.options.save_debug and debug_dir is not None:
            save_debug_images(original_rgb, result_rgb, faces, debug_dir, output_path.stem)
        return len(predictions)
