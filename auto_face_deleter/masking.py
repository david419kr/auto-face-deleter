from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .types import Aggression, Detection, FaceMask


@dataclass(frozen=True)
class MaskTuning:
    skin_dist: float
    feature_dist: float
    face_close: int
    feature_close: int
    feature_dilate: int
    face_dilate: int


TUNING = {
    Aggression.low: MaskTuning(30.0, 34.0, 5, 5, 2, 0),
    Aggression.normal: MaskTuning(38.0, 28.0, 7, 7, 4, 1),
    Aggression.high: MaskTuning(48.0, 22.0, 9, 9, 6, 2),
}


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def expand_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    x_scale: float = 0.12,
    y_top_scale: float = 0.14,
    y_bottom_scale: float = 0.10,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    bw = x1 - x0
    bh = y1 - y0
    return _clip_box(
        (
            int(round(x0 - bw * x_scale)),
            int(round(y0 - bh * y_top_scale)),
            int(round(x1 + bw * x_scale)),
            int(round(y1 + bh * y_bottom_scale)),
        ),
        width,
        height,
    )


def _detector_face_shape(
    crop_h: int,
    crop_w: int,
    local_box: tuple[int, int, int, int],
    aggression: Aggression,
) -> np.ndarray:
    x0, y0, x1, y1 = local_box
    bw = max(2, x1 - x0)
    bh = max(2, y1 - y0)
    yy, xx = np.ogrid[:crop_h, :crop_w]

    widen = 1.0 if aggression == Aggression.low else 1.05 if aggression == Aggression.normal else 1.10
    cx = x0 + bw * 0.50
    cy = y0 + bh * 0.51
    rx = max(2.0, bw * 0.405 * widen)
    ry = max(2.0, bh * 0.455)
    upper_oval = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0

    jaw = np.zeros((crop_h, crop_w), dtype=np.uint8)
    points = np.array(
        [
            [int(round(cx - bw * 0.375 * widen)), int(round(y0 + bh * 0.40))],
            [int(round(cx - bw * 0.405 * widen)), int(round(y0 + bh * 0.66))],
            [int(round(cx - bw * 0.235 * widen)), int(round(y0 + bh * 0.87))],
            [int(round(cx)), int(round(y0 + bh * 1.005))],
            [int(round(cx + bw * 0.235 * widen)), int(round(y0 + bh * 0.87))],
            [int(round(cx + bw * 0.405 * widen)), int(round(y0 + bh * 0.66))],
            [int(round(cx + bw * 0.375 * widen)), int(round(y0 + bh * 0.40))],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(jaw, [points], 1)

    face = upper_oval | (jaw > 0)
    face &= yy >= y0 + bh * 0.04
    face &= yy <= y0 + bh * 1.03
    return face


def _estimate_skin_color(crop: np.ndarray, geometry: np.ndarray | None = None) -> tuple[int, int, int]:
    h, w = crop.shape[:2]
    if geometry is None:
        region = np.zeros((h, w), dtype=bool)
        region[int(h * 0.22) : int(h * 0.88), int(w * 0.18) : int(w * 0.82)] = True
    else:
        region = geometry

    rgb = crop.astype(np.int16)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    candidate = (
        region
        & (val > 55)
        & (val < 252)
        & (sat > 3)
        & (sat < 135)
        & (r >= g - 28)
        & (r >= b - 4)
        & (g >= b - 48)
        & ((r - b) < 140)
        & ~((val > 240) & (sat < 18))
    )

    if int(candidate.sum()) < max(20, int(region.sum() * 0.03)):
        candidate = region & (val > 48) & (val < 255) & (sat < 175) & (r >= b - 45)
    if int(candidate.sum()) < 10:
        cx0, cx1 = int(w * 0.32), max(int(w * 0.68), int(w * 0.32) + 1)
        cy0, cy1 = int(h * 0.35), max(int(h * 0.74), int(h * 0.35) + 1)
        samples = crop[cy0:cy1, cx0:cx1].reshape(-1, 3)
    else:
        samples = crop[candidate]

    color = np.median(samples.reshape(-1, 3).astype(np.int16), axis=0)
    return _normalize_skin_color(tuple(int(np.clip(v, 0, 255)) for v in color.tolist()))


def _normalize_skin_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0, 0]
    hsv[1] = min(int(hsv[1]), 82)
    hsv[2] = max(70, int(hsv[2]))
    rgb = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _skin_similarity_mask(crop: np.ndarray, skin_color: tuple[int, int, int], threshold: float) -> np.ndarray:
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - skin_lab, axis=2)
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    rgb = crop.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    plausible = (
        (val > 42)
        & (sat < 190)
        & (r >= b - 54)
        & (g >= b - 68)
        & ~((val > 244) & (sat < 12))
    )
    return (dist <= threshold) & plausible


def _anime_skin_rule(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    rgb = crop.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return (
        (val > 56)
        & (val < 252)
        & (sat > 2)
        & (sat < 130)
        & (r >= g - 30)
        & (r >= b - 3)
        & (g >= b - 46)
        & ((r - b) < 145)
        & ~((val > 241) & (sat < 18))
    )


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8) * 255
    flood = cv2.bitwise_not(binary)
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 0)
    holes = flood > 0
    return mask | holes


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    labels, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[1:3]
    keep = np.zeros(stats.shape[0], dtype=bool)
    for label in range(1, stats.shape[0]):
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _largest_center_components(mask: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)[1:4]
    if stats.shape[0] <= 1:
        return mask

    h, w = mask.shape
    center = np.array([w * 0.50, h * 0.56], dtype=np.float32)
    scores: list[tuple[float, int]] = []
    min_area = max(20, int(geometry.sum() * 0.01))
    for label in range(1, stats.shape[0]):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        centroid = centroids[label].astype(np.float32)
        dx = abs(centroid[0] - center[0]) / max(1.0, w)
        dy = abs(centroid[1] - center[1]) / max(1.0, h)
        scores.append((area * (1.0 - min(0.85, dx * 1.35 + dy * 0.8)), label))
    if not scores:
        return mask

    scores.sort(reverse=True)
    best_area = float(stats[scores[0][1], cv2.CC_STAT_AREA])
    keep = np.zeros(stats.shape[0], dtype=bool)
    for _, label in scores:
        area = float(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        central = (w * 0.08) <= cx <= (w * 0.92) and (h * 0.08) <= cy <= (h * 0.95)
        if central and (area >= best_area * 0.08 or area >= geometry.sum() * 0.035):
            keep[label] = True
    return keep[labels]


def _convex_fill(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if xs.size < 12:
        return mask
    points = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(points)
    filled = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled > 0


def _hair_protect_mask(
    crop: np.ndarray,
    geometry: np.ndarray,
    skin_color: tuple[int, int, int],
    local_box: tuple[int, int, int, int],
    tuning: MaskTuning,
) -> np.ndarray:
    x0, y0, x1, y1 = local_box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - skin_lab, axis=2)

    dark_hair = (val < 92) & (dist > tuning.skin_dist * 0.75)
    color_hair = (sat > 70) & (val < 178) & (dist > tuning.skin_dist * 1.1)
    candidate = geometry & (dark_hair | color_hair)

    top_zone = yy <= y0 + bh * 0.38
    side_zone = (xx <= x0 + bw * 0.09) | (xx >= x1 - bw * 0.09)
    seed = candidate & (top_zone | side_zone)
    seed = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)) > 0

    labels, stats = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)[1:3]
    protected = np.zeros_like(seed, dtype=bool)
    min_area = max(12, int(geometry.sum() * 0.001))
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        touches_hairline = y <= y0 + bh * 0.18
        side_strand = (x <= x0 + bw * 0.08 or x + w >= x1 - bw * 0.08) and h >= bh * 0.13
        if touches_hairline or side_strand or area > geometry.sum() * 0.015:
            protected[labels == label] = True

    if int(protected.sum()) == 0:
        return protected

    protected = cv2.dilate(protected.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=2) > 0
    return protected & geometry


def _feature_mask(
    crop: np.ndarray,
    face_mask: np.ndarray,
    skin_color: tuple[int, int, int],
    local_box: tuple[int, int, int, int],
    tuning: MaskTuning,
) -> np.ndarray:
    x0, y0, x1, y1 = local_box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    rgb = crop.astype(np.int16)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    skin_lab = cv2.cvtColor(np.uint8([[skin_color]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    dist = np.linalg.norm(lab - skin_lab, axis=2)

    upper = (yy >= y0 + bh * 0.14) & (yy <= y0 + bh * 0.67)
    center_lower = (yy >= y0 + bh * 0.38) & (yy <= y0 + bh * 0.88)
    face_core = face_mask & (xx >= x0 - bw * 0.03) & (xx <= x1 + bw * 0.03)

    skin_like = _skin_similarity_mask(crop, skin_color, tuning.skin_dist * 0.72) | _anime_skin_rule(crop)
    non_skin_holes = face_core & ~skin_like & (dist > tuning.feature_dist * 0.52)
    dark_lines = face_core & (val < 135) & (dist > tuning.feature_dist * 0.42)
    eye_whites = face_core & upper & (val > 150) & (sat < 80) & (dist > tuning.feature_dist * 0.40)
    iris_or_tears = face_core & upper & (sat > 24) & (val > 42) & (dist > tuning.feature_dist * 0.45)
    mouth_or_blush = face_core & center_lower & (sat > 16) & (r > g + 5) & (r > b + 2)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 32, 112) > 0
    edge_features = face_core & edges & (dist > tuning.feature_dist * 0.38)

    candidate = non_skin_holes | dark_lines | eye_whites | iris_or_tears | mouth_or_blush | edge_features
    candidate &= upper | center_lower
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((_odd(tuning.feature_close), _odd(tuning.feature_close)), dtype=np.uint8),
    ) > 0
    candidate = _remove_small_components(candidate, max(5, int(face_mask.sum() * 0.00012)))

    labels, stats, centroids = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)[1:4]
    features = np.zeros_like(candidate, dtype=bool)
    min_area = max(6, int(face_mask.sum() * 0.00010))
    for label in range(1, stats.shape[0]):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        rel_y = (cy - y0) / bh
        rel_x = (cx - x0) / bw

        in_eye_band = 0.18 <= rel_y <= 0.68
        in_mouth_band = 0.48 <= rel_y <= 0.92
        central = -0.08 <= rel_x <= 1.08
        likely_eye = in_eye_band and central and (area >= min_area or w >= bw * 0.035 or h >= bh * 0.025)
        likely_mouth = in_mouth_band and 0.18 <= rel_x <= 0.82 and (area >= min_area or w >= bw * 0.035)
        likely_blush_or_tear = central and area >= min_area and h <= bh * 0.32
        if not (likely_eye or likely_mouth or likely_blush_or_tear):
            continue

        if likely_eye:
            pad_x = max(4, int(w * 0.42), int(bw * 0.025))
            pad_y = max(3, int(h * 0.46), int(bh * 0.025))
        elif likely_mouth:
            pad_x = max(3, int(w * 0.32), int(bw * 0.018))
            pad_y = max(3, int(h * 0.38), int(bh * 0.018))
        else:
            pad_x = max(2, int(w * 0.25))
            pad_y = max(2, int(h * 0.28))

        ex0, ey0 = max(0, x - pad_x), max(0, y - pad_y)
        ex1, ey1 = min(crop.shape[1], x + w + pad_x), min(crop.shape[0], y + h + pad_y)
        features[ey0:ey1, ex0:ex1] |= face_mask[ey0:ey1, ex0:ex1]

    if int(features.sum()) > 0:
        features = cv2.dilate(
            features.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=tuning.feature_dilate,
        ) > 0
        features = cv2.morphologyEx(features.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)) > 0
    return features & face_mask


def build_face_mask(
    image_rgb: np.ndarray,
    detection: Detection,
    index: int,
    aggression: Aggression = Aggression.normal,
    extra_dilate: int = 0,
) -> FaceMask:
    height, width = image_rgb.shape[:2]
    crop_box = expand_box(detection.xyxy, width, height)
    x0, y0, x1, y1 = crop_box
    crop = image_rgb[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    tuning = TUNING[aggression]

    local_box = (
        detection.xyxy[0] - x0,
        detection.xyxy[1] - y0,
        detection.xyxy[2] - x0,
        detection.xyxy[3] - y0,
    )

    geometry = _detector_face_shape(crop_h, crop_w, local_box, aggression)
    skin_color = _estimate_skin_color(crop, geometry)
    protected = _hair_protect_mask(crop, geometry, skin_color, local_box, tuning)

    skin_seed = (_anime_skin_rule(crop) | _skin_similarity_mask(crop, skin_color, tuning.skin_dist)) & geometry & ~protected
    skin_seed = cv2.morphologyEx(
        skin_seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((_odd(tuning.face_close), _odd(tuning.face_close)), dtype=np.uint8),
    ) > 0
    skin_seed = cv2.dilate(skin_seed.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    skin_seed = _largest_center_components(skin_seed, geometry)

    face = (_fill_holes(skin_seed) | _convex_fill(skin_seed)) & geometry
    face &= ~protected
    if int(face.sum()) < max(32, int(geometry.sum() * 0.24)):
        face = geometry & ~protected

    if tuning.face_dilate + extra_dilate > 0:
        face = cv2.dilate(
            face.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=max(0, tuning.face_dilate + int(extra_dilate)),
        ) > 0
        face &= geometry
        face &= ~protected
    face = cv2.morphologyEx(face.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)) > 0
    face = _fill_holes(face) & geometry & ~protected

    skin_color = _estimate_skin_color(crop, face)
    features = _feature_mask(crop, face, skin_color, local_box, tuning)

    # The output target is faceless, so the actual paint area is the visible face skin.
    # The raw mask remains the feature detector debug view.
    erase = face.copy()

    full_raw = np.zeros((height, width), dtype=np.uint8)
    full_refined = np.zeros((height, width), dtype=np.uint8)
    full_erase = np.zeros((height, width), dtype=np.uint8)
    full_raw[y0:y1, x0:x1] = (features.astype(np.uint8) * 255)
    full_refined[y0:y1, x0:x1] = (face.astype(np.uint8) * 255)
    full_erase[y0:y1, x0:x1] = (erase.astype(np.uint8) * 255)

    return FaceMask(
        index=index,
        detection=detection,
        crop_box=crop_box,
        skin_color=skin_color,
        raw_mask=full_raw,
        refined_mask=full_refined,
        erase_mask=full_erase,
    )
