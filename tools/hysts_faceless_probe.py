from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

HYSTS_WORKER = r'''
import json
import sys
from pathlib import Path

import anime_face_detector
import cv2


def main() -> None:
    device = sys.argv[1]
    paths = [Path(arg) for arg in sys.argv[2:]]
    detector = anime_face_detector.create_detector("yolov3", device=device)
    result = {}
    for path in paths:
        image = cv2.imread(str(path))
        preds = detector(image)
        result[str(path)] = [
            {
                "bbox": pred["bbox"].astype(float).tolist(),
                "keypoints": pred["keypoints"].astype(float).tolist(),
            }
            for pred in preds
        ]
    print("AFD_JSON_START")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
'''


def default_hysts_python() -> Path:
    return Path(".hysts-venv") / "Scripts" / "python.exe"


def gather_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    images = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "faceless" not in path.stem.lower()
    ]
    return images


def run_landmarks(paths: list[Path], hysts_python: Path, device: str) -> dict[str, list[dict]]:
    if not hysts_python.exists():
        raise RuntimeError(f"Hysts python not found: {hysts_python}. Run install_hysts_probe.bat first.")

    proc = subprocess.run(
        [str(hysts_python), "-", device, *[str(path) for path in paths]],
        input=HYSTS_WORKER,
        text=True,
        capture_output=True,
        check=False,
    )
    marker = "AFD_JSON_START"
    marker_index = proc.stdout.rfind(marker)
    if proc.returncode != 0 or marker_index < 0:
        raise RuntimeError(
            "hysts landmark worker failed\n"
            f"returncode={proc.returncode}\n"
            f"stdout={proc.stdout[-4000:]}\n"
            f"stderr={proc.stderr[-4000:]}"
        )
    payload = proc.stdout[marker_index + len(marker) :].strip()
    return json.loads(payload)


def feather(mask: np.ndarray, amount: int) -> np.ndarray:
    k = max(3, amount * 2 + 1)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur((mask > 0).astype(np.float32), (k, k), amount / 2.0).clip(0, 1)


def plausible_skin(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    arr = rgb.astype(np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return (
        (val > 55)
        & (val < 252)
        & (sat > 2)
        & (sat < 145)
        & (r >= g - 34)
        & (r >= b - 8)
        & (g >= b - 54)
        & ((r - b) < 150)
        & ~((val > 240) & (sat < 18))
    )


def landmark_masks(shape: tuple[int, int], keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    pts = keypoints[:, :2].astype(np.float32)

    face_poly = pts[[5, 6, 7, 8, 9, 10, 4, 3, 2, 1, 0]]
    face = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(face, [np.round(face_poly).astype(np.int32)], 255)
    face = cv2.morphologyEx(face, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    feature = np.zeros((height, width), dtype=np.uint8)
    for group, scale in (
        ([11, 12, 13, 14, 15, 16], 0.26),
        ([17, 18, 19, 20, 21, 22], 0.26),
        ([24, 25, 26, 27], 0.34),
    ):
        g = pts[group]
        x0, y0 = g.min(axis=0)
        x1, y1 = g.max(axis=0)
        bw, bh = x1 - x0, y1 - y0
        pad_x = max(5.0, bw * scale)
        pad_y = max(5.0, bh * scale)
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

    nose = pts[23]
    mouth_mid = pts[[24, 25, 26, 27]].mean(axis=0)
    eye_mid = pts[[14, 15, 16, 20, 21, 22]].mean(axis=0)
    nose_end = (eye_mid + mouth_mid) / 2.0
    radius = max(4, int(np.linalg.norm(pts[4] - pts[0]) * 0.018))
    cv2.line(feature, tuple(np.round(nose).astype(int)), tuple(np.round(nose_end).astype(int)), 255, radius * 2)
    feature = cv2.dilate(feature, np.ones((5, 5), np.uint8), iterations=2)
    feature = cv2.bitwise_and(feature, face)
    return face, feature


def make_skin_field(rgb: np.ndarray, face_mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    height, width = face_mask.shape
    samples = face_mask & plausible_skin(rgb) & ~feature_mask
    if int(samples.sum()) < 30:
        samples = face_mask & ~feature_mask
    if int(samples.sum()) < 30:
        samples = face_mask

    fallback = np.median(rgb[samples].reshape(-1, 3), axis=0).astype(np.float32)
    rows = np.full((height, 3), np.nan, dtype=np.float32)
    for y in range(height):
        row = rgb[y, samples[y]]
        if len(row) >= 3:
            rows[y] = np.median(row.astype(np.float32), axis=0)

    valid = ~np.isnan(rows[:, 0])
    if int(valid.sum()) == 0:
        rows[:] = fallback
    elif int(valid.sum()) == 1:
        rows[~valid] = rows[valid][0]
    else:
        x = np.arange(height, dtype=np.float32)
        interp = np.zeros_like(rows)
        for channel in range(3):
            interp[:, channel] = np.interp(x, x[valid], rows[valid, channel])
        rows = cv2.GaussianBlur(interp[:, None, :], (1, 41), 0)[:, 0, :]

    field = np.repeat(rows[:, None, :], width, axis=1)
    ys, xs = np.where(face_mask)
    if xs.size:
        cx = xs.mean()
        cy = ys.mean()
        rx = max(1.0, (xs.max() - xs.min() + 1) * 0.62)
        ry = max(1.0, (ys.max() - ys.min() + 1) * 0.76)
        yy, xx = np.ogrid[:height, :width]
        radial = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        field *= np.clip(1.045 - radial * 0.065, 0.92, 1.06)[:, :, None]

    return cv2.bilateralFilter(field.clip(0, 255).astype(np.uint8), 17, 24, 18)


def fill_face(rgb: np.ndarray, face_mask: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
    field = make_skin_field(rgb, face_mask, feature_mask)
    alpha = feather(face_mask, 7)[:, :, None] * 0.98
    out = rgb.astype(np.float32) * (1.0 - alpha) + field.astype(np.float32) * alpha
    return out.clip(0, 255).astype(np.uint8)


def save_landmark_overlay(path: Path, output_path: Path, preds: list[dict]) -> None:
    bgr = cv2.imread(str(path))
    overlay = bgr.copy()
    for pred in preds:
        bbox = np.round(np.asarray(pred["bbox"][:4])).astype(int)
        cv2.rectangle(overlay, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
        keypoints = np.asarray(pred["keypoints"], dtype=np.float32)
        for idx, (x, y, score) in enumerate(keypoints):
            color = (0, 0, 255) if score >= 0.3 else (0, 255, 255)
            pt = (int(round(x)), int(round(y)))
            cv2.circle(overlay, pt, 4, color, -1)
            cv2.putText(overlay, str(idx), (pt[0] + 4, pt[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def process(
    input_path: Path,
    output: Path,
    landmarks_output: Path | None,
    hysts_python: Path,
    device: str,
) -> None:
    paths = gather_images(input_path)
    if not paths:
        raise RuntimeError(f"No input images found: {input_path}")

    landmarks = run_landmarks(paths, hysts_python, device)
    output.mkdir(parents=True, exist_ok=True)
    if landmarks_output is not None:
        landmarks_output.mkdir(parents=True, exist_ok=True)

    for path in paths:
        preds = landmarks.get(str(path), [])
        if not preds:
            print(f"No face: {path}", file=sys.stderr)
            continue

        rgba = np.array(Image.open(path).convert("RGBA"))
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3].copy()
        pred = max(preds, key=lambda item: float(item["bbox"][4]))
        keypoints = np.asarray(pred["keypoints"], dtype=np.float32)
        face, feature = landmark_masks(rgb.shape[:2], keypoints)
        result = fill_face(rgb, face > 0, feature > 0)

        stem = path.stem
        Image.fromarray(np.dstack([result, alpha])).save(output / f"{stem}_faceless_probe.png")
        cv2.imwrite(str(output / f"{stem}_face_mask.png"), face)
        cv2.imwrite(str(output / f"{stem}_feature_mask.png"), feature)
        if landmarks_output is not None:
            save_landmark_overlay(path, landmarks_output / f"{stem}_landmarks.png", preds)
        print(f"Processed {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the hysts landmark faceless probe.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("examples"))
    parser.add_argument("--output", type=Path, default=Path("qa_outputs") / "hysts_probe")
    parser.add_argument("--landmarks-output", type=Path, default=Path("qa_outputs") / "hysts_landmarks")
    parser.add_argument("--hysts-python", type=Path, default=Path(os.environ.get("HYSTS_PYTHON", default_hysts_python())))
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.input, args.output, args.landmarks_output, args.hysts_python, args.device)


if __name__ == "__main__":
    main()
