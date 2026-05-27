from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from .models import ensure_device, yolo_model_path
from .types import Detection


class AnimeFaceDetector:
    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda",
        conf: float = 0.25,
        imgsz: int = 1024,
        max_faces: int | None = None,
    ) -> None:
        ensure_device(device)
        model_path = yolo_model_path(model_dir)
        if not model_path.exists():
            raise RuntimeError(f"Detector model not found: {model_path}. Run: afd models download")
        self.model = YOLO(str(model_path))
        self.device = "0" if device == "cuda" else device
        self.conf = conf
        self.imgsz = imgsz
        self.max_faces = max_faces

    def detect(self, image_path: Path) -> list[Detection]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else [0] * len(boxes)
            for box, score, cls in zip(boxes, confs, classes):
                x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
                if x1 <= x0 or y1 <= y0:
                    continue
                detections.append(Detection((x0, y0, x1, y1), float(score), int(cls)))

        detections.sort(key=lambda item: item.confidence, reverse=True)
        if self.max_faces is not None:
            detections = detections[: self.max_faces]
        return detections
