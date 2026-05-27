from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = Path(os.environ.get("AFD_MODEL_DIR", PROJECT_ROOT / "models"))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

YOLO_REPO_ID = "Fuyucchi/yolov8_animeface"
YOLO_FILENAME = "yolov8x6_animeface.pt"

ANIME_LAMA_URL = (
    "https://github.com/Sanster/models/releases/download/"
    "AnimeMangaInpainting/anime-manga-big-lama.pt"
)
ANIME_LAMA_FILENAME = "anime-manga-big-lama.pt"
ANIME_LAMA_MD5 = "29f284f36a0a510bcacf39ecf4c4d54f"
