from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .models import ensure_device, lama_model_path


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
        height, width = image.shape[:2]
        pad_h = (mod - height % mod) % mod
        pad_w = (mod - width % mod) % mod
        if pad_h == 0 and pad_w == 0:
            return image, mask, (height, width)
        image_padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        mask_padded = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        return image_padded, mask_padded, (height, width)

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
        height, width = original_size
        return output_np[:height, :width]
