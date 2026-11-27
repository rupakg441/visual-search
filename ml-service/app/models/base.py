from __future__ import annotations

from typing import Protocol
import numpy as np
from PIL import Image


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D or 2D numpy array along the last axis."""
    norms = np.linalg.norm(vec, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vec / norms


class ImageEncoder(Protocol):
    dims: int

    def load(self) -> None: ...

    def encode_images(self, images: list[Image.Image]) -> np.ndarray: ...
