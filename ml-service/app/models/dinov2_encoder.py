from __future__ import annotations

import logging
import numpy as np
from PIL import Image
import torch

from app.models.base import l2_normalize
from app.models.clip_encoder import resolve_device

logger = logging.getLogger(__name__)


class DINOv2Encoder:
    """Optional self-supervised DINOv2 encoder for fine-grained visual similarity."""

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "auto") -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.processor = None
        self.model = None
        self.dims: int = 768

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.dims = int(self.model.config.hidden_size)
        logger.info("Loaded DINOv2 %s on %s | dims=%d", self.model_name, self.device, self.dims)

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("DINOv2Encoder is not loaded.")

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_tokens = outputs.last_hidden_state[:, 0, :]
            embeddings = cls_tokens.detach().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)
