from __future__ import annotations

import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detects the primary foreground object using YOLOv8 and crops with padding."""

    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25) -> None:
        self.model_name = model_name
        self.conf = conf
        self._model = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_name)
            logger.info("Loaded object detector model: %s", self.model_name)
        except Exception as exc:  # pragma: no cover - optional runtime dependency download
            logger.warning("Object detector unavailable (%s); falling back to full image.", exc)
            self._model = None

    def crop_main_object(self, image: Image.Image) -> Image.Image:
        """Crop the highest-confidence/largest object; return original image on no detection."""
        if self._model is None:
            return image

        try:
            results = self._model.predict(image, conf=self.conf, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return image

            boxes = results[0].boxes
            # Pick highest confidence box
            best_idx = int(boxes.conf.argmax().item())
            x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

            width, height = image.size
            box_w = x2 - x1
            box_h = y2 - y1
            # Ignore degenerate tiny crops (< 5% of image area)
            if (box_w * box_h) < (0.05 * width * height):
                return image

            pad_x = box_w * 0.08
            pad_y = box_h * 0.08
            crop_box = (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(width, int(x2 + pad_x)),
                min(height, int(y2 + pad_y)),
            )
            return image.crop(crop_box)
        except Exception as exc:
            logger.warning("Object detection failed (%s); using full image.", exc)
            return image
