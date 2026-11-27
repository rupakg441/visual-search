from __future__ import annotations

import logging
from PIL import Image

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """Removes background with rembg and composites onto a neutral white canvas."""

    def __init__(self) -> None:
        self._session = None
        self._remove_fn = None

    def load(self) -> None:
        try:
            from rembg import new_session, remove

            self._session = new_session("u2net")
            self._remove_fn = remove
            logger.info("Loaded rembg u2net session.")
        except Exception as exc:  # pragma: no cover - optional model download
            logger.warning("rembg unavailable (%s); skipping background removal.", exc)
            self._session = None
            self._remove_fn = None

    def remove_to_white(self, image: Image.Image) -> Image.Image:
        """Composite foreground onto pure white RGB background; return original on failure."""
        if self._remove_fn is None:
            return image

        try:
            rgba = self._remove_fn(image, session=self._session)
            if rgba.mode != "RGBA":
                rgba = rgba.convert("RGBA")
            # Check if alpha channel accidentally erased almost everything
            alpha_extrema = rgba.getchannel("A").getextrema()
            if alpha_extrema[1] < 16:
                return image

            white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
            white_bg.paste(rgba, mask=rgba.split()[3])
            return white_bg
        except Exception as exc:
            logger.warning("Background removal failed (%s); using original image.", exc)
            return image
