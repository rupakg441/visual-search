from __future__ import annotations

from PIL import Image, ImageOps


def build_tta_views(image: Image.Image) -> list[Image.Image]:
    """Generate deterministic test-time augmentation views: original, horizontal flip, slight center crop."""
    w, h = image.size
    margin_x = max(1, int(w * 0.06))
    margin_y = max(1, int(h * 0.06))
    cropped = (
        image.crop((margin_x, margin_y, w - margin_x, h - margin_y))
        if w > 32 and h > 32
        else image
    )
    return [
        image,
        ImageOps.mirror(image),
        cropped,
    ]
