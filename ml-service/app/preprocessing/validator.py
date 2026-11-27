from __future__ import annotations

import io
from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails size, type, or decode checks."""


def validate_and_decode_image(
    raw_bytes: bytes,
    content_type: str | None,
    max_bytes: int = 10 * 1024 * 1024,
) -> Image.Image:
    """Validate image size, MIME type, and decode into an orientation-corrected RGB PIL Image."""
    if not raw_bytes:
        raise ImageValidationError("Empty image payload.")

    if len(raw_bytes) > max_bytes:
        raise ImageValidationError(
            f"Image exceeds maximum size of {max_bytes // (1024 * 1024)} MB."
        )

    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        # Allow octet-stream only if PIL can actually decode a supported format
        if content_type.lower() != "application/octet-stream":
            raise ImageValidationError(f"Unsupported image content type: {content_type}")

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.verify()
        with Image.open(io.BytesIO(raw_bytes)) as img:
            if img.format and f"image/{img.format.lower()}" not in ALLOWED_MIME_TYPES:
                raise ImageValidationError(f"Unsupported image format: {img.format}")
            corrected = ImageOps.exif_transpose(img)
            return corrected.convert("RGB")
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError("Unable to decode image file.") from exc
