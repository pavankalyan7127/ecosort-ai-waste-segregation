"""Image preprocessing and validation utilities for EcoSort AI vision pipeline."""

import io
import logging
from typing import Tuple
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB limit
MIN_DIMENSION = 32
MAX_DIMENSION = 4096
SUPPORTED_FORMATS = {"JPEG", "JPG", "PNG", "WEBP", "BMP"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation or is malformed."""
    pass


def validate_and_load_image(image_bytes: bytes) -> Tuple[Image.Image, Tuple[int, int, int]]:
    """Validate image bytes, dimensions, and format, returning an RGB PIL Image

    and its dimensions (width, height, channels).

    Raises ImageValidationError on any corrupt, empty, or unsupported input.
    """
    if not image_bytes:
        raise ImageValidationError("Empty image payload received (0 bytes).")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(image_bytes) / (1024 * 1024)
        raise ImageValidationError(
            f"Image payload size ({size_mb:.2f} MB) exceeds maximum limit of 15 MB."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()  # Force load image data to catch truncated / corrupted streams
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        logger.warning(f"Malformed or corrupt image stream: {exc}")
        raise ImageValidationError(f"Invalid or corrupted image stream: {exc}")

    img_format = (image.format or "").upper()
    if img_format not in SUPPORTED_FORMATS:
        raise ImageValidationError(
            f"Unsupported image format '{img_format}'. Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )

    width, height = image.size
    if width < MIN_DIMENSION or height < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) are too small. Minimum required is {MIN_DIMENSION}x{MIN_DIMENSION}."
        )

    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_DIMENSION}."
        )

    # Convert to standard RGB (3 channels) safely handling alpha, paletted, or grayscale images
    if image.mode != "RGB":
        try:
            image = image.convert("RGB")
        except Exception as exc:
            raise ImageValidationError(f"Failed to convert image mode '{image.mode}' to RGB: {exc}")

    channels = 3
    return image, (width, height, channels)
