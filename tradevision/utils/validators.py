"""
tradevision/utils/validators.py — Input validation for uploaded images.

All validators raise descriptive, user-friendly exceptions rather
than propagating raw Python errors to the Streamlit UI.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Custom exceptions ────────────────────────────────────────────────────────


class UnsupportedFormatError(ValueError):
    """Raised when the uploaded file has an unsupported extension."""


class FileTooLargeError(ValueError):
    """Raised when the uploaded file exceeds the size limit."""


class CorruptImageError(ValueError):
    """Raised when the image cannot be decoded or is corrupt."""


# ─── Validator functions ──────────────────────────────────────────────────────


def validate_file_extension(filename: str, supported: list[str] | None = None) -> None:
    """Ensure *filename* has a supported image extension.

    Args:
        filename: The original filename of the uploaded file.
        supported: Allowed extensions (with leading dot). Defaults to
            :data:`config.SUPPORTED_FORMATS`.

    Raises:
        UnsupportedFormatError: If the extension is not in *supported*.
    """
    if supported is None:
        from config import SUPPORTED_FORMATS  # noqa: PLC0415

        supported = SUPPORTED_FORMATS

    ext = Path(filename).suffix.lower()
    if ext not in supported:
        allowed = ", ".join(supported)
        raise UnsupportedFormatError(
            f"File type '{ext}' is not supported. "
            f"Please upload one of: {allowed}"
        )

    logger.debug("File extension OK: %s", ext)


def validate_file_size(file_bytes: bytes, max_mb: int | None = None) -> None:
    """Ensure *file_bytes* does not exceed *max_mb* megabytes.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        max_mb: Maximum allowed size in megabytes. Defaults to
            :data:`config.MAX_FILE_SIZE_MB`.

    Raises:
        FileTooLargeError: If the file size exceeds the limit.
    """
    if max_mb is None:
        from config import MAX_FILE_SIZE_MB  # noqa: PLC0415

        max_mb = MAX_FILE_SIZE_MB

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise FileTooLargeError(
            f"This file is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {max_mb} MB."
        )

    logger.debug("File size OK: %.2f MB", size_mb)


def validate_image_integrity(file_bytes: bytes) -> None:
    """Verify that *file_bytes* is a valid, decodable image.

    Uses ``PIL.Image.open`` + ``Image.verify()`` to catch truncated
    or corrupt files. Opens a second handle to check pixel readability.

    Args:
        file_bytes: Raw bytes of the uploaded file.

    Raises:
        CorruptImageError: If the image cannot be decoded.
    """
    if len(file_bytes) == 0:
        raise CorruptImageError("The uploaded file is empty (0 bytes).")

    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()  # verify() closes the file after checking
    except UnidentifiedImageError:
        raise CorruptImageError(
            "This file could not be identified as a valid image. "
            "Please upload a real chart screenshot or photo."
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Image integrity check failed: %s", exc)
        raise CorruptImageError(
            "The image appears to be corrupt or incomplete. "
            "Please re-export or re-save the chart image and try again."
        ) from exc

    # Re-open for pixel loading (verify() leaves the file consumed)
    try:
        img2 = Image.open(io.BytesIO(file_bytes))
        img2.load()
    except Exception as exc:  # noqa: BLE001
        raise CorruptImageError(
            "Could not load image pixel data. The file may be truncated."
        ) from exc

    logger.debug("Image integrity OK")


def validate_upload(
    filename: str,
    file_bytes: bytes,
    supported: list[str] | None = None,
    max_mb: int | None = None,
) -> None:
    """Run all three validations in sequence.

    Convenience wrapper used by the Streamlit UI so that a single
    call covers extension, size, and integrity checks.

    Args:
        filename: Original filename from the uploader widget.
        file_bytes: Raw file content.
        supported: Allowed extensions. Falls back to config.
        max_mb: Max size in MB. Falls back to config.

    Raises:
        UnsupportedFormatError | FileTooLargeError | CorruptImageError
    """
    validate_file_extension(filename, supported)
    validate_file_size(file_bytes, max_mb)
    validate_image_integrity(file_bytes)
