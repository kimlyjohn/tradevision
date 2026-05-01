"""
tests/test_validators.py — Unit tests for input validation utilities.

Tests:
- validate_file_extension (valid/invalid/edge cases)
- validate_file_size (within limit / over limit / zero)
- validate_image_integrity (valid PNG / corrupt / empty)
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from tradevision.utils.validators import (
    CorruptImageError,
    FileTooLargeError,
    UnsupportedFormatError,
    validate_file_extension,
    validate_file_size,
    validate_image_integrity,
    validate_upload,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_valid_png(width: int = 100, height: int = 100) -> bytes:
    """Return bytes of a valid minimal PNG image."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


SUPPORTED = [".jpg", ".jpeg", ".png", ".webp"]


# ── validate_file_extension ────────────────────────────────────────────────────


class TestValidateFileExtension:
    def test_valid_jpg(self) -> None:
        validate_file_extension("chart.jpg", SUPPORTED)

    def test_valid_jpeg(self) -> None:
        validate_file_extension("chart.JPEG", SUPPORTED)  # case-insensitive

    def test_valid_png(self) -> None:
        validate_file_extension("my_chart.png", SUPPORTED)

    def test_valid_webp(self) -> None:
        validate_file_extension("chart.webp", SUPPORTED)

    def test_invalid_pdf(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            validate_file_extension("chart.pdf", SUPPORTED)

    def test_invalid_gif(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            validate_file_extension("chart.gif", SUPPORTED)

    def test_invalid_no_extension(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            validate_file_extension("chartfile", SUPPORTED)

    def test_invalid_txt(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            validate_file_extension("data.txt", SUPPORTED)

    def test_error_message_mentions_file_type(self) -> None:
        with pytest.raises(UnsupportedFormatError, match=r"\.bmp"):
            validate_file_extension("chart.bmp", SUPPORTED)


# ── validate_file_size ─────────────────────────────────────────────────────────


class TestValidateFileSize:
    def test_small_file_passes(self) -> None:
        data = b"x" * 1024  # 1 KB
        validate_file_size(data, max_mb=10)

    def test_exact_limit_passes(self) -> None:
        data = b"x" * (10 * 1024 * 1024)  # exactly 10 MB
        validate_file_size(data, max_mb=10)

    def test_over_limit_raises(self) -> None:
        data = b"x" * (11 * 1024 * 1024)  # 11 MB
        with pytest.raises(FileTooLargeError):
            validate_file_size(data, max_mb=10)

    def test_zero_bytes_passes_size_check(self) -> None:
        # Size check passes; integrity check would fail separately
        validate_file_size(b"", max_mb=10)

    def test_error_message_shows_size(self) -> None:
        data = b"x" * (15 * 1024 * 1024)
        with pytest.raises(FileTooLargeError, match="15"):
            validate_file_size(data, max_mb=10)


# ── validate_image_integrity ──────────────────────────────────────────────────


class TestValidateImageIntegrity:
    def test_valid_png_passes(self) -> None:
        validate_image_integrity(_make_valid_png())

    def test_valid_jpeg_passes(self) -> None:
        img = Image.new("RGB", (100, 100), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        validate_image_integrity(buf.getvalue())

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(CorruptImageError, match="empty"):
            validate_image_integrity(b"")

    def test_random_bytes_raises(self) -> None:
        with pytest.raises(CorruptImageError):
            validate_image_integrity(b"\x00\x01\x02\x03" * 100)

    def test_truncated_png_raises(self) -> None:
        data = _make_valid_png()
        truncated = data[: len(data) // 3]  # cut off last 2/3
        with pytest.raises(CorruptImageError):
            validate_image_integrity(truncated)

    def test_text_file_disguised_as_png_raises(self) -> None:
        with pytest.raises(CorruptImageError):
            validate_image_integrity(b"This is not an image\n" * 100)


# ── validate_upload (integration) ────────────────────────────────────────────


class TestValidateUpload:
    def test_valid_png_upload(self) -> None:
        validate_upload("chart.png", _make_valid_png(), SUPPORTED, 10)

    def test_bad_extension_fails_first(self) -> None:
        with pytest.raises(UnsupportedFormatError):
            validate_upload("doc.pdf", _make_valid_png(), SUPPORTED, 10)

    def test_large_file_caught_second(self) -> None:
        huge = b"x" * (20 * 1024 * 1024)
        with pytest.raises(FileTooLargeError):
            validate_upload("chart.png", huge, SUPPORTED, 10)
