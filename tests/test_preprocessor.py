"""
tests/test_preprocessor.py — Unit tests for image preprocessing pipeline.

Tests:
- Output shape after resize
- Pixel normalisation range [0, 1]
- Grayscale → RGB conversion
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_png_bytes(width: int, height: int, mode: str = "RGB") -> bytes:
    """Create a minimal in-memory PNG image and return its raw bytes."""
    if mode == "RGB":
        color = (100, 150, 200)
    elif mode == "L":
        color = 128  # type: ignore[assignment]
    elif mode == "RGBA":
        color = (100, 150, 200, 255)  # type: ignore[assignment]
    else:
        color = (100, 150, 200)  # type: ignore[assignment]

    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestPreprocessImageBytes:
    """Tests for tradevision.data.preprocessor.preprocess_image_bytes."""

    def test_output_shape_is_correct(self) -> None:
        """Output tensor must have shape (1, 224, 224, 3)."""
        import config
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(640, 480)
        result = preprocess_image_bytes(png)

        h, w = config.INPUT_SIZE
        assert result.shape == (1, h, w, 3), f"Expected (1,{h},{w},3), got {result.shape}"

    def test_pixel_values_are_finite(self) -> None:
        """Preprocessed pixels must remain finite after backbone preprocessing."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(300, 300)
        result = preprocess_image_bytes(png)

        assert np.isfinite(result).all(), "Preprocessed image contains non-finite values"

    def test_grayscale_converted_to_rgb(self) -> None:
        """Grayscale images must be converted to 3-channel output."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(100, 100, mode="L")  # grayscale
        result = preprocess_image_bytes(png)

        assert result.shape[-1] == 3, "Expected 3 channels (RGB) after conversion"

    def test_rgba_converted_to_rgb(self) -> None:
        """RGBA images must be converted to 3-channel RGB output."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(100, 100, mode="RGBA")
        result = preprocess_image_bytes(png)

        assert result.shape[-1] == 3, "Expected 3 channels (RGB) after RGBA conversion"

    def test_large_image_resized_correctly(self) -> None:
        """A 4000×3000 image should still resize to the correct target dimensions."""
        import config
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(4000, 3000)
        result = preprocess_image_bytes(png)

        h, w = config.INPUT_SIZE
        assert result.shape == (1, h, w, 3)

    def test_small_image_resized_correctly(self) -> None:
        """A 10×10 (very small) image should be upscaled to the target dimensions."""
        import config
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(10, 10)
        result = preprocess_image_bytes(png)

        h, w = config.INPUT_SIZE
        assert result.shape == (1, h, w, 3)

    def test_dtype_is_float32(self) -> None:
        """Output dtype must be float32."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(224, 224)
        result = preprocess_image_bytes(png)

        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

    def test_batch_dim_is_one(self) -> None:
        """The batch dimension must always be 1."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        png = _make_png_bytes(224, 224)
        result = preprocess_image_bytes(png)

        assert result.shape[0] == 1, "Batch dimension must be 1"

    def test_efficientnet_preserves_zero_to_255_scale(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EfficientNetB0 inputs should remain in the [0, 255] range."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        monkeypatch.setattr("config.BACKBONE", "EfficientNetB0")
        png = _make_png_bytes(224, 224)
        result = preprocess_image_bytes(png)

        assert result.min() >= 0.0
        assert result.max() > 1.0

    def test_mobilenetv2_scales_to_minus_one_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MobileNetV2 inputs should be preprocessed to the [-1, 1] range."""
        from tradevision.data.preprocessor import preprocess_image_bytes

        monkeypatch.setattr("config.BACKBONE", "MobileNetV2")
        png = _make_png_bytes(224, 224)
        result = preprocess_image_bytes(png)

        assert result.min() >= -1.0
        assert result.max() <= 1.0
