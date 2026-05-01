"""
tests/test_predictor.py — Unit tests for the inference Predictor.

Uses unittest.mock to avoid loading a real model during CI.

Tests:
- predict() output structure and key presence
- low_confidence flag triggers correctly
- ModelNotFoundError raised when no model file exists
- InvalidImageError raised for bad image bytes
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from tradevision.inference.predictor import (
    InferenceError,
    InvalidImageError,
    ModelMetadataError,
    ModelNotFoundError,
    Predictor,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


CLASS_NAMES = ["Double Top", "Head and Shoulders", "Flag", "Wedge"]
NUM_CLASSES = len(CLASS_NAMES)


def _make_valid_png() -> bytes:
    img = Image.new("RGB", (100, 100), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_mock_model(probs: list[float]) -> MagicMock:
    """Return a MagicMock whose .predict() returns a fixed probability array."""
    model = MagicMock()
    model.count_params.return_value = 100_000
    model.predict.return_value = np.array([probs], dtype=np.float32)
    return model


def _make_predictor_with_mock_model(probs: list[float]) -> Predictor:
    """Build a Predictor pre-loaded with a mock model."""
    predictor = Predictor.__new__(Predictor)
    predictor._model_path = Path("models/tradevision_best.keras")  # noqa: SLF001
    predictor._model = _make_mock_model(probs)  # noqa: SLF001
    predictor._class_names = CLASS_NAMES  # noqa: SLF001
    return predictor


# ── Output structure ───────────────────────────────────────────────────────────


class TestPredictOutputStructure:
    def test_result_has_required_keys(self) -> None:
        probs = [0.05, 0.80, 0.10, 0.05]
        predictor = _make_predictor_with_mock_model(probs)

        result = predictor.predict(_make_valid_png())

        assert "pattern" in result
        assert "confidence" in result
        assert "all_scores" in result
        assert "low_confidence" in result

    def test_pattern_is_string(self) -> None:
        probs = [0.05, 0.80, 0.10, 0.05]
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        assert isinstance(result["pattern"], str)

    def test_confidence_is_float(self) -> None:
        probs = [0.05, 0.80, 0.10, 0.05]
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        assert isinstance(result["confidence"], float)

    def test_all_scores_has_all_classes(self) -> None:
        probs = [0.05, 0.80, 0.10, 0.05]
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        assert set(result["all_scores"].keys()) == set(CLASS_NAMES)

    def test_confidence_matches_top_class(self) -> None:
        probs = [0.05, 0.80, 0.10, 0.05]
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        # Index 1 = "Head and Shoulders", prob 0.80
        assert result["pattern"] == "Head and Shoulders"
        assert pytest.approx(result["confidence"], abs=1e-4) == 0.80


# ── low_confidence flag ────────────────────────────────────────────────────────


class TestLowConfidenceFlag:
    def test_high_confidence_not_flagged(self) -> None:
        import config
        # Confidence above threshold → low_confidence should be False
        high = config.CONFIDENCE_THRESHOLD + 0.20
        low_rest = (1.0 - high) / (NUM_CLASSES - 1)
        probs = [high] + [low_rest] * (NUM_CLASSES - 1)
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        assert result["low_confidence"] is False

    def test_below_threshold_flagged(self) -> None:
        import config
        # Confidence below threshold → low_confidence must be True
        low = config.CONFIDENCE_THRESHOLD - 0.05
        rest = (1.0 - low) / (NUM_CLASSES - 1)
        probs = [low] + [rest] * (NUM_CLASSES - 1)
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        assert result["low_confidence"] is True

    def test_exactly_at_threshold_not_flagged(self) -> None:
        import config
        threshold = config.CONFIDENCE_THRESHOLD
        rest = (1.0 - threshold) / (NUM_CLASSES - 1)
        probs = [threshold] + [rest] * (NUM_CLASSES - 1)
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())
        # Exactly at threshold → confidence >= threshold → NOT low
        assert result["low_confidence"] is False

    def test_low_margin_is_rejected(self) -> None:
        import config

        probs = [0.45, 0.43, 0.07, 0.05]
        predictor = _make_predictor_with_mock_model(probs)
        result = predictor.predict(_make_valid_png())

        assert result["low_confidence"] is True
        assert result["pattern"] == config.REJECTION_LABEL
        assert result["predicted_class"] == "Double Top"


# ── Error handling ─────────────────────────────────────────────────────────────


class TestPredictorErrors:
    def test_model_not_found_error(self) -> None:
        """ModelNotFoundError raised when model file does not exist."""
        predictor = Predictor(model_path=Path("/nonexistent/model.keras"))
        with pytest.raises(ModelNotFoundError):
            predictor.predict(_make_valid_png())

    def test_invalid_image_error_for_garbage_bytes(self) -> None:
        """InvalidImageError raised when image bytes cannot be decoded."""
        probs = [0.25, 0.25, 0.25, 0.25]
        predictor = _make_predictor_with_mock_model(probs)
        with pytest.raises(InvalidImageError):
            predictor.predict(b"\x00\x01\x02\x03" * 10)

    def test_inference_error_on_model_failure(self) -> None:
        """InferenceError raised when the model's predict() throws."""
        probs = [0.25, 0.25, 0.25, 0.25]
        predictor = _make_predictor_with_mock_model(probs)
        # Make the model's predict() method raise an exception
        predictor._model.predict.side_effect = RuntimeError("GPU OOM")  # noqa: SLF001
        with pytest.raises(InferenceError):
            predictor.predict(_make_valid_png())

    def test_inference_error_on_class_name_mismatch(self) -> None:
        """InferenceError raised when model output size ≠ num class names."""
        # Model returns 6 classes but we only have 4 class names
        model = _make_mock_model([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
        predictor = Predictor.__new__(Predictor)
        predictor._model_path = Path("models/tradevision_best.keras")
        predictor._model = model
        predictor._class_names = CLASS_NAMES  # only 4 names, model has 6 outputs
        with pytest.raises(InferenceError, match="does not match"):
            predictor.predict(_make_valid_png())

    def test_missing_class_names_metadata_raises(self, tmp_path: Path) -> None:
        model_path = tmp_path / "tradevision_best.keras"
        model_path.write_text("stub model")
        predictor = Predictor(model_path=model_path)

        with patch("tensorflow.keras.models.load_model", return_value=_make_mock_model([0.25] * 4)):
            with pytest.raises(ModelMetadataError, match="class_names.json"):
                predictor.predict(_make_valid_png())
