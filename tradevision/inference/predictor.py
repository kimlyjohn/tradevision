"""
tradevision/inference/predictor.py — Model loading and inference.

Implements a singleton pattern so the model is loaded from disk only once
and reused across all Streamlit reruns / requests.

Public API
----------
- ``predict(image_bytes)`` → dict with pattern, confidence, all_scores, low_confidence
- ``get_predictor()`` → shared :class:`Predictor` instance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import config
from tradevision.data.preprocessor import preprocess_image_bytes
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Custom exceptions ────────────────────────────────────────────────────────


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model file cannot be found at MODEL_PATH."""


class InvalidImageError(ValueError):
    """Raised when the image cannot be decoded for inference."""


class InferenceError(RuntimeError):
    """Raised when the model fails during forward pass."""


class ModelMetadataError(RuntimeError):
    """Raised when model sidecar metadata is missing or inconsistent."""


# ─── Result type ──────────────────────────────────────────────────────────────


PredictionResult = dict[str, Any]
"""
{
    "pattern":        str,
    "confidence":     float,
    "all_scores":     dict[str, float],
    "low_confidence": bool,
}
"""


# ─── Predictor ────────────────────────────────────────────────────────────────


class Predictor:
    """Singleton wrapper around a loaded Keras model.

    Load once, predict many times.  Thread-safe for Streamlit's single-
    threaded execution model.

    Args:
        model_path: Path to the saved ``.keras`` file.
        class_names: List of class labels matching the model's output order.
    """

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        class_names: list[str] | None = None,
    ) -> None:
        self._model_path = model_path
        self._model: tf.keras.Model | None = None
        self._class_names: list[str] = class_names or []

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load the Keras model from disk if not already loaded."""
        if not self._model_path.exists():
            raise ModelNotFoundError(
                f"No model found at {self._model_path}.\n"
                "Train the model first:\n"
                "  python -m tradevision.model.trainer"
            )

        logger.info("Loading model from %s …", self._model_path)
        self._model = tf.keras.models.load_model(str(self._model_path))
        logger.info("Model loaded successfully (%s params)", f"{self._model.count_params():,}")

        # Try to load class names saved alongside the model
        if not self._class_names:
            class_names_file = self._model_path.parent / "class_names.json"
            if not class_names_file.exists():
                raise ModelMetadataError(
                    f"Missing model metadata: {class_names_file}. "
                    "Retrain the model so class_names.json is saved alongside the checkpoint."
                )

            self._class_names = json.loads(class_names_file.read_text())
            logger.info("Class names loaded: %s", self._class_names)

    def _ensure_loaded(self) -> None:
        """Lazily load the model on first inference call."""
        if self._model is None:
            self._load_model()

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded into memory."""
        return self._model is not None

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names (loads model if needed)."""
        self._ensure_loaded()
        return self._class_names

    def predict(self, image_bytes: bytes) -> PredictionResult:
        """Run inference on *image_bytes* and return a structured result.

        Args:
            image_bytes: Raw bytes of the uploaded chart image.

        Returns:
            Dict with keys: ``pattern``, ``confidence``, ``all_scores``,
            ``low_confidence``.

        Raises:
            ModelNotFoundError: If the model file does not exist.
            InvalidImageError: If the image bytes cannot be decoded.
            InferenceError: If the forward pass fails.
        """
        self._ensure_loaded()

        # Preprocess
        try:
            tensor = preprocess_image_bytes(image_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Image preprocessing failed")
            raise InvalidImageError(
                f"Could not preprocess the uploaded image: {exc}"
            ) from exc

        # Forward pass
        try:
            raw_output: np.ndarray = self._model.predict(tensor, verbose=0)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model forward pass failed")
            raise InferenceError(
                f"The model encountered an error during prediction: {exc}"
            ) from exc

        probabilities: np.ndarray = raw_output[0]  # shape (num_classes,)

        # Validate shape matches class names
        if len(probabilities) != len(self._class_names):
            raise InferenceError(
                f"Model output size ({len(probabilities)}) does not match "
                f"number of class names ({len(self._class_names)}). "
                "Re-train the model or update config.CLASS_NAMES."
            )

        top_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[top_idx])
        predicted_class = self._class_names[top_idx]
        sorted_indices = np.argsort(probabilities)[::-1]
        second_confidence = (
            float(probabilities[sorted_indices[1]])
            if len(sorted_indices) > 1
            else 0.0
        )
        confidence_margin = confidence - second_confidence
        low_confidence = (
            confidence < config.CONFIDENCE_THRESHOLD
            or confidence_margin < config.CONFIDENCE_MARGIN_THRESHOLD
        )
        pattern = (
            config.REJECTION_LABEL
            if low_confidence
            else predicted_class
        )

        all_scores = {
            name: float(prob)
            for name, prob in zip(self._class_names, probabilities)
        }

        result: PredictionResult = {
            "pattern": pattern,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_scores": all_scores,
            "low_confidence": low_confidence,
            "confidence_margin": confidence_margin,
        }

        logger.info(
            "Prediction → display='%s' top='%s' confidence=%.3f margin=%.3f low=%s",
            pattern, predicted_class, confidence, confidence_margin, result["low_confidence"],
        )
        return result


# ─── Singleton accessor ───────────────────────────────────────────────────────

_predictor_instance: Predictor | None = None


def get_predictor() -> Predictor:
    """Return the shared :class:`Predictor` singleton.

    This function is called by :func:`app.py`'s ``@st.cache_resource``
    wrapper so the model is loaded only once per Streamlit server process.

    Returns:
        The shared :class:`Predictor` instance.
    """
    global _predictor_instance  # noqa: PLW0603
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance
