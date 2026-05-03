"""
tradevision/data/preprocessor.py — Image preprocessing pipeline.

Handles:
- Resize to INPUT_SIZE
- Normalise pixel values to [0, 1]
- Data augmentation (training only)
- Grayscale → RGB conversion
"""

from __future__ import annotations

import io

import numpy as np
import tensorflow as tf
from PIL import Image

import config
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Augmentation layers (training only) ─────────────────────────────────────

def build_augmentation_layer() -> tf.keras.Sequential:
    """Return a Keras Sequential of random augmentation layers.

    Only applied to the training set.  Uses Keras built-in preprocessing
    layers so augmentation runs on the GPU when available.

    Returns:
        A compiled :class:`tf.keras.Sequential` of augmentation ops.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.20),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.20),
            tf.keras.layers.RandomBrightness(0.20),
            tf.keras.layers.RandomContrast(0.20),
        ],
        name="augmentation",
    )


def _preprocess_array_by_backbone(arr: np.ndarray) -> np.ndarray:
    """Apply backbone-specific preprocessing to an image batch array."""
    if config.BACKBONE == "EfficientNetB0":
        return arr
    if config.BACKBONE == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    raise ValueError(f"Unsupported backbone '{config.BACKBONE}'")


def preprocess_tensor_for_backbone(image: tf.Tensor) -> tf.Tensor:
    """Apply backbone-specific preprocessing to a single image tensor."""
    image = tf.cast(image, tf.float32)

    if config.BACKBONE == "EfficientNetB0":
        return image
    if config.BACKBONE == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)
    raise ValueError(f"Unsupported backbone '{config.BACKBONE}'")


# ─── Single-image preprocessing for inference ─────────────────────────────────


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes into a model-ready numpy array.

    Steps:
    1. Decode via Pillow (handles JPEG / PNG / WEBP automatically).
    2. Convert to RGB (handles grayscale, RGBA, palette modes).
    3. Resize to ``config.INPUT_SIZE``.
    4. Apply backbone-specific preprocessing.
    5. Add batch dimension → shape ``(1, H, W, 3)``.

    Args:
        image_bytes: Raw file bytes from an uploaded image.

    Returns:
        Float32 numpy array of shape ``(1, H, W, 3)`` ready for model input.
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Grayscale / palette → RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
        logger.debug("Converted image mode to RGB")

    img = img.resize(config.INPUT_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)

    # Shape: (H, W, 3) → (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)
    arr = _preprocess_array_by_backbone(arr)
    logger.debug("Preprocessed image → shape %s, range [%.3f, %.3f]",
                 arr.shape, arr.min(), arr.max())
    return arr


# ─── tf.data dataset preprocessors ───────────────────────────────────────────


def preprocess_for_training(
    image: tf.Tensor,
    label: tf.Tensor,
    augmentation_layer: tf.keras.Sequential,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess a single (image, label) pair for the training split.

    Augmentation is applied here so it happens at batch-build time
    inside the ``tf.data`` pipeline.

    Args:
        image: Float32 tensor, already decoded and scaled to [0, 1].
        label: Integer label tensor.
        augmentation_layer: Pre-built augmentation layer.

    Returns:
        Tuple of (augmented_image, label).
    """
    image = augmentation_layer(image, training=True)
    image = preprocess_tensor_for_backbone(image)
    return image, label


def preprocess_for_eval(
    image: tf.Tensor,
    label: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Preprocess a (image, label) pair for val/test (no augmentation).

    Args:
        image: Float32 tensor already scaled to [0, 1].
        label: Integer label tensor.

    Returns:
        Unchanged (image, label).
    """
    image = preprocess_tensor_for_backbone(image)
    return image, label
