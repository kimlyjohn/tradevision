"""
tradevision/model/builder.py — Transfer learning model architecture.

Builds a binary/multi-class classifier on top of a frozen ImageNet
backbone (EfficientNetB0 by default; swap via config.BACKBONE).

Architecture
------------
    Frozen backbone (EfficientNetB0 / MobileNetV2)
        ↓
    GlobalAveragePooling2D
        ↓
    Dense(DENSE_UNITS, activation='relu')
        ↓
    BatchNormalization
        ↓
    Dropout(DROPOUT_RATE)
        ↓
    Dense(num_classes, activation='softmax')
"""

from __future__ import annotations

import tensorflow as tf

import config
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


def _get_backbone(input_tensor: tf.Tensor) -> tuple[tf.keras.Model, tf.Tensor]:
    """Instantiate the pretrained backbone from *config.BACKBONE*.

    The backbone weights are frozen (``trainable = False``) so that only
    the new classification head is trained in the first phase.

    Args:
        input_tensor: The Keras input tensor that feeds into the backbone.

    Returns:
        Tuple of (backbone_model, backbone_output_tensor).

    Raises:
        ValueError: If *config.BACKBONE* is not a supported name.
    """
    backbone_name = config.BACKBONE
    kwargs = dict(
        include_top=False,
        weights="imagenet",
        input_tensor=input_tensor,
    )

    if backbone_name == "EfficientNetB0":
        backbone = tf.keras.applications.EfficientNetB0(**kwargs)
    elif backbone_name == "MobileNetV2":
        backbone = tf.keras.applications.MobileNetV2(**kwargs)
    else:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. "
            "Set config.BACKBONE to 'EfficientNetB0' or 'MobileNetV2'."
        )

    backbone.trainable = False
    logger.info("Backbone: %s | frozen layers: %d", backbone_name, len(backbone.layers))
    return backbone, backbone.output


def build_model(num_classes: int) -> tf.keras.Model:
    """Build and compile the TradeVision classification model.

    Args:
        num_classes: Number of chart-pattern classes to predict.

    Returns:
        A compiled :class:`tf.keras.Model` ready for ``model.fit()``.
    """
    h, w = config.INPUT_SIZE
    inputs = tf.keras.Input(shape=(h, w, 3), name="chart_image")

    backbone, x = _get_backbone(inputs)

    # ── Classification head ────────────────────────────────────────────────
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(
        config.DENSE_UNITS, activation="relu", name="dense_head"
    )(x)
    x = tf.keras.layers.BatchNormalization(name="batch_norm")(x)
    x = tf.keras.layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="predictions"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tradevision")

    # ── Compile ───────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    logger.info(
        "Model compiled: backbone=%s | classes=%d | params=%s",
        config.BACKBONE,
        num_classes,
        f"{model.count_params():,}",
    )
    return model
