"""
tradevision/data/loader.py — Dataset loading and tf.data pipeline.

Loads a preserved train/valid/test split from ``data/processed/`` or re-splits a flat class-folder dataset.

Returns ``tf.data.Dataset`` objects ready for model.fit().
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import config
from tradevision.data.preprocessor import (
    build_augmentation_layer,
    preprocess_for_eval,
    preprocess_for_training,
)
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)

# Supported image globs when scanning the processed directory
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")

AUTOTUNE = tf.data.AUTOTUNE


class DatasetBundle(NamedTuple):
    """Holds all three splits and the detected class names."""

    train: tf.data.Dataset
    val: tf.data.Dataset
    test: tf.data.Dataset
    class_names: list[str]
    num_classes: int


def _collect_split_dirs(raw_dir: Path) -> dict[str, Path] | None:
    """Return canonical split directories when *raw_dir* is already split."""
    split_dirs = {
        "train": raw_dir / "train",
        "valid": raw_dir / "valid",
        "test": raw_dir / "test",
    }
    if all(path.is_dir() for path in split_dirs.values()):
        return split_dirs
    return None


def _discover_class_names(split_dirs: dict[str, Path]) -> list[str]:
    """Collect the sorted union of class-folder names across all splits."""
    class_names = sorted(
        {
            class_dir.name
            for split_dir in split_dirs.values()
            for class_dir in split_dir.iterdir()
            if class_dir.is_dir()
        }
    )
    if not class_names:
        raise FileNotFoundError(
            "No class subdirectories found in split-aware processed dataset.\n"
            "Expected folders like data/processed/train/<class-name>/..."
        )
    return class_names


def _collect_paths_for_class_dirs(
    root_dir: Path,
    class_names: list[str],
) -> tuple[list[Path], list[int]]:
    """Collect image paths and labels from a class-folder dataset root."""
    paths: list[Path] = []
    labels: list[int] = []

    for idx, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        found: list[Path] = []
        if class_dir.is_dir():
            for glob in _IMAGE_EXTS:
                found.extend(class_dir.glob(glob))

        logger.info("Split '%s' class '%s' → %d images", root_dir.name, class_name, len(found))
        paths.extend(found)
        labels.extend([idx] * len(found))

    return paths, labels


def _collect_preserved_splits(
    raw_dir: Path,
) -> tuple[
    list[Path],
    list[int],
    list[Path],
    list[int],
    list[Path],
    list[int],
    list[str],
]:
    """Load a processed dataset that already preserves train/valid/test splits."""
    split_dirs = _collect_split_dirs(raw_dir)
    if split_dirs is None:
        raise FileNotFoundError(
            f"Expected split-aware dataset under {raw_dir}, but one or more split directories are missing."
        )

    class_names = _discover_class_names(split_dirs)
    train_paths, train_labels = _collect_paths_for_class_dirs(split_dirs["train"], class_names)
    val_paths, val_labels = _collect_paths_for_class_dirs(split_dirs["valid"], class_names)
    test_paths, test_labels = _collect_paths_for_class_dirs(split_dirs["test"], class_names)

    return (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
        class_names,
    )


def _collect_image_paths(raw_dir: Path) -> tuple[list[Path], list[int], list[str]]:
    """Walk *raw_dir* and collect image paths, integer labels, and class names.

    Expects the following layout::

        raw_dir/
          class_a/
            img1.jpg
            img2.png
          class_b/
            img3.jpg
          ...

    Args:
        raw_dir: Root directory of the dataset.

    Returns:
        Tuple of (paths, labels, class_names).

    Raises:
        FileNotFoundError: If *raw_dir* does not exist or contains no class folders.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {raw_dir}\n"
            "Rebuild it with:  python data/reorganize_yolov8.py"
        )

    class_dirs = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subdirectories found in {raw_dir}.\n"
            "Each chart pattern should be in its own folder."
        )

    class_names = [d.name for d in class_dirs]
    paths: list[Path] = []
    labels: list[int] = []

    for idx, cls_dir in enumerate(class_dirs):
        found = []
        for glob in _IMAGE_EXTS:
            found.extend(cls_dir.glob(glob))

        logger.info("Class '%s' → %d images", cls_dir.name, len(found))
        paths.extend(found)
        labels.extend([idx] * len(found))

    logger.info("Dataset: %d images across %d classes", len(paths), len(class_names))
    return paths, labels, class_names


def _load_and_decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """tf.data map fn: read an image file and decode to float32 pixels."""
    raw = tf.io.read_file(path)
    image = tf.image.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, config.INPUT_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def _validate_split_inputs(labels: list[int]) -> None:
    """Validate split ratios and per-class counts before stratified splitting."""
    total_ratio = config.TRAIN_SPLIT + config.VAL_SPLIT + config.TEST_SPLIT
    if not math.isclose(total_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("TRAIN_SPLIT, VAL_SPLIT, and TEST_SPLIT must sum to 1.0.")

    counts = Counter(labels)
    undersized = {label: count for label, count in counts.items() if count < 3}
    if undersized:
        raise ValueError(
            "Each class needs at least 3 images for a stratified 70/15/15 split. "
            f"Undersized classes: {undersized}"
        )


def _stratified_split(
    paths: list[Path],
    labels: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset paths and labels into stratified train/val/test partitions."""
    _validate_split_inputs(labels)

    paths_np = np.array([str(path) for path in paths], dtype=str)
    labels_np = np.array(labels, dtype=np.int32)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths_np,
        labels_np,
        train_size=config.TRAIN_SPLIT,
        stratify=labels_np,
        random_state=42,
    )
    temp_ratio = config.VAL_SPLIT + config.TEST_SPLIT
    val_ratio_within_temp = config.VAL_SPLIT / temp_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_ratio_within_temp,
        stratify=temp_labels,
        random_state=42,
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def load_datasets(raw_dir: Path | None = None) -> DatasetBundle:
    """Load the processed dataset, split it, and log class distribution.

    Args:
        raw_dir: Root of the processed class-folder dataset. Defaults to
            :data:`config.DATA_PROCESSED_DIR`.

    Returns:
        A DatasetBundle containing train, validation, and test datasets.
    """
    if raw_dir is None:
        raw_dir = config.DATA_PROCESSED_DIR

    split_dirs = _collect_split_dirs(raw_dir)
    if split_dirs is not None:
        logger.info("Detected split-aware processed dataset; preserving original train/valid/test boundaries.")
        (
            train_paths,
            train_labels,
            val_paths,
            val_labels,
            test_paths,
            test_labels,
            class_names,
        ) = _collect_preserved_splits(raw_dir)
    else:
        logger.warning(
            "Using legacy flat processed dataset layout. "
            "Regenerate data/processed with python data/reorganize_yolov8.py to preserve source splits."
        )
        image_paths, labels, class_names = _collect_image_paths(raw_dir)
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
            _stratified_split(
                image_paths,
                labels,
            )
        )

    if min(len(train_paths), len(val_paths), len(test_paths)) == 0:
        raise ValueError(
            "Train/validation/test split produced an empty partition. "
            "Add more images per class or adjust the split ratios."
        )

    logger.info(
        "Splits → train: %d | val: %d | test: %d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    augmentation = build_augmentation_layer()

    # ── Build tf.data pipelines ───────────────────────────────────────────
    def make_ds(paths_: tf.Tensor, labels_: tf.Tensor) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((paths_, labels_))

    train_paths_t = tf.constant([str(path) for path in train_paths])
    train_labels_t = tf.constant(train_labels, dtype=tf.int32)
    val_paths_t = tf.constant([str(path) for path in val_paths])
    val_labels_t = tf.constant(val_labels, dtype=tf.int32)
    test_paths_t = tf.constant([str(path) for path in test_paths])
    test_labels_t = tf.constant(test_labels, dtype=tf.int32)

    train_ds = (
        make_ds(train_paths_t, train_labels_t)
        .map(_load_and_decode, num_parallel_calls=AUTOTUNE)
        .map(
            lambda img, lbl: preprocess_for_training(img, lbl, augmentation),
            num_parallel_calls=AUTOTUNE,
        )
        .shuffle(buffer_size=min(1000, len(train_paths)), seed=42)
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        make_ds(val_paths_t, val_labels_t)
        .map(_load_and_decode, num_parallel_calls=AUTOTUNE)
        .map(preprocess_for_eval, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        make_ds(test_paths_t, test_labels_t)
        .map(_load_and_decode, num_parallel_calls=AUTOTUNE)
        .map(preprocess_for_eval, num_parallel_calls=AUTOTUNE)
        .batch(config.BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    return DatasetBundle(
        train=train_ds,
        val=val_ds,
        test=test_ds,
        class_names=class_names,
        num_classes=len(class_names),
    )
