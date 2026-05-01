"""
tests/test_loader.py — Tests for dataset loading and splitting.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from tradevision.data.loader import _stratified_split, load_datasets


def _write_png(path: Path, pixel_value: int) -> None:
    image = Image.fromarray(np.full((8, 8, 3), pixel_value, dtype=np.uint8), mode="RGB")
    image.save(path, format="PNG")


def _make_dataset(root: Path, images_per_class: int = 10) -> None:
    for class_name, pixel_value in (
        ("ascending_triangle", 10),
        ("double_top", 20),
        ("flag", 30),
    ):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for idx in range(images_per_class):
            _write_png(class_dir / f"{idx}.png", pixel_value)


class AddFiveLayer(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        return inputs + 5.0


def test_stratified_split_is_disjoint_and_preserves_all_classes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, images_per_class=10)

    paths = sorted(raw_dir.rglob("*.png"))
    class_names = sorted({path.parent.name for path in paths})
    labels = [class_names.index(path.parent.name) for path in paths]

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = _stratified_split(paths, labels)

    assert not (set(train_paths) & set(val_paths))
    assert not (set(train_paths) & set(test_paths))
    assert not (set(val_paths) & set(test_paths))
    assert set(train_paths) | set(val_paths) | set(test_paths) == {str(path) for path in paths}
    assert set(train_labels) == {0, 1, 2}
    assert set(val_labels) == {0, 1, 2}
    assert set(test_labels) == {0, 1, 2}

    assert Counter(train_labels) == Counter({0: 7, 1: 7, 2: 7})
    assert len(val_paths) == 4
    assert len(test_paths) == 5


def test_load_datasets_applies_augmentation_only_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, images_per_class=10)

    monkeypatch.setattr("tradevision.data.loader.build_augmentation_layer", lambda: AddFiveLayer())
    monkeypatch.setattr("config.BACKBONE", "EfficientNetB0")
    monkeypatch.setattr("config.BATCH_SIZE", 4)

    bundle = load_datasets(raw_dir)

    train_images, _ = next(iter(bundle.train.take(1)))
    val_images, _ = next(iter(bundle.val.take(1)))
    test_images, _ = next(iter(bundle.test.take(1)))

    assert float(tf.reduce_min(train_images).numpy()) == 15.0
    assert float(tf.reduce_min(val_images).numpy()) in {10.0, 20.0, 30.0}
    assert float(tf.reduce_min(test_images).numpy()) in {10.0, 20.0, 30.0}


def test_load_datasets_rejects_classes_with_too_few_images(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_dataset(raw_dir, images_per_class=2)

    with pytest.raises(ValueError, match="at least 3 images"):
        load_datasets(raw_dir)
