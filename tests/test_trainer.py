"""
tests/test_trainer.py — Tests for training callbacks and artifact handling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import tradevision.model.trainer as trainer


class FakeHistory:
    history = {
        "accuracy": [0.5, 0.7],
        "val_accuracy": [0.4, 0.8],
        "loss": [1.0, 0.6],
        "val_loss": [1.2, 0.7],
    }


class FakeModel:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path

    def summary(self, print_fn) -> None:
        print_fn("fake model")

    def fit(self, *_args, **_kwargs) -> FakeHistory:
        self.checkpoint_path.write_text("best model")
        return FakeHistory()


class FakeBestModel:
    def predict(self, images: tf.Tensor, verbose: int = 0) -> np.ndarray:
        batch_size = int(images.shape[0])
        return np.tile(np.array([[0.7, 0.2, 0.1]], dtype=np.float32), (batch_size, 1))


def test_build_callbacks_monitor_best_validation_accuracy() -> None:
    callbacks = trainer._build_callbacks()

    early_stop = next(cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.EarlyStopping))
    checkpoint = next(cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ModelCheckpoint))
    reduce_lr = next(cb for cb in callbacks if isinstance(cb, tf.keras.callbacks.ReduceLROnPlateau))

    assert early_stop.monitor == "val_accuracy"
    assert early_stop.mode == "max"
    assert checkpoint.monitor == "val_accuracy"
    assert checkpoint.mode == "max"
    assert reduce_lr.monitor == "val_loss"


def test_train_saves_class_names_after_checkpoint_and_evaluates_best_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "tradevision_best.keras"
    monkeypatch.setattr("config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("config.MODEL_PATH", model_path)
    monkeypatch.setattr(trainer, "_plot_training_curves", lambda history: None)
    monkeypatch.setattr(trainer, "_plot_confusion_matrix", lambda y_true, y_pred, class_names: None)

    bundle = SimpleNamespace(
        train=[],
        val=[],
        test=[(tf.zeros((2, 224, 224, 3), dtype=tf.float32), tf.constant([0, 1], dtype=tf.int32))],
        class_names=["ascending_triangle", "double_top", "flag"],
        num_classes=3,
    )
    monkeypatch.setattr(trainer, "load_datasets", lambda: bundle)
    monkeypatch.setattr(trainer, "build_model", lambda num_classes: FakeModel(model_path))

    report_calls: dict[str, object] = {}

    def fake_report(y_true, y_pred, labels, target_names, zero_division):
        report_calls["labels"] = labels
        report_calls["target_names"] = target_names
        report_calls["zero_division"] = zero_division
        return "classification report"

    monkeypatch.setattr(trainer, "classification_report", fake_report)
    monkeypatch.setattr(trainer.tf.keras.models, "load_model", lambda path: FakeBestModel())

    trainer.train()

    assert model_path.exists()
    assert (tmp_path / "class_names.json").exists()
    assert report_calls["labels"] == [0, 1, 2]
    assert report_calls["target_names"] == bundle.class_names
    assert report_calls["zero_division"] == 0


def test_train_fails_when_checkpoint_is_not_saved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoCheckpointModel(FakeModel):
        def fit(self, *_args, **_kwargs) -> FakeHistory:
            return FakeHistory()

    model_path = tmp_path / "tradevision_best.keras"
    monkeypatch.setattr("config.MODELS_DIR", tmp_path)
    monkeypatch.setattr("config.MODEL_PATH", model_path)
    monkeypatch.setattr(trainer, "_plot_training_curves", lambda history: None)
    monkeypatch.setattr(trainer, "_plot_confusion_matrix", lambda y_true, y_pred, class_names: None)
    monkeypatch.setattr(
        trainer,
        "load_datasets",
        lambda: SimpleNamespace(
            train=[],
            val=[],
            test=[],
            class_names=["ascending_triangle", "double_top", "flag"],
            num_classes=3,
        ),
    )
    monkeypatch.setattr(trainer, "build_model", lambda num_classes: NoCheckpointModel(model_path))

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        trainer.train()
