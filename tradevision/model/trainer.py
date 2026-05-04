"""
tradevision/model/trainer.py — Full training loop with callbacks.

Run directly::

    python -m tradevision.model.trainer

This script:
1. Loads and splits the dataset via loader.py
2. Builds the model via builder.py
3. Trains with EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
4. Plots training curves
5. Evaluates on the test set and prints a confusion matrix
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import config
from tradevision.data.loader import load_datasets
from tradevision.model.builder import build_model
from tradevision.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Callbacks ────────────────────────────────────────────────────────────────


def _build_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Return the standard set of training callbacks.

    Returns:
        List containing EarlyStopping, ReduceLROnPlateau, and
        ModelCheckpoint callbacks.
    """
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    return [early_stop, reduce_lr, checkpoint]


# ─── Plot helpers ─────────────────────────────────────────────────────────────


def _plot_training_curves(history: tf.keras.callbacks.History) -> None:
    """Save training/validation accuracy and loss curves to disk.

    Args:
        history: Keras history object returned by ``model.fit()``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("TradeVision — Training Curves", fontsize=14)

    epochs = range(1, len(history.history["accuracy"]) + 1)

    axes[0].plot(epochs, history.history["accuracy"], label="Train Acc")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Train Loss")
    axes[1].plot(epochs, history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.TRAINING_CURVES_PATH, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved → %s", config.TRAINING_CURVES_PATH)


def _plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> None:
    """Save a labelled confusion matrix heat-map.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        class_names: List of human-readable class names.
    """
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", config.CONFUSION_MATRIX_PATH)


# ─── Main training routine ────────────────────────────────────────────────────


def train() -> None:
    """Execute the full training pipeline end-to-end."""
    logger.info("═══ TradeVision Training Start ═══")

    # 1. Load data
    bundle = load_datasets()
    class_names = bundle.class_names

    # 2. Build model
    model = build_model(num_classes=bundle.num_classes)
    model.summary(print_fn=logger.info)

    # 3. Train
    logger.info(
        "Training: epochs=%d | batch=%d | backbone=%s",
        config.EPOCHS, config.BATCH_SIZE, config.BACKBONE,
    )
    history = model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=config.EPOCHS,
        callbacks=_build_callbacks(),
    )

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Expected best-model checkpoint at {config.MODEL_PATH}, but none was saved."
        )

    class_names_path = config.MODELS_DIR / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2))
    logger.info("Class names saved → %s", class_names_path)

    logger.info("Loading best checkpoint for evaluation …")
    best_model = tf.keras.models.load_model(str(config.MODEL_PATH))

    # 4. Fine-tuning Phase
    logger.info("═══ Starting Fine-Tuning Phase ═══")
    logger.info("Unfreezing backbone layers (keeping BatchNormalization frozen) ...")
    
    for layer in best_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.FINETUNE_LEARNING_RATE)
    best_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    best_model.summary(print_fn=logger.info)

    logger.info(
        "Fine-tuning: epochs=%d | lr=%s",
        config.FINETUNE_EPOCHS, config.FINETUNE_LEARNING_RATE,
    )
    
    ft_history = best_model.fit(
        bundle.train,
        validation_data=bundle.val,
        epochs=config.FINETUNE_EPOCHS,
        callbacks=_build_callbacks(),
    )

    # Reload best model from fine-tuning
    logger.info("Loading best fine-tuned checkpoint for evaluation …")
    best_model = tf.keras.models.load_model(str(config.MODEL_PATH))

    # Append fine-tuning history to initial history for full curve plot
    for key in history.history:
        history.history[key].extend(ft_history.history[key])

    # 5. Training curves
    _plot_training_curves(history)

    # 5. Evaluate on test set
    logger.info("Evaluating on test set …")
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in bundle.test:
        preds = best_model.predict(images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    label_indices = list(range(len(class_names)))

    report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=label_indices,
        target_names=class_names,
        zero_division=0,
    )
    logger.info("\n%s", report)

    _plot_confusion_matrix(y_true_arr, y_pred_arr, class_names)

    test_acc = np.mean(y_true_arr == y_pred_arr)
    logger.info("Test accuracy: %.4f", test_acc)
    logger.info("═══ Training Complete ═══")


if __name__ == "__main__":
    train()
