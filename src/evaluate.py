from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from .utils import ensure_dir, load_json, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained MRI classifier on the test split.")
    parser.add_argument("--model-path", default="models/best_model.keras", type=Path)
    parser.add_argument("--data-dir", default="data/processed/test", type=Path)
    parser.add_argument("--class-names-path", default="models/class_names.json", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    return parser.parse_args()


def load_test_dataset(data_dir: Path, image_size: tuple[int, int], batch_size: int) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )


def save_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    auc = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return float(auc)


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_dir)

    model = tf.keras.models.load_model(args.model_path)
    image_size = (args.image_size, args.image_size)
    test_ds = load_test_dataset(args.data_dir, image_size, args.batch_size)
    class_names = test_ds.class_names

    if args.class_names_path.exists():
        class_names = load_json(args.class_names_path).get("class_names", class_names)

    loss, accuracy = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([labels.numpy().ravel() for _, labels in test_ds]).astype(int)
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    save_confusion_matrix(cm, class_names, args.results_dir / "confusion_matrix.png")
    auc = save_roc_curve(y_true, y_prob, args.results_dir / "roc_curve.png")

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    save_json(metrics, args.results_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
