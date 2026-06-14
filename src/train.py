from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from .model import build_model
from .utils import ensure_dir, plot_training_curves, save_json, set_seed


AUTOTUNE = tf.data.AUTOTUNE


def load_split(split_dir: Path, image_size: tuple[int, int], batch_size: int, seed: int, shuffle: bool) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="binary",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary MRI tumor classifier.")
    parser.add_argument("--data-dir", default="data/processed", type=Path)
    parser.add_argument("--model-dir", default="models", type=Path)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--model-name", choices=["cnn", "mobilenetv2"], default="cnn")
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=25, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no-augmentation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    image_size = (args.image_size, args.image_size)
    train_ds = load_split(args.data_dir / "train", image_size, args.batch_size, args.seed, shuffle=True)
    val_ds = load_split(args.data_dir / "val", image_size, args.batch_size, args.seed, shuffle=False)
    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    ensure_dir(args.model_dir)
    ensure_dir(args.results_dir)

    model = build_model(
        model_name=args.model_name,
        image_size=image_size,
        learning_rate=args.learning_rate,
        use_augmentation=not args.no_augmentation,
    )

    callbacks = [
        ModelCheckpoint(args.model_dir / "best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"),
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-6),
        CSVLogger(args.results_dir / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.model_dir / "final_model.keras")
    save_json({"class_names": class_names}, args.model_dir / "class_names.json")
    save_json(history.history, args.results_dir / "training_history.json")
    plot_training_curves(history.history, args.results_dir / "training_curves.png")


if __name__ == "__main__":
    main()

