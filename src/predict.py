from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .utils import load_json


def load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(model_path: Path, image_path: Path, class_names: list[str], image_size: tuple[int, int]) -> dict[str, float | str]:
    model = tf.keras.models.load_model(model_path)
    probability_class_1 = float(model.predict(load_image(image_path, image_size), verbose=0)[0][0])
    predicted_index = int(probability_class_1 >= 0.5)
    confidence = probability_class_1 if predicted_index == 1 else 1 - probability_class_1

    return {
        "label": class_names[predicted_index],
        "confidence": confidence,
        "probability_class_1": probability_class_1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one MRI image.")
    parser.add_argument("--model-path", default="models/best_model.keras", type=Path)
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument("--class-names-path", default="models/class_names.json", type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = load_json(args.class_names_path)["class_names"]
    result = predict_image(
        model_path=args.model_path,
        image_path=args.image_path,
        class_names=class_names,
        image_size=(args.image_size, args.image_size),
    )
    print(result)


if __name__ == "__main__":
    main()

