from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

from src.utils import load_json


def build_app(model_path: Path, class_names_path: Path, image_size: int) -> gr.Interface:
    model = tf.keras.models.load_model(model_path)
    class_names = load_json(class_names_path)["class_names"]

    def recognize_image(image: np.ndarray) -> dict[str, float]:
        resized = Image.fromarray(image).convert("RGB").resize((image_size, image_size))
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        probability_class_1 = float(model.predict(batch, verbose=0)[0][0])
        predicted_index = int(probability_class_1 >= 0.5)
        confidence = probability_class_1 if predicted_index == 1 else 1 - probability_class_1
        return {class_names[predicted_index]: confidence}

    return gr.Interface(
        fn=recognize_image,
        inputs=gr.Image(type="numpy", label="MRI image"),
        outputs=gr.Label(label="Prediction"),
        title="Brain Tumor MRI Classification",
        description="Upload a brain MRI image to classify it with the trained binary CNN model.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Gradio MRI classifier app.")
    parser.add_argument("--model-path", default="models/best_model.keras", type=Path)
    parser.add_argument("--class-names-path", default="models/class_names.json", type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app(args.model_path, args.class_names_path, args.image_size).launch(share=args.share)

