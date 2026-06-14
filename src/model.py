from __future__ import annotations

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from tensorflow.keras.optimizers import Adam


def augmentation_layers() -> Sequential:
    return Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.05),
            RandomZoom(0.1),
            RandomContrast(0.1),
        ],
        name="augmentation",
    )


def compile_binary_model(model: Model, learning_rate: float) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    image_size: tuple[int, int] = (128, 128),
    learning_rate: float = 1e-3,
    use_augmentation: bool = True,
) -> Model:
    """CNN inspired by the original notebook, with normalization baked into the model."""
    inputs = Input(shape=(*image_size, 3), name="mri_image")
    x = Rescaling(1.0 / 255, name="rescale")(inputs)
    if use_augmentation:
        x = augmentation_layers()(x)

    x = Conv2D(64, kernel_size=(2, 2), padding="same")(x)
    x = Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = Conv2D(64, kernel_size=(2, 2), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid", name="tumor_probability")(x)

    return compile_binary_model(Model(inputs=inputs, outputs=outputs, name="notebook_style_cnn"), learning_rate)


def build_model(
    image_size: tuple[int, int] = (128, 128),
    learning_rate: float = 1e-3,
    use_augmentation: bool = True,
) -> Model:
    return build_cnn(image_size=image_size, learning_rate=learning_rate, use_augmentation=use_augmentation)
