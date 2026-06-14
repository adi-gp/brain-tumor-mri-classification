# Brain Tumor MRI Classification

Brain MRI image classification project built with a Convolutional Neural Network (CNN) using TensorFlow/Keras.

The model classifies MRI images into two classes:

- `no`: no tumor
- `yes`: tumor

> This project is for learning and portfolio demonstration only. It is not a medical diagnostic tool.

## Overview

This project implements a simple end-to-end machine learning workflow for binary MRI image classification:

- preprocessing MRI images into train, validation, and test folders
- building a CNN model in TensorFlow/Keras
- training the model with validation monitoring
- evaluating the model on a test set
- running single-image prediction
- launching a small Gradio demo app

## Project Structure

```text
brain-tumor-mri-classification/
├── app.py
├── DATASET.md
├── data/
│   └── README.md
├── models/
│   └── README.md
├── results/
│   └── README.md
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The dataset used for development has this structure:

```text
Brain_Tumor_Dataset/
├── no/   # MRI images without tumor
└── yes/  # MRI images with tumor
```

Dataset inventory:

- `98` no-tumor images
- `155` tumor images
- `253` usable images
- supported formats: `.jpg`, `.jpeg`, `.png`

Raw MRI images are not committed to this repository because medical-image datasets can have licensing or redistribution restrictions. To reproduce the project, place the images locally under `data/raw/`.

Expected local format:

```text
data/raw/
├── no/
│   ├── image_001.jpg
│   └── image_002.jpg
└── yes/
    ├── image_001.jpg
    └── image_002.jpg
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Create train, validation, and test splits:

```bash
python -m src.data_preprocessing --raw-dir data/raw --output-dir data/processed
```

Train the CNN:

```bash
python -m src.train --epochs 25 --batch-size 16
```

Evaluate the trained model:

```bash
python -m src.evaluate --model-path models/best_model.keras
```

Predict a single MRI image:

```bash
python -m src.predict --image-path path/to/mri.jpg --model-path models/best_model.keras
```

Launch the Gradio app:

```bash
python app.py --model-path models/best_model.keras
```

## Model

The CNN uses:

- image resizing to `128x128`
- normalization with a Keras `Rescaling` layer
- convolution layers for feature extraction
- batch normalization, max pooling, and dropout
- dense layer for classification
- sigmoid output for binary prediction

## Results

Evaluation outputs are saved in the `results/` folder:

- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`
- `training_curves.png`

Trained model files are saved in the `models/` folder:

- `best_model.keras`
- `final_model.keras`
- `class_names.json`

