# Brain Tumor MRI Classification

Beginner-friendly brain MRI image classification project using a Convolutional Neural Network (CNN) in TensorFlow/Keras.

The model classifies MRI images into two classes:

- `no`: no tumor
- `yes`: tumor

> This is a learning project and is not a medical diagnostic tool.

## Project Summary

The original work was done in a Jupyter/Colab notebook. This repository keeps the same simple CNN idea, but organizes the code so it is easier for recruiters to read:

- dataset preprocessing
- CNN model definition
- training script
- evaluation script
- single-image prediction
- simple Gradio demo app

The project uses class folders, so labels are automatically taken from the folder names.

## Repository Structure

```text
brain-tumor-mri-classification/
├── app.py
├── DATASET.md
├── data/
│   └── README.md
├── models/
│   └── README.md
├── notebooks/
│   ├── original_teacher_notebook.ipynb
│   └── NOTEBOOK_ANALYSIS.md
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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Format

The original dataset used this structure:

```text
Brain_Tumor_Dataset/
├── no/   # MRI images without tumor
└── yes/  # MRI images with tumor
```

Local dataset inventory:

- `98` no-tumor images
- `155` tumor images
- `253` unique images in the usable class-folder copy
- image formats: `.jpg`, `.jpeg`, `.png`

The local download also contains a duplicated nested copy at `brain_tumor_dataset/no` and `brain_tumor_dataset/yes`, so only one copy should be used for training.

Raw images are not committed to this repository because medical-image datasets can have licensing and redistribution restrictions. To reproduce the project, place the dataset locally under `data/raw/`.

Place images in class folders:

```text
data/raw/
├── no/
│   ├── image_001.jpg
│   └── image_002.jpg
└── yes/
    ├── image_001.jpg
    └── image_002.jpg
```

The code can also work with other binary class names such as:

```text
data/raw/
├── benign/
└── malignant/
```

## Usage

Create reproducible train/validation/test splits:

```bash
python -m src.data_preprocessing --raw-dir data/raw --output-dir data/processed
```

Train the CNN:

```bash
python -m src.train --epochs 25 --batch-size 16
```

Evaluate on the held-out test split:

```bash
python -m src.evaluate --model-path models/best_model.keras
```

Predict a single MRI image:

```bash
python -m src.predict --image-path path/to/mri.jpg --model-path models/best_model.keras
```

Launch the demo app:

```bash
python app.py --model-path models/best_model.keras
```

## CNN Model

The CNN is based on the original notebook:

- image normalization is inside the model through a `Rescaling` layer
- convolution blocks use batch normalization, max pooling, and dropout
- final layer uses sigmoid activation for binary tumor/no-tumor classification
- training uses binary cross-entropy and Adam optimizer

Small data augmentation is included during training to help the model generalize better.

## Evaluation

Running `src.evaluate` writes the following artifacts to `results/`:

- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`

Because trained weights are not included, this repository does not hard-code an accuracy claim. Train the model on the dataset, run evaluation, and then use the generated metrics from `results/metrics.json`.

## Resume Alignment

Suggested resume wording after you train and evaluate this repo:

```text
Brain Tumor MRI Classification — Medical Image Classification
Python • TensorFlow/Keras • CNN

- Built a CNN-based MRI image classifier to detect tumor vs no-tumor images.
- Added preprocessing, train/validation/test split creation, model training, and test evaluation scripts.
- Created a simple Gradio demo for single-image prediction.
```

Replace the final bullet or add the measured test accuracy only after `results/metrics.json` is generated.
