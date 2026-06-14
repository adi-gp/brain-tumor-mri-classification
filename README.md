# Brain Tumor MRI Classification

Binary brain MRI image classification using TensorFlow/Keras. This project turns an original Colab notebook into a reproducible machine learning repository with dataset preparation, CNN training, evaluation, single-image prediction, and a Gradio demo app.

> This repository is for learning and portfolio demonstration only. It is not a medical diagnostic tool.

## Project Summary

The original notebook loaded MRI images from Google Drive, resized them to `128x128`, trained a binary CNN-style classifier, and tested predictions through a small Gradio app. This version keeps the same core idea but restructures it as a professional ML project:

- reusable preprocessing script for class-folder datasets
- train/validation/test split creation
- notebook-inspired CNN architecture
- optional MobileNetV2 transfer-learning baseline
- model checkpoints and training logs
- evaluation metrics, confusion matrix, and ROC curve generation
- command-line prediction and Gradio interface

The code supports both folder conventions:

- `data/raw/benign` and `data/raw/malignant`
- `data/raw/no` and `data/raw/yes`

Class names are inferred from folder names, so the trained model reports labels that match the dataset you provide.

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
├── LICENSE
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

The original project used a two-class MRI dataset with the following class folders:

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

Train the notebook-style CNN:

```bash
python -m src.train --model-name cnn --epochs 25 --batch-size 16
```

Train the transfer-learning baseline:

```bash
python -m src.train --model-name mobilenetv2 --epochs 15 --batch-size 16 --learning-rate 0.0001
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

## Model

The default CNN follows the original notebook architecture while fixing production issues:

- image normalization is inside the model through a `Rescaling` layer
- data augmentation is applied only during training
- convolution blocks use batch normalization, max pooling, and dropout
- final layer uses sigmoid activation for binary classification
- training uses binary cross-entropy and Adam optimizer

An optional MobileNetV2 transfer-learning model is included for small datasets where a pretrained image backbone can improve generalization.

## Evaluation

Running `src.evaluate` writes the following artifacts to `results/`:

- `metrics.json`
- `confusion_matrix.png`
- `roc_curve.png`

Because the original trained weights and dataset are not included, this repository does not hard-code an accuracy claim. Train the model on your dataset, run evaluation, then update this section and your resume with the generated test accuracy.

## Resume Alignment

Suggested resume wording after you train and evaluate this repo:

```text
Brain Tumor MRI Classification — Medical Image Classification
Python • TensorFlow/Keras • CNN • Transfer Learning

- Built a reproducible MRI image classification pipeline with preprocessing, train/validation/test splits, CNN training, and test-set evaluation.
- Implemented a custom CNN and MobileNetV2 transfer-learning baseline, generating confusion matrix, ROC curve, and classification metrics.
- Deployed a Gradio demo for single-image inference using the trained Keras model.
```

Replace the final bullet or add the measured test accuracy only after `results/metrics.json` is generated.
