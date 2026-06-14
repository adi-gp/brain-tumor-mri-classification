# Original Notebook Analysis

The teacher-provided notebook was a Colab workflow with 24 cells.

## Workflow Observed

- Mounted Google Drive and read images from `/content/drive/MyDrive/Brain_Tumor_Dataset/yes` and `/no`.
- Loaded `.jpg` files with PIL, resized each image to `128x128`, and kept only RGB images.
- Assigned labels as `1` for tumor and `0` for no tumor.
- Normalized images with `/255.0`, although the subsequent train/test split used the unnormalized `data` array.
- Split data with `train_test_split(test_size=0.2, shuffle=True, random_state=0)`.
- Built a Keras Sequential CNN:
  - two convolution blocks with batch normalization, max pooling, and dropout
  - dense layer with dropout
  - sigmoid output for binary classification
- Included single-image inference and a Gradio interface.

## Production Changes Made

- Added an explicit preprocessing module for reusable train/validation/test splits.
- Moved model architecture into `src/model.py`.
- Added a real training script with `model.fit`, callbacks, model checkpointing, and training curves.
- Added a test evaluation script with accuracy, classification report, confusion matrix, and ROC curve.
- Fixed inference preprocessing by placing image normalization inside the model.
- Kept class labels inferred from folder names, so the same code works for `benign/malignant` or `yes/no` folders.

