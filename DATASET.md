# Dataset Notes

The dataset used during development contains a duplicated folder structure:

```text
Brain_Tumor_Dataset/
├── brain_tumor_dataset/
│   ├── no/   # 98 images
│   └── yes/  # 155 images
├── no/       # duplicate copy, 98 images
└── yes/      # duplicate copy, 155 images
```

Use only one copy for training. The original notebook used the direct `yes` and `no` folders.

## Class Meaning

- `no`: MRI image without a visible tumor
- `yes`: MRI image with a visible tumor

## Why Images Are Not Committed

The raw MRI images are intentionally not uploaded to GitHub. Even when a dataset is small enough for GitHub, medical-image datasets may have licensing or redistribution restrictions. Keeping raw data out of the repository is the safer professional choice.

Recruiters can still review the complete ML workflow through the source code, notebook analysis, preprocessing pipeline, training script, evaluation script, prediction script, and Gradio app.

## Reproducing Locally

Copy one dataset copy into `data/raw`:

```bash
mkdir -p data/raw
cp -R /Users/mymac/Downloads/Brain_Tumor_Dataset/no data/raw/no
cp -R /Users/mymac/Downloads/Brain_Tumor_Dataset/yes data/raw/yes
```

Then create train/validation/test splits:

```bash
python -m src.data_preprocessing --raw-dir data/raw --output-dir data/processed
```
