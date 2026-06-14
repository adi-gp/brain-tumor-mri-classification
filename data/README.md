# Data

Place MRI images in class-specific folders before preprocessing.

For a benign/malignant dataset:

```text
data/raw/
├── benign/
└── malignant/
```

For the original notebook dataset:

```text
data/raw/
├── no/
└── yes/
```

The preprocessing command creates `data/processed/train`, `data/processed/val`, and `data/processed/test`.
Raw and processed image files are ignored by Git to avoid committing medical images or large datasets.

