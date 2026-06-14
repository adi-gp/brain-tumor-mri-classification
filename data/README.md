# Data

Place MRI images in class-specific folders before preprocessing.

The original local dataset used for this project contains one usable two-class copy:

```text
Brain_Tumor_Dataset/
├── no/   # 98 images
└── yes/  # 155 images
```

The download also contains a duplicated nested copy at `brain_tumor_dataset/no` and `brain_tumor_dataset/yes`. Use only one copy to avoid training and testing on duplicate images.

For this repository, copy one dataset copy into:

```text
data/raw/
├── no/
└── yes/
```

The code can also work with other binary class names:

```text
data/raw/
├── benign/
└── malignant/
```

The preprocessing command creates `data/processed/train`, `data/processed/val`, and `data/processed/test`.
Raw and processed image files are ignored by Git to avoid committing medical images or datasets with unclear redistribution rights.
