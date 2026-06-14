from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random

from PIL import Image

from .utils import ensure_dir, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitConfig:
    raw_dir: Path
    output_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    copy_files: bool = True


def is_valid_rgb_image(path: Path) -> bool:
    """Return True when an image can be opened and converted to RGB."""
    try:
        with Image.open(path) as image:
            image.convert("RGB")
        return True
    except Exception:
        return False


def find_class_folders(raw_dir: Path) -> list[Path]:
    classes = sorted(path for path in raw_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if len(classes) < 2:
        raise ValueError(
            f"Expected at least two class folders inside {raw_dir}. "
            "Example: data/raw/benign and data/raw/malignant."
        )
    return classes


def collect_images(class_dir: Path) -> list[Path]:
    images = [
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and is_valid_rgb_image(path)
    ]
    if not images:
        raise ValueError(f"No valid images found in class folder: {class_dir}")
    return sorted(images)


def split_items(items: list[Path], config: SplitConfig) -> dict[str, list[Path]]:
    shuffled = items[:]
    Random(config.seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = max(1, int(total * config.train_ratio))
    val_end = train_end + max(1, int(total * config.val_ratio))

    if total >= 3 and val_end >= total:
        val_end = total - 1

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def copy_or_link_file(source: Path, destination: Path, copy_files: bool) -> None:
    ensure_dir(destination.parent)
    if destination.exists():
        destination.unlink()
    if copy_files:
        shutil.copy2(source, destination)
    else:
        destination.symlink_to(source)


def create_dataset_splits(config: SplitConfig) -> dict[str, dict[str, int]]:
    """Create train/val/test folders from class folders in data/raw."""
    if round(config.train_ratio + config.val_ratio + config.test_ratio, 6) != 1:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    ensure_dir(config.output_dir)
    summary: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for class_dir in find_class_folders(config.raw_dir):
        class_name = class_dir.name
        class_splits = split_items(collect_images(class_dir), config)

        for split_name, paths in class_splits.items():
            summary[split_name][class_name] = len(paths)
            for index, image_path in enumerate(paths, start=1):
                destination = config.output_dir / split_name / class_name / f"{class_name}_{index:04d}{image_path.suffix.lower()}"
                copy_or_link_file(image_path, destination, config.copy_files)

    save_json(summary, config.output_dir / "split_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/validation/test splits for MRI image classification.")
    parser.add_argument("--raw-dir", default="data/raw", type=Path, help="Folder containing one subfolder per class.")
    parser.add_argument("--output-dir", default="data/processed", type=Path, help="Destination for processed splits.")
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--test-ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_dataset_splits(
        SplitConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            copy_files=not args.symlink,
        )
    )
    print(summary)


if __name__ == "__main__":
    main()

