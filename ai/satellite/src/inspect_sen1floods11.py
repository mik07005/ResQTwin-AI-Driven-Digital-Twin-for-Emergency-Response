"""Read-only structure and label inspection for Sen1Floods11 v1.1."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"
IMAGE_DIR = DATA_ROOT / "S1Hand"
LABEL_DIR = DATA_ROOT / "LabelHand"

SPLIT_FILES = {
    "train": DATA_ROOT / "flood_train_data.csv",
    "validation": DATA_ROOT / "flood_valid_data.csv",
    "test": DATA_ROOT / "flood_test_data.csv",
    "bolivia_holdout": DATA_ROOT / "flood_bolivia_data.csv",
}


def read_split(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if not all(len(row) == 2 for row in rows):
        raise ValueError(f"Unexpected split-file format: {path.name}")

    return [(image_name, label_name) for image_name, label_name in rows]


image_files = sorted(IMAGE_DIR.glob("*_S1Hand.tif"))
label_files = sorted(LABEL_DIR.glob("*_LabelHand.tif"))

print("=== Sen1Floods11 v1.1: read-only inspection ===")
print(f"Sentinel-1 chips: {len(image_files)}")
print(f"Hand labels:       {len(label_files)}")

image_ids = {path.name.replace("_S1Hand.tif", "") for path in image_files}
label_ids = {path.name.replace("_LabelHand.tif", "") for path in label_files}

if image_ids != label_ids:
    raise ValueError("Image/label filename pairs do not match.")

print("Image-label pairing: PASS")

all_split_images = []
for split_name, split_path in SPLIT_FILES.items():
    rows = read_split(split_path)

    for image_name, label_name in rows:
        if not (IMAGE_DIR / image_name).exists():
            raise FileNotFoundError(f"Missing image: {image_name}")
        if not (LABEL_DIR / label_name).exists():
            raise FileNotFoundError(f"Missing label: {label_name}")

    all_split_images.extend(image_name for image_name, _ in rows)
    print(f"{split_name:16}: {len(rows)} chips")

if len(all_split_images) != 446 or len(set(all_split_images)) != 446:
    raise ValueError("The official split files are incomplete or overlap.")

print("Split integrity: PASS")

with rasterio.open(image_files[0]) as image:
    print("\nSample Sentinel-1 metadata")
    print(f"Shape: {image.height} x {image.width}")
    print(f"Bands: {image.count} ({', '.join(image.descriptions)})")
    print(f"Data type: {image.dtypes[0]}")
    print(f"CRS: {image.crs}")

label_counts = Counter()
for label_path in label_files:
    with rasterio.open(label_path) as label:
        values, counts = np.unique(label.read(1), return_counts=True)
        label_counts.update(dict(zip(values.tolist(), counts.tolist())))

unexpected_values = set(label_counts) - {-1, 0, 1}
if unexpected_values:
    raise ValueError(f"Unexpected label values: {unexpected_values}")

valid_pixels = label_counts[0] + label_counts[1]

print("\nLabel-pixel totals")
print(f"Ignore (-1): {label_counts[-1]:,}")
print(f"Not water (0): {label_counts[0]:,}")
print(f"Water (1): {label_counts[1]:,}")
print(f"Water among valid pixels: {100 * label_counts[1] / valid_pixels:.2f}%")
print("\nInspection complete. No model training was performed.")