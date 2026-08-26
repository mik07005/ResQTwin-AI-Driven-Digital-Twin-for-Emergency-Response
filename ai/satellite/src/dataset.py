"""PyTorch Dataset for Sen1Floods11 v1.1."""

import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


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


# Frozen statistics calculated ONLY from the 252 training chips.
VV_MEAN = -10.3929
VV_STD = 4.0388

VH_MEAN = -17.2411
VH_STD = 4.7540

IGNORE_INDEX = -1


class Sen1Floods11Dataset(Dataset):
    """Sen1Floods11 dataset with fixed SAR preprocessing."""

    def __init__(self, split: str):
        if split not in SPLIT_FILES:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Choose from: {list(SPLIT_FILES)}"
            )

        self.split = split
        self.samples = self._read_split(SPLIT_FILES[split])

    @staticmethod
    def _read_split(path: Path) -> list[tuple[str, str]]:
        """Read image-label pairs from an official split CSV."""

        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))

        if not all(len(row) == 2 for row in rows):
            raise ValueError(
                f"Unexpected split-file format: {path.name}"
            )

        return [
            (image_name, label_name)
            for image_name, label_name in rows
        ]

    def __len__(self) -> int:
        """Return number of image-label pairs."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess one image-label pair."""

        image_name, label_name = self.samples[index]

        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / label_name

        # Read Sentinel-1 VV/VH data.
        with rasterio.open(image_path) as image_file:
            image = image_file.read().astype(np.float32)

            # Rasterio mask: True = valid SAR observation.
            sar_valid = image_file.read_masks(1) > 0

        # Read flood label.
        with rasterio.open(label_path) as label_file:
            label = label_file.read(1).astype(np.int64)

        if image.shape[0] != 2:
            raise ValueError(
                f"Expected VV/VH bands in {image_name}, "
                f"found {image.shape[0]} bands."
            )

        if image.shape[1:] != label.shape:
            raise ValueError(
                f"Image/label shape mismatch for {image_name}: "
                f"{image.shape[1:]} vs {label.shape}"
            )

        # Ground-truth validity.
        label_valid = (label == 0) | (label == 1)

        # A pixel is usable for loss/metrics only when:
        # 1. SAR observation is valid.
        # 2. Ground-truth label is 0 or 1.
        valid_for_loss = sar_valid & label_valid

        # Replace SAR NoData/NaN values before normalization so that
        # NaNs never enter the neural network.
        image[0][~sar_valid] = VV_MEAN
        image[1][~sar_valid] = VH_MEAN

        # Per-channel standardization using frozen training statistics.
        image[0] = (image[0] - VV_MEAN) / VV_STD
        image[1] = (image[1] - VH_MEAN) / VH_STD

        # Mark all pixels that cannot contribute to the loss as IGNORE.
        processed_label = label.copy()
        processed_label[~valid_for_loss] = IGNORE_INDEX

        # Convert NumPy arrays to PyTorch tensors.
        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(processed_label)

        return image_tensor, label_tensor