"""Training-only statistical profiling for Sen1Floods11 v1.1."""

import csv
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"

IMAGE_DIR = DATA_ROOT / "S1Hand"
TRAIN_SPLIT = DATA_ROOT / "flood_train_data.csv"


def read_training_images() -> list[str]:
    """Read image filenames from the official training split."""

    with TRAIN_SPLIT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    return [image_name for image_name, _ in rows]


def describe_channel(values: np.ndarray, name: str) -> None:
    """Print descriptive statistics for one channel."""

    if values.size == 0:
        print(f"\n{name}: no valid pixels found.")
        return

    percentiles = np.percentile(
        values,
        [0, 1, 5, 25, 50, 75, 95, 99, 100],
    )

    print(f"\n{name}")
    print("-" * 50)
    print(f"Valid pixels: {values.size:,}")
    print(f"Min:          {percentiles[0]:.4f}")
    print(f"1st %ile:     {percentiles[1]:.4f}")
    print(f"5th %ile:     {percentiles[2]:.4f}")
    print(f"25th %ile:    {percentiles[3]:.4f}")
    print(f"Median:       {percentiles[4]:.4f}")
    print(f"75th %ile:    {percentiles[5]:.4f}")
    print(f"95th %ile:    {percentiles[6]:.4f}")
    print(f"99th %ile:    {percentiles[7]:.4f}")
    print(f"Max:          {percentiles[8]:.4f}")
    print(f"Mean:         {values.mean():.4f}")
    print(f"Std:          {values.std():.4f}")


def main() -> None:
    image_names = read_training_images()

    print("=== Sen1Floods11 v1.1: training-only profiling ===")
    print(f"Training images: {len(image_names)}")

    vv_values = []
    vh_values = []

    total_pixels = 0
    invalid_pixels = 0

    for image_name in image_names:
        image_path = IMAGE_DIR / image_name

        with rasterio.open(image_path) as image:
            data = image.read().astype(np.float32)

            if data.shape[0] != 2:
                raise ValueError(
                    f"Expected VV/VH bands in {image_path.name}"
                )

            # Dataset mask: True = valid, False = NoData.
            valid_mask = image.read_masks(1) > 0

            total_pixels += valid_mask.size
            invalid_pixels += (~valid_mask).sum()

            vv = data[0][valid_mask]
            vh = data[1][valid_mask]

            vv_values.append(vv)
            vh_values.append(vh)

    vv = np.concatenate(vv_values)
    vh = np.concatenate(vh_values)

    print(f"Total pixels:   {total_pixels:,}")
    print(f"NoData pixels:  {invalid_pixels:,}")
    print(
        f"NoData ratio:   "
        f"{100 * invalid_pixels / total_pixels:.2f}%"
    )

    describe_channel(vv, "VV channel")
    describe_channel(vh, "VH channel")

    print("\nTraining-only profiling complete.")
    print("No normalization or model training was performed.")


if __name__ == "__main__":
    main()