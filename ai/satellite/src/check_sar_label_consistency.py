"""Check Sentinel-1 NoData and flood-label consistency."""

import csv
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"

IMAGE_DIR = DATA_ROOT / "S1Hand"
LABEL_DIR = DATA_ROOT / "LabelHand"

TRAIN_SPLIT = DATA_ROOT / "flood_train_data.csv"


def read_training_pairs() -> list[tuple[str, str]]:
    """Read image-label pairs from the official training split."""

    with TRAIN_SPLIT.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    return [(image_name, label_name) for image_name, label_name in rows]


def main() -> None:
    pairs = read_training_pairs()

    print("=== Sen1Floods11: SAR-label consistency check ===")
    print(f"Training pairs: {len(pairs)}")

    total_pixels = 0

    sar_nodata_pixels = 0
    sar_valid_pixels = 0

    sar_nodata_label_ignore = 0
    sar_nodata_label_valid = 0

    sar_valid_label_ignore = 0
    sar_valid_label_valid = 0

    shape_mismatches = 0

    for image_name, label_name in pairs:
        image_path = IMAGE_DIR / image_name
        label_path = LABEL_DIR / label_name

        with rasterio.open(image_path) as image:
            sar = image.read()
            sar_mask = image.read_masks(1) > 0

        with rasterio.open(label_path) as label:
            labels = label.read(1)

        if sar.shape[1:] != labels.shape:
            shape_mismatches += 1
            print(f"Shape mismatch: {image_name}")
            continue

        total_pixels += labels.size

        sar_valid = sar_mask
        sar_nodata = ~sar_valid

        label_ignore = labels == -1
        label_valid = (labels == 0) | (labels == 1)

        sar_nodata_pixels += sar_nodata.sum()
        sar_valid_pixels += sar_valid.sum()

        sar_nodata_label_ignore += np.sum(
            sar_nodata & label_ignore
        )

        sar_nodata_label_valid += np.sum(
            sar_nodata & label_valid
        )

        sar_valid_label_ignore += np.sum(
            sar_valid & label_ignore
        )

        sar_valid_label_valid += np.sum(
            sar_valid & label_valid
        )

    print("\nPixel totals")
    print("-" * 50)

    print(f"Total pixels:                 {total_pixels:,}")
    print(f"SAR NoData pixels:            {sar_nodata_pixels:,}")
    print(f"SAR valid pixels:             {sar_valid_pixels:,}")

    print("\nSAR NoData vs label")
    print("-" * 50)

    print(
        f"SAR NoData + label -1:        "
        f"{sar_nodata_label_ignore:,}"
    )

    print(
        f"SAR NoData + valid label:     "
        f"{sar_nodata_label_valid:,}"
    )

    print("\nSAR valid vs label")
    print("-" * 50)

    print(
        f"SAR valid + label -1:         "
        f"{sar_valid_label_ignore:,}"
    )

    print(
        f"SAR valid + valid label:      "
        f"{sar_valid_label_valid:,}"
    )

    print("\nIntegrity checks")
    print("-" * 50)

    print(f"Shape mismatches:             {shape_mismatches}")

    if sar_nodata_label_valid == 0:
        print(
            "SAR NoData / valid-label check: PASS"
        )
    else:
        print(
            "SAR NoData / valid-label check: REVIEW"
        )

    valid_for_loss = sar_valid_label_valid

    

    print("\nEffective training pixels")
    print("-" * 50)
    print(f"Valid for loss:               {valid_for_loss:,}")
    print(
        f"Excluded pixels:              "
        f"{total_pixels - valid_for_loss:,}"
    )
    print(
        f"Effective usable ratio:       "
        f"{100 * valid_for_loss / total_pixels:.2f}%"
    )

    print("\nConsistency check complete.")
    print("No data was modified.")
    print("No model training was performed.")


if __name__ == "__main__":
    main()