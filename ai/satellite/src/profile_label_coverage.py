"""Profile usable flood-label coverage across Sen1Floods11 splits."""

import csv
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "sen1floods11"

LABEL_DIR = DATA_ROOT / "LabelHand"

SPLIT_FILES = {
    "train": DATA_ROOT / "flood_train_data.csv",
    "validation": DATA_ROOT / "flood_valid_data.csv",
    "test": DATA_ROOT / "flood_test_data.csv",
    "bolivia_holdout": DATA_ROOT / "flood_bolivia_data.csv",
}


def read_split(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def main() -> None:
    print("=== Sen1Floods11: label coverage profiling ===")

    for split_name, split_path in SPLIT_FILES.items():
        rows = read_split(split_path)

        usable_chips = 0
        ignored_chips = 0

        total_pixels = 0
        usable_pixels = 0
        water_pixels = 0
        non_water_pixels = 0
        ignore_pixels = 0

        for _, label_name in rows:
            label_path = LABEL_DIR / label_name

            with rasterio.open(label_path) as label_file:
                label = label_file.read(1)

            values, counts = np.unique(label, return_counts=True)
            counts_by_value = dict(zip(values.tolist(), counts.tolist()))

            water = counts_by_value.get(1, 0)
            non_water = counts_by_value.get(0, 0)
            ignore = counts_by_value.get(-1, 0)

            total = label.size
            usable = water + non_water

            total_pixels += total
            usable_pixels += usable
            water_pixels += water
            non_water_pixels += non_water
            ignore_pixels += ignore

            if usable > 0:
                usable_chips += 1
            else:
                ignored_chips += 1

        print(f"\n{split_name}")
        print("-" * 50)
        print(f"Total chips:             {len(rows)}")
        print(f"Chips with labels:       {usable_chips}")
        print(f"All-ignore chips:        {ignored_chips}")
        print(f"Total pixels:            {total_pixels:,}")
        print(f"Usable pixels:           {usable_pixels:,}")
        print(f"Ignore pixels:           {ignore_pixels:,}")
        print(f"Water pixels:            {water_pixels:,}")
        print(f"Non-water pixels:        {non_water_pixels:,}")
        print(
            f"Usable pixel ratio:      "
            f"{100 * usable_pixels / total_pixels:.2f}%"
        )

    print("\nProfiling complete.")
    print("No data was modified.")


if __name__ == "__main__":
    main()