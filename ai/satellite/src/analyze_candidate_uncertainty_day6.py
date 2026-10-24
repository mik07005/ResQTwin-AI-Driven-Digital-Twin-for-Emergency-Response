
from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "candidate_new_inundation_filtered.tif"
)

ENTROPY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "day6"
    / "candidate_inundation_entropy.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "day6"
)

CATEGORY_MASK_PATH = OUTPUT_DIR / "candidate_uncertainty_categories.tif"

LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66

NODATA_CATEGORY = 255
BACKGROUND_CATEGORY = 0


def same_grid(src_a, src_b):
    return (
        src_a.crs == src_b.crs
        and src_a.width == src_b.width
        and src_a.height == src_b.height
        and np.allclose(
            tuple(src_a.transform),
            tuple(src_b.transform),
            atol=1e-9,
        )
    )


def main():
    with rasterio.open(CANDIDATE_PATH) as candidate_src, \
            rasterio.open(ENTROPY_PATH) as entropy_src:

        if not same_grid(candidate_src, entropy_src):
            raise ValueError(
                "Candidate and entropy rasters do not share a grid."
            )

        candidate_values = candidate_src.read(1)
        entropy = entropy_src.read(1).astype(np.float32)

        candidate_valid = candidate_src.dataset_mask() > 0
        entropy_valid = (
            (entropy_src.dataset_mask() > 0)
            & np.isfinite(entropy)
            & (entropy != entropy_src.nodata)
        )

        # Only candidate pixels with valid entropy are classified.
        valid = (
            (candidate_values == 1)
            & candidate_valid
            & entropy_valid
        )

        if not valid.any():
            raise ValueError("No valid candidate entropy pixels found.")

        categories = np.full(
            candidate_values.shape,
            NODATA_CATEGORY,
            dtype=np.uint8,
        )

        # Valid non-candidate background is category 0.
        valid_background = candidate_valid & ~(
            (candidate_values == 1) & entropy_valid
        )
        categories[valid_background] = BACKGROUND_CATEGORY

        low = valid & (entropy < LOW_THRESHOLD)
        medium = valid & (
            (entropy >= LOW_THRESHOLD)
            & (entropy < HIGH_THRESHOLD)
        )
        high = valid & (entropy >= HIGH_THRESHOLD)

        categories[low] = 1
        categories[medium] = 2
        categories[high] = 3

        low_pixels = int(low.sum())
        medium_pixels = int(medium.sum())
        high_pixels = int(high.sum())
        total_pixels = int(valid.sum())

        # Area is calculated only in a projected CRS with metre units.
        if candidate_src.crs is None or not candidate_src.crs.is_projected:
            raise ValueError("Area calculation requires a projected CRS.")

        linear_units = (candidate_src.crs.linear_units or "").lower()
        if linear_units not in ("metre", "meter"):
            raise ValueError(
                f"Expected projected metre units, got: {linear_units}"
            )

        transform = candidate_src.transform
        pixel_area_km2 = abs(
            transform.a * transform.e
            - transform.b * transform.d
        ) / 1_000_000.0

        low_area = low_pixels * pixel_area_km2
        medium_area = medium_pixels * pixel_area_km2
        high_area = high_pixels * pixel_area_km2
        total_area = total_pixels * pixel_area_km2

        profile = candidate_src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            nodata=NODATA_CATEGORY,
            compress="deflate",
        )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with rasterio.open(CATEGORY_MASK_PATH, "w", **profile) as dst:
            dst.write(categories, 1)

        print()
        print("=" * 64)
        print("DAY 6 - CANDIDATE UNCERTAINTY CATEGORIES")
        print("=" * 64)
        print("CRS:", candidate_src.crs)
        print("NoData category:", NODATA_CATEGORY)
        print("Thresholds: low < 0.33; medium < 0.66; high >= 0.66")
        print("Valid candidate pixels:", total_pixels)
        print("Candidate area (km2):", round(total_area, 4))

        print()
        print("Low uncertainty proxy:")
        print("  Pixels:", low_pixels)
        print("  Area (km2):", round(low_area, 4))
        print("  Percentage:", round(low_pixels / total_pixels * 100, 2))

        print()
        print("Medium uncertainty proxy:")
        print("  Pixels:", medium_pixels)
        print("  Area (km2):", round(medium_area, 4))
        print("  Percentage:", round(medium_pixels / total_pixels * 100, 2))

        print()
        print("High uncertainty proxy:")
        print("  Pixels:", high_pixels)
        print("  Area (km2):", round(high_area, 4))
        print("  Percentage:", round(high_pixels / total_pixels * 100, 2))

        print()
        print("Category total check:",
              low_pixels + medium_pixels + high_pixels,
              "of", total_pixels)
        print("Output:", CATEGORY_MASK_PATH)
        print()
        print("Note: categories are heuristic entropy groupings,")
        print("not calibrated probabilities of model error.")
        print("=" * 64)


if __name__ == "__main__":
    main()
