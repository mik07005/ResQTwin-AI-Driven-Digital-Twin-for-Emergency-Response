from pathlib import Path

import numpy as np
import rasterio


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

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
    / "candidate_inundation_entropy.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
)

CATEGORY_MASK_PATH = (
    OUTPUT_DIR
    / "candidate_uncertainty_categories.tif"
)


# ---------------------------------------------------------------------
# Uncertainty thresholds
# ---------------------------------------------------------------------

LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(CANDIDATE_PATH) as candidate_src, \
         rasterio.open(ENTROPY_PATH) as entropy_src:

        print("Candidate CRS:", candidate_src.crs)
        print("Entropy CRS:", entropy_src.crs)

        print(
            "Candidate size:",
            candidate_src.width,
            candidate_src.height
        )

        print(
            "Entropy size:",
            entropy_src.width,
            entropy_src.height
        )

        # -------------------------------------------------------------
        # Verify that both rasters use exactly the same grid.
        # -------------------------------------------------------------

        if candidate_src.crs != entropy_src.crs:
            raise ValueError("CRS mismatch.")

        if candidate_src.width != entropy_src.width:
            raise ValueError("Width mismatch.")

        if candidate_src.height != entropy_src.height:
            raise ValueError("Height mismatch.")

        if not np.allclose(
            tuple(candidate_src.transform),
            tuple(entropy_src.transform),
            atol=1e-9,
        ):
            raise ValueError("Transform/grid mismatch.")

        # -------------------------------------------------------------
        # Read data
        # -------------------------------------------------------------

        candidate = candidate_src.read(1) == 1
        entropy = entropy_src.read(1).astype(np.float32)

        valid = candidate & np.isfinite(entropy)

        if not valid.any():
            raise ValueError("No valid candidate pixels found.")

        candidate_entropy = entropy[valid]

        # -------------------------------------------------------------
        # Classify uncertainty
        #
        # 1 = Low uncertainty
        # 2 = Medium uncertainty
        # 3 = High uncertainty
        #
        # Only candidate inundation pixels are classified.
        # Non-candidate pixels remain 0.
        # -------------------------------------------------------------

        categories = np.zeros(
            candidate.shape,
            dtype=np.uint8
        )

        low = valid & (entropy < LOW_THRESHOLD)

        medium = valid & (
            (entropy >= LOW_THRESHOLD)
            & (entropy < HIGH_THRESHOLD)
        )

        high = valid & (entropy >= HIGH_THRESHOLD)

        categories[low] = 1
        categories[medium] = 2
        categories[high] = 3

        # -------------------------------------------------------------
        # Area calculation
        # -------------------------------------------------------------

        pixel_area_km2 = (
            abs(candidate_src.transform.a)
            * abs(candidate_src.transform.e)
            / 1_000_000.0
        )

        total_pixels = int(valid.sum())

        low_pixels = int(low.sum())
        medium_pixels = int(medium.sum())
        high_pixels = int(high.sum())

        total_area = total_pixels * pixel_area_km2
        low_area = low_pixels * pixel_area_km2
        medium_area = medium_pixels * pixel_area_km2
        high_area = high_pixels * pixel_area_km2

        low_pct = (
            low_pixels / total_pixels * 100.0
        )

        medium_pct = (
            medium_pixels / total_pixels * 100.0
        )

        high_pct = (
            high_pixels / total_pixels * 100.0
        )

        # -------------------------------------------------------------
        # Save category raster
        # -------------------------------------------------------------

        profile = candidate_src.profile.copy()

        profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(
            CATEGORY_MASK_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(categories, 1)

        # -------------------------------------------------------------
        # Print results
        # -------------------------------------------------------------

        print()
        print("=" * 65)
        print("CANDIDATE INUNDATION UNCERTAINTY ANALYSIS")
        print("=" * 65)

        print()
        print("Total candidate pixels:", total_pixels)
        print(
            "Total candidate area (km²):",
            round(total_area, 4)
        )

        print()
        print("Uncertainty thresholds:")
        print("Low:    entropy < 0.33")
        print("Medium: 0.33 <= entropy < 0.66")
        print("High:   entropy >= 0.66")

        print()
        print("-" * 65)
        print("LOW UNCERTAINTY")
        print("-" * 65)
        print("Pixels:", low_pixels)
        print("Area (km²):", round(low_area, 4))
        print("Percentage:", round(low_pct, 2), "%")

        print()
        print("-" * 65)
        print("MEDIUM UNCERTAINTY")
        print("-" * 65)
        print("Pixels:", medium_pixels)
        print("Area (km²):", round(medium_area, 4))
        print("Percentage:", round(medium_pct, 2), "%")

        print()
        print("-" * 65)
        print("HIGH UNCERTAINTY")
        print("-" * 65)
        print("Pixels:", high_pixels)
        print("Area (km²):", round(high_area, 4))
        print("Percentage:", round(high_pct, 2), "%")

        print()
        print("-" * 65)
        print("CHECK")
        print("-" * 65)

        print(
            "Category pixels:",
            low_pixels + medium_pixels + high_pixels
        )

        print(
            "Expected pixels:",
            total_pixels
        )

        print(
            "Category area (km²):",
            round(
                low_area + medium_area + high_area,
                4
            )
        )

        print(
            "Expected area (km²):",
            round(total_area, 4)
        )

        print()
        print("Category raster:")
        print(CATEGORY_MASK_PATH)

        print()
        print("=" * 65)


if __name__ == "__main__":
    main()