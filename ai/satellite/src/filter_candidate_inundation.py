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
    / "sentinel1"
    / "gcc"
    / "change_analysis"
    / "candidate_new_inundation_mask.tif"
)

PERMANENT_WATER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "jrc_permanent_water_80_gcc_utm44n.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
)

FILTERED_OUTPUT = (
    OUTPUT_DIR
    / "candidate_new_inundation_filtered.tif"
)

REMOVED_OUTPUT = (
    OUTPUT_DIR
    / "candidate_persistent_water_removed.tif"
)


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(CANDIDATE_PATH) as candidate_src, \
         rasterio.open(PERMANENT_WATER_PATH) as water_src:

        print("Candidate CRS:", candidate_src.crs)
        print("Permanent-water CRS:", water_src.crs)

        print(
            "Candidate size:",
            candidate_src.width,
            candidate_src.height
        )

        print(
            "Permanent-water size:",
            water_src.width,
            water_src.height
        )

        # -------------------------------------------------------------
        # Confirm grids match
        # -------------------------------------------------------------

        if candidate_src.crs != water_src.crs:
            raise ValueError("CRS mismatch.")

        if candidate_src.width != water_src.width:
            raise ValueError("Width mismatch.")

        if candidate_src.height != water_src.height:
            raise ValueError("Height mismatch.")

        if not np.allclose(
            tuple(candidate_src.transform),
            tuple(water_src.transform),
            atol=1e-9
        ):
            raise ValueError("Transform/grid mismatch.")

        # -------------------------------------------------------------
        # Read masks
        # -------------------------------------------------------------

        candidate = candidate_src.read(1)

        permanent_water = water_src.read(1)

        candidate_mask = candidate == 1
        permanent_mask = permanent_water == 1

        # -------------------------------------------------------------
        # Calculate overlap
        # -------------------------------------------------------------

        persistent_overlap = candidate_mask & permanent_mask

        filtered_candidate = (
            candidate_mask & ~permanent_mask
        )

        # -------------------------------------------------------------
        # Area calculations
        # -------------------------------------------------------------

        pixel_area_m2 = (
            abs(candidate_src.transform.a)
            * abs(candidate_src.transform.e)
        )

        pixel_area_km2 = pixel_area_m2 / 1_000_000.0

        candidate_pixels = int(candidate_mask.sum())
        persistent_pixels = int(persistent_overlap.sum())
        filtered_pixels = int(filtered_candidate.sum())

        candidate_area = candidate_pixels * pixel_area_km2
        persistent_area = persistent_pixels * pixel_area_km2
        filtered_area = filtered_pixels * pixel_area_km2

        overlap_percentage = (
            persistent_pixels / candidate_pixels * 100
            if candidate_pixels > 0
            else 0
        )

        # -------------------------------------------------------------
        # Save filtered candidate
        # -------------------------------------------------------------

        profile = candidate_src.profile.copy()

        profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate"
        )

        with rasterio.open(
            FILTERED_OUTPUT,
            "w",
            **profile
        ) as dst:

            dst.write(filtered_candidate.astype(np.uint8), 1)

        # -------------------------------------------------------------
        # Save removed persistent-water portion
        # -------------------------------------------------------------

        with rasterio.open(
            REMOVED_OUTPUT,
            "w",
            **profile
        ) as dst:

            dst.write(
                persistent_overlap.astype(np.uint8),
                1
            )

        # -------------------------------------------------------------
        # Report
        # -------------------------------------------------------------

        print()
        print("=" * 60)
        print("PERMANENT-WATER FILTERING RESULTS")
        print("=" * 60)

        print()
        print("Candidate new inundation pixels:", candidate_pixels)
        print(
            "Candidate new inundation area (km²):",
            round(candidate_area, 4)
        )

        print()
        print(
            "Candidate pixels overlapping JRC >=80% water:",
            persistent_pixels
        )

        print(
            "Persistent-water overlap area (km²):",
            round(persistent_area, 4)
        )

        print(
            "Candidate area classified as persistent water (%):",
            round(overlap_percentage, 2)
        )

        print()
        print(
            "Remaining candidate inundation pixels:",
            filtered_pixels
        )

        print(
            "Remaining candidate inundation area (km²):",
            round(filtered_area, 4)
        )

        print()
        print("Filtered output:")
        print(FILTERED_OUTPUT)

        print()
        print("Removed persistent-water output:")
        print(REMOVED_OUTPUT)

        print()
        print("=" * 60)


if __name__ == "__main__":
    main()