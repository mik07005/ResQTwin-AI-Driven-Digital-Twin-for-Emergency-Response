from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

JRC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chennai"
    / "persistent_water"
    / "jrc_gsw"
    / "occurrence_80E_20N_v1_5_2024.tif"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "flood_probability.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
)

OCCURRENCE_OUTPUT = OUTPUT_DIR / "jrc_occurrence_gcc_utm44n.tif"
PERMANENT_OUTPUT = OUTPUT_DIR / "jrc_permanent_water_80_gcc_utm44n.tif"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PERMANENT_WATER_THRESHOLD = 80.0


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(JRC_PATH) as src_jrc, rasterio.open(TARGET_PATH) as target:

        print("JRC CRS:", src_jrc.crs)
        print("JRC bounds:", src_jrc.bounds)
        print("Target CRS:", target.crs)
        print("Target size:", target.width, target.height)
        print("Target resolution:", target.res)

        # Read JRC occurrence as float32.
        jrc_data = src_jrc.read(1).astype(np.float32)

        # Target grid.
        target_height = target.height
        target_width = target.width
        target_transform = target.transform
        target_crs = target.crs

        # Reproject occurrence onto EXACT target grid.
        occurrence = np.zeros(
            (target_height, target_width),
            dtype=np.float32
        )

        reproject(
            source=jrc_data,
            destination=occurrence,
            src_transform=src_jrc.transform,
            src_crs=src_jrc.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=0.0,
        )

        # Numerical safety.
        occurrence = np.clip(occurrence, 0.0, 100.0)

        # Save occurrence raster.
        occurrence_profile = target.profile.copy()
        occurrence_profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=2,
        )

        with rasterio.open(
            OCCURRENCE_OUTPUT,
            "w",
            **occurrence_profile
        ) as dst:

            dst.write(occurrence, 1)

        # Permanent-water mask.
        permanent_water = (
            occurrence >= PERMANENT_WATER_THRESHOLD
        ).astype(np.uint8)

        permanent_profile = target.profile.copy()
        permanent_profile.update(
            dtype="uint8",
            count=1,
            nodata=0,
            compress="deflate",
        )

        with rasterio.open(
            PERMANENT_OUTPUT,
            "w",
            **permanent_profile
        ) as dst:

            dst.write(permanent_water, 1)

        # Diagnostics.
        valid_occurrence = np.isfinite(occurrence)

        occurrence_valid_count = int(valid_occurrence.sum())
        permanent_count = int(permanent_water.sum())

        pixel_area_km2 = (
            abs(target.transform.a)
            * abs(target.transform.e)
            / 1_000_000.0
        )

        print()
        print("Finished.")
        print("Occurrence output:", OCCURRENCE_OUTPUT)
        print("Permanent-water output:", PERMANENT_OUTPUT)
        print()
        print("Occurrence valid pixels:", occurrence_valid_count)
        print("Occurrence min:", float(occurrence.min()))
        print("Occurrence max:", float(occurrence.max()))
        print("Occurrence mean:", float(occurrence.mean()))
        print()
        print("Permanent-water pixels:", permanent_count)
        print("Target grid area (km²):",
              round(target.width * target.height * pixel_area_km2, 4))
        print("Permanent-water area on grid (km²):",
              round(permanent_count * pixel_area_km2, 4))


if __name__ == "__main__":
    main()