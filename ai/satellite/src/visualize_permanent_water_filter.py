from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


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

REMOVED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "candidate_persistent_water_removed.tif"
)

FILTERED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "candidate_new_inundation_filtered.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "persistent_water"
    / "permanent_water_filter_comparison.png"
)


def main():

    with rasterio.open(CANDIDATE_PATH) as src_candidate, \
         rasterio.open(REMOVED_PATH) as src_removed, \
         rasterio.open(FILTERED_PATH) as src_filtered:

        candidate = src_candidate.read(1) == 1
        removed = src_removed.read(1) == 1
        filtered = src_filtered.read(1) == 1

        # Verify all three grids match.
        for name, src in [
            ("removed", src_removed),
            ("filtered", src_filtered),
        ]:
            if src.crs != src_candidate.crs:
                raise ValueError(f"{name} CRS mismatch")

            if src.width != src_candidate.width:
                raise ValueError(f"{name} width mismatch")

            if src.height != src_candidate.height:
                raise ValueError(f"{name} height mismatch")

            if not np.allclose(
                tuple(src.transform),
                tuple(src_candidate.transform),
                atol=1e-9,
            ):
                raise ValueError(f"{name} transform mismatch")

        # Pixel dimensions.
        pixel_area_km2 = (
            abs(src_candidate.transform.a)
            * abs(src_candidate.transform.e)
            / 1_000_000.0
        )

        candidate_area = candidate.sum() * pixel_area_km2
        removed_area = removed.sum() * pixel_area_km2
        filtered_area = filtered.sum() * pixel_area_km2

        # Plot extent.
        bounds = src_candidate.bounds
        extent = [
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(
        candidate,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[0].set_title(
        f"Raw Candidate New Inundation\n"
        f"{candidate_area:.4f} km²"
    )

    axes[1].imshow(
        removed,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[1].set_title(
        f"Persistent-Water Overlap (JRC ≥80%)\n"
        f"{removed_area:.4f} km²"
    )

    axes[2].imshow(
        filtered,
        extent=extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[2].set_title(
        f"Filtered Candidate Inundation\n"
        f"{filtered_area:.4f} km²"
    )

    for ax in axes:
        ax.set_xlabel("UTM Easting (m)")
        ax.set_ylabel("UTM Northing (m)")

    fig.suptitle(
        "Chennai GCC — Permanent-Water Filtering",
        fontsize=16,
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(OUTPUT_PATH)
    print()
    print(f"Raw candidate area: {candidate_area:.4f} km²")
    print(f"Persistent-water overlap: {removed_area:.4f} km²")
    print(f"Filtered candidate area: {filtered_area:.4f} km²")


if __name__ == "__main__":
    main()