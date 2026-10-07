from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CATEGORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "candidate_uncertainty_categories.tif"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "candidate_uncertainty_map.png"
)


def main():

    with rasterio.open(CATEGORY_PATH) as src:

        categories = src.read(1)

        bounds = src.bounds

        extent = [
            bounds.left,
            bounds.right,
            bounds.bottom,
            bounds.top,
        ]

        print("CRS:", src.crs)
        print("Size:", src.width, src.height)
        print("Resolution:", src.res)

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    low = categories == 1
    medium = categories == 2
    high = categories == 3

    pixel_area_km2 = 0.0001  # 10 m × 10 m

    low_area = low.sum() * pixel_area_km2
    medium_area = medium.sum() * pixel_area_km2
    high_area = high.sum() * pixel_area_km2

    total_area = low_area + medium_area + high_area

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    # 0 = background
    # 1 = low
    # 2 = medium
    # 3 = high
    cmap = ListedColormap([
        "black",
        "green",
        "gold",
        "red",
    ])

    fig, ax = plt.subplots(figsize=(10, 9))

    image = ax.imshow(
        categories,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=3,
    )

    ax.set_title(
        "Chennai GCC — Candidate Inundation Uncertainty",
        fontsize=17,
        pad=15,
    )

    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")

    # -----------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="green",
            label=f"Low uncertainty: {low_area:.4f} km² "
                  f"({low_area / total_area * 100:.2f}%)",
        ),
        Patch(
            facecolor="gold",
            label=f"Medium uncertainty: {medium_area:.4f} km² "
                  f"({medium_area / total_area * 100:.2f}%)",
        ),
        Patch(
            facecolor="red",
            label=f"High uncertainty: {high_area:.4f} km² "
                  f"({high_area / total_area * 100:.2f}%)",
        ),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        title="Model-derived uncertainty proxy",
    )

    ax.text(
        0.01,
        0.01,
        f"Total filtered candidate inundation: {total_area:.4f} km²",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="none",
        ),
    )

    plt.tight_layout()

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    print()
    print("=" * 60)
    print("UNCERTAINTY MAP CREATED")
    print("=" * 60)

    print()
    print("Low uncertainty:")
    print(f"  {low.sum()} pixels")
    print(f"  {low_area:.4f} km²")

    print()
    print("Medium uncertainty:")
    print(f"  {medium.sum()} pixels")
    print(f"  {medium_area:.4f} km²")

    print()
    print("High uncertainty:")
    print(f"  {high.sum()} pixels")
    print(f"  {high_area:.4f} km²")

    print()
    print(f"Total: {total_area:.4f} km²")

    print()
    print("Saved:")
    print(OUTPUT_PATH)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()