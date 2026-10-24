from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]

UNCERTAINTY_DIR = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "uncertainty"
    / "day6"
)

CATEGORY_PATH = UNCERTAINTY_DIR / "candidate_uncertainty_categories.tif"
OUTPUT_PATH = UNCERTAINTY_DIR / "candidate_uncertainty_map.png"

NODATA_CATEGORY = 255

CATEGORY_LABELS = {
    0: "Background",
    1: "Low entropy",
    2: "Medium entropy",
    3: "High entropy",
}

CATEGORY_COLORS = {
    0: "#D9DDE3",
    1: "#2E8B57",
    2: "#F2C14E",
    3: "#D1495B",
}


def main():
    if not CATEGORY_PATH.exists():
        raise FileNotFoundError(
            f"Category raster not found: {CATEGORY_PATH}\n"
            "Run analyze_candidate_uncertainty_day6.py first."
        )

    with rasterio.open(CATEGORY_PATH) as src:
        categories = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        dataset_mask = src.dataset_mask() > 0

        if crs is None:
            raise ValueError("Category raster has no CRS.")

        if not crs.is_projected:
            raise ValueError(
                f"Expected a projected CRS for area calculation; got {crs}."
            )

        units = (crs.linear_units or "").lower()
        if units not in {"metre", "meter", "metres", "meters"}:
            raise ValueError(
                f"Expected metre-based projected units; got {crs.linear_units}."
            )

        pixel_area_m2 = abs(
            transform.a * transform.e - transform.b * transform.d
        )

        if pixel_area_m2 <= 0:
            raise ValueError("Invalid raster pixel area.")

    valid = dataset_mask.copy()

    if nodata is not None:
        valid &= categories != nodata

    valid &= categories != NODATA_CATEGORY

    candidate_mask = valid & np.isin(categories, [1, 2, 3])
    candidate_pixels = int(candidate_mask.sum())

    if candidate_pixels == 0:
        raise ValueError("No categorized candidate pixels were found.")

    pixel_area_km2 = pixel_area_m2 / 1_000_000
    candidate_area_km2 = candidate_pixels * pixel_area_km2

    category_counts = {
        category: int(np.sum(candidate_mask & (categories == category)))
        for category in (1, 2, 3)
    }

    category_areas = {
        category: count * pixel_area_km2
        for category, count in category_counts.items()
    }

    # Mask NoData and background so they remain visually distinct.
    display_data = np.ma.masked_where(
        ~valid | (categories == 0),
        categories,
    )

    cmap = ListedColormap(
        [
            CATEGORY_COLORS[1],
            CATEGORY_COLORS[2],
            CATEGORY_COLORS[3],
        ]
    )
    cmap.set_bad(color="white", alpha=0)

    fig, ax = plt.subplots(figsize=(12, 9))

    image = ax.imshow(
        display_data,
        cmap=cmap,
        vmin=1,
        vmax=3,
        interpolation="nearest",
    )

    legend_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[1],
            label=f"Low entropy: {category_areas[1]:.3f} km² "
                  f"({category_counts[1]:,} pixels)",
        ),
        Patch(
            facecolor=CATEGORY_COLORS[2],
            label=f"Medium entropy: {category_areas[2]:.3f} km² "
                  f"({category_counts[2]:,} pixels)",
        ),
        Patch(
            facecolor=CATEGORY_COLORS[3],
            label=f"High entropy: {category_areas[3]:.3f} km² "
                  f"({category_counts[3]:,} pixels)",
        ),
        Patch(
            facecolor=CATEGORY_COLORS[0],
            label="Background (not a candidate pixel)",
        ),
        Patch(
            facecolor="white",
            edgecolor="#777777",
            label="NoData / invalid source coverage",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        loc="lower right",
        framealpha=0.95,
        fontsize=9,
    )

    ax.set_title(
        "Chennai Candidate Inundation: Model-Derived Entropy Proxy\n"
        f"Candidate area: {candidate_area_km2:.4f} km² | CRS: {crs}",
        fontsize=13,
        pad=14,
    )
    ax.set_xlabel("Raster column (pixels)")
    ax.set_ylabel("Raster row (pixels)")
    ax.set_aspect("equal")

    fig.text(
        0.5,
        0.025,
        "Interpretation caution: candidate inundation is not confirmed flooding. "
        "Entropy categories are heuristic model-derived indicators, "
        "not calibrated probabilities of prediction error.",
        ha="center",
        va="bottom",
        fontsize=9,
        wrap=True,
    )

    fig.tight_layout(rect=(0, 0.065, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=" * 64)
    print("DAY 6 - CANDIDATE UNCERTAINTY VISUALIZATION")
    print("=" * 64)
    print(f"CRS: {crs}")
    print(f"Pixel area: {pixel_area_m2:.2f} m²")
    print(f"Candidate pixels: {candidate_pixels:,}")
    print(f"Candidate area: {candidate_area_km2:.4f} km²")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Category area check:")
    for category in (1, 2, 3):
        percentage = 100 * category_counts[category] / candidate_pixels
        print(
            f"  {CATEGORY_LABELS[category]}: "
            f"{category_counts[category]:,} pixels, "
            f"{category_areas[category]:.4f} km², "
            f"{percentage:.2f}%"
        )
    print()
    print("Caution: this map visualizes model-derived entropy proxies,")
    print("not verified flood extent or calibrated prediction-error probabilities.")
    print("=" * 64)


if __name__ == "__main__":
    main()