from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "sentinel1_vv_vh_chennai_utm44n.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parameters
# ============================================================

# Percentile stretch prevents extreme SAR values from
# dominating the visualization.
LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98

# Downsample for visualization only.
MAX_DISPLAY_SIZE = 1600


# ============================================================
# Utility functions
# ============================================================

def percentile_stretch(image, low=2, high=98):
    """
    Robust percentile normalization for visualization.
    Does NOT modify the original data.
    """

    valid = np.isfinite(image) & (image > 0)

    if not np.any(valid):
        raise ValueError("No valid pixels found.")

    low_value = np.percentile(image[valid], low)
    high_value = np.percentile(image[valid], high)

    stretched = np.zeros_like(image, dtype=np.float32)

    stretched[valid] = (
        (image[valid] - low_value)
        / (high_value - low_value + 1e-12)
    )

    stretched = np.clip(stretched, 0, 1)

    return stretched, low_value, high_value


def log_scale(image):
    """
    Convert linear Sigma0 to dB-like logarithmic representation.

    Visualization only:
        dB = 10 * log10(Sigma0)
    """

    valid = np.isfinite(image) & (image > 0)

    output = np.full(
        image.shape,
        np.nan,
        dtype=np.float32
    )

    output[valid] = 10.0 * np.log10(image[valid])

    return output


def downsample(image, max_size=1600):
    """
    Simple nearest-neighbor downsampling for visualization.
    """

    height, width = image.shape

    scale = min(
        1.0,
        max_size / max(height, width)
    )

    if scale >= 1.0:
        return image

    new_height = max(1, int(height * scale))
    new_width = max(1, int(width * scale))

    row_idx = np.linspace(
        0,
        height - 1,
        new_height
    ).astype(int)

    col_idx = np.linspace(
        0,
        width - 1,
        new_width
    ).astype(int)

    return image[np.ix_(row_idx, col_idx)]


def save_image(image, path, title, cmap="gray",
               vmin=None, vmax=None):

    plt.figure(figsize=(10, 8))

    plt.imshow(
        image,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("CHENNAI SENTINEL-1 VISUAL / GEOSPATIAL SANITY CHECK")
    print("=" * 70)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input stack not found:\n{INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Read raster
    # --------------------------------------------------------

    with rasterio.open(INPUT_PATH) as src:

        print("\nRaster:")
        print(f"CRS        : {src.crs}")
        print(f"Size       : {src.width} x {src.height}")
        print(f"Resolution : {src.res}")
        print(f"Bounds     : {src.bounds}")
        print(f"Bands      : {src.count}")

        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)

        transform = src.transform
        bounds = src.bounds

    # --------------------------------------------------------
    # Valid masks
    # --------------------------------------------------------

    vv_valid = np.isfinite(vv) & (vv > 0)
    vh_valid = np.isfinite(vh) & (vh > 0)

    common_valid = vv_valid & vh_valid

    print("\nValid pixels:")
    print(f"VV     : {vv_valid.sum():,}")
    print(f"VH     : {vh_valid.sum():,}")
    print(f"Common : {common_valid.sum():,}")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\nVV statistics:")
    print(f"  Mean   : {vv[vv_valid].mean():.8f}")
    print(f"  Median : {np.median(vv[vv_valid]):.8f}")
    print(f"  Std    : {vv[vv_valid].std():.8f}")

    print("\nVH statistics:")
    print(f"  Mean   : {vh[vh_valid].mean():.8f}")
    print(f"  Median : {np.median(vh[vh_valid]):.8f}")
    print(f"  Std    : {vh[vh_valid].std():.8f}")

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    vv_small = downsample(vv, MAX_DISPLAY_SIZE)
    vh_small = downsample(vh, MAX_DISPLAY_SIZE)

    # --------------------------------------------------------
    # VV visualization
    # --------------------------------------------------------

    vv_vis, vv_low, vv_high = percentile_stretch(
        vv_small,
        LOW_PERCENTILE,
        HIGH_PERCENTILE
    )

    print("\nVV display stretch:")
    print(f"  {LOW_PERCENTILE}th percentile  : {vv_low:.8f}")
    print(f"  {HIGH_PERCENTILE}th percentile : {vv_high:.8f}")

    save_image(
        vv_vis,
        OUTPUT_DIR / "chennai_vv_sigma0.png",
        "Chennai Sentinel-1 VV Sigma0",
        cmap="gray",
        vmin=0,
        vmax=1
    )

    # --------------------------------------------------------
    # VH visualization
    # --------------------------------------------------------

    vh_vis, vh_low, vh_high = percentile_stretch(
        vh_small,
        LOW_PERCENTILE,
        HIGH_PERCENTILE
    )

    print("\nVH display stretch:")
    print(f"  {LOW_PERCENTILE}th percentile  : {vh_low:.8f}")
    print(f"  {HIGH_PERCENTILE}th percentile : {vh_high:.8f}")

    save_image(
        vh_vis,
        OUTPUT_DIR / "chennai_vh_sigma0.png",
        "Chennai Sentinel-1 VH Sigma0",
        cmap="gray",
        vmin=0,
        vmax=1
    )

    # --------------------------------------------------------
    # VV dB
    # --------------------------------------------------------

    vv_db = log_scale(vv_small)

    vv_db_valid = np.isfinite(vv_db)

    if np.any(vv_db_valid):

        vv_db_low = np.percentile(
            vv_db[vv_db_valid],
            LOW_PERCENTILE
        )

        vv_db_high = np.percentile(
            vv_db[vv_db_valid],
            HIGH_PERCENTILE
        )

        print("\nVV dB range for display:")
        print(f"  Low  : {vv_db_low:.3f} dB")
        print(f"  High : {vv_db_high:.3f} dB")

        save_image(
            vv_db,
            OUTPUT_DIR / "chennai_vv_db.png",
            "Chennai Sentinel-1 VV (dB)",
            cmap="gray",
            vmin=vv_db_low,
            vmax=vv_db_high
        )

    # --------------------------------------------------------
    # VH dB
    # --------------------------------------------------------

    vh_db = log_scale(vh_small)

    vh_db_valid = np.isfinite(vh_db)

    if np.any(vh_db_valid):

        vh_db_low = np.percentile(
            vh_db[vh_db_valid],
            LOW_PERCENTILE
        )

        vh_db_high = np.percentile(
            vh_db[vh_db_valid],
            HIGH_PERCENTILE
        )

        print("\nVH dB range for display:")
        print(f"  Low  : {vh_db_low:.3f} dB")
        print(f"  High : {vh_db_high:.3f} dB")

        save_image(
            vh_db,
            OUTPUT_DIR / "chennai_vh_db.png",
            "Chennai Sentinel-1 VH (dB)",
            cmap="gray",
            vmin=vh_db_low,
            vmax=vh_db_high
        )

    # --------------------------------------------------------
    # VV / VH ratio
    # --------------------------------------------------------

    # Ratio is calculated only where both channels are valid.
    ratio = np.full_like(vv_small, np.nan)

    ratio_valid = (
        (vv_small > 0)
        & (vh_small > 0)
        & np.isfinite(vv_small)
        & np.isfinite(vh_small)
    )

    ratio[ratio_valid] = (
        vv_small[ratio_valid]
        / vh_small[ratio_valid]
    )

    ratio_db = np.full_like(ratio, np.nan)

    ratio_db[ratio_valid] = (
        10.0 * np.log10(ratio[ratio_valid])
    )

    ratio_db_valid = np.isfinite(ratio_db)

    if np.any(ratio_db_valid):

        ratio_low = np.percentile(
            ratio_db[ratio_db_valid],
            LOW_PERCENTILE
        )

        ratio_high = np.percentile(
            ratio_db[ratio_db_valid],
            HIGH_PERCENTILE
        )

        print("\nVV/VH ratio dB display range:")
        print(f"  Low  : {ratio_low:.3f} dB")
        print(f"  High : {ratio_high:.3f} dB")

        save_image(
            ratio_db,
            OUTPUT_DIR / "chennai_vv_vh_ratio_db.png",
            "Chennai Sentinel-1 VV/VH Ratio (dB)",
            cmap="gray",
            vmin=ratio_low,
            vmax=ratio_high
        )

    # --------------------------------------------------------
    # SAR composite
    # --------------------------------------------------------

    # Normalize VV and VH independently for visualization.
    vv_comp, _, _ = percentile_stretch(
        vv_small,
        LOW_PERCENTILE,
        HIGH_PERCENTILE
    )

    vh_comp, _, _ = percentile_stretch(
        vh_small,
        LOW_PERCENTILE,
        HIGH_PERCENTILE
    )

    # A simple pseudo-RGB composite:
    #
    # R = VV
    # G = VH
    # B = VV
    #
    # This is ONLY for visual inspection.
    composite = np.zeros(
        (*vv_comp.shape, 3),
        dtype=np.float32
    )

    composite[:, :, 0] = vv_comp
    composite[:, :, 1] = vh_comp
    composite[:, :, 2] = vv_comp

    save_image(
        composite,
        OUTPUT_DIR / "chennai_sentinel1_vv_vh_composite.png",
        "Chennai Sentinel-1 VV/VH Composite"
    )

    # --------------------------------------------------------
    # Report geospatial center
    # --------------------------------------------------------

    center_x = (bounds.left + bounds.right) / 2
    center_y = (bounds.bottom + bounds.top) / 2

    print("\nScene center in EPSG:32644:")
    print(f"  X : {center_x:.3f}")
    print(f"  Y : {center_y:.3f}")

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")

    for file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {file.name}")

    print("\nThese images are visualization products only.")
    print("The original VV/VH GeoTIFF has NOT been modified.")


if __name__ == "__main__":
    main()