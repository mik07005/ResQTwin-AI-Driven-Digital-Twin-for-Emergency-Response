from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "sentinel1_vv_vh_chennai_utm44n.tif"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "mosaic_visualization"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile_stretch(data, low=2, high=98):
    valid = data[data > 0]

    p_low, p_high = np.percentile(valid, [low, high])

    stretched = (data - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1)

    stretched[data <= 0] = np.nan

    return stretched, p_low, p_high


def linear_to_db(data):
    db = np.full_like(data, np.nan, dtype=np.float32)

    valid = data > 0
    db[valid] = 10.0 * np.log10(data[valid])

    return db


def save_image(data, path, title, cmap="gray", vmin=None, vmax=None):
    plt.figure(figsize=(12, 10))

    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


def main():

    print("=== Final Chennai Sentinel-1 Mosaic Visualization ===")
    print()

    print("Input:", INPUT)
    print("Output:", OUTPUT_DIR)
    print()

    with rasterio.open(INPUT) as src:

        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)

        print("CRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print()

        vv_valid = vv > 0
        vh_valid = vh > 0

        print("VV valid pixels:", int(vv_valid.sum()))
        print("VH valid pixels:", int(vh_valid.sum()))
        print(
            "Coverage: %.4f%%"
            % (100.0 * vv_valid.sum() / vv_valid.size)
        )
        print()

    # ---------------------------------------------------------
    # Convert to dB for visualization
    # ---------------------------------------------------------

    vv_db = linear_to_db(vv)
    vh_db = linear_to_db(vh)

    vv_db_valid = vv_db[np.isfinite(vv_db)]
    vh_db_valid = vh_db[np.isfinite(vh_db)]

    vv_low, vv_high = np.percentile(vv_db_valid, [2, 98])
    vh_low, vh_high = np.percentile(vh_db_valid, [2, 98])

    print("VV dB display range:", vv_low, "to", vv_high)
    print("VH dB display range:", vh_low, "to", vh_high)
    print()

    # ---------------------------------------------------------
    # VV
    # ---------------------------------------------------------

    save_image(
        vv_db,
        OUTPUT_DIR / "chennai_mosaic_vv_db.png",
        "Chennai Sentinel-1 VV σ⁰ — Two-Slice Mosaic",
        cmap="gray",
        vmin=vv_low,
        vmax=vv_high,
    )

    # ---------------------------------------------------------
    # VH
    # ---------------------------------------------------------

    save_image(
        vh_db,
        OUTPUT_DIR / "chennai_mosaic_vh_db.png",
        "Chennai Sentinel-1 VH σ⁰ — Two-Slice Mosaic",
        cmap="gray",
        vmin=vh_low,
        vmax=vh_high,
    )

    # ---------------------------------------------------------
    # VV/VH ratio
    # ---------------------------------------------------------

    ratio_db = np.full_like(vv, np.nan, dtype=np.float32)

    common = (vv > 0) & (vh > 0)

    ratio_db[common] = (
        10.0
        * np.log10(
            vv[common] / vh[common]
        )
    )

    ratio_valid = ratio_db[np.isfinite(ratio_db)]

    ratio_low, ratio_high = np.percentile(
        ratio_valid,
        [2, 98]
    )

    save_image(
        ratio_db,
        OUTPUT_DIR / "chennai_mosaic_vv_vh_ratio_db.png",
        "Chennai Sentinel-1 VV/VH Ratio — Two-Slice Mosaic",
        cmap="gray",
        vmin=ratio_low,
        vmax=ratio_high,
    )

    # ---------------------------------------------------------
    # VV/VH composite
    # ---------------------------------------------------------

    vv_stretched, _, _ = percentile_stretch(vv)
    vh_stretched, _, _ = percentile_stretch(vh)

    composite = np.zeros(
        (vv.shape[0], vv.shape[1], 3),
        dtype=np.float32
    )

    valid = common

    # VV → red
    composite[:, :, 0] = np.nan_to_num(
        vv_stretched,
        nan=0.0
    )

    # VH → green
    composite[:, :, 1] = np.nan_to_num(
        vh_stretched,
        nan=0.0
    )

    # VV/VH ratio → blue
    ratio_norm = np.full_like(ratio_db, 0.0)

    ratio_norm[valid] = np.clip(
        (ratio_db[valid] - ratio_low)
        / (ratio_high - ratio_low),
        0,
        1
    )

    composite[:, :, 2] = ratio_norm

    save_image(
        composite,
        OUTPUT_DIR / "chennai_mosaic_vv_vh_composite.png",
        "Chennai Sentinel-1 VV/VH Composite — Two-Slice Mosaic",
    )

    # ---------------------------------------------------------
    # Valid-data footprint
    # ---------------------------------------------------------

    footprint = valid.astype(np.uint8)

    save_image(
        footprint,
        OUTPUT_DIR / "chennai_mosaic_valid_footprint.png",
        "Chennai Sentinel-1 Valid Data Footprint",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    print("Created:")
    print(
        "  ",
        OUTPUT_DIR / "chennai_mosaic_vv_db.png"
    )
    print(
        "  ",
        OUTPUT_DIR / "chennai_mosaic_vh_db.png"
    )
    print(
        "  ",
        OUTPUT_DIR / "chennai_mosaic_vv_vh_ratio_db.png"
    )
    print(
        "  ",
        OUTPUT_DIR / "chennai_mosaic_vv_vh_composite.png"
    )
    print(
        "  ",
        OUTPUT_DIR / "chennai_mosaic_valid_footprint.png"
    )

    print()
    print("VISUALIZATION: PASS")


if __name__ == "__main__":
    main()