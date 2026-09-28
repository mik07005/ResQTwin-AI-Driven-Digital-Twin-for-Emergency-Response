from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

INPUT_SAR = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
)

PROBABILITY = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "flood_probability.tif"
)

MASK = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "flood_mask.tif"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLD = 0.80


def save_image(data, path, title, cmap="gray", vmin=None, vmax=None):

    plt.figure(figsize=(12, 10))

    plt.imshow(
        data,
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


def main():

    print("=== Validate Chennai GCC Flood Prediction ===")
    print()

    # ---------------------------------------------------------
    # Read SAR input
    # ---------------------------------------------------------

    with rasterio.open(INPUT_SAR) as src:

        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32)

        sar_profile = src.profile.copy()

        sar_valid = (
            (vv > 0) &
            (vh > 0)
        )

        print("SAR input:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print("  Valid pixels:", int(sar_valid.sum()))
        print()

    # ---------------------------------------------------------
    # Read probability
    # ---------------------------------------------------------

    with rasterio.open(PROBABILITY) as src:

        probability = src.read(1).astype(np.float32)

        print("Probability raster:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print("  NoData:", src.nodata)
        print()

        if src.crs != sar_profile["crs"]:
            raise ValueError("Probability CRS mismatch")

        if (
            src.width != sar_profile["width"]
            or src.height != sar_profile["height"]
        ):
            raise ValueError("Probability dimensions mismatch")

        probability_valid = probability > 0

    # ---------------------------------------------------------
    # Read mask
    # ---------------------------------------------------------

    with rasterio.open(MASK) as src:

        flood_mask = src.read(1)

        print("Flood mask:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print("  NoData:", src.nodata)
        print()

        if src.crs != sar_profile["crs"]:
            raise ValueError("Mask CRS mismatch")

        if (
            src.width != sar_profile["width"]
            or src.height != sar_profile["height"]
        ):
            raise ValueError("Mask dimensions mismatch")

    # ---------------------------------------------------------
    # Validate probability values
    # ---------------------------------------------------------

    probability_valid = sar_valid

    valid_probability = probability[probability_valid]

    print("=== Probability Validation ===")

    print(
        "Min:",
        float(valid_probability.min())
    )

    print(
        "Max:",
        float(valid_probability.max())
    )

    print(
        "Mean:",
        float(valid_probability.mean())
    )

    print(
        "Median:",
        float(np.median(valid_probability))
    )

    print(
        "P01:",
        float(np.percentile(valid_probability, 1))
    )

    print(
        "P05:",
        float(np.percentile(valid_probability, 5))
    )

    print(
        "P10:",
        float(np.percentile(valid_probability, 10))
    )

    print(
        "P25:",
        float(np.percentile(valid_probability, 25))
    )

    print(
        "P50:",
        float(np.percentile(valid_probability, 50))
    )

    print(
        "P75:",
        float(np.percentile(valid_probability, 75))
    )

    print(
        "P90:",
        float(np.percentile(valid_probability, 90))
    )

    print(
        "P95:",
        float(np.percentile(valid_probability, 95))
    )

    print(
        "P99:",
        float(np.percentile(valid_probability, 99))
    )

    # Check probability bounds.
    outside_probability_range = (
        (valid_probability < 0)
        |
        (valid_probability > 1)
    ).sum()

    print(
        "Values outside [0,1]:",
        int(outside_probability_range)
    )

    if outside_probability_range > 0:
        raise ValueError(
            "Probability raster contains values outside [0,1]"
        )

    # ---------------------------------------------------------
    # Validate mask
    # ---------------------------------------------------------

    print()
    print("=== Mask Validation ===")

    valid_mask = flood_mask != 255

    unique_values = np.unique(
        flood_mask
    )

    print(
        "Unique values:",
        unique_values.tolist()
    )

    allowed_values = {0, 1, 255}

    if not set(unique_values).issubset(
        allowed_values
    ):
        raise ValueError(
            "Unexpected values found in flood mask"
        )

    mask_valid = (
        flood_mask != 255
    )

    # Mask and SAR valid pixels should agree.
    mask_mismatch = (
        mask_valid != sar_valid
    ).sum()

    print(
        "SAR/mask validity mismatches:",
        int(mask_mismatch)
    )

    if mask_mismatch > 0:
        print(
            "WARNING: SAR and mask validity masks differ."
        )

    flood_pixels = (
        flood_mask == 1
    )

    non_flood_pixels = (
        flood_mask == 0
    )

    print(
        "Flood pixels:",
        int(flood_pixels.sum())
    )

    print(
        "Non-flood pixels:",
        int(non_flood_pixels.sum())
    )

    print(
        "NoData pixels:",
        int((flood_mask == 255).sum())
    )

    print(
        "Flood fraction: %.4f%%"
        % (
            100.0
            * flood_pixels.sum()
            / sar_valid.sum()
        )
    )

    flood_area_km2 = (
        flood_pixels.sum()
        * 10.0
        * 10.0
        / 1_000_000.0
    )

    print(
        "Predicted flood area: %.4f km²"
        % flood_area_km2
    )

    # ---------------------------------------------------------
    # Probability vs mask consistency
    # ---------------------------------------------------------

    expected_mask = np.zeros_like(
        flood_mask,
        dtype=np.uint8
    )

    expected_mask[sar_valid] = (
        probability[sar_valid] >= THRESHOLD
    ).astype(np.uint8)

    expected_mask[~sar_valid] = 255

    mismatch = (
        expected_mask != flood_mask
    ).sum()

    print()
    print(
        "Probability/mask mismatches:",
        int(mismatch)
    )

    if mismatch > 0:
        raise ValueError(
            "Flood mask does not match probability threshold"
        )

    # ---------------------------------------------------------
    # Visualization 1 — Probability
    # ---------------------------------------------------------

    probability_display = probability.copy()
    probability_display[~sar_valid] = np.nan

    save_image(
        probability_display,
        OUTPUT_DIR / "gcc_flood_probability.png",
        "Chennai GCC Flood Probability",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # ---------------------------------------------------------
    # Visualization 2 — Binary flood mask
    # ---------------------------------------------------------

    mask_display = flood_mask.astype(float)
    mask_display[flood_mask == 255] = np.nan

    save_image(
        mask_display,
        OUTPUT_DIR / "gcc_flood_mask.png",
        "Chennai GCC Predicted Flood Mask (Threshold 0.80)",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # ---------------------------------------------------------
    # Visualization 3 — VV background + flood overlay
    # ---------------------------------------------------------

    vv_valid = vv[sar_valid]

    vv_low, vv_high = np.percentile(
        vv_valid,
        [2, 98]
    )

    vv_display = np.full_like(
        vv,
        np.nan,
        dtype=np.float32
    )

    vv_display[sar_valid] = (
        10.0
        * np.log10(
            np.maximum(
                vv[sar_valid],
                1e-8
            )
        )
    )

    vv_db_valid = vv_display[
        np.isfinite(vv_display)
    ]

    vv_db_low, vv_db_high = np.percentile(
        vv_db_valid,
        [2, 98]
    )

    plt.figure(figsize=(12, 10))

    plt.imshow(
        vv_display,
        cmap="gray",
        vmin=vv_db_low,
        vmax=vv_db_high
    )

    overlay = np.ma.masked_where(
        flood_mask != 1,
        flood_mask
    )

    plt.imshow(
        overlay,
        cmap="autumn",
        alpha=0.45,
        vmin=0,
        vmax=1
    )

    plt.title(
        "Chennai GCC Sentinel-1 VV + Predicted Flood"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gcc_flood_overlay.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # Visualization 4 — Flood probability histogram
    # ---------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.hist(
        valid_probability,
        bins=50
    )

    plt.axvline(
        THRESHOLD,
        linestyle="--",
        label="Threshold = 0.80"
    )

    plt.xlabel("Flood probability")
    plt.ylabel("Pixel count")
    plt.title(
        "Chennai GCC Flood Probability Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gcc_probability_histogram.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print()
    print("Created validation outputs in:")
    print(OUTPUT_DIR)

    print()
    print("=== VALIDATION COMPLETE ===")


if __name__ == "__main__":
    main()  