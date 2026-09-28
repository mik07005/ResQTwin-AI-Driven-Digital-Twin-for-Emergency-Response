from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------

PRE_PROBABILITY = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "pre_event"
    / "gcc"
    / "inference"
    / "flood_probability.tif"
)

PRE_MASK = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "pre_event"
    / "gcc"
    / "inference"
    / "flood_mask.tif"
)

POST_PROBABILITY = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
    / "flood_probability.tif"
)

POST_MASK = (
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
    / "change_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLD = 0.80


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def validate_geometry(pre, post, name):
    """Verify that two rasters are spatially compatible."""

    if pre.crs != post.crs:
        raise ValueError(f"{name}: CRS mismatch")

    if pre.width != post.width or pre.height != post.height:
        raise ValueError(f"{name}: dimensions mismatch")

    if pre.transform != post.transform:
        raise ValueError(f"{name}: transform mismatch")

    if pre.res != post.res:
        raise ValueError(f"{name}: resolution mismatch")


def save_image(data, path, title, cmap="gray", vmin=None, vmax=None):
    """Save a simple raster visualization."""

    plt.figure(figsize=(12, 10))

    plt.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=== Chennai Pre/Post Flood Change Analysis ===")
    print()

    print("Pre-event probability:")
    print(PRE_PROBABILITY)

    print("Post-event probability:")
    print(POST_PROBABILITY)

    print("Pre-event mask:")
    print(PRE_MASK)

    print("Post-event mask:")
    print(POST_MASK)

    print()

    # -----------------------------------------------------
    # Read probability rasters
    # -----------------------------------------------------

    with rasterio.open(PRE_PROBABILITY) as src:
        pre_probability = src.read(1).astype(np.float32)
        pre_profile = src.profile.copy()

        print("Pre-event probability geometry:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print("  Bounds:", src.bounds)
        print()

        with rasterio.open(PRE_PROBABILITY) as pre_src, \
         rasterio.open(POST_PROBABILITY) as post_src:

            post_probability = post_src.read(1).astype(np.float32)

        print("Post-event probability geometry:")
        print("  CRS:", post_src.crs)
        print("  Size:", post_src.width, "x", post_src.height)
        print("  Resolution:", post_src.res)
        print("  Bounds:", post_src.bounds)
        print()

        validate_geometry(
            pre_src,
            post_src,
            "Probability rasters",
        )

    # -----------------------------------------------------
    # Read binary masks
    # -----------------------------------------------------

    with rasterio.open(PRE_MASK) as src:
        pre_mask = src.read(1)

        print("Pre-event mask geometry:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print()

    with rasterio.open(POST_MASK) as src:
        post_mask = src.read(1)

        print("Post-event mask geometry:")
        print("  CRS:", src.crs)
        print("  Size:", src.width, "x", src.height)
        print("  Resolution:", src.res)
        print()

    # -----------------------------------------------------
    # Geometry checks
    # -----------------------------------------------------

    with rasterio.open(PRE_MASK) as pre_src, \
         rasterio.open(POST_MASK) as post_src:

        validate_geometry(
            pre_src,
            post_src,
            "Mask rasters",
        )

    print("=== Geometry Validation ===")
    print("Probability geometry: PASS")
    print("Mask geometry: PASS")
    print()

    # -----------------------------------------------------
    # Validity masks
    # -----------------------------------------------------

    pre_valid = pre_mask != 255
    post_valid = post_mask != 255

    common_valid = pre_valid & post_valid

    print("=== Validity ===")
    print("Pre-event valid:", int(pre_valid.sum()))
    print("Post-event valid:", int(post_valid.sum()))
    print("Common valid:", int(common_valid.sum()))

    print(
        "Pre-only valid:",
        int((pre_valid & ~post_valid).sum()),
    )

    print(
        "Post-only valid:",
        int((post_valid & ~pre_valid).sum()),
    )

    print()

    # -----------------------------------------------------
    # Probability validation
    # -----------------------------------------------------

    pre_valid_probability = pre_probability[pre_valid]
    post_valid_probability = post_probability[post_valid]

    print("=== Probability Validation ===")

    print(
        "Pre probability range:",
        float(pre_valid_probability.min()),
        "->",
        float(pre_valid_probability.max()),
    )

    print(
        "Post probability range:",
        float(post_valid_probability.min()),
        "->",
        float(post_valid_probability.max()),
    )

    pre_outside = (
        (pre_valid_probability < 0)
        | (pre_valid_probability > 1)
    ).sum()

    post_outside = (
        (post_valid_probability < 0)
        | (post_valid_probability > 1)
    ).sum()

    print("Pre values outside [0,1]:", int(pre_outside))
    print("Post values outside [0,1]:", int(post_outside))

    if pre_outside > 0 or post_outside > 0:
        raise ValueError(
            "Probability values outside [0,1] detected."
        )

    print("Probability validation: PASS")
    print()

    # -----------------------------------------------------
    # Binary mask validation
    # -----------------------------------------------------

    print("=== Mask Validation ===")

    pre_unique = np.unique(pre_mask)
    post_unique = np.unique(post_mask)

    print("Pre-event unique values:", pre_unique.tolist())
    print("Post-event unique values:", post_unique.tolist())

    allowed = {0, 1, 255}

    if not set(pre_unique).issubset(allowed):
        raise ValueError(
            "Unexpected value in pre-event mask."
        )

    if not set(post_unique).issubset(allowed):
        raise ValueError(
            "Unexpected value in post-event mask."
        )

    print("Mask values: PASS")
    print()

    # -----------------------------------------------------
    # Check masks against probabilities
    # -----------------------------------------------------

    expected_pre = np.zeros_like(pre_mask, dtype=np.uint8)
    expected_post = np.zeros_like(post_mask, dtype=np.uint8)

    expected_pre[pre_valid] = (
        pre_probability[pre_valid] >= THRESHOLD
    ).astype(np.uint8)

    expected_post[post_valid] = (
        post_probability[post_valid] >= THRESHOLD
    ).astype(np.uint8)

    expected_pre[~pre_valid] = 255
    expected_post[~post_valid] = 255

    pre_mask_mismatch = (
        expected_pre != pre_mask
    ).sum()

    post_mask_mismatch = (
        expected_post != post_mask
    ).sum()

    print("Pre probability/mask mismatches:",
          int(pre_mask_mismatch))

    print("Post probability/mask mismatches:",
          int(post_mask_mismatch))

    if pre_mask_mismatch > 0:
        raise ValueError(
            "Pre-event mask does not match threshold."
        )

    if post_mask_mismatch > 0:
        raise ValueError(
            "Post-event mask does not match threshold."
        )

    print("Probability/mask consistency: PASS")
    print()

    # -----------------------------------------------------
    # Basic pre/post change
    # -----------------------------------------------------

    pre_flood = (
        pre_mask == 1
    )

    post_flood = (
        post_mask == 1
    )

    # Only compare pixels observed on both dates.
    pre_flood_common = pre_flood & common_valid
    post_flood_common = post_flood & common_valid

    persistent = (
        pre_flood_common
        & post_flood_common
    )

    newly_detected = (
        post_flood_common
        & ~pre_flood_common
    )

    pre_lost = (
        pre_flood_common
        & ~post_flood_common
    )

    unchanged_dry = (
        ~pre_flood_common
        & ~post_flood_common
        & common_valid
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    pixel_area_km2 = 10.0 * 10.0 / 1_000_000.0

    print("=== Pre/Post Change Statistics ===")
    print()

    print(
        "Pre-event predicted water/flood pixels:",
        int(pre_flood_common.sum()),
    )

    print(
        "Post-event predicted water/flood pixels:",
        int(post_flood_common.sum()),
    )

    print(
        "Persistent predicted water pixels:",
        int(persistent.sum()),
    )

    print(
        "Newly detected pixels:",
        int(newly_detected.sum()),
    )

    print(
        "Pre-event pixels no longer detected:",
        int(pre_lost.sum()),
    )

    print(
        "Unchanged dry pixels:",
        int(unchanged_dry.sum()),
    )

    print()

    print(
        "Pre-event predicted area: %.4f km²"
        % (pre_flood_common.sum() * pixel_area_km2)
    )

    print(
        "Post-event predicted area: %.4f km²"
        % (post_flood_common.sum() * pixel_area_km2)
    )

    print(
        "Persistent predicted area: %.4f km²"
        % (persistent.sum() * pixel_area_km2)
    )

    print(
        "Newly detected candidate area: %.4f km²"
        % (newly_detected.sum() * pixel_area_km2)
    )

    print(
        "Area lost from prediction: %.4f km²"
        % (pre_lost.sum() * pixel_area_km2)
    )

    print()

    # -----------------------------------------------------
    # Probability change
    # -----------------------------------------------------

    probability_change = np.full_like(
        pre_probability,
        np.nan,
        dtype=np.float32,
    )

    probability_change[common_valid] = (
        post_probability[common_valid]
        - pre_probability[common_valid]
    )

    print("=== Probability Change ===")

    valid_change = probability_change[
        np.isfinite(probability_change)
    ]

    print(
        "Mean probability change:",
        float(valid_change.mean()),
    )

    print(
        "Median probability change:",
        float(np.median(valid_change)),
    )

    print(
        "Minimum probability change:",
        float(valid_change.min()),
    )

    print(
        "Maximum probability change:",
        float(valid_change.max()),
    )

    print()

    # -----------------------------------------------------
    # Save candidate change mask
    # -----------------------------------------------------

    candidate_change = np.zeros_like(
        post_mask,
        dtype=np.uint8,
    )

    candidate_change[newly_detected] = 1

    candidate_change[~common_valid] = 255

    candidate_output = (
        OUTPUT_DIR
        / "candidate_new_inundation_mask.tif"
    )

    candidate_profile = pre_profile.copy()

    candidate_profile.update(
        dtype="uint8",
        count=1,
        nodata=255,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(
        candidate_output,
        "w",
        **candidate_profile,
    ) as dst:

        dst.write(candidate_change, 1)

        dst.set_band_description(
            1,
            "Candidate new inundation: "
            "0=none, 1=new, 255=NoData",
        )

    print("Candidate change mask:")
    print(candidate_output)
    print()

    # -----------------------------------------------------
    # Save probability change raster
    # -----------------------------------------------------

    probability_output = (
        OUTPUT_DIR
        / "probability_change_post_minus_pre.tif"
    )

    probability_profile = pre_profile.copy()

    probability_profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(
        probability_output,
        "w",
        **probability_profile,
    ) as dst:

        dst.write(
            probability_change,
            1,
        )

        dst.set_band_description(
            1,
            "Post-event minus pre-event "
            "flood probability",
        )

    print("Probability change:")
    print(probability_output)
    print()

    # -----------------------------------------------------
    # Visualizations
    # -----------------------------------------------------

    pre_display = np.where(
        common_valid,
        pre_probability,
        np.nan,
    )

    post_display = np.where(
        common_valid,
        post_probability,
        np.nan,
    )

    change_display = np.where(
        common_valid,
        probability_change,
        np.nan,
    )

    candidate_display = np.where(
        candidate_change == 1,
        1,
        np.nan,
    )

    save_image(
        pre_display,
        OUTPUT_DIR / "pre_event_probability.png",
        "4-Nov-2021 Pre-Event Flood Probability",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    save_image(
        post_display,
        OUTPUT_DIR / "post_event_probability.png",
        "16-Nov-2021 Post-Event Flood Probability",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    save_image(
        change_display,
        OUTPUT_DIR / "probability_change.png",
        "Flood Probability Change: Post - Pre",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    save_image(
        candidate_display,
        OUTPUT_DIR / "candidate_new_inundation.png",
        "Candidate New Inundation",
        cmap="autumn",
        vmin=0,
        vmax=1,
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    print("Created change-analysis outputs in:")
    print(OUTPUT_DIR)

    print()
    print("=== PRE/POST COMPARISON COMPLETE ===")


if __name__ == "__main__":
    main()