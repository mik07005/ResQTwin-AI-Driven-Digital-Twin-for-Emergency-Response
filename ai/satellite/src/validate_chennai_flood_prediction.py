from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROBABILITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "inference"
    / "flood_probability.tif"
)

MASK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "inference"
    / "flood_mask.tif"
)

STACK_PATH = (
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
    / "inference"
    / "validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Display parameters
# ============================================================

MAX_DISPLAY_SIZE = 1800
THRESHOLD = 0.80


# ============================================================
# Functions
# ============================================================

def downsample(image, max_size=1800):
    """
    Downsample an image for visualization only.
    """

    height, width = image.shape

    scale = min(
        1.0,
        max_size / max(height, width)
    )

    if scale >= 1.0:
        return image

    new_height = max(
        1,
        int(height * scale)
    )

    new_width = max(
        1,
        int(width * scale)
    )

    rows = np.linspace(
        0,
        height - 1,
        new_height
    ).astype(int)

    cols = np.linspace(
        0,
        width - 1,
        new_width
    ).astype(int)

    return image[np.ix_(rows, cols)]


def normalize(image):
    """
    Percentile normalization for visualization.
    """

    valid = (
        np.isfinite(image)
        & (image > 0)
    )

    if not np.any(valid):
        return np.zeros_like(
            image,
            dtype=np.float32
        )

    low = np.percentile(
        image[valid],
        2
    )

    high = np.percentile(
        image[valid],
        98
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    output[valid] = (
        image[valid] - low
    ) / (
        high - low + 1e-12
    )

    return np.clip(
        output,
        0,
        1
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("CHENNAI FLOOD PREDICTION VALIDATION")
    print("#" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        PROBABILITY_PATH,
        MASK_PATH,
        STACK_PATH
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Read probability map
    # --------------------------------------------------------

    with rasterio.open(
        PROBABILITY_PATH
    ) as src:

        probability = src.read(
            1
        ).astype(
            np.float32
        )

        profile = src.profile.copy()
        bounds = src.bounds
        transform = src.transform
        crs = src.crs

    # --------------------------------------------------------
    # Read flood mask
    # --------------------------------------------------------

    with rasterio.open(
        MASK_PATH
    ) as src:

        mask = src.read(
            1
        )

    # --------------------------------------------------------
    # Read Sentinel-1 stack
    # --------------------------------------------------------

    with rasterio.open(
        STACK_PATH
    ) as src:

        vv = src.read(
            1
        ).astype(
            np.float32
        )

        vh = src.read(
            2
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Valid mask
    # --------------------------------------------------------

    valid = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & (vv > 0)
        & (vh > 0)
        & (mask != 255)
    )

    flood = (
        mask == 1
    ) & valid

    non_flood = (
        mask == 0
    ) & valid

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PREDICTION STATISTICS")
    print("=" * 70)

    valid_probability = probability[
        valid
    ]

    print(
        f"CRS                 : {crs}"
    )

    print(
        f"Raster size         : "
        f"{probability.shape[1]} x "
        f"{probability.shape[0]}"
    )

    print(
        f"Valid pixels        : "
        f"{valid.sum():,}"
    )

    print(
        f"Flood pixels        : "
        f"{flood.sum():,}"
    )

    print(
        f"Non-flood pixels    : "
        f"{non_flood.sum():,}"
    )

    print(
        f"Flood fraction      : "
        f"{100 * flood.sum() / valid.sum():.4f}%"
    )

    print(
        f"Probability mean    : "
        f"{valid_probability.mean():.6f}"
    )

    print(
        f"Probability median  : "
        f"{np.median(valid_probability):.6f}"
    )

    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    print("\nProbability percentiles:")

    for p in [
        1,
        5,
        10,
        25,
        50,
        75,
        90,
        95,
        99
    ]:

        value = np.percentile(
            valid_probability,
            p
        )

        print(
            f"  P{p:02d}: {value:.6f}"
        )

    # --------------------------------------------------------
    # Downsample
    # --------------------------------------------------------

    probability_small = downsample(
        probability,
        MAX_DISPLAY_SIZE
    )

    mask_small = downsample(
        mask,
        MAX_DISPLAY_SIZE
    )

    vv_small = downsample(
        vv,
        MAX_DISPLAY_SIZE
    )

    vh_small = downsample(
        vh,
        MAX_DISPLAY_SIZE
    )

    # --------------------------------------------------------
    # Probability preview
    # --------------------------------------------------------

    probability_display = probability_small.copy()

    probability_display[
        probability_display == 0
    ] = np.nan

    plt.figure(
        figsize=(11, 8)
    )

    plt.imshow(
        probability_display,
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    plt.colorbar(
        label="Flood probability"
    )

    plt.title(
        "Chennai Sentinel-1 Flood Probability"
    )

    plt.axis("off")
    plt.tight_layout()

    probability_output = (
        OUTPUT_DIR
        / "flood_probability_preview.png"
    )

    plt.savefig(
        probability_output,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n"
        f"{probability_output}"
    )

    # --------------------------------------------------------
    # Binary mask preview
    # --------------------------------------------------------

    mask_display = np.full(
        mask_small.shape,
        np.nan,
        dtype=np.float32
    )

    mask_display[
        mask_small == 0
    ] = 0

    mask_display[
        mask_small == 1
    ] = 1

    plt.figure(
        figsize=(11, 8)
    )

    plt.imshow(
        mask_display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Chennai Flood Mask "
        f"(Threshold = {THRESHOLD:.2f})"
    )

    plt.axis("off")
    plt.tight_layout()

    mask_output = (
        OUTPUT_DIR
        / "flood_mask_preview.png"
    )

    plt.savefig(
        mask_output,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n"
        f"{mask_output}"
    )

    # --------------------------------------------------------
    # VV/VH composite
    # --------------------------------------------------------

    vv_display = normalize(
        vv_small
    )

    vh_display = normalize(
        vh_small
    )

    composite = np.zeros(
        (
            vv_display.shape[0],
            vv_display.shape[1],
            3
        ),
        dtype=np.float32
    )

    composite[:, :, 0] = vv_display
    composite[:, :, 1] = vh_display
    composite[:, :, 2] = vv_display

    # --------------------------------------------------------
    # Flood overlay
    # --------------------------------------------------------

    flood_small = (
        mask_small == 1
    )

    valid_small = (
        mask_small != 255
    )

    plt.figure(
        figsize=(11, 8)
    )

    plt.imshow(
        composite
    )

    # Transparent flood overlay.
    overlay = np.zeros(
        (
            flood_small.shape[0],
            flood_small.shape[1],
            4
        ),
        dtype=np.float32
    )

    # Red overlay for predicted flood.
    overlay[:, :, 0] = 1.0
    overlay[:, :, 1] = 0.0
    overlay[:, :, 2] = 0.0

    overlay[:, :, 3] = (
        flood_small
        & valid_small
    ).astype(
        np.float32
    ) * 0.55

    plt.imshow(
        overlay
    )

    plt.title(
        "Chennai Sentinel-1 "
        "Flood Prediction Overlay"
    )

    plt.axis("off")
    plt.tight_layout()

    overlay_output = (
        OUTPUT_DIR
        / "sentinel1_flood_overlay.png"
    )

    plt.savefig(
        overlay_output,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n"
        f"{overlay_output}"
    )

    # --------------------------------------------------------
    # Flood probability > threshold distribution
    # --------------------------------------------------------

    high_confidence = (
        valid_probability >= 0.90
    )

    medium_confidence = (
        (valid_probability >= 0.80)
        & (valid_probability < 0.90)
    )

    print("\n" + "=" * 70)
    print("HIGH-PROBABILITY FLOOD PIXELS")
    print("=" * 70)

    print(
        f">= 0.90 probability : "
        f"{high_confidence.sum():,}"
    )

    print(
        f"0.80–0.90           : "
        f"{medium_confidence.sum():,}"
    )

    print(
        f">= 0.80 total       : "
        f"{(high_confidence | medium_confidence).sum():,}"
    )

    # --------------------------------------------------------
    # Approximate area
    # --------------------------------------------------------

    pixel_area_m2 = 10 * 10

    flood_area_km2 = (
        flood.sum()
        * pixel_area_m2
        / 1_000_000
    )

    print(
        f"\nPredicted flood area "
        f"within valid scene: "
        f"{flood_area_km2:.3f} km²"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "#" * 70)
    print("VALIDATION VISUALS CREATED")
    print("#" * 70)

    print(
        "\nOutput directory:"
        f"\n{OUTPUT_DIR}"
    )

    print("\nGenerated:")

    for path in sorted(
        OUTPUT_DIR.glob("*.png")
    ):

        print(
            f"  - {path.name}"
        )


if __name__ == "__main__":
    main()