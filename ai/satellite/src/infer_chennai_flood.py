from pathlib import Path

import numpy as np
import rasterio
import torch

from rasterio.windows import Window

import sys
from pathlib import Path

# Add ai/satellite to Python's import path
SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))

from models.unet import UNet


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

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "positive_weight_5"
    / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "inference"
)

PROBABILITY_PATH = OUTPUT_DIR / "flood_probability.tif"
MASK_PATH = OUTPUT_DIR / "flood_mask.tif"


# ============================================================
# Model / preprocessing configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TILE_SIZE = 512

# 25% overlap between neighbouring tiles.
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP

BATCH_SIZE = 4

# Locked operating point from Sen1Floods11 evaluation.
THRESHOLD = 0.80


# Sen1Floods11 statistics.
#
# IMPORTANT:
# Chennai input is Sigma0 linear.
# We convert it to dB first and then use these statistics.
#
# VV:
# mean = -10.3929
# std  =   4.0388
#
# VH:
# mean = -17.2411
# std  =   4.7540

VV_MEAN = -10.3929
VV_STD = 4.0388

VH_MEAN = -17.2411
VH_STD = 4.7540


# ============================================================
# Utility functions
# ============================================================

def linear_to_db(image):
    """
    Convert linear Sigma0 to dB.

        dB = 10 * log10(Sigma0)

    Invalid / zero pixels remain zero.
    """

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    valid = (
        np.isfinite(image)
        & (image > 0)
    )

    output[valid] = (
        10.0 * np.log10(
            image[valid]
        )
    )

    return output


def normalize_channels(vv_db, vh_db):
    """
    Apply the exact channel-wise normalization used
    for the Sen1Floods11 model.
    """

    vv_norm = (
        vv_db - VV_MEAN
    ) / VV_STD

    vh_norm = (
        vh_db - VH_MEAN
    ) / VH_STD

    return vv_norm, vh_norm


def make_positions(length, tile_size, stride):
    """
    Generate tile start positions while guaranteeing
    that the final portion of the image is covered.
    """

    if length <= tile_size:
        return [0]

    positions = list(
        range(
            0,
            length - tile_size + 1,
            stride
        )
    )

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def create_weight_window(size):
    """
    Create a smooth blending window.

    Overlapping predictions are blended so that tile
    boundaries are less visible.
    """

    # Hann window.
    one_dim = np.hanning(size).astype(
        np.float32
    )

    window = np.outer(
        one_dim,
        one_dim
    )

    # Avoid completely zero weights.
    window = np.maximum(
        window,
        1e-3
    )

    return window


def load_model(checkpoint_path):
    """
    Load the trained PW5 U-Net checkpoint.
    """

    print("\n" + "=" * 70)
    print("LOADING MODEL")
    print("=" * 70)

    print(f"Checkpoint : {checkpoint_path}")
    print(f"Device     : {DEVICE}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Construct model
    # --------------------------------------------------------

    model = UNet(
        in_channels=2,
        out_channels=1
    )

    # --------------------------------------------------------
    # Handle common checkpoint formats
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        else:
            # Some training scripts save the state
            # dictionary directly.
            state_dict = checkpoint

    else:
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    # Remove possible DataParallel prefix.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        print("\nWARNING: Missing model keys:")
        for key in missing:
            print(f"  {key}")

    if unexpected:
        print("\nWARNING: Unexpected checkpoint keys:")
        for key in unexpected:
            print(f"  {key}")

    model.to(DEVICE)
    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{parameter_count:,}"
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# Main inference
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("RESQTWIN — CHENNAI SENTINEL-1 FLOOD INFERENCE")
    print("#" * 70)

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input Sentinel-1 stack not found:\n"
            f"{INPUT_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(
        CHECKPOINT_PATH
    )

    # --------------------------------------------------------
    # Open Sentinel-1 stack
    # --------------------------------------------------------

    with rasterio.open(
        INPUT_PATH
    ) as src:

        width = src.width
        height = src.height

        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

        print("\n" + "=" * 70)
        print("INPUT SENTINEL-1 STACK")
        print("=" * 70)

        print(f"CRS        : {crs}")
        print(
            f"Size       : "
            f"{width} x {height}"
        )

        print(
            f"Resolution : "
            f"{src.res}"
        )

        print(
            f"Bounds     : "
            f"{src.bounds}"
        )

        # ----------------------------------------------------
        # Allocate output arrays
        # ----------------------------------------------------

        probability_sum = np.zeros(
            (height, width),
            dtype=np.float32
        )

        weight_sum = np.zeros(
            (height, width),
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Tile positions
        # ----------------------------------------------------

        row_positions = make_positions(
            height,
            TILE_SIZE,
            STRIDE
        )

        col_positions = make_positions(
            width,
            TILE_SIZE,
            STRIDE
        )

        total_tiles = (
            len(row_positions)
            * len(col_positions)
        )

        print("\n" + "=" * 70)
        print("TILING")
        print("=" * 70)

        print(f"Tile size  : {TILE_SIZE}")
        print(f"Overlap    : {OVERLAP}")
        print(f"Stride     : {STRIDE}")
        print(
            f"Rows       : "
            f"{len(row_positions)}"
        )
        print(
            f"Columns    : "
            f"{len(col_positions)}"
        )
        print(
            f"Total tiles: "
            f"{total_tiles}"
        )

        # ----------------------------------------------------
        # Blending window
        # ----------------------------------------------------

        weight_window = create_weight_window(
            TILE_SIZE
        )

        # ----------------------------------------------------
        # Batch storage
        # ----------------------------------------------------

        batch_tensors = []
        batch_locations = []

        processed_tiles = 0

        # ----------------------------------------------------
        # Process tiles
        # ----------------------------------------------------

        for row in row_positions:

            for col in col_positions:

                window = Window(
                    col,
                    row,
                    min(
                        TILE_SIZE,
                        width - col
                    ),
                    min(
                        TILE_SIZE,
                        height - row
                    )
                )

                # --------------------------------------------
                # Read VV + VH
                # --------------------------------------------

                data = src.read(
                    indexes=[1, 2],
                    window=window
                ).astype(
                    np.float32
                )

                actual_height = data.shape[1]
                actual_width = data.shape[2]

                # --------------------------------------------
                # Convert linear Sigma0 → dB
                # --------------------------------------------

                vv_linear = data[0]
                vh_linear = data[1]

                vv_db = linear_to_db(
                    vv_linear
                )

                vh_db = linear_to_db(
                    vh_linear
                )

                # --------------------------------------------
                # Normalize
                # --------------------------------------------

                vv_norm, vh_norm = (
                    normalize_channels(
                        vv_db,
                        vh_db
                    )
                )

                # --------------------------------------------
                # Stack channels
                # --------------------------------------------

                tile = np.stack(
                    [
                        vv_norm,
                        vh_norm
                    ],
                    axis=0
                ).astype(
                    np.float32
                )

                # --------------------------------------------
                # Pad border tiles
                # --------------------------------------------

                padded = np.zeros(
                    (
                        2,
                        TILE_SIZE,
                        TILE_SIZE
                    ),
                    dtype=np.float32
                )

                padded[
                    :,
                    :actual_height,
                    :actual_width
                ] = tile

                tensor = torch.from_numpy(
                    padded
                )

                batch_tensors.append(
                    tensor
                )

                batch_locations.append(
                    (
                        row,
                        col,
                        actual_height,
                        actual_width
                    )
                )

                # --------------------------------------------
                # Run batch when full
                # --------------------------------------------

                if len(batch_tensors) >= BATCH_SIZE:

                    batch = torch.stack(
                        batch_tensors
                    ).to(
                        DEVICE,
                        non_blocking=True
                    )

                    with torch.no_grad():

                        logits = model(
                            batch
                        )

                        probabilities = torch.sigmoid(
                            logits
                        )

                    probabilities = (
                        probabilities
                        .squeeze(1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    # ----------------------------------------
                    # Accumulate predictions
                    # ----------------------------------------

                    for probability, location in zip(
                        probabilities,
                        batch_locations
                    ):

                        (
                            r,
                            c,
                            h,
                            w
                        ) = location

                        probability = (
                            probability[:h, :w]
                        )

                        weight = (
                            weight_window[:h, :w]
                        )

                        probability_sum[
                            r:r+h,
                            c:c+w
                        ] += (
                            probability
                            * weight
                        )

                        weight_sum[
                            r:r+h,
                            c:c+w
                        ] += weight

                    processed_tiles += (
                        len(batch_tensors)
                    )

                    print(
                        f"Processed "
                        f"{processed_tiles}/"
                        f"{total_tiles} tiles"
                    )

                    batch_tensors = []
                    batch_locations = []

        # ----------------------------------------------------
        # Process final partial batch
        # ----------------------------------------------------

        if batch_tensors:

            batch = torch.stack(
                batch_tensors
            ).to(
                DEVICE,
                non_blocking=True
            )

            with torch.no_grad():

                logits = model(
                    batch
                )

                probabilities = torch.sigmoid(
                    logits
                )

            probabilities = (
                probabilities
                .squeeze(1)
                .detach()
                .cpu()
                .numpy()
            )

            for probability, location in zip(
                probabilities,
                batch_locations
            ):

                (
                    r,
                    c,
                    h,
                    w
                ) = location

                probability = (
                    probability[:h, :w]
                )

                weight = (
                    weight_window[:h, :w]
                )

                probability_sum[
                    r:r+h,
                    c:c+w
                ] += (
                    probability
                    * weight
                )

                weight_sum[
                    r:r+h,
                    c:c+w
                ] += weight

            processed_tiles += (
                len(batch_tensors)
            )

            print(
                f"Processed "
                f"{processed_tiles}/"
                f"{total_tiles} tiles"
            )

    # ========================================================
    # Build final probability map
    # ========================================================

    print("\n" + "=" * 70)
    print("BUILDING FLOOD PROBABILITY MAP")
    print("=" * 70)

    probability_map = np.zeros(
        (height, width),
        dtype=np.float32
    )

    valid_output = weight_sum > 0

    probability_map[
        valid_output
    ] = (
        probability_sum[valid_output]
        / weight_sum[valid_output]
    )

    # --------------------------------------------------------
    # Valid Sentinel-1 footprint
    # --------------------------------------------------------

    # Read original valid mask.
    with rasterio.open(
        INPUT_PATH
    ) as src:

        vv = src.read(
            1
        )

        vh = src.read(
            2
        )

    valid_sar = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & (vv > 0)
        & (vh > 0)
    )

    # Outside SAR footprint should not be treated
    # as "non-flood".
    probability_map[
        ~valid_sar
    ] = 0.0

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    valid_prob = probability_map[
        valid_sar
    ]

    print(
        f"Valid pixels       : "
        f"{valid_prob.size:,}"
    )

    print(
        f"Probability min    : "
        f"{valid_prob.min():.6f}"
    )

    print(
        f"Probability max    : "
        f"{valid_prob.max():.6f}"
    )

    print(
        f"Probability mean   : "
        f"{valid_prob.mean():.6f}"
    )

    print(
        f"Probability median : "
        f"{np.median(valid_prob):.6f}"
    )

    # ========================================================
    # Save probability GeoTIFF
    # ========================================================

    probability_profile = profile.copy()

    probability_profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=0.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(
        PROBABILITY_PATH,
        "w",
        **probability_profile
    ) as dst:

        output = probability_map.copy()

        dst.write(
            output,
            1
        )

        dst.set_band_description(
            1,
            "Flood Probability"
        )

        dst.update_tags(
            model="U-Net",
            model_variant="positive_weight_5",
            input="Sentinel-1 VV+VH",
            threshold=str(THRESHOLD),
            normalization="Sen1Floods11 dB statistics",
            vv_mean=str(VV_MEAN),
            vv_std=str(VV_STD),
            vh_mean=str(VH_MEAN),
            vh_std=str(VH_STD),
            acquisition="2021-11-16",
            city="Chennai",
            projection="EPSG:32644",
            resolution="10m"
        )

    print(
        f"\nSaved probability map:\n"
        f"{PROBABILITY_PATH}"
    )

    # ========================================================
    # Binary flood mask
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING FLOOD MASK")
    print("=" * 70)

    flood_mask = np.full(
        (height, width),
        255,
        dtype=np.uint8
    )

    # Valid SAR pixels:
    # 0 = non-flood
    # 1 = flood
    flood_mask[
        valid_sar
    ] = (
        probability_map[
            valid_sar
        ] >= THRESHOLD
    ).astype(
        np.uint8
    )

    valid_mask = flood_mask[
        valid_sar
    ]

    flood_pixels = (
        valid_mask == 1
    ).sum()

    non_flood_pixels = (
        valid_mask == 0
    ).sum()

    print(
        f"Threshold          : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Flood pixels       : "
        f"{flood_pixels:,}"
    )

    print(
        f"Non-flood pixels   : "
        f"{non_flood_pixels:,}"
    )

    print(
        f"Flood area fraction: "
        f"{100.0 * flood_pixels / valid_mask.size:.4f}%"
    )

    # ========================================================
    # Save binary flood mask
    # ========================================================

    mask_profile = profile.copy()

    mask_profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        nodata=255,
        compress="deflate",
        tiled=True,
        BIGTIFF="IF_SAFER"
    )

    with rasterio.open(
        MASK_PATH,
        "w",
        **mask_profile
    ) as dst:

        dst.write(
            flood_mask,
            1
        )

        dst.set_band_description(
            1,
            "Flood Mask (0=Non-Flood, 1=Flood)"
        )

        dst.update_tags(
            model="U-Net",
            model_variant="positive_weight_5",
            threshold=str(THRESHOLD),
            input="Sentinel-1 VV+VH",
            acquisition="2021-11-16",
            city="Chennai",
            projection="EPSG:32644",
            resolution="10m",
            nodata_value="255"
        )

    print(
        f"\nSaved flood mask:\n"
        f"{MASK_PATH}"
    )

    # ========================================================
    # Final verification
    # ========================================================

    print("\n" + "#" * 70)
    print("CHENNAI FLOOD INFERENCE COMPLETE")
    print("#" * 70)

    print("\nOutputs:")

    print(
        f"1. Probability:\n"
        f"   {PROBABILITY_PATH}"
    )

    print(
        f"\n2. Binary mask:\n"
        f"   {MASK_PATH}"
    )

    print("\nMask convention:")
    print("  0   = non-flood")
    print("  1   = flood")
    print("  255 = NoData / outside SAR footprint")

    print("\nGeospatial properties:")
    print("  CRS        = EPSG:32644")
    print("  Resolution = 10 m")
    print("  Grid       = same as Sentinel-1 input")

    print("\nThese files are ready for the downstream")
    print("Digital Twin / geospatial impact pipeline.")


if __name__ == "__main__":
    main()