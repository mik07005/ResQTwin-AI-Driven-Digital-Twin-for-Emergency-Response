from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy.signal.windows import hann


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))

from models.unet import UNet


INPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
)

CHECKPOINT = (
    ROOT
    / "ai"
    / "satellite"
    / "checkpoints"
    / "positive_weight_5"
    / "best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "gcc"
    / "inference"
)

PROBABILITY_OUTPUT = OUTPUT_DIR / "flood_probability.tif"
MASK_OUTPUT = OUTPUT_DIR / "flood_mask.tif"


# ---------------------------------------------------------
# Locked model configuration
# ---------------------------------------------------------

THRESHOLD = 0.80

TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP

BATCH_SIZE = 4

VV_MEAN = -10.3929
VV_STD = 4.0388

VH_MEAN = -17.2411
VH_STD = 4.7548


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def linear_to_db(x):
    """
    Convert Sentinel-1 sigma0 linear values to dB.
    """
    return 10.0 * np.log10(np.maximum(x, 1e-8))


def make_hann_window(size):
    """
    2D Hann window for overlapping tile blending.
    """
    w = hann(size, sym=False).astype(np.float32)

    # Prevent exact zero weights at the border.
    w = np.maximum(w, 1e-3)

    return np.outer(w, w).astype(np.float32)


def get_positions(length, tile_size, stride):
    """
    Generate tile start positions while ensuring the
    final tile reaches the end of the dimension.
    """
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))

    final_position = length - tile_size

    if positions[-1] != final_position:
        positions.append(final_position)

    return positions


def prepare_tile(vv, vh):
    """
    Convert linear sigma0 → dB and normalize using
    Sen1Floods11 training statistics.
    """

    vv_db = linear_to_db(vv)
    vh_db = linear_to_db(vh)

    vv_norm = (vv_db - VV_MEAN) / VV_STD
    vh_norm = (vh_db - VH_MEAN) / VH_STD

    return np.stack(
        [vv_norm, vh_norm],
        axis=0
    ).astype(np.float32)


# ---------------------------------------------------------
# Main inference
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Chennai GCC Flood Inference ===")
    print()

    print("Input:")
    print(INPUT)

    print("Checkpoint:")
    print(CHECKPOINT)

    print("Threshold:", THRESHOLD)
    print("Tile size:", TILE_SIZE)
    print("Overlap:", OVERLAP)
    print("Batch size:", BATCH_SIZE)
    print()

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    model = UNet(
        in_channels=2,
        out_channels=1
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print("Model loaded.")
    print()

    # -----------------------------------------------------
    # Open input
    # -----------------------------------------------------

    with rasterio.open(INPUT) as src:

        profile = src.profile.copy()

        width = src.width
        height = src.height

        print("Input CRS:", src.crs)
        print("Input size:", width, "x", height)
        print("Input resolution:", src.res)
        print("Input bounds:", src.bounds)
        print()

        # -------------------------------------------------
        # Tile positions
        # -------------------------------------------------

        rows = get_positions(
            height,
            TILE_SIZE,
            STRIDE
        )

        cols = get_positions(
            width,
            TILE_SIZE,
            STRIDE
        )

        total_tiles = len(rows) * len(cols)

        print("Rows:", len(rows))
        print("Columns:", len(cols))
        print("Total tiles:", total_tiles)
        print()

        # -------------------------------------------------
        # Output accumulation
        # -------------------------------------------------

        probability_sum = np.zeros(
            (height, width),
            dtype=np.float32
        )

        weight_sum = np.zeros(
            (height, width),
            dtype=np.float32
        )

        blend_window = make_hann_window(
            TILE_SIZE
        )

        batch_inputs = []
        batch_locations = []

        processed = 0

        def process_batch():

            nonlocal batch_inputs
            nonlocal batch_locations
            nonlocal processed

            if not batch_inputs:
                return

            batch = np.stack(
                batch_inputs,
                axis=0
            )

            tensor = torch.from_numpy(
                batch
            ).to(device)

            with torch.no_grad():

                logits = model(tensor)

                probabilities = torch.sigmoid(
                    logits
                ).squeeze(1)

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
            )

            for probability, (row, col, valid_mask) in zip(
                probabilities,
                batch_locations
            ):

                # The clipped GCC raster may contain
                # NoData inside the bounding rectangle.
                probability = probability.astype(
                    np.float32
                )

                probability_sum[
                    row:row + TILE_SIZE,
                    col:col + TILE_SIZE
                ] += probability * blend_window * valid_mask

                weight_sum[
                    row:row + TILE_SIZE,
                    col:col + TILE_SIZE
                ] += blend_window * valid_mask

                processed += 1

                if processed % 25 == 0 or processed == total_tiles:
                    print(
                        f"Processed {processed}/{total_tiles} tiles"
                    )

            batch_inputs = []
            batch_locations = []

        # -------------------------------------------------
        # Process tiles
        # -------------------------------------------------

        for row in rows:

            for col in cols:

                window = Window(
                    col,
                    row,
                    TILE_SIZE,
                    TILE_SIZE
                )

                # Read both bands.
                data = src.read(
                    [1, 2],
                    window=window,
                    boundless=True,
                    fill_value=0
                ).astype(np.float32)

                vv = data[0]
                vh = data[1]

                # Valid only where both channels contain data.
                valid_mask = (
                    (vv > 0) &
                    (vh > 0)
                ).astype(np.float32)

                # Avoid inference on completely empty tiles.
                if valid_mask.sum() == 0:
                    continue

                # Convert and normalize.
                tile = prepare_tile(
                    vv,
                    vh
                )

                # Explicitly zero invalid pixels after normalization.
                tile[:, valid_mask == 0] = 0.0

                batch_inputs.append(tile)

                batch_locations.append(
                    (
                        row,
                        col,
                        valid_mask
                    )
                )

                if len(batch_inputs) >= BATCH_SIZE:
                    process_batch()

        process_batch()

        # -------------------------------------------------
        # Blend overlapping predictions
        # -------------------------------------------------

        valid_output = weight_sum > 0

        probability = np.zeros(
            (height, width),
            dtype=np.float32
        )

        probability[valid_output] = (
            probability_sum[valid_output]
            /
            weight_sum[valid_output]
        )

        # -------------------------------------------------
        # Binary flood mask
        # -------------------------------------------------

        flood_mask = np.zeros(
            (height, width),
            dtype=np.uint8
        )

        flood_mask[valid_output] = (
            probability[valid_output] >= THRESHOLD
        ).astype(np.uint8)

        # 255 = NoData
        flood_mask[~valid_output] = 255

        # -------------------------------------------------
        # Write probability
        # -------------------------------------------------

        probability_profile = profile.copy()

        probability_profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512
        )

        with rasterio.open(
            PROBABILITY_OUTPUT,
            "w",
            **probability_profile
        ) as dst:

            dst.write(
                probability,
                1
            )

            dst.set_band_description(
                1,
                "Flood probability"
            )

        # -------------------------------------------------
        # Write mask
        # -------------------------------------------------

        mask_profile = profile.copy()

        mask_profile.update(
            dtype="uint8",
            count=1,
            nodata=255,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512
        )

        with rasterio.open(
            MASK_OUTPUT,
            "w",
            **mask_profile
        ) as dst:

            dst.write(
                flood_mask,
                1
            )

            dst.set_band_description(
                1,
                "Flood mask: 0=non-flood, 1=flood, 255=NoData"
            )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    valid_probability = probability[valid_output]

    flood_pixels = (
        flood_mask[valid_output] == 1
    )

    print()
    print("=== Inference Results ===")

    print(
        "Valid pixels:",
        int(valid_output.sum())
    )

    print(
        "Probability min:",
        float(valid_probability.min())
    )

    print(
        "Probability max:",
        float(valid_probability.max())
    )

    print(
        "Probability mean:",
        float(valid_probability.mean())
    )

    print(
        "Probability median:",
        float(np.median(valid_probability))
    )

    print(
        "Flood pixels:",
        int(flood_pixels.sum())
    )

    print(
        "Non-flood pixels:",
        int((~flood_pixels).sum())
    )

    print(
        "Flood fraction: %.4f%%"
        % (
            100.0
            * flood_pixels.sum()
            / valid_output.sum()
        )
    )

    print(
        "Predicted flood area: %.4f km²"
        % (
            flood_pixels.sum()
            * 10.0
            * 10.0
            / 1_000_000.0
        )
    )

    print()
    print("Probability output:")
    print(PROBABILITY_OUTPUT)

    print("Mask output:")
    print(MASK_OUTPUT)

    print()
    print("GCC INFERENCE: PASS")


if __name__ == "__main__":
    main()