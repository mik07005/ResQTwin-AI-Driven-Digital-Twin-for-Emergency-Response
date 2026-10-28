
"""Reusable flood inference for two-band Sentinel-1 VV/VH GeoTIFF scenes."""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from scipy.signal.windows import hann
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.satellite.models.unet import UNet



DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "ai" / "satellite" / "checkpoints"
    / "positive_weight_5" / "best.pt"
)

TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
BATCH_SIZE = 4

VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7548


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run U-Net flood inference on a two-band VV/VH GeoTIFF."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input GeoTIFF: band 1=VV, band 2=VH, linear sigma0.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def positions(length):
    if length <= TILE_SIZE:
        return [0]
    result = list(range(0, length - TILE_SIZE + 1, STRIDE))
    final = length - TILE_SIZE
    if result[-1] != final:
        result.append(final)
    return result


def make_window():
    w = np.maximum(hann(TILE_SIZE, sym=False).astype(np.float32), 1e-3)
    return np.outer(w, w).astype(np.float32)


def prepare_tile(vv, vh, valid):
    # Clip before log10 so invalid zeros never produce infinities.
    vv_db = 10.0 * np.log10(np.maximum(vv, 1e-8))
    vh_db = 10.0 * np.log10(np.maximum(vh, 1e-8))
    tile = np.stack([
        (vv_db - VV_MEAN) / VV_STD,
        (vh_db - VH_MEAN) / VH_STD,
    ]).astype(np.float32)
    tile[:, ~valid] = 0.0
    return tile


def main():
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input raster not found: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = args.output_dir / "flood_probability.tif"
    mask_path = args.output_dir / "flood_mask.tif"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=2, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    print("Device:", device)
    print("Input:", args.input)
    print("Checkpoint:", args.checkpoint)
    print("Threshold:", args.threshold)

    with rasterio.open(args.input) as src:
        if src.count < 2:
            raise ValueError("Input must contain VV in band 1 and VH in band 2.")
        if src.crs is None:
            raise ValueError("Input raster has no CRS; georeferenced output is unsafe.")

        profile = src.profile.copy()
        height, width = src.height, src.width
        probability_sum = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        blend = make_window()

        row_positions = positions(height)
        col_positions = positions(width)
        total_tiles = len(row_positions) * len(col_positions)
        batch_inputs, batch_meta = [], []
        processed = 0

        def process_batch():
            nonlocal processed
            if not batch_inputs:
                return

            tensor = torch.from_numpy(np.stack(batch_inputs)).to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(tensor)).squeeze(1)
            probs = probs.cpu().numpy()

            for prob, (row, col, valid) in zip(probs, batch_meta):
                h = min(TILE_SIZE, height - row)
                w = min(TILE_SIZE, width - col)
                local_blend = blend[:h, :w] * valid[:h, :w]
                probability_sum[row:row+h, col:col+w] += (
                    prob[:h, :w] * local_blend
                )
                weight_sum[row:row+h, col:col+w] += local_blend
                processed += 1

            batch_inputs.clear()
            batch_meta.clear()
            print(f"Processed {processed}/{total_tiles} tiles")

        for row in row_positions:
            for col in col_positions:
                h = min(TILE_SIZE, height - row)
                w = min(TILE_SIZE, width - col)
                window = Window(col, row, w, h)

                data = src.read([1, 2], window=window).astype(np.float32)
                vv, vh = data[0], data[1]

                raster_valid = (
                    (src.read_masks(1, window=window) > 0)
                    & (src.read_masks(2, window=window) > 0)
                )
                valid = (
                    raster_valid
                    & np.isfinite(vv)
                    & np.isfinite(vh)
                    & (vv > 0)
                    & (vh > 0)
                )

                if not valid.any():
                    continue

                tile = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)
                tile[:, :h, :w] = prepare_tile(vv, vh, valid)

                # Edge tiles are padded to the model's fixed input size.
                batch_inputs.append(tile)
                padded_valid = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                padded_valid[:h, :w] = valid.astype(np.float32)
                batch_meta.append((row, col, padded_valid))

                if len(batch_inputs) >= args.batch_size:
                    process_batch()

        process_batch()

        valid_output = weight_sum > 0
        probability = np.full((height, width), np.nan, dtype=np.float32)
        probability[valid_output] = (
            probability_sum[valid_output] / weight_sum[valid_output]
        )

        flood_mask = np.full((height, width), 255, dtype=np.uint8)
        flood_mask[valid_output] = (
            probability[valid_output] >= args.threshold
        ).astype(np.uint8)

        prob_profile = profile.copy()
        prob_profile.update(
            driver="GTiff", count=1, dtype="float32", nodata=np.nan,
            compress="deflate", predictor=3
        )
        with rasterio.open(probability_path, "w", **prob_profile) as dst:
            dst.write(probability, 1)
            dst.set_band_description(1, "Flood probability")

        mask_profile = profile.copy()
        mask_profile.update(
            driver="GTiff", count=1, dtype="uint8", nodata=255,
            compress="deflate"
        )
        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(flood_mask, 1)
            dst.set_band_description(1, "Flood mask: 0=non-flood, 1=flood, 255=NoData")

    n_valid = int(valid_output.sum())
    n_flood = int((flood_mask[valid_output] == 1).sum())
    print("\nInference complete.")
    print("Valid pixels:", n_valid)
    print("Invalid pixels:", int(valid_output.size - n_valid))
    print("Predicted flood area (10 m pixel assumption):",
          f"{n_flood * 100 / 1_000_000:.4f} km²")
    print("Probability output:", probability_path)
    print("Mask output:", mask_path)


if __name__ == "__main__":
    main()
