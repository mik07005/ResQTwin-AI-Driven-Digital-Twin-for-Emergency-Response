
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rasterio.windows import Window

# ------------------------------------------------------------
# Paths and configuration
# ------------------------------------------------------------

SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))

from models.unet import UNet

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai"
    / "sentinel1" / "gcc"
    / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
)

PROBABILITY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai"
    / "sentinel1" / "gcc" / "inference"
    / "flood_probability.tif"
)

MASK_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chennai"
    / "sentinel1" / "gcc" / "inference"
    / "flood_mask.tif"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "ai" / "satellite" / "checkpoints"
    / "positive_weight_5" / "best.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "chennai"
    / "explainability"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
THRESHOLD = 0.80

VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7544


# ------------------------------------------------------------
# Preprocessing: match existing inference pipeline
# ------------------------------------------------------------

def linear_to_db(image):
    output = np.zeros_like(image, dtype=np.float32)
    valid = np.isfinite(image) & (image > 0)
    output[valid] = 10.0 * np.log10(image[valid])
    return output


def preprocess(data):
    vv_db = linear_to_db(data[0])
    vh_db = linear_to_db(data[1])

    vv_norm = (vv_db - VV_MEAN) / VV_STD
    vh_norm = (vh_db - VH_MEAN) / VH_STD

    tile = np.stack([vv_norm, vh_norm]).astype(np.float32)

    padded = np.zeros(
        (2, TILE_SIZE, TILE_SIZE), dtype=np.float32
    )
    h = min(tile.shape[1], TILE_SIZE)
    w = min(tile.shape[2], TILE_SIZE)
    padded[:, :h, :w] = tile[:, :h, :w]

    return padded


def load_model():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = UNet(in_channels=2, out_channels=1)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ------------------------------------------------------------
# Grad-CAM for a selected target region
# ------------------------------------------------------------

def calculate_gradcam(model, model_input, target_region, label):
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Same decoder layer as the Day 1 experiment.
    target_layer = model.dec1

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        logits = model(model_input)
        target = torch.from_numpy(target_region).to(DEVICE)

        if not target.any():
            raise ValueError(f"No valid pixels in {label} target region.")

        # Flood target: increase flood logits.
        # Non-flood target: decrease flood logits by explaining -logit.
        signed_logits = logits[0, 0] if label == "flood" else -logits[0, 0]
        score = signed_logits[target].mean()
        score.backward()

        features = activations["value"].detach()
        grads = gradients["value"].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * features).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()

        if not np.isfinite(cam).all() or cam.max() <= 0:
            raise RuntimeError(
                f"{label} Grad-CAM has no positive finite attribution. "
                "Do not interpret this map; inspect the target or layer."
            )

        cam /= cam.max()
        return cam

    finally:
        handle_f.remove()
        handle_b.remove()


def display_scale(image, valid):
    values = image[valid & np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lo, hi = np.percentile(values, [2, 98])
    scaled = np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)
    scaled[~valid] = np.nan
    return scaled


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    for path in (
        INPUT_PATH, PROBABILITY_PATH, MASK_PATH, CHECKPOINT_PATH
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading trained U-Net...")
    model = load_model()

    with rasterio.open(INPUT_PATH) as src:
        sar = src.read([1, 2]).astype(np.float32)

    with rasterio.open(PROBABILITY_PATH) as src:
        probability = src.read(1)

    with rasterio.open(MASK_PATH) as src:
        flood_mask = src.read(1)

    if sar.shape[1:] != probability.shape or probability.shape != flood_mask.shape:
        raise ValueError("Input, probability and mask dimensions do not match.")

    valid_sar = (
        np.isfinite(sar[0]) & np.isfinite(sar[1])
        & (sar[0] > 0) & (sar[1] > 0)
    )

    # Select a tile containing both predicted classes.
    height, width = valid_sar.shape
    best = None
    best_count = 0

    row_positions = list(range(0, max(1, height - TILE_SIZE + 1), STRIDE))
    col_positions = list(range(0, max(1, width - TILE_SIZE + 1), STRIDE))

    if row_positions[-1] != max(0, height - TILE_SIZE):
        row_positions.append(max(0, height - TILE_SIZE))
    if col_positions[-1] != max(0, width - TILE_SIZE):
        col_positions.append(max(0, width - TILE_SIZE))

    for row in sorted(set(row_positions)):
        for col in sorted(set(col_positions)):
            h = min(TILE_SIZE, height - row)
            w = min(TILE_SIZE, width - col)

            local_valid = valid_sar[row:row+h, col:col+w]
            local_mask = flood_mask[row:row+h, col:col+w]

            flood_count = np.count_nonzero(local_valid & (local_mask == 1))
            nonflood_count = np.count_nonzero(local_valid & (local_mask == 0))

            # Prefer tiles with both classes; maximize the smaller class.
            if flood_count > 0 and nonflood_count > 0:
                score = min(flood_count, nonflood_count)
                if score > best_count:
                    best_count = score
                    best = (row, col, h, w)

    if best is None:
        raise RuntimeError(
            "Could not find a tile with both predicted flood and "
            "non-flood pixels. Check the mask and valid SAR footprint."
        )

    row, col, h, w = best
    print(f"Selected tile: row={row}, col={col}, size={h}x{w}")

    tile_data = sar[:, row:row+h, col:col+w]
    model_input = torch.from_numpy(
        preprocess(tile_data)
    ).unsqueeze(0).to(DEVICE)

    local_valid = valid_sar[row:row+h, col:col+w]
    local_mask = flood_mask[row:row+h, col:col+w]

    flood_target = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    nonflood_target = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)

    flood_target[:h, :w] = local_valid & (local_mask == 1)
    nonflood_target[:h, :w] = local_valid & (local_mask == 0)

    print(f"Flood target pixels: {flood_target.sum():,}")
    print(f"Non-flood target pixels: {nonflood_target.sum():,}")

    # Compute the two maps independently using the same model and input.
    print("Calculating flood-target Grad-CAM...")
    flood_cam_full = calculate_gradcam(
        model, model_input, flood_target, "flood"
    )

    print("Calculating non-flood-target Grad-CAM...")
    nonflood_cam_full = calculate_gradcam(
        model, model_input, nonflood_target, "non-flood"
    )

    flood_cam = flood_cam_full[:h, :w]
    nonflood_cam = nonflood_cam_full[:h, :w]

    # Exclude invalid pixels from both maps.
    flood_cam[~local_valid] = np.nan
    nonflood_cam[~local_valid] = np.nan

    vv_db = linear_to_db(tile_data[0])
    vv_display = display_scale(vv_db, local_valid)

    local_probability = probability[row:row+h, col:col+w].astype(np.float32)
    local_probability[~local_valid] = np.nan

    mask_display = local_mask.astype(np.float32)
    mask_display[~local_valid] = np.nan

    # Save separate attribution rasters with geospatial metadata.
    with rasterio.open(INPUT_PATH) as src:
        profile = src.profile.copy()
        transform = src.window_transform(Window(col, row, w, h))

    profile.update(
        driver="GTiff",
        count=1,
        height=h,
        width=w,
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
    )

    for name, cam in [
        ("gradcam_day2_flood.tif", flood_cam),
        ("gradcam_day2_nonflood.tif", nonflood_cam),
    ]:
        output = np.where(np.isfinite(cam), cam, -9999.0).astype(np.float32)
        with rasterio.open(OUTPUT_DIR / name, "w", **profile) as dst:
            dst.write(output, 1)
            dst.set_band_description(1, "Normalized Grad-CAM attribution")
            dst.update_tags(
                method="Grad-CAM",
                target_layer="dec1",
                note="Relative attribution; not ground truth or proof of causality.",
            )

    # Comparison figure: identical spatial tile, separate target maps.
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(vv_display, cmap="gray")
    axes[0, 0].set_title("Sentinel-1 VV (dB)")

    axes[0, 1].imshow(local_probability, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Existing flood probability")

    axes[0, 2].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Existing flood mask (threshold 0.80)")

    axes[1, 0].imshow(vv_display, cmap="gray")
    heat1 = axes[1, 0].imshow(flood_cam, cmap="jet", alpha=0.60, vmin=0, vmax=1)
    axes[1, 0].set_title("Grad-CAM: flood target")
    fig.colorbar(heat1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(vv_display, cmap="gray")
    heat2 = axes[1, 1].imshow(nonflood_cam, cmap="jet", alpha=0.60, vmin=0, vmax=1)
    axes[1, 1].set_title("Grad-CAM: non-flood target")
    fig.colorbar(heat2, ax=axes[1, 1], fraction=0.046, pad=0.04)

    difference = flood_cam - nonflood_cam
    axes[1, 2].imshow(difference, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 2].set_title("Flood CAM − non-flood CAM")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"ResQTwin Day 2 Grad-CAM comparison | row={row}, col={col}",
        fontsize=14,
    )
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "gradcam_day2_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nDay 2 attribution comparison completed.")
    print(f"Comparison figure: {figure_path}")
    print(f"Flood CAM raster: {OUTPUT_DIR / 'gradcam_day2_flood.tif'}")
    print(f"Non-flood CAM raster: {OUTPUT_DIR / 'gradcam_day2_nonflood.tif'}")
    print(
        "\nCaution: these maps show relative Grad-CAM attribution for "
        "selected predicted classes. They do not establish ground truth, "
        "causality, or explanation reliability."
    )


if __name__ == "__main__":
    main()
