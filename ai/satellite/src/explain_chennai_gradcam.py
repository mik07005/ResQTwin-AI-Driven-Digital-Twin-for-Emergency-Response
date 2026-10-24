
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rasterio.windows import Window

# ------------------------------------------------------------
# Paths and imports
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
THRESHOLD = 0.80

VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7544

# ------------------------------------------------------------
# Preprocessing: match the existing inference pipeline
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

    # Match inference padding for partial edge tiles.
    padded = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)
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

    # Strict loading prevents silently explaining the wrong model.
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


# ------------------------------------------------------------
# Grad-CAM for a selected flood-prediction region
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

    # Read the current prediction mask and input raster.
    with rasterio.open(INPUT_PATH) as src:
        sar = src.read([1, 2]).astype(np.float32)
        profile = src.profile.copy()

    with rasterio.open(PROBABILITY_PATH) as src:
        probability = src.read(1)

    with rasterio.open(MASK_PATH) as src:
        flood_mask = src.read(1)

    valid_sar = (
        np.isfinite(sar[0]) & np.isfinite(sar[1])
        & (sar[0] > 0) & (sar[1] > 0)
    )

    # Choose the tile with the most predicted flood pixels.
    # This is a diagnostic example, not a claim that it is
    # the most important or most representative location.
    best = None
    best_count = 0

    height, width = valid_sar.shape

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

            count = int(np.count_nonzero(local_valid & (local_mask == 1)))

            if count > best_count:
                best_count = count
                best = (row, col, h, w)

    if best is None or best_count == 0:
        raise RuntimeError(
            "No tile contains predicted flood pixels. "
            "Check the flood mask and valid SAR footprint."
        )

    row, col, h, w = best
    print(f"Selected tile: row={row}, col={col}, size={h}x{w}")
    print(f"Predicted flood pixels in tile: {best_count:,}")

    tile_data = sar[:, row:row+h, col:col+w]
    model_input = torch.from_numpy(preprocess(tile_data)).unsqueeze(0).to(DEVICE)

    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    # Final decoder block before the 1x1 output convolution.
    target_layer = model.dec1
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(model_input)

        # Select predicted flood pixels inside the tile using the
        # existing saved mask, excluding invalid SAR pixels.
        local_mask = flood_mask[row:row+h, col:col+w]
        local_valid = valid_sar[row:row+h, col:col+w]

        target_region = (
            (local_mask == 1) & local_valid
        )

        if not target_region.any():
            raise RuntimeError("Selected tile has no valid predicted flood region.")

        target = torch.zeros(
            (TILE_SIZE, TILE_SIZE), dtype=torch.bool, device=DEVICE
        )
        target[:h, :w] = torch.from_numpy(target_region).to(DEVICE)

        # Average raw logits over the selected predicted-flood region.
        score = logits[0, 0][target].mean()
        score.backward()

        feature_maps = activations["value"].detach()
        grads = gradients["value"].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(TILE_SIZE, TILE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.detach().cpu().numpy()
        cam = cam[:h, :w]

        if not np.isfinite(cam).all() or cam.max() <= 0:
            raise RuntimeError(
                "Grad-CAM produced no positive finite attribution. "
                "We should test another target layer or method."
            )

        cam = cam / cam.max()

    finally:
        handle_f.remove()
        handle_b.remove()

    # Build an interpretable VV dB background.
    vv_db = linear_to_db(tile_data[0])
    vh_db = linear_to_db(tile_data[1])

    def scale_for_display(image):
        finite = np.isfinite(image)
        if not finite.any():
            return np.zeros_like(image, dtype=np.float32)
        lo, hi = np.percentile(image[finite], [2, 98])
        return np.clip((image - lo) / (hi - lo + 1e-8), 0, 1)

    vv_display = scale_for_display(vv_db)
    pred_display = probability[row:row+h, col:col+w]
    mask_display = flood_mask[row:row+h, col:col+w]

    # Save the attribution raster with the correct spatial metadata.
    cam_profile = profile.copy()
    cam_profile.update(
        driver="GTiff",
        count=1,
        height=h,
        width=w,
        transform=rasterio.windows.transform(
            Window(col, row, w, h), profile_transform(profile)
        ),
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
    )

    cam_to_save = cam.astype(np.float32)
    cam_to_save[~valid_sar[row:row+h, col:col+w]] = -9999.0

    cam_path = OUTPUT_DIR / "gradcam_tile.tif"
    with rasterio.open(cam_path, "w", **cam_profile) as dst:
        dst.write(cam_to_save, 1)
        dst.set_band_description(1, "Normalized Grad-CAM attribution")
        dst.update_tags(
            method="Grad-CAM",
            target_layer="dec1",
            target="mean flood logit over existing predicted flood region",
            note="Attribution indicates model sensitivity, not ground truth.",
        )

    # Save a four-panel diagnostic figure.
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    axes[0, 0].imshow(vv_display, cmap="gray")
    axes[0, 0].set_title("Sentinel-1 VV (dB, display-scaled)")

    axes[0, 1].imshow(pred_display, cmap="viridis", vmin=0, vmax=1)
    axes[0, 1].set_title("Existing model flood scores")

    axes[1, 0].imshow(mask_display, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Existing flood mask (threshold 0.80)")

    axes[1, 1].imshow(vv_display, cmap="gray")
    heat = axes[1, 1].imshow(cam, cmap="jet", alpha=0.55, vmin=0, vmax=1)
    axes[1, 1].set_title("Grad-CAM overlay (initial experiment)")
    fig.colorbar(heat, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Relative attribution")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        f"ResQTwin Grad-CAM | tile row={row}, col={col}",
        fontsize=14,
    )
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "gradcam_diagnostic.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("\nGrad-CAM experiment completed.")
    print(f"Attribution raster: {cam_path}")
    print(f"Diagnostic figure : {figure_path}")
    print(f"CAM min/max       : {cam.min():.4f} / {cam.max():.4f}")
    print(f"CAM mean          : {cam.mean():.4f}")
    print(
        "\nInterpretation caution: this heatmap shows relative model "
        "attribution for the selected predicted region. It does not "
        "verify real-world flooding or establish causation."
    )


def profile_transform(profile):
    # Helper retained for clarity if called from older Rasterio profiles.
    return profile["transform"]


if __name__ == "__main__":
    main()
