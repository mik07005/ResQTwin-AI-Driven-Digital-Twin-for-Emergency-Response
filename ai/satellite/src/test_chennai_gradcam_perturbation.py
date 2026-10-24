
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch

SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))
from models.unet import UNet

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"
INPUT = BASE / "sentinel1" / "gcc" / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
MASK = BASE / "sentinel1" / "gcc" / "inference" / "flood_mask.tif"
CAM = BASE / "explainability" / "gradcam_day2_flood.tif"
CHECKPOINT = ROOT / "ai" / "satellite" / "checkpoints" / "positive_weight_5" / "best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512
ROW, COL = 0, 1635
VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7544


def linear_to_db(x):
    out = np.zeros_like(x, dtype=np.float32)
    valid = np.isfinite(x) & (x > 0)
    out[valid] = 10.0 * np.log10(x[valid])
    return out


def preprocess(data):
    vv = (linear_to_db(data[0]) - VV_MEAN) / VV_STD
    vh = (linear_to_db(data[1]) - VH_MEAN) / VH_STD
    out = np.zeros((2, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    h, w = data.shape[1:]
    out[:, :h, :w] = np.stack([vv, vh]).astype(np.float32)
    return out


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = UNet(in_channels=2, out_channels=1)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()
    return model


def target_score(model, x, target):
    with torch.no_grad():
        logits = model(x)[0, 0]
        return logits[target].mean().item()


def main():
    for p in (INPUT, MASK, CAM, CHECKPOINT):
        if not p.exists():
            raise FileNotFoundError(p)

    with rasterio.open(INPUT) as src:
        sar = src.read([1, 2]).astype(np.float32)

    with rasterio.open(MASK) as src:
        mask = src.read(1)

    with rasterio.open(CAM) as src:
        cam = src.read(1)
        nodata = src.nodata

    tile = sar[:, ROW:ROW + TILE_SIZE, COL:COL + TILE_SIZE]
    with rasterio.open(MASK) as src:
        local_mask = src.read(
            1,
            window=(
                (ROW, ROW + tile.shape[1]),
                (COL, COL + tile.shape[2]),
            ),
        )

    local_cam = cam[:tile.shape[1], :tile.shape[2]]

    valid = (
        np.isfinite(tile[0]) & np.isfinite(tile[1])
        & (tile[0] > 0) & (tile[1] > 0)
    )
    valid &= np.isfinite(local_cam)
    if nodata is not None:
        valid &= local_cam != nodata

    flood = valid & (local_mask == 1)
    if flood.sum() < 100:
        raise RuntimeError("Too few valid flood pixels for this test.")

    # Use a fixed-size set of pixels from the top and bottom of
    # the flood-target attribution distribution.
    # Calculate thresholds only over valid predicted flood pixels.
    flood_cam = local_cam[flood]

    high_cut = np.quantile(flood_cam, 0.90)
    low_cut = np.quantile(flood_cam, 0.10)

    high = flood & (local_cam >= high_cut)
    low = flood & (local_cam <= low_cut)

    print(f"Eligible flood pixels: {flood.sum():,}")
    print(f"High-attribution candidates: {high.sum():,}")
    print(f"Low-attribution candidates: {low.sum():,}")
    # Equalize the number of perturbed pixels for a fairer comparison.
    rng = np.random.default_rng(42)
    high_idx = np.flatnonzero(high)
    low_idx = np.flatnonzero(low)
    n = min(len(high_idx), len(low_idx), 3000)

    if n < 100:
        raise RuntimeError("Too few high/low attribution pixels.")

    high_idx = rng.choice(high_idx, n, replace=False)
    low_idx = rng.choice(low_idx, n, replace=False)

    model = load_model()
    x_np = preprocess(tile)
    x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

    target = torch.zeros((TILE_SIZE, TILE_SIZE), dtype=torch.bool, device=DEVICE)
    target[:tile.shape[1], :tile.shape[2]] = torch.from_numpy(flood).to(DEVICE)

    original = target_score(model, x, target)

    # Perturb both VV and VH at selected pixels using the same
    # neutral value (normalized zero) for this diagnostic.
    x_high = x.clone()
    x_low = x.clone()

    h, w = tile.shape[1:]
    high_local = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
    low_local = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)

    high_local[:h, :w].flat[high_idx] = True
    low_local[:h, :w].flat[low_idx] = True

    high_t = torch.from_numpy(high_local).to(DEVICE)
    low_t = torch.from_numpy(low_local).to(DEVICE)

    x_high[0, :, high_t] = 0.0
    x_low[0, :, low_t] = 0.0

    high_score = target_score(model, x_high, target)
    low_score = target_score(model, x_low, target)

    print("\nGrad-CAM perturbation sanity test")
    print("--------------------------------")
    print(f"Device: {DEVICE}")
    print(f"Target: mean flood logit over existing predicted flood pixels")
    print(f"Original target score: {original:.6f}")
    print(f"After high-attribution perturbation: {high_score:.6f}")
    print(f"After low-attribution perturbation:  {low_score:.6f}")
    print(f"High-attribution score change: {high_score - original:+.6f}")
    print(f"Low-attribution score change:  {low_score - original:+.6f}")
    print(f"Pixels perturbed in each test: {n}")

    high_change = abs(high_score - original)
    low_change = abs(low_score - original)

    if high_change > low_change:
        print("Observation: high-attribution perturbation changed the score more.")
    elif high_change < low_change:
        print("Observation: low-attribution perturbation changed the score more.")
    else:
        print("Observation: both perturbations changed the score equally.")

    print(
        "\nCaution: zeroing normalized inputs can create unrealistic SAR patterns. "
        "This is a preliminary sanity check, not proof of causal explanation."
    )


if __name__ == "__main__":
    main()
