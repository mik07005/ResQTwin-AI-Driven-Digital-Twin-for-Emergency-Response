
from pathlib import Path
import sys
import csv

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter
from rasterio.windows import Window

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))

from models.unet import UNet

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"

INPUT = (
    BASE / "sentinel1" / "gcc"
    / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
)
MASK = (
    BASE / "sentinel1" / "gcc" / "inference"
    / "flood_mask.tif"
)
CHECKPOINT = (
    ROOT / "ai" / "satellite" / "checkpoints"
    / "positive_weight_5" / "best.pt"
)

OUT = BASE / "explainability" / "day5_v2"
CSV_PATH = OUT / "gradcam_day5_v2_robustness_summary.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TILE = 512
VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7544

MIN_TARGET_PIXELS = 500
MAX_PERTURB = 3000
SEED = 42
MEDIAN_SIZE = 5


# ============================================================
# PREPROCESSING
# ============================================================

def linear_to_db(x):
    result = np.zeros_like(x, dtype=np.float32)
    valid = np.isfinite(x) & (x > 0)
    result[valid] = 10.0 * np.log10(x[valid])
    return result


def preprocess(data):
    vv = (linear_to_db(data[0]) - VV_MEAN) / VV_STD
    vh = (linear_to_db(data[1]) - VH_MEAN) / VH_STD

    result = np.zeros((2, TILE, TILE), dtype=np.float32)
    h, w = data.shape[1:]

    result[:, :h, :w] = np.stack([vv, vh]).astype(np.float32)
    return result


def median_replacement_input(x_np, h, w, size=MEDIAN_SIZE):
    """
    Create local-median replacements for both normalized SAR channels.
    Returns an array with shape (2, TILE, TILE).

    Median filtering is applied to the tile, including padded regions.
    Therefore, replacements near tile edges should be interpreted
    cautiously.
    """
    result = x_np.copy()

    for channel in range(2):
        result[channel] = median_filter(
            x_np[channel],
            size=size,
            mode="nearest"
        )

    return result


# ============================================================
# MODEL
# ============================================================

def load_model():
    ckpt = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = UNet(in_channels=2, out_channels=1)

    state = ckpt.get(
        "model_state_dict",
        ckpt.get("state_dict", ckpt)
    )

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }

    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


# ============================================================
# GRAD-CAM
# ============================================================

def gradcam(model, x, target):
    activations = {}
    gradients = {}

    layer = model.dec1

    def forward_hook(module, inputs, output):
        activations["value"] = output

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    hf = layer.register_forward_hook(forward_hook)
    hb = layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)

        logits = model(x)
        target_t = torch.from_numpy(target).to(DEVICE)

        if not target_t.any():
            raise ValueError("Empty flood target.")

        logits[0, 0][target_t].mean().backward()

        features = activations["value"].detach()
        grads = gradients["value"].detach()

        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = torch.relu(
            (weights * features).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=(TILE, TILE),
            mode="bilinear",
            align_corners=False
        )[0, 0].detach().cpu().numpy()

        if not np.isfinite(cam).all() or cam.max() <= 0:
            raise RuntimeError(
                "Grad-CAM has no positive finite attribution."
            )

        return (cam / cam.max()).astype(np.float32)

    finally:
        hf.remove()
        hb.remove()


def target_score(model, x, target):
    """Mean flood logit over the original predicted-flood target."""
    with torch.no_grad():
        logits = model(x)[0, 0]
        return logits[target].mean().item()


# ============================================================
# HELPERS
# ============================================================

def make_perturbed_input(x_np, selected, replacement):
    """
    Replace selected pixels in BOTH VV and VH channels.
    x_np and replacement: (2, TILE, TILE)
    selected: (TILE, TILE) boolean mask.
    """
    result = x_np.copy()

    for channel in range(2):
        result[channel, selected] = replacement[channel, selected]

    return torch.from_numpy(result).unsqueeze(0).to(DEVICE)


def save_cam(cam, valid, h, w, row, col):
    cam_out = np.full((h, w), -9999.0, dtype=np.float32)
    cam_out[valid] = cam[:h, :w][valid]

    output_path = OUT / f"gradcam_flood_r{row}_c{col}.tif"

    with rasterio.open(INPUT) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            height=h,
            width=w,
            transform=src.window_transform(
                Window(col, row, w, h)
            ),
            dtype="float32",
            nodata=-9999.0,
            compress="deflate"
        )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cam_out, 1)
        dst.set_band_description(
            1, "Normalized flood-target Grad-CAM"
        )
        dst.update_tags(
            target_layer="dec1",
            method="Grad-CAM"
        )

    return str(output_path.relative_to(ROOT))


# ============================================================
# TILE EVALUATION
# ============================================================

def evaluate_tile(model, sar, mask, row, col, rng):
    h = min(TILE, sar.shape[1] - row)
    w = min(TILE, sar.shape[2] - col)

    data = sar[:, row:row + h, col:col + w]
    local_mask = mask[row:row + h, col:col + w]

    valid = (
        np.isfinite(data[0])
        & np.isfinite(data[1])
        & (data[0] > 0)
        & (data[1] > 0)
        & (local_mask != 255)
    )

    flood = valid & (local_mask == 1)

    if int(flood.sum()) < MIN_TARGET_PIXELS:
        return {
            "row": row,
            "col": col,
            "status": "skipped",
            "reason": (
                f"Only {int(flood.sum())} valid flood pixels"
            )
        }

    target = np.zeros((TILE, TILE), dtype=bool)
    target[:h, :w] = flood

    x_np = preprocess(data)
    x = torch.from_numpy(x_np).unsqueeze(0).to(DEVICE)

    # Generate a fresh Grad-CAM for this tile.
    cam_full = gradcam(model, x, target)
    cam = cam_full[:h, :w]

    values = cam[flood]
    low_cut, high_cut = np.quantile(values, [0.10, 0.90])

    high = flood & (cam >= high_cut)
    low = flood & (cam <= low_cut)

    high_idx = np.flatnonzero(high)
    low_idx = np.flatnonzero(low)

    n = min(len(high_idx), len(low_idx), MAX_PERTURB)

    if n < 100:
        return {
            "row": row,
            "col": col,
            "status": "skipped",
            "reason": (
                f"Too few candidates: high={len(high_idx)}, "
                f"low={len(low_idx)}"
            )
        }

    # Sample the same number of high- and low-attribution pixels.
    high_idx = rng.choice(high_idx, n, replace=False)
    low_idx = rng.choice(low_idx, n, replace=False)

    high_local = np.zeros((TILE, TILE), dtype=bool)
    low_local = np.zeros((TILE, TILE), dtype=bool)
    random_local = np.zeros((TILE, TILE), dtype=bool)

    high_local[:h, :w].flat[high_idx] = True
    low_local[:h, :w].flat[low_idx] = True

    # Random control sampled from eligible predicted-flood pixels.
    # It is not forced to be disjoint from the high/low groups.
    # Exclude both attribution-selected groups from the random control.
    excluded = high | low
    control_pool = flood & ~excluded
    control_idx = np.flatnonzero(control_pool)

    if len(control_idx) < n:
        return {
            "row": row,
            "col": col,
            "status": "skipped",
            "reason": (
                f"Insufficient disjoint random-control pixels: "
                f"available={len(control_idx)}, required={n}"
            )
     }

    random_idx = rng.choice(control_idx, n, replace=False)
    random_local[:h, :w].flat[random_idx] = True

    # Median replacements are calculated on normalized channels.
    replacement = median_replacement_input(x_np, h, w)

    x_high = make_perturbed_input(
        x_np, high_local, replacement
    )
    x_low = make_perturbed_input(
        x_np, low_local, replacement
    )
    x_random = make_perturbed_input(
        x_np, random_local, replacement
    )

    target_t = torch.from_numpy(target).to(DEVICE)

    original_score = target_score(model, x, target_t)
    high_score = target_score(model, x_high, target_t)
    low_score = target_score(model, x_low, target_t)
    random_score = target_score(model, x_random, target_t)

    high_change = abs(high_score - original_score)
    low_change = abs(low_score - original_score)
    random_change = abs(random_score - original_score)

    cam_path = save_cam(
        cam, valid, h, w, row, col
    )

    return {
        "row": row,
        "col": col,
        "status": "ok",
        "reason": "",
        "valid_flood_pixels": int(flood.sum()),
        "pixels_perturbed_each": n,
        "replacement_method": "5x5_local_median",
        "original_score": original_score,
        "high_perturbed_score": high_score,
        "low_perturbed_score": low_score,
        "random_perturbed_score": random_score,
        "high_abs_change": high_change,
        "low_abs_change": low_change,
        "random_abs_change": random_change,
        "high_vs_low_ratio": (
            high_change / low_change
            if low_change > 1e-12 else ""
        ),
        "high_vs_random_ratio": (
            high_change / random_change
            if random_change > 1e-12 else ""
        ),
        "high_change_larger_than_low": (
            high_change > low_change
        ),
        "high_change_larger_than_random": (
            high_change > random_change
        ),
        "cam_path": cam_path,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    for path in (INPUT, MASK, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output folder: {OUT}")

    with rasterio.open(INPUT) as src:
        sar = src.read([1, 2]).astype(np.float32)

    with rasterio.open(MASK) as src:
        mask = src.read(1)

    if sar.shape[1:] != mask.shape:
        raise ValueError(
            "SAR and flood-mask dimensions do not match."
        )

    candidates = [
        (0, 1635),
        (384, 1536),
        (3072, 768),
        (768, 384),
    ]

    model = load_model()
    rows = []

    for index, (row, col) in enumerate(candidates, start=1):
        print(
            f"\nTile {index}/{len(candidates)}: "
            f"row={row}, col={col}"
        )

        try:
            result = evaluate_tile(
                model,
                sar,
                mask,
                row,
                col,
                np.random.default_rng(SEED + index)
            )
        except Exception as exc:
            result = {
                "row": row,
                "col": col,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"
            }

        rows.append(result)

        if result["status"] == "ok":
            print(
                f"  Pixels perturbed per condition: "
                f"{result['pixels_perturbed_each']:,}"
            )
            print(
                f"  High attribution change: "
                f"{result['high_abs_change']:.6f}"
            )
            print(
                f"  Low attribution change: "
                f"{result['low_abs_change']:.6f}"
            )
            print(
                f"  Random control change: "
                f"{result['random_abs_change']:.6f}"
            )
        else:
            print(
                f"  {result['status']}: {result['reason']}"
            )

    columns = sorted({
        key for result in rows for key in result
    })

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    successful = [
        result for result in rows
        if result["status"] == "ok"
    ]

    print(f"\nCSV report: {CSV_PATH}")
    print(
        f"Successful tiles: {len(successful)}/{len(rows)}"
    )

    if successful:
        high_vs_low = sum(
            bool(result["high_change_larger_than_low"])
            for result in successful
        )
        high_vs_random = sum(
            bool(result["high_change_larger_than_random"])
            for result in successful
        )

        print(
            "High-attribution change > low-attribution change: "
            f"{high_vs_low}/{len(successful)} tiles"
        )
        print(
            "High-attribution change > random-control change: "
            f"{high_vs_random}/{len(successful)} tiles"
        )


print(
    "\nCaution: local-median replacement is a synthetic perturbation. "
    "Random-control pixels are disjoint from the high- and low-attribution "
    "groups. Results are preliminary sensitivity diagnostics, not proof "
    "of causal correctness."
)



if __name__ == "__main__":
    main()
