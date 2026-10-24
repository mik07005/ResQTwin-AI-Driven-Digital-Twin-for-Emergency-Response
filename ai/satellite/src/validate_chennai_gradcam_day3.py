
from pathlib import Path
import sys
import csv

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.windows import Window

SATELLITE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SATELLITE_DIR))
from models.unet import UNet

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"
INPUT = BASE / "sentinel1" / "gcc" / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
MASK = BASE / "sentinel1" / "gcc" / "inference" / "flood_mask.tif"
CHECKPOINT = ROOT / "ai" / "satellite" / "checkpoints" / "positive_weight_5" / "best.pt"
OUT = BASE / "explainability" / "day3"
CSV_PATH = OUT / "gradcam_robustness_summary.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE = 512
VV_MEAN, VV_STD = -10.3929, 4.0388
VH_MEAN, VH_STD = -17.2411, 4.7544
MIN_TARGET_PIXELS = 500
MAX_PERTURB = 3000
SEED = 42


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


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = UNet(in_channels=2, out_channels=1)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


def gradcam(model, x, target):
    activations, gradients = {}, {}
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
        cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(TILE, TILE), mode="bilinear",
            align_corners=False
        )[0, 0].detach().cpu().numpy()

        if not np.isfinite(cam).all() or cam.max() <= 0:
            raise RuntimeError("Grad-CAM has no positive finite attribution.")

        return (cam / cam.max()).astype(np.float32)
    finally:
        hf.remove()
        hb.remove()


def target_score(model, x, target):
    with torch.no_grad():
        logits = model(x)[0, 0]
        return logits[target].mean().item()


def evaluate_tile(model, sar, mask, row, col, rng):
    h = min(TILE, sar.shape[1] - row)
    w = min(TILE, sar.shape[2] - col)
    data = sar[:, row:row+h, col:col+w]
    local_mask = mask[row:row+h, col:col+w]

    valid = (
        np.isfinite(data[0]) & np.isfinite(data[1])
        & (data[0] > 0) & (data[1] > 0)
    )
    flood = valid & (local_mask == 1)

    if flood.sum() < MIN_TARGET_PIXELS:
        return {
            "row": row, "col": col, "status": "skipped",
            "reason": f"Only {int(flood.sum())} valid flood pixels"
        }

    target = np.zeros((TILE, TILE), dtype=bool)
    target[:h, :w] = flood

    x = torch.from_numpy(preprocess(data)).unsqueeze(0).to(DEVICE)
    cam_full = gradcam(model, x, target)
    cam = cam_full[:h, :w]

    # Select top/bottom attribution deciles among eligible flood pixels.
    values = cam[flood]
    low_cut, high_cut = np.quantile(values, [0.10, 0.90])
    high = flood & (cam >= high_cut)
    low = flood & (cam <= low_cut)

    high_idx = np.flatnonzero(high)
    low_idx = np.flatnonzero(low)
    n = min(len(high_idx), len(low_idx), MAX_PERTURB)

    if n < 100:
        return {
            "row": row, "col": col, "status": "skipped",
            "reason": f"Too few candidates: high={len(high_idx)}, low={len(low_idx)}"
        }

    high_idx = rng.choice(high_idx, n, replace=False)
    low_idx = rng.choice(low_idx, n, replace=False)

    high_local = np.zeros((TILE, TILE), dtype=bool)
    low_local = np.zeros((TILE, TILE), dtype=bool)
    high_local[:h, :w].flat[high_idx] = True
    low_local[:h, :w].flat[low_idx] = True

    high_t = torch.from_numpy(high_local).to(DEVICE)
    low_t = torch.from_numpy(low_local).to(DEVICE)
    original = target_score(model, x, torch.from_numpy(target).to(DEVICE))

    x_high, x_low = x.clone(), x.clone()
    x_high[0, :, high_t] = 0.0
    x_low[0, :, low_t] = 0.0

    high_score = target_score(model, x_high, torch.from_numpy(target).to(DEVICE))
    low_score = target_score(model, x_low, torch.from_numpy(target).to(DEVICE))
    high_change = abs(high_score - original)
    low_change = abs(low_score - original)

    # Save this tile's independently generated CAM.
    cam_out = np.full((h, w), -9999.0, dtype=np.float32)
    cam_out[valid] = cam[valid]
    output_path = OUT / f"gradcam_flood_r{row}_c{col}.tif"

    with rasterio.open(INPUT) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff", count=1, height=h, width=w,
            transform=src.window_transform(Window(col, row, w, h)),
            dtype="float32", nodata=-9999.0, compress="deflate"
        )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cam_out, 1)
        dst.set_band_description(1, "Normalized flood-target Grad-CAM")
        dst.update_tags(target_layer="dec1", method="Grad-CAM")

    return {
        "row": row, "col": col, "status": "ok", "reason": "",
        "valid_flood_pixels": int(flood.sum()),
        "pixels_perturbed_each": n,
        "original_score": original,
        "high_perturbed_score": high_score,
        "low_perturbed_score": low_score,
        "high_abs_change": high_change,
        "low_abs_change": low_change,
        "change_ratio": high_change / low_change if low_change > 1e-12 else "",
        "high_change_larger": high_change > low_change,
        "cam_path": str(output_path.relative_to(ROOT)),
    }


def main():
    for path in (INPUT, MASK, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(path)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    with rasterio.open(INPUT) as src:
        sar = src.read([1, 2]).astype(np.float32)
    with rasterio.open(MASK) as src:
        mask = src.read(1)

    if sar.shape[1:] != mask.shape:
        raise ValueError("SAR and flood-mask dimensions do not match.")

    height, width = mask.shape
    # Four separated, deterministic candidate locations.
    # Eligible candidates found by the scene-wide diagnostic.
# These are spatially separated, but still overlap some neighboring
# tiles because the inference grid uses overlapping windows.
    candidates = [
        (0, 1635),
        (384, 1536),
        (3072, 768),
        (768, 384),
    ]
    model = load_model()
    rows = []
    for index, (row, col) in enumerate(candidates, start=1):
        print(f"\nTile {index}/{len(candidates)}: row={row}, col={col}")
        try:
            result = evaluate_tile(
                model, sar, mask, row, col,
                np.random.default_rng(SEED + index)
            )
        except Exception as exc:
            result = {
                "row": row, "col": col, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"
            }
        rows.append(result)
        if result["status"] == "ok":
            print(
                f"  flood pixels={result['valid_flood_pixels']:,}; "
                f"high change={result['high_abs_change']:.6f}; "
                f"low change={result['low_abs_change']:.6f}; "
                f"ratio={result['change_ratio']}"
            )
        else:
            print(f"  {result['status']}: {result['reason']}")

    columns = sorted({key for row in rows for key in row})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    successful = [r for r in rows if r["status"] == "ok"]
    print(f"\nCSV report: {CSV_PATH}")
    print(f"Successful tiles: {len(successful)}/{len(rows)}")
    if successful:
        count = sum(bool(r["high_change_larger"]) for r in successful)
        print(
            f"High-attribution perturbation changed the score more "
            f"in {count}/{len(successful)} successful tiles."
        )
    print(
        "Caution: zeroing normalized inputs may create unrealistic SAR "
        "patterns. Results are preliminary sensitivity checks, not causal proof."
    )


if __name__ == "__main__":
    main()
