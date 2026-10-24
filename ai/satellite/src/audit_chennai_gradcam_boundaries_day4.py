
from pathlib import Path
import csv

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"
INPUT = BASE / "sentinel1" / "gcc" / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
PROB = BASE / "sentinel1" / "gcc" / "inference" / "flood_probability.tif"
DAY3 = BASE / "explainability" / "day3"
OUT = BASE / "explainability" / "day4"
CSV_OUT = OUT / "day4_boundary_diagnostic_v2.csv"

TILE_SIZE = 512
BUFFER = 10
TILES = [(0, 1635), (384, 1536), (3072, 768), (768, 384)]


def pearson(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_corr(x, y):
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with rasterio.open(INPUT) as src:
        sar = src.read([1, 2]).astype(np.float32)
        transform, crs = src.transform, src.crs

    with rasterio.open(PROB) as src:
        probability = src.read(1).astype(np.float32)
        if src.transform != transform or src.crs != crs:
            raise ValueError("Probability raster georeferencing differs.")
        if src.shape != sar.shape[1:]:
            raise ValueError("Probability and SAR dimensions differ.")

    try:
        from scipy.ndimage import distance_transform_edt
        has_scipy = True
    except ImportError:
        has_scipy = False

    records = []

    for row, col in TILES:
        h = min(TILE_SIZE, sar.shape[1] - row)
        w = min(TILE_SIZE, sar.shape[2] - col)
        sl = (slice(row, row + h), slice(col, col + w))
        data = sar[:, sl[0], sl[1]]
        p = probability[sl].copy()

        valid = (
            np.isfinite(data[0]) & np.isfinite(data[1])
            & (data[0] > 0) & (data[1] > 0)
            & np.isfinite(p) & (p >= 0) & (p <= 1)
        )

        cam_path = DAY3 / f"gradcam_flood_r{row}_c{col}.tif"
        with rasterio.open(cam_path) as src:
            cam = src.read(1).astype(np.float32)
            nodata = src.nodata

        if cam.shape != (h, w):
            raise ValueError(f"CAM shape mismatch: {cam_path.name}")

        cam_valid = valid & np.isfinite(cam)
        if nodata is not None:
            cam_valid &= cam != nodata

        # Only measure proximity to the external edge of the valid footprint.
        # Fill internal holes for this geometry calculation so that each
        # isolated invalid patch does not expand over the entire tile.
        padded = np.pad(valid, 1, constant_values=False)

        if has_scipy:
            from scipy.ndimage import binary_fill_holes
            footprint = binary_fill_holes(padded)[1:-1, 1:-1]
            # Distance to invalid space outside the filled footprint.
            distance = distance_transform_edt(footprint)
        else:
            # Conservative fallback: use distance from tile edges.
            yy, xx = np.indices(valid.shape)
            distance = np.minimum.reduce([
                yy + 1, xx + 1, h - yy, w - xx
            ]).astype(float)

        edge_zone = valid & (distance <= BUFFER)
        interior = valid & (distance > BUFFER)

        near_cam = cam_valid & edge_zone
        interior_cam = cam_valid & interior

        vals = cam[cam_valid]
        if vals.size:
            cutoff = np.quantile(vals, 0.90)
            top = cam_valid & (cam >= cutoff)
            top_count = int(top.sum())
            top_edge_fraction = (
                float(np.count_nonzero(top & edge_zone) / top_count)
                if top_count else float("nan")
            )
        else:
            top_edge_fraction = float("nan")

        corr_mask = cam_valid & np.isfinite(p)
        x = cam[corr_mask].astype(float)
        y = p[corr_mask].astype(float)

        records.append({
            "row": row,
            "col": col,
            "boundary_method": (
                "filled_valid_footprint_distance"
                if has_scipy else "tile_edge_distance_fallback"
            ),
            "valid_cam_pixels": int(cam_valid.sum()),
            "edge_zone_pixels": int(edge_zone.sum()),
            "interior_pixels": int(interior.sum()),
            "mean_cam_edge_zone": (
                float(cam[near_cam].mean()) if near_cam.any() else ""
            ),
            "mean_cam_interior": (
                float(cam[interior_cam].mean())
                if interior_cam.any() else ""
            ),
            "top10_cam_pixels_in_edge_zone_fraction": top_edge_fraction,
            "spearman_cam_probability": rank_corr(x, y),
            "pearson_cam_probability": pearson(x, y),
        })

    columns = list(records[0].keys())
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    for r in records:
        print(
            f"row={r['row']}, col={r['col']}: "
            f"edge={r['edge_zone_pixels']:,}, "
            f"interior={r['interior_pixels']:,}, "
            f"top-10% edge fraction="
            f"{r['top10_cam_pixels_in_edge_zone_fraction']:.3f}"
        )

    print(f"\nSaved: {CSV_OUT}")
    if not has_scipy:
        print(
            "Note: SciPy unavailable; fallback measures distance from tile "
            "edges only and does not detect the full scene footprint."
        )


if __name__ == "__main__":
    main()
