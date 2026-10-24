
from pathlib import Path
import csv

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"

INPUT = (
    BASE / "sentinel1" / "gcc"
    / "sentinel1_vv_vh_gcc_2025_utm44n.tif"
)
PROBABILITY = (
    BASE / "sentinel1" / "gcc" / "inference"
    / "flood_probability.tif"
)
MASK = (
    BASE / "sentinel1" / "gcc" / "inference"
    / "flood_mask.tif"
)
DAY3 = BASE / "explainability" / "day3"
OUT = BASE / "explainability" / "day4"

TILES = [
    (0, 1635),
    (384, 1536),
    (3072, 768),
    (768, 384),
]
TILE_SIZE = 512
NODATA_CAM = -9999.0


def to_db(image):
    result = np.full(image.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(image) & (image > 0)
    result[valid] = 10.0 * np.log10(image[valid])
    return result


def scale_image(image, valid):
    values = image[valid & np.isfinite(image)]
    result = np.full(image.shape, np.nan, dtype=np.float32)

    if values.size == 0:
        return result

    low, high = np.percentile(values, [2, 98])
    if high <= low:
        result[valid] = 0.5
    else:
        result[valid] = np.clip(
            (image[valid] - low) / (high - low), 0, 1
        )
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    required = [INPUT, PROBABILITY, MASK]
    required += [
        DAY3 / f"gradcam_flood_r{r}_c{c}.tif"
        for r, c in TILES
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    with rasterio.open(INPUT) as src:
        sar = src.read([1, 2]).astype(np.float32)
        profile = src.profile.copy()
        height, width = src.height, src.width
        input_transform = src.transform
        input_crs = src.crs

    with rasterio.open(PROBABILITY) as src:
        probability = src.read(1).astype(np.float32)
        probability_transform = src.transform
        probability_crs = src.crs

    with rasterio.open(MASK) as src:
        flood_mask = src.read(1)
        mask_transform = src.transform
        mask_crs = src.crs

    if not (
        sar.shape[1:] == probability.shape == flood_mask.shape
    ):
        raise ValueError("Input, probability and mask shapes differ.")

    if not (
        input_transform == probability_transform == mask_transform
        and input_crs == probability_crs == mask_crs
    ):
        raise ValueError(
            "Input, probability and mask georeferencing differs. "
            "Check alignment before interpreting the figure."
        )

    fig, axes = plt.subplots(
        len(TILES), 4, figsize=(18, 4.5 * len(TILES)),
        squeeze=False
    )
    records = []

    for i, (row, col) in enumerate(TILES):
        h = min(TILE_SIZE, height - row)
        w = min(TILE_SIZE, width - col)

        if h <= 0 or w <= 0:
            raise ValueError(f"Tile outside scene: row={row}, col={col}")

        window = (slice(row, row + h), slice(col, col + w))
        vv = to_db(sar[0][window])
        local_probability = probability[window].copy()
        local_mask = flood_mask[window].copy()

        valid = (
            np.isfinite(sar[0][window])
            & np.isfinite(sar[1][window])
            & (sar[0][window] > 0)
            & (sar[1][window] > 0)
            & np.isfinite(local_probability)
            & np.isin(local_mask, [0, 1])
        )

        vv_display = scale_image(vv, valid)
        local_probability[~valid] = np.nan

        mask_display = local_mask.astype(np.float32)
        mask_display[~valid] = np.nan

        cam_path = DAY3 / f"gradcam_flood_r{row}_c{col}.tif"
        with rasterio.open(cam_path) as src:
            cam = src.read(1).astype(np.float32)
            cam_transform = src.transform
            cam_crs = src.crs
            cam_nodata = src.nodata

        # Verify that the CAM belongs to the correct spatial window.
        expected_transform = rasterio.windows.transform(
            Window(col, row, w, h), input_transform
        )
        if (
            cam.shape != (h, w)
            or cam_transform != expected_transform
            or cam_crs != input_crs
        ):
            raise ValueError(
                f"CAM georeferencing mismatch for row={row}, col={col}"
            )

        cam_valid = valid & np.isfinite(cam)
        if cam_nodata is not None:
            cam_valid &= cam != cam_nodata
        else:
            cam_valid &= cam != NODATA_CAM

        cam_display = cam.copy()
        cam_display[~cam_valid] = np.nan

        panels = [
            (vv_display, "Sentinel-1 VV (scaled dB)", "gray", None, None),
            (
                local_probability, "Existing flood probability",
                "viridis", 0, 1
            ),
            (mask_display, "Existing flood mask", "gray", 0, 1),
            (cam_display, "Flood-target Grad-CAM", "jet", 0, 1),
        ]

        for j, (image, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[i, 0].set_ylabel(
            f"Tile {i + 1}\nrow={row}, col={col}",
            fontsize=10
        )

        flood = cam_valid & (local_mask == 1)
        nonflood = cam_valid & (local_mask == 0)
        flood_values = cam[flood]
        nonflood_values = cam[nonflood]
        valid_cam_values = cam[cam_valid]

        top10 = (
            valid_cam_values.size > 0
            and np.quantile(valid_cam_values, 0.90)
        )

        if valid_cam_values.size:
            cutoff = np.quantile(valid_cam_values, 0.90)
            top_pixels = cam_valid & (cam >= cutoff)
            top_count = int(top_pixels.sum())
            top_in_flood = (
                float(np.count_nonzero(top_pixels & (local_mask == 1)))
                / top_count if top_count else float("nan")
            )
            cam_mean = float(np.mean(valid_cam_values))
            cam_p90 = float(cutoff)
        else:
            top_in_flood = float("nan")
            cam_mean = float("nan")
            cam_p90 = float("nan")

        records.append({
            "row": row,
            "col": col,
            "valid_sar_pixels": int(valid.sum()),
            "valid_cam_pixels": int(cam_valid.sum()),
            "predicted_flood_pixels": int(flood.sum()),
            "predicted_nonflood_pixels": int(nonflood.sum()),
            "cam_mean": cam_mean,
            "cam_p90": cam_p90,
            "top10_cam_pixels_in_flood_fraction": top_in_flood,
            "cam_mean_flood": (
                float(flood_values.mean()) if flood_values.size else ""
            ),
            "cam_mean_nonflood": (
                float(nonflood_values.mean()) if nonflood_values.size else ""
            ),
            "cam_path": str(cam_path.relative_to(ROOT)),
        })

    fig.suptitle(
        "ResQTwin Day 4 — Multi-Tile Grad-CAM Spatial Audit",
        fontsize=15
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    figure_path = OUT / "day4_four_tile_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    csv_path = OUT / "day4_spatial_audit_summary.csv"
    columns = list(records[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    print("\nDay 4 visual audit created.")
    print(f"Figure: {figure_path}")
    print(f"Summary: {csv_path}")
    print("\nPer-tile audit:")
    for item in records:
        print(
            f"row={item['row']}, col={item['col']}: "
            f"valid CAM pixels={item['valid_cam_pixels']:,}, "
            f"flood={item['predicted_flood_pixels']:,}, "
            f"nonflood={item['predicted_nonflood_pixels']:,}, "
            f"CAM mean={item['cam_mean']}"
        )
    print(
        "\nNote: spatial metrics are descriptive diagnostics, "
        "not measures of explanation correctness."
    )


if __name__ == "__main__":
    main()
