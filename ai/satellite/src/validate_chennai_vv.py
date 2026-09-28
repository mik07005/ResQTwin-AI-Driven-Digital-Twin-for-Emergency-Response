from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vv_sigma0_chennai_utm44n.tif"
)

PREVIEW = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vv_sigma0_chennai_preview.tif"
)


print("=== Chennai VV Product Validation ===")

with rasterio.open(INPUT) as src:

    print("\nMetadata:")
    print(f"CRS: {src.crs}")
    print(f"Size: {src.width} x {src.height}")
    print(f"Resolution: {src.res}")
    print(f"Bounds: {src.bounds}")
    print(f"Nodata: {src.nodata}")

    # Downsample to a manageable validation raster
    scale = 20

    out_width = max(1, src.width // scale)
    out_height = max(1, src.height // scale)

    data = src.read(
        1,
        out_shape=(out_height, out_width),
        resampling=Resampling.average,
    )

    valid = np.isfinite(data) & (data > 0)

    print("\nDownsampled validation:")
    print(f"Size: {out_width} x {out_height}")
    print(f"Valid pixels: {valid.sum():,}")
    print(f"Coverage: {100 * valid.mean():.2f}%")

    if valid.any():

        values = data[valid]

        print(f"Min: {values.min():.8f}")
        print(f"Max: {values.max():.8f}")
        print(f"Mean: {values.mean():.8f}")
        print(f"Median: {np.median(values):.8f}")

    profile = src.profile.copy()

    profile.update(
        width=out_width,
        height=out_height,
        transform=src.transform * src.transform.scale(
            src.width / out_width,
            src.height / out_height,
        ),
        compress="deflate",
    )

    with rasterio.open(PREVIEW, "w", **profile) as dst:
        dst.write(data.astype(np.float32), 1)

print("\nPreview written:")
print(PREVIEW)

print("\n=== Validation Complete ===")