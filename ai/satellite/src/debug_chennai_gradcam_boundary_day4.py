
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data" / "processed" / "chennai"
INPUT = BASE / "sentinel1" / "gcc" / "sentinel1_vv_vh_gcc_2025_utm44n.tif"

TILES = [(0, 1635), (384, 1536), (3072, 768), (768, 384)]
TILE_SIZE = 512


def expand(mask, iterations):
    result = mask.copy()
    for _ in range(iterations):
        p = np.pad(result, 1, constant_values=False)
        result = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
            | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )
    return result


with rasterio.open(INPUT) as src:
    sar = src.read([1, 2]).astype(np.float32)

for row, col in TILES:
    data = sar[:, row:row + TILE_SIZE, col:col + TILE_SIZE]

    valid = (
        np.isfinite(data[0]) & np.isfinite(data[1])
        & (data[0] > 0) & (data[1] > 0)
    )

    invalid = ~valid
    expanded = expand(invalid, 10)
    interior = valid & ~expanded

    total = valid.size
    valid_count = int(valid.sum())
    invalid_count = int(invalid.sum())
    expanded_count = int(expanded.sum())
    interior_count = int(interior.sum())

    print(f"\nTile row={row}, col={col}")
    print(f"Tile pixels: {total:,}")
    print(f"Valid SAR pixels: {valid_count:,} ({valid_count / total:.1%})")
    print(f"Invalid SAR pixels: {invalid_count:,} ({invalid_count / total:.1%})")
    print(f"Pixels in expanded invalid mask: {expanded_count:,}")
    print(f"Valid interior pixels after buffer: {interior_count:,}")
