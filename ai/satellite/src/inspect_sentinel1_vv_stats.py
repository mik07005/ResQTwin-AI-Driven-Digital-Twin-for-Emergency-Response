from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "data/processed/chennai/sentinel1"
    / "vv_sigma0_linear.tif"
)

SAMPLE_SIZE = 2_000_000
SEED = 42


def main():
    print("=== Sentinel-1 VV Sigma0 Statistical Sanity Check ===")
    print(f"Input: {INPUT}")
    print()

    rng = np.random.default_rng(SEED)

    with rasterio.open(INPUT) as src:
        print(f"Raster size: {src.width} x {src.height}")
        print(f"dtype: {src.dtypes[0]}")
        print(f"nodata: {src.nodata}")
        print()

        # Randomly sample windows/pixels rather than loading the
        # entire ~1.7 GB raster into memory.
        values = []

        rows_per_chunk = 512

        for row_start in range(0, src.height, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, src.height)

            window = rasterio.windows.Window(
                0,
                row_start,
                src.width,
                row_end - row_start,
            )

            arr = src.read(1, window=window)

            valid = arr[np.isfinite(arr) & (arr > 0)]

            if valid.size:
                # Keep a bounded random sample from this chunk.
                take = min(valid.size, 10_000)

                if valid.size > take:
                    idx = rng.choice(
                        valid.size,
                        size=take,
                        replace=False,
                    )
                    valid = valid[idx]

                values.append(valid)

    values = np.concatenate(values)

    if values.size > SAMPLE_SIZE:
        idx = rng.choice(
            values.size,
            size=SAMPLE_SIZE,
            replace=False,
        )
        values = values[idx]

    percentiles = [
        50,
        90,
        95,
        99,
        99.9,
        99.99,
        99.999,
    ]

    print(f"Sampled valid pixels: {values.size:,}")
    print()

    print("Sigma0 linear percentiles:")

    for p in percentiles:
        value = np.percentile(values, p)
        print(f"  P{p:<7}: {value:.8f}")

    print()

    thresholds = [
        0.01,
        0.1,
        1,
        10,
        100,
        1000,
        5000,
    ]

    print("Extreme-value fractions:")

    for threshold in thresholds:
        fraction = np.mean(values > threshold) * 100
        print(
            f"  sigma0 > {threshold:<7}: "
            f"{fraction:.6f}%"
        )

    print()

    print("Equivalent sigma0 dB:")

    db_values = [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100,
        1000,
    ]

    for value in db_values:
        db = 10 * np.log10(value)
        print(f"  {value:<8} -> {db:8.3f} dB")

    print()

    print("Sampled-value dB percentiles:")

    db = 10 * np.log10(values)

    for p in percentiles:
        value = np.percentile(db, p)
        print(f"  P{p:<7}: {value:8.3f} dB")

    print()
    print("=== Sanity Check Complete ===")


if __name__ == "__main__":
    main()