from pathlib import Path

import numpy as np
import rasterio


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

VV_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vv_sigma0_chennai_utm44n.tif"
)

VH_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "vh_sigma0_chennai_utm44n.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
)

STACK_PATH = OUTPUT_DIR / "sentinel1_vv_vh_chennai_utm44n.tif"


# ============================================================
# Helper functions
# ============================================================

def print_raster_info(name, path):
    """Print basic raster metadata and statistics."""

    print("\n" + "=" * 70)
    print(f"{name} RASTER")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with rasterio.open(path) as src:
        print(f"Path       : {path}")
        print(f"CRS        : {src.crs}")
        print(f"Size       : {src.width} x {src.height}")
        print(f"Count      : {src.count}")
        print(f"Resolution : {src.res}")
        print(f"Bounds     : {src.bounds}")
        print(f"Dtype      : {src.dtypes}")
        print(f"NoData     : {src.nodata}")

        data = src.read(1)

        valid = np.isfinite(data) & (data > 0)

        if src.nodata is not None:
            valid &= data != src.nodata

        valid_values = data[valid]

        print(f"Valid      : {valid_values.size:,}")
        print(f"Coverage   : {100 * valid.mean():.4f}%")

        if valid_values.size > 0:
            print(f"Min        : {valid_values.min():.8f}")
            print(f"Max        : {valid_values.max():.8f}")
            print(f"Mean       : {valid_values.mean():.8f}")
            print(f"Median     : {np.median(valid_values):.8f}")
            print(f"Std        : {valid_values.std():.8f}")

        return src.profile.copy(), src.transform, src.crs, src.bounds, data


# ============================================================
# Main validation
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("CHENNAI SENTINEL-1 VV/VH ALIGNMENT VALIDATION")
    print("#" * 70)

    # --------------------------------------------------------
    # 1. Inspect VV
    # --------------------------------------------------------

    vv_profile, vv_transform, vv_crs, vv_bounds, vv = print_raster_info(
        "VV",
        VV_PATH
    )

    # --------------------------------------------------------
    # 2. Inspect VH
    # --------------------------------------------------------

    vh_profile, vh_transform, vh_crs, vh_bounds, vh = print_raster_info(
        "VH",
        VH_PATH
    )

    # --------------------------------------------------------
    # 3. Metadata comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VV / VH ALIGNMENT CHECK")
    print("=" * 70)

    checks = {}

    checks["CRS"] = vv_crs == vh_crs
    checks["Width"] = vv.shape[1] == vh.shape[1]
    checks["Height"] = vv.shape[0] == vh.shape[0]
    checks["Resolution"] = np.allclose(
        vv_profile["transform"].to_gdal(),
        vh_profile["transform"].to_gdal()
    )
    checks["Bounds"] = np.allclose(
        vv_bounds,
        vh_bounds,
        atol=1e-6
    )

    for check_name, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"{check_name:<15}: {status}")

    if not all(checks.values()):
        print("\nWARNING: VV and VH are not perfectly aligned.")
        print("Do NOT create the final two-channel stack yet.")
        return

    # --------------------------------------------------------
    # 4. Compare valid-data masks
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALID-DATA MASK CHECK")
    print("=" * 70)

    vv_valid = np.isfinite(vv) & (vv > 0)
    vh_valid = np.isfinite(vh) & (vh > 0)

    common_valid = vv_valid & vh_valid
    vv_only = vv_valid & ~vh_valid
    vh_only = vh_valid & ~vv_valid

    total_pixels = vv.size

    print(f"Total pixels          : {total_pixels:,}")
    print(f"VV valid              : {vv_valid.sum():,}")
    print(f"VH valid              : {vh_valid.sum():,}")
    print(f"Common valid          : {common_valid.sum():,}")
    print(f"VV-only valid         : {vv_only.sum():,}")
    print(f"VH-only valid         : {vh_only.sum():,}")

    print(
        f"Common coverage       : "
        f"{100 * common_valid.mean():.4f}%"
    )

    print(
        f"VV-only coverage      : "
        f"{100 * vv_only.mean():.4f}%"
    )

    print(
        f"VH-only coverage      : "
        f"{100 * vh_only.mean():.4f}%"
    )

    # --------------------------------------------------------
    # 5. Compare spatial metadata numerically
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPATIAL METADATA")
    print("=" * 70)

    print(f"VV CRS       : {vv_crs}")
    print(f"VH CRS       : {vh_crs}")

    print(f"VV transform : {vv_transform}")
    print(f"VH transform : {vh_transform}")

    print(f"\nVV bounds:")
    print(f"  left   = {vv_bounds.left:.6f}")
    print(f"  bottom = {vv_bounds.bottom:.6f}")
    print(f"  right  = {vv_bounds.right:.6f}")
    print(f"  top    = {vv_bounds.top:.6f}")

    print(f"\nVH bounds:")
    print(f"  left   = {vh_bounds.left:.6f}")
    print(f"  bottom = {vh_bounds.bottom:.6f}")
    print(f"  right  = {vh_bounds.right:.6f}")
    print(f"  top    = {vh_bounds.top:.6f}")

    # --------------------------------------------------------
    # 6. Compare VV/VH statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VV / VH STATISTICS")
    print("=" * 70)

    vv_values = vv[vv_valid]
    vh_values = vh[vh_valid]

    print("\nVV:")
    print(f"  Mean   : {vv_values.mean():.8f}")
    print(f"  Median : {np.median(vv_values):.8f}")
    print(f"  Std    : {vv_values.std():.8f}")
    print(f"  Min    : {vv_values.min():.8f}")
    print(f"  Max    : {vv_values.max():.8f}")

    print("\nVH:")
    print(f"  Mean   : {vh_values.mean():.8f}")
    print(f"  Median : {np.median(vh_values):.8f}")
    print(f"  Std    : {vh_values.std():.8f}")
    print(f"  Min    : {vh_values.min():.8f}")
    print(f"  Max    : {vh_values.max():.8f}")

    # --------------------------------------------------------
    # 7. Correlation on common valid pixels
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VV / VH COMMON-PIXEL ANALYSIS")
    print("=" * 70)

    if common_valid.sum() > 100:

        vv_common = vv[common_valid].astype(np.float64)
        vh_common = vh[common_valid].astype(np.float64)

        correlation = np.corrcoef(
            vv_common,
            vh_common
        )[0, 1]

        print(
            f"Pearson correlation : "
            f"{correlation:.6f}"
        )

        print(
            "\nNote: VV and VH are different polarizations, "
            "so they are NOT expected to be identical."
        )

        print(
            "The correlation is reported only as a diagnostic; "
            "it is not an alignment criterion."
        )

    # --------------------------------------------------------
    # 8. Check exact transform equality
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIXEL GRID CHECK")
    print("=" * 70)

    transform_difference = np.abs(
        np.array(vv_transform.to_gdal())
        - np.array(vh_transform.to_gdal())
    )

    print(
        "Maximum transform coefficient difference: "
        f"{transform_difference.max():.12f}"
    )

    if np.allclose(
        vv_transform.to_gdal(),
        vh_transform.to_gdal(),
        atol=1e-9
    ):
        print("Pixel grid alignment: PASS")
    else:
        print("Pixel grid alignment: FAIL")

    # --------------------------------------------------------
    # 9. Create common-valid two-channel stack
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING VV + VH TWO-CHANNEL STACK")
    print("=" * 70)

    # Use float32 to keep the final stack manageable.
    vv_float = vv.astype(np.float32)
    vh_float = vh.astype(np.float32)

    # Pixels without valid data in either channel become 0.
    vv_float[~common_valid] = 0
    vh_float[~common_valid] = 0

    stack_profile = vv_profile.copy()

    stack_profile.update(
        driver="GTiff",
        dtype="float32",
        count=2,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
        nodata=0.0
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        STACK_PATH,
        "w",
        **stack_profile
    ) as dst:

        dst.write(vv_float, 1)
        dst.write(vh_float, 2)

        dst.set_band_description(
            1,
            "Sentinel-1 VV Sigma0 Linear"
        )

        dst.set_band_description(
            2,
            "Sentinel-1 VH Sigma0 Linear"
        )

        dst.update_tags(
            acquisition="2021-11-16T00:32:12Z",
            satellite="Sentinel-1A",
            product_type="IW GRD",
            polarization="VV+VH",
            calibration="Sigma0 linear",
            projection="EPSG:32644",
            pixel_spacing="10m",
            event="Chennai November 2021 flood"
        )

    # --------------------------------------------------------
    # 10. Re-open final stack and verify
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL STACK VALIDATION")
    print("=" * 70)

    with rasterio.open(STACK_PATH) as src:

        print(f"Output       : {STACK_PATH}")
        print(f"CRS          : {src.crs}")
        print(f"Size         : {src.width} x {src.height}")
        print(f"Bands        : {src.count}")
        print(f"Resolution   : {src.res}")
        print(f"Bounds       : {src.bounds}")
        print(f"Dtype        : {src.dtypes}")

        print(f"\nBand 1: {src.descriptions[0]}")
        print(f"Band 2: {src.descriptions[1]}")

        assert src.count == 2
        assert src.crs == vv_crs
        assert src.width == vv.shape[1]
        assert src.height == vv.shape[0]

    print("\n" + "#" * 70)
    print("FINAL RESULT: VV + VH STACK CREATED SUCCESSFULLY")
    print("#" * 70)

    print(f"\nSaved to:")
    print(STACK_PATH)


if __name__ == "__main__":
    main()