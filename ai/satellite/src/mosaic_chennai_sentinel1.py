from pathlib import Path

import numpy as np
import rasterio


ROOT = Path(__file__).resolve().parents[3]

SENTINEL_DIR = ROOT / "data" / "processed" / "chennai" / "sentinel1"

AD56_VV = SENTINEL_DIR / "pre_event" / "ad56_vv_sigma0_chennai_utm44n.tif"
AD56_VH = SENTINEL_DIR / "pre_event" / "ad56_vh_sigma0_chennai_utm44n.tif"

F20_VV = SENTINEL_DIR / "pre_event" / "20f6_vv_sigma0_chennai_utm44n.tif"
F20_VH = SENTINEL_DIR / "pre_event" / "20f6_vh_sigma0_chennai_utm44n.tif"

OUTPUT_VV = SENTINEL_DIR / "pre_event" / "sentinel1_pre_event_vv_chennai_utm44n.tif"
OUTPUT_VH = SENTINEL_DIR / "pre_event" / "sentinel1_pre_event_vh_chennai_utm44n.tif"
OUTPUT_STACK = SENTINEL_DIR / "pre_event" / "sentinel1_pre_event_vv_vh_chennai_utm44n.tif"


def main():
    print("=== Chennai Sentinel-1 Two-Slice Mosaic ===")
    print()

    print("F20 VV:", F20_VV)
    print("F20 VH:", F20_VH)
    print("AD56 VV:", AD56_VV)
    print("AD56 VH:", AD56_VH)
    print()

    with rasterio.open(F20_VV) as f20_vv, \
         rasterio.open(F20_VH) as f20_vh, \
         rasterio.open(AD56_VV) as ad56_vv, \
         rasterio.open(AD56_VH) as ad56_vh:

        # Check that all rasters share the same geometry.
        datasets = {
            "F20 VV": f20_vv,
            "F20 VH": f20_vh,
            "AD56 VV": ad56_vv,
            "AD56 VH": ad56_vh,
        }

        reference = f20_vv

        for name, ds in datasets.items():
            if ds.crs != reference.crs:
                raise ValueError(f"{name}: CRS mismatch")

            if ds.width != reference.width or ds.height != reference.height:
                raise ValueError(f"{name}: size mismatch")

            if ds.transform != reference.transform:
                raise ValueError(f"{name}: transform mismatch")

        print("Geometry validation: PASS")
        print("CRS:", reference.crs)
        print("Size:", reference.width, "x", reference.height)
        print("Resolution:", reference.res)
        print("Bounds:", reference.bounds)
        print()

        f20_vv_data = f20_vv.read(1)
        f20_vh_data = f20_vh.read(1)
        ad56_vv_data = ad56_vv.read(1)
        ad56_vh_data = ad56_vh.read(1)

        # Validity is determined independently for each slice.
        f20_valid = f20_vv_data > 0
        ad56_valid = ad56_vv_data > 0

        # The VV/VH validation already confirmed that each slice has
        # identical VV/VH validity masks.
        f20_vh_valid = f20_vh_data > 0
        ad56_vh_valid = ad56_vh_data > 0

        if not np.array_equal(f20_valid, f20_vh_valid):
            raise ValueError("f20 VV/VH validity masks differ")

        if not np.array_equal(ad56_valid, ad56_vh_valid):
            raise ValueError("ad56 VV/VH validity masks differ")

        # Start with zeros representing NoData.
        mosaic_vv = np.zeros_like(f20_vv_data, dtype=np.float32)
        mosaic_vh = np.zeros_like(f20_vh_data, dtype=np.float32)

        # f20 is the first slice.
        mosaic_vv[f20_valid] = f20_vv_data[f20_valid]
        mosaic_vh[f20_valid] = f20_vh_data[f20_valid]

        # ad56 fills pixels not already occupied by f20.
        ad56_only = ad56_valid & ~f20_valid

        mosaic_vv[ad56_only] = ad56_vv_data[ad56_only]
        mosaic_vh[ad56_only] = ad56_vh_data[ad56_only]

        overlap = f20_valid & ad56_valid
        union = f20_valid | ad56_valid

        print("f20 valid pixels:", int(f20_valid.sum()))
        print("ad56 valid pixels:", int(ad56_valid.sum()))
        print("Overlap:", int(overlap.sum()))
        print("ad56-only pixels:", int(ad56_only.sum()))
        print("Union:", int(union.sum()))
        print("Union coverage: %.4f%%" %
              (100.0 * union.sum() / union.size))
        print()

        # Verify that the two products really have almost no overlap.
        if overlap.sum() > 1000:
            raise ValueError(
                f"Unexpectedly large slice overlap: {int(overlap.sum())}"
            )

        profile = reference.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(OUTPUT_VV, "w", **profile) as dst:
            dst.write(mosaic_vv, 1)
            dst.set_band_description(1, "Sentinel-1 VV sigma0 linear")

        with rasterio.open(OUTPUT_VH, "w", **profile) as dst:
            dst.write(mosaic_vh, 1)
            dst.set_band_description(1, "Sentinel-1 VH sigma0 linear")

        stack_profile = reference.profile.copy()

        stack_profile.update(
            dtype="float32",
            count=2,
            nodata=0.0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        with rasterio.open(OUTPUT_STACK, "w", **stack_profile) as dst:
            dst.write(mosaic_vv, 1)
            dst.write(mosaic_vh, 2)

            dst.set_band_description(1, "VV sigma0 linear")
            dst.set_band_description(2, "VH sigma0 linear")

        print("Created:")
        print("  VV:", OUTPUT_VV)
        print("  VH:", OUTPUT_VH)
        print("  Stack:", OUTPUT_STACK)
        print()

    # Final validation
    with rasterio.open(OUTPUT_STACK) as src:
        data_vv = src.read(1)
        data_vh = src.read(2)

        valid_vv = data_vv > 0
        valid_vh = data_vh > 0

        print("=== Final Validation ===")
        print("CRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("Bands:", src.count)

        print("VV valid:", int(valid_vv.sum()))
        print("VH valid:", int(valid_vh.sum()))
        print("Common valid:", int((valid_vv & valid_vh).sum()))

        print(
            "Coverage: %.4f%%"
            % (100.0 * valid_vv.sum() / valid_vv.size)
        )

        print("VV min/max:",
              float(data_vv[valid_vv].min()),
              float(data_vv[valid_vv].max()))

        print("VH min/max:",
              float(data_vh[valid_vh].min()),
              float(data_vh[valid_vh].max()))

        print()
        print("FINAL MOSAIC: PASS")


if __name__ == "__main__":
    main()