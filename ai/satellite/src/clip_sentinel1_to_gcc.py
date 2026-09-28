from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "pre_event"
    / "sentinel1_pre_event_vv_vh_chennai_utm44n.tif"
)

GCC_BOUNDARY = (
    ROOT
    / "data"
    / "raw"
    / "chennai"
    / "aoi"
    / "gcc_ward_boundary_2025.geojson"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "pre_event"
    / "gcc"
    / "sentinel1_pre_event_vv_vh_gcc_2025_utm44n.tif"
)


def main():

    print("=== Clip Sentinel-1 Mosaic to GCC Boundary ===")
    print()

    print("Input:")
    print(INPUT)

    print("GCC boundary:")
    print(GCC_BOUNDARY)

    print("Output:")
    print(OUTPUT)
    print()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load GCC boundary
    # ---------------------------------------------------------

    gcc = gpd.read_file(GCC_BOUNDARY)

    print("GCC features:", len(gcc))
    print("GCC CRS:", gcc.crs)

    if gcc.empty:
        raise ValueError("GCC boundary is empty")

    # Dissolve all 200 wards into one GCC geometry.
    gcc_geometry = gcc.dissolve().geometry.iloc[0]

    print("GCC geometry type:", gcc_geometry.geom_type)
    print()

    # ---------------------------------------------------------
    # Open Sentinel-1 mosaic
    # ---------------------------------------------------------

    with rasterio.open(INPUT) as src:

        print("Sentinel-1 CRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("Bands:", src.count)
        print()

        # Reproject GCC boundary to Sentinel-1 CRS.
        gcc_projected = (
            gpd.GeoSeries(
                [gcc_geometry],
                crs=gcc.crs
            )
            .to_crs(src.crs)
            .iloc[0]
        )

        # Mask and crop.
        clipped, clipped_transform = mask(
            src,
            [gcc_projected.__geo_interface__],
            crop=True,
            filled=True,
            nodata=0.0,
        )

        profile = src.profile.copy()

        profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            count=2,
            dtype="float32",
            nodata=0.0,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        # -----------------------------------------------------
        # Validate the two bands
        # -----------------------------------------------------

        vv = clipped[0]
        vh = clipped[1]

        vv_valid = vv > 0
        vh_valid = vh > 0

        common_valid = vv_valid & vh_valid

        print("Clipped dimensions:")
        print("  Width:", clipped.shape[2])
        print("  Height:", clipped.shape[1])
        print()

        print("Valid VV pixels:", int(vv_valid.sum()))
        print("Valid VH pixels:", int(vh_valid.sum()))
        print("Common valid pixels:", int(common_valid.sum()))

        if not np.array_equal(vv_valid, vh_valid):
            raise ValueError(
                "VV and VH validity masks differ after GCC clipping"
            )

        # -----------------------------------------------------
        # Calculate GCC coverage
        # -----------------------------------------------------

        # GCC polygon raster area is represented by pixels inside
        # the geometry. Since outside-GCC pixels are zero, valid
        # pixels give the SAR-covered portion.
        #
        # The clipped bounding box itself is larger than GCC, so
        # calculate the actual GCC pixel count separately.
        gcc_mask, _ = mask(
            src,
            [gcc_projected.__geo_interface__],
            crop=True,
            filled=True,
            nodata=0,
        )

        gcc_area_mask = gcc_mask[0] >= 0

        # More reliable GCC pixel mask using geometry directly.
                # The clipped raster has already been masked to the GCC
        # geometry. Therefore every valid Sentinel-1 pixel in the
        # clipped output lies inside GCC.
        from rasterio.features import geometry_mask

        gcc_inside = ~geometry_mask(
            [gcc_projected.__geo_interface__],
            out_shape=(clipped.shape[1], clipped.shape[2]),
            transform=clipped_transform,
            invert=False,
        )

        gcc_pixels = int(gcc_inside.sum())

        # All valid pixels remaining after rasterio.mask are inside GCC.
        covered_gcc_pixels = int(common_valid.sum())

        coverage = (
            100.0 * covered_gcc_pixels / gcc_pixels
            if gcc_pixels > 0
            else 0.0
        )

        print()
        print("GCC raster pixels:", gcc_pixels)
        print("SAR-covered GCC pixels:", covered_gcc_pixels)
        print("SAR coverage of GCC: %.4f%%" % coverage)
        print("GCC NoData/uncovered: %.4f%%" % (100.0 - coverage))
        print()

        # -----------------------------------------------------
        # Write output
        # -----------------------------------------------------

        with rasterio.open(OUTPUT, "w", **profile) as dst:

            dst.write(clipped)

            dst.set_band_description(
                1,
                "Sentinel-1 VV sigma0 linear"
            )

            dst.set_band_description(
                2,
                "Sentinel-1 VH sigma0 linear"
            )

        print("Created:")
        print(OUTPUT)
        print()

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    with rasterio.open(OUTPUT) as src:

        vv = src.read(1)
        vh = src.read(2)

        vv_valid = vv > 0
        vh_valid = vh > 0

        print("=== Final Validation ===")
        print("CRS:", src.crs)
        print("Size:", src.width, "x", src.height)
        print("Resolution:", src.res)
        print("Bounds:", src.bounds)
        print("Bands:", src.count)

        print("VV valid:", int(vv_valid.sum()))
        print("VH valid:", int(vh_valid.sum()))
        print("Common valid:", int((vv_valid & vh_valid).sum()))

        print()
        print("GCC CLIP: PASS")


if __name__ == "__main__":
    main()