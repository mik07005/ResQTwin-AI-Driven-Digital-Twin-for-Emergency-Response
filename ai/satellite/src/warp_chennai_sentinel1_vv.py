from pathlib import Path
import xml.etree.ElementTree as ET
import math

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT
    / "data/processed/chennai/sentinel1"
    / "vv_sigma0_linear.tif"
)

ANNOTATION = (
    ROOT
    / "data/raw/chennai/sentinel1"
    / "cog_safe/S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation/s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)

OUTPUT = (
    ROOT
    / "data/processed/chennai/sentinel1"
    / "vv_sigma0_chennai_utm44n.tif"
)

SRC_CRS = CRS.from_epsg(4326)
DST_CRS = CRS.from_epsg(32644)

# Chennai basin / study-area bounds from the selected
# Chennai flood-monitoring AOI.
#
# lon_min, lat_min, lon_max, lat_max
CHENNAI_BBOX = (
    79.1666667,
    12.6666667,
    80.4166667,
    13.6666667,
)

OUTPUT_RESOLUTION = 10.0


def parse_gcps(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    gcps = []

    for elem in root.iter():

        if elem.tag.split("}")[-1] != "geolocationGridPoint":
            continue

        values = {}

        for child in elem.iter():

            name = child.tag.split("}")[-1]

            if child.text is None:
                continue

            text = child.text.strip()

            if name == "line":
                values["line"] = int(text)

            elif name == "pixel":
                values["pixel"] = int(text)

            elif name == "latitude":
                values["lat"] = float(text)

            elif name == "longitude":
                values["lon"] = float(text)

        if {"line", "pixel", "lat", "lon"} <= values.keys():

            gcps.append(
                GroundControlPoint(
                    row=values["line"],
                    col=values["pixel"],
                    x=values["lon"],
                    y=values["lat"],
                )
            )

    if not gcps:
        raise RuntimeError("No geolocation GCPs found.")

    return gcps


def main():

    print("=== Sentinel-1 VV Chennai GCP Warp ===")
    print(f"Input:      {INPUT}")
    print(f"Annotation: {ANNOTATION}")
    print(f"Output:     {OUTPUT}")
    print()

    gcps = parse_gcps(ANNOTATION)

    print(f"GCP count: {len(gcps)}")

    # Transform the requested Chennai geographic bounding box
    # into UTM Zone 44N.
    transformer = Transformer.from_crs(
        SRC_CRS,
        DST_CRS,
        always_xy=True,
    )

    lon_min, lat_min, lon_max, lat_max = CHENNAI_BBOX

    x1, y1 = transformer.transform(
        lon_min,
        lat_min,
    )

    x2, y2 = transformer.transform(
        lon_max,
        lat_max,
    )

    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    width = math.ceil(
        (max_x - min_x) / OUTPUT_RESOLUTION
    )

    height = math.ceil(
        (max_y - min_y) / OUTPUT_RESOLUTION
    )

    transform = from_origin(
        min_x,
        max_y,
        OUTPUT_RESOLUTION,
        OUTPUT_RESOLUTION,
    )

    print()
    print("Chennai AOI:")
    print(
        f"  Longitude: {lon_min} -> {lon_max}"
    )
    print(
        f"  Latitude:  {lat_min} -> {lat_max}"
    )

    print()
    print("Destination:")
    print("  CRS: EPSG:32644")
    print(f"  Resolution: {OUTPUT_RESOLUTION} m")
    print(f"  Size: {width} x {height}")
    print(
        f"  Approx pixels: "
        f"{width * height:,}"
    )

    with rasterio.open(INPUT) as src:

        print()
        print("Source:")
        print(f"  Size: {src.width} x {src.height}")
        print(f"  dtype: {src.dtypes[0]}")
        print(f"  CRS: {src.crs}")

        profile = src.profile.copy()

        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            crs=DST_CRS,
            transform=transform,
            width=width,
            height=height,
            nodata=0.0,
            compress="deflate",
            predictor=3,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        OUTPUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            OUTPUT,
            "w",
            **profile,
        ) as dst:

            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),

                # Critical: GCP-based source geolocation.
                gcps=gcps,

                src_crs=SRC_CRS,
                dst_crs=DST_CRS,

                src_nodata=0.0,
                dst_nodata=0.0,

                resampling=Resampling.bilinear,

                num_threads=4,
                warp_mem_limit=1024,
                init_dest_nodata=True,

                # Required when source has no affine transform.
                SRC_METHOD="NO_GEOTRANSFORM",
            )

    # Validate output.
    with rasterio.open(OUTPUT) as check:

        print()
        print("=== Warp Complete ===")
        print(f"Output: {OUTPUT}")
        print()

        print("Output metadata:")
        print(f"  CRS: {check.crs}")
        print(f"  Size: {check.width} x {check.height}")
        print(
            f"  Resolution: "
            f"{check.res[0]:.3f} x "
            f"{check.res[1]:.3f} m"
        )
        print(f"  Bounds: {check.bounds}")
        print()

        # Read a decimated validation sample instead of
        # loading the entire output into RAM.
        sample = check.read(
            1,
            out_shape=(
                1,
                min(1000, check.height),
                min(1000, check.width),
            ),
        ).astype(np.float32)

        valid = sample[
            np.isfinite(sample) &
            (sample > 0)
        ]

        print("Validation sample:")

        if valid.size:

            print(
                f"  Valid pixels: "
                f"{valid.size:,}"
            )

            print(
                f"  Min sigma0: "
                f"{valid.min():.8f}"
            )

            print(
                f"  Max sigma0: "
                f"{valid.max():.8f}"
            )

            print(
                f"  Mean sigma0: "
                f"{valid.mean():.8f}"
            )

            print()
            print("Checks:")

            if check.crs == DST_CRS:
                print("  EPSG:32644 CRS: PASS")
            else:
                print("  EPSG:32644 CRS: FAIL")

            if valid.size > 0:
                print("  Non-empty output: PASS")
            else:
                print("  Non-empty output: FAIL")

            if np.all(np.isfinite(valid)):
                print("  Finite values: PASS")
            else:
                print("  Finite values: FAIL")

        else:

            print("  No valid pixels found.")
            print("  Non-empty output: FAIL")

    print()
    print("VV Chennai GCP warp finished.")


if __name__ == "__main__":
    main()