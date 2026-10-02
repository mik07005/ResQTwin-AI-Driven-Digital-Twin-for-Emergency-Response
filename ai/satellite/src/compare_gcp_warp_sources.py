from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds


ROOT = Path(__file__).resolve().parents[3]

ORIGINAL = (
    ROOT / "data" / "raw" / "chennai" / "sentinel1"
    / "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.tiff"
)

CALIBRATED = (
    ROOT / "data" / "processed" / "chennai" / "sentinel1"
    / "vv_sigma0_linear.tif"
)

ANNOTATION = (
    ROOT / "data" / "raw" / "chennai" / "sentinel1"
    / "cog_safe"
    / "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation"
    / "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)

SRC_CRS = CRS.from_epsg(4326)
DST_CRS = CRS.from_epsg(32644)

# Same small AOI that previously worked with the original COG
BBOX = (80.20, 12.95, 80.30, 13.05)


def parse_gcps(xml_path):
    root = ET.parse(xml_path).getroot()
    gcps = []

    for node in root.iter():
        if node.tag.endswith("geolocationGridPoint"):
            line = node.find(".//{*}line")
            pixel = node.find(".//{*}pixel")
            lat = node.find(".//{*}latitude")
            lon = node.find(".//{*}longitude")

            if all(v is not None for v in (line, pixel, lat, lon)):
                gcps.append(
                    GroundControlPoint(
                        row=float(line.text),
                        col=float(pixel.text),
                        x=float(lon.text),
                        y=float(lat.text),
                    )
                )

    return gcps


def run_test(label, path):

    print(f"\n{'=' * 60}")
    print(label)
    print(f"{'=' * 60}")
    print(f"Input: {path}")

    with rasterio.open(path) as src:

        print(f"Size: {src.width} x {src.height}")
        print(f"dtype: {src.dtypes[0]}")
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")

        # Read a small source window
        window = rasterio.windows.Window(
            0, 0,
            min(src.width, 1000),
            min(src.height, 1000)
        )

        arr = src.read(1, window=window)

        print(
            f"Source window: "
            f"min={arr.min():.8f}, "
            f"max={arr.max():.8f}, "
            f"nonzero={np.count_nonzero(arr):,}"
        )

        # Destination grid
        left, bottom, right, top = transform_bounds(
            SRC_CRS,
            DST_CRS,
            *BBOX
        )

        resolution = 10.0

        width = int(np.ceil((right - left) / resolution))
        height = int(np.ceil((top - bottom) / resolution))

        dst_transform = from_origin(
            left,
            top,
            resolution,
            resolution
        )

        destination = np.zeros(
            (height, width),
            dtype=np.float32
        )

        print(f"Destination: {width} x {height}")

        gcps = parse_gcps(ANNOTATION)

        print(f"GCPs: {len(gcps)}")

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,

            # Explicitly tell Rasterio that the source has
            # no affine geotransform.
            

            # Use the Sentinel-1 GCP geolocation.
            gcps=gcps,

            src_crs=SRC_CRS,

            dst_transform=dst_transform,
            dst_crs=DST_CRS,

            resampling=Resampling.nearest,

            num_threads=4,
            warp_mem_limit=512,

            init_dest_nodata=True,

            SRC_METHOD="NO_GEOTRANSFORM",
        )

        valid = np.isfinite(destination) & (destination != 0)

        print("\nResult:")
        print(f"Valid: {valid.sum():,}")
        print(f"Total: {destination.size:,}")
        print(f"Coverage: {100 * valid.mean():.4f}%")

        if valid.any():
            values = destination[valid]
            print(f"Min: {values.min():.8f}")
            print(f"Max: {values.max():.8f}")
            print(f"Mean: {values.mean():.8f}")
            print("STATUS: PASS")
        else:
            print("STATUS: FAIL")


print("=== Sentinel-1 GCP Source Comparison ===")

gcps = parse_gcps(ANNOTATION)

print(f"\nGCP count: {len(gcps)}")

run_test("TEST 1 — ORIGINAL COG", ORIGINAL)

run_test("TEST 2 — CALIBRATED VV", CALIBRATED)

print("\n=== Diagnostic complete ===")