from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.control import GroundControlPoint
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]

SRC = (
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

DST_CRS = CRS.from_epsg(32644)

# EXACT same AOI used in the successful earlier test
BBOX = (
    80.20,   # min lon
    12.95,   # min lat
    80.30,   # max lon
    13.05,   # max lat
)


def parse_gcps(xml_path):
    root = ET.parse(xml_path).getroot()
    gcps = []

    for node in root.iter():
        if node.tag.endswith("geolocationGridPoint"):
            line = node.find(".//{*}line")
            pixel = node.find(".//{*}pixel")
            lat = node.find(".//{*}latitude")
            lon = node.find(".//{*}longitude")

            if all(x is not None for x in (line, pixel, lat, lon)):
                gcps.append(
                    GroundControlPoint(
                        row=float(line.text),
                        col=float(pixel.text),
                        x=float(lon.text),
                        y=float(lat.text),
                    )
                )

    return gcps


print("=== Calibrated VV GCP Warp Test ===")
print(f"Source: {SRC}")
print(f"Annotation: {ANNOTATION}")

gcps = parse_gcps(ANNOTATION)

print(f"GCP count: {len(gcps)}")

with rasterio.open(SRC) as src:

    print("\nSource:")
    print(f"  Size: {src.width} x {src.height}")
    print(f"  dtype: {src.dtypes[0]}")
    print(f"  CRS: {src.crs}")

    # Confirm calibrated data actually exists
    sample = src.read(
        1,
        out_shape=(1, 1000, 1000),
        masked=False,
    ).astype(np.float32)

    valid_source = np.isfinite(sample) & (sample > 0)

    print("\nSource sample:")
    print(f"  min: {sample.min():.8f}")
    print(f"  max: {sample.max():.8f}")
    print(f"  mean: {sample.mean():.8f}")
    print(f"  nonzero: {np.count_nonzero(valid_source):,}")

    # Geographic AOI -> UTM
    left, bottom, right, top = transform_bounds(
        CRS.from_epsg(4326),
        DST_CRS,
        *BBOX,
    )

    resolution = 10.0

    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))

    transform = from_origin(
        left,
        top,
        resolution,
        resolution,
    )

    print("\nDestination:")
    print(f"  Geographic: {BBOX}")
    print(
        f"  UTM bounds: "
        f"{left:.2f}, {bottom:.2f}, "
        f"{right:.2f}, {top:.2f}"
    )
    print(f"  Size: {width} x {height}")
    print(f"  Pixels: {width * height:,}")

    destination = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    print("\nRunning GCP reproject...")

    reproject(
        source=rasterio.band(src, 1),
        destination=destination,
        gcps=gcps,
        src_crs=CRS.from_epsg(4326),
        dst_transform=transform,
        dst_crs=DST_CRS,
        resampling=Resampling.bilinear,
        num_threads=4,
        warp_mem_limit=1024,
        init_dest_nodata=True,
        SRC_METHOD="NO_GEOTRANSFORM",
    )

    valid = np.isfinite(destination) & (destination > 0)

    print("\n=== Result ===")
    print(f"Valid pixels: {valid.sum():,}")
    print(f"Total pixels: {destination.size:,}")
    print(f"Valid percentage: {100 * valid.mean():.4f}%")

    if valid.any():
        values = destination[valid]

        print(f"Min: {values.min():.8f}")
        print(f"Max: {values.max():.8f}")
        print(f"Mean: {values.mean():.8f}")

        print("\n*** CALIBRATED VV GCP WARP: PASS ***")
    else:
        print("\n*** CALIBRATED VV GCP WARP: FAIL ***")