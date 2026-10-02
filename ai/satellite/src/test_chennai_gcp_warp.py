from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]

SRC = ROOT / (
    "data/raw/chennai/sentinel1/"
    "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.tiff"
)

XML = ROOT / (
    "data/raw/chennai/sentinel1/cog_safe/"
    "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE/"
    "annotation/"
    "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)

OUT = ROOT / "data/processed/chennai/sentinel1/test_chennai_gcp_warp.tif"


# Small Chennai validation AOI.
# Approximately 11 km x 11 km.
MIN_LON, MAX_LON = 80.20, 80.30
MIN_LAT, MAX_LAT = 12.95, 13.05

DST_CRS = CRS.from_epsg(32644)  # WGS84 / UTM zone 44N
RESOLUTION = 10.0


def parse_gcps(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Sentinel-1 annotation uses namespaces, so search by local tag name.
    gcps = []

    for elem in root.iter():
        if elem.tag.endswith("geolocationGridPoint"):
            line = None
            pixel = None
            lat = None
            lon = None

            for child in elem.iter():
                tag = child.tag.split("}")[-1]

                if tag == "line":
                    line = int(child.text)
                elif tag == "pixel":
                    pixel = int(child.text)
                elif tag == "latitude":
                    lat = float(child.text)
                elif tag == "longitude":
                    lon = float(child.text)

            if None not in (line, pixel, lat, lon):
                gcps.append(
                    GroundControlPoint(
                        row=line,
                        col=pixel,
                        x=lon,
                        y=lat,
                    )
                )

    return gcps


def main():
    print("=== ResQTwin Chennai GCP Warp Test ===")
    print(f"Source: {SRC}")
    print(f"XML:    {XML}")

    if not SRC.exists():
        raise FileNotFoundError(f"Source COG not found: {SRC}")

    if not XML.exists():
        raise FileNotFoundError(f"Annotation XML not found: {XML}")

    gcps = parse_gcps(XML)

    print(f"\nGCP count: {len(gcps)}")

    if len(gcps) != 210:
        raise RuntimeError(
            f"Expected 210 GCPs, found {len(gcps)}"
        )

    with rasterio.open(SRC) as src:
        print(f"Source size: {src.width} x {src.height}")
        print(f"Source dtype: {src.dtypes[0]}")

        # Convert the requested geographic AOI to UTM 44N.
        transformer = Transformer.from_crs(
            "EPSG:4326",
            DST_CRS,
            always_xy=True,
        )

        min_x, min_y = transformer.transform(MIN_LON, MIN_LAT)
        max_x, max_y = transformer.transform(MAX_LON, MAX_LAT)

        width = int(np.ceil((max_x - min_x) / RESOLUTION))
        height = int(np.ceil((max_y - min_y) / RESOLUTION))

        dst_transform = from_origin(
            min_x,
            max_y,
            RESOLUTION,
            RESOLUTION,
        )

        print("\nDestination:")
        print(f"  CRS: EPSG:32644")
        print(f"  Bounds: {min_x:.2f}, {min_y:.2f}, {max_x:.2f}, {max_y:.2f}")
        print(f"  Size: {width} x {height}")
        print(f"  Resolution: {RESOLUTION} m")

        destination = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        print("\nRunning GCP warp...")

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            gcps=gcps,
            src_crs=CRS.from_epsg(4326),
            dst_transform=dst_transform,
            dst_crs=DST_CRS,
            resampling=Resampling.bilinear,
            num_threads=4,
            warp_mem_limit=1024,
            init_dest_nodata=True,
            dst_nodata=0,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": DST_CRS,
        "transform": dst_transform,
        "nodata": 0,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(destination, 1)

    valid = destination != 0

    print("\n=== RESULT ===")
    print(f"Output: {OUT}")
    print(f"CRS: {DST_CRS}")
    print(f"Size: {width} x {height}")
    print(f"Valid pixels: {valid.sum():,}")
    print(f"Valid fraction: {valid.mean() * 100:.2f}%")

    if valid.any():
        values = destination[valid]

        print(f"Min: {values.min():.3f}")
        print(f"Max: {values.max():.3f}")
        print(f"Mean: {values.mean():.3f}")

        print("\nGCP/TPS validation produced non-empty output.")
    else:
        raise RuntimeError(
            "Warp completed but produced no valid pixels."
        )


if __name__ == "__main__":
    main()