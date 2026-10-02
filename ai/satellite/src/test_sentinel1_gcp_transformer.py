from pathlib import Path
import xml.etree.ElementTree as ET

from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[3]

ANNOTATION = (
    ROOT / "data" / "raw" / "chennai" / "sentinel1"
    / "cog_safe"
    / "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation"
    / "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
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


print("=== Sentinel-1 GCP Transformer Test ===")

gcps = parse_gcps(ANNOTATION)

print(f"GCP count: {len(gcps)}")

print("\nGCP geographic extent:")

lons = [g.x for g in gcps]
lats = [g.y for g in gcps]

print(f"Longitude: {min(lons):.6f} -> {max(lons):.6f}")
print(f"Latitude:  {min(lats):.6f} -> {max(lats):.6f}")


# Create GDAL/Rasterio GCP transformer
transformer = GCPTransformer(gcps)

# Geographic -> UTM
to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32644",
    always_xy=True,
)

# Test geographic locations around Chennai
test_points = [
    (80.20, 12.95),
    (80.25, 13.00),
    (80.30, 13.05),
    (80.00, 13.00),
    (79.50, 13.00),
]


print("\n=== Geographic -> UTM -> Image Pixel Test ===")

for lon, lat in test_points:

    x, y = to_utm.transform(lon, lat)

    try:
        row, col = transformer.rowcol(x, y)
    except Exception as e:
        print(
            f"{lon:.4f}, {lat:.4f} -> "
            f"ERROR: {e}"
        )
        continue

    print(
        f"Lon/Lat ({lon:.4f}, {lat:.4f})"
        f" -> UTM ({x:.1f}, {y:.1f})"
        f" -> pixel/line ({col}, {row})"
    )

    if (
        0 <= row <= 16759
        and 0 <= col <= 25320
    ):
        print("    INSIDE Sentinel-1 image")
    else:
        print("    OUTSIDE Sentinel-1 image")


print("\n=== Direct Image -> Geographic Test ===")

test_pixels = [
    (0, 0),
    (25320, 0),
    (0, 16759),
    (25320, 16759),
    (12660, 8380),
]

for col, row in test_pixels:

    try:
        x, y = transformer.xy(row, col)

        print(
            f"Pixel/line ({col}, {row})"
            f" -> Lon/Lat ({x:.6f}, {y:.6f})"
        )

    except Exception as e:

        print(
            f"Pixel/line ({col}, {row})"
            f" -> ERROR: {e}"
        )


print("\n=== Test Complete ===")