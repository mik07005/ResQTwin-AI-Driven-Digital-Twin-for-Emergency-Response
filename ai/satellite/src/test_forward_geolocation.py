from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from rasterio.control import GroundControlPoint
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator


ROOT = Path(__file__).resolve().parents[3]

ANNOTATION = (
    ROOT / "data" / "raw" / "chennai" / "sentinel1"
    / "cog_safe"
    / "S1A_IW_GRDH_1SDV_20211116T003212_20211116T003237_040589_04D094_667C_COG.SAFE"
    / "annotation"
    / "s1a-iw-grd-vv-20211116t003212-20211116t003237-040589-04d094-001-cog.xml"
)

WIDTH = 25321
HEIGHT = 16760

# Chennai test points
TEST_POINTS = [
    (80.20, 12.95),
    (80.25, 13.00),
    (80.30, 13.05),
    (80.00, 13.00),
    (79.50, 13.00),
]


def parse_gcps(path):
    root = ET.parse(path).getroot()
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


print("=== Sentinel-1 Forward Geolocation Diagnostic ===")

gcps = parse_gcps(ANNOTATION)

print(f"GCP count: {len(gcps)}")

# ---------------------------------------------------------
# Build source pixel/line -> geographic interpolators
# ---------------------------------------------------------

cols = np.array([g.col for g in gcps], dtype=np.float64)
rows = np.array([g.row for g in gcps], dtype=np.float64)
lons = np.array([g.x for g in gcps], dtype=np.float64)
lats = np.array([g.y for g in gcps], dtype=np.float64)

points = np.column_stack([cols, rows])

lon_interp = LinearNDInterpolator(
    points,
    lons
)

lat_interp = LinearNDInterpolator(
    points,
    lats
)

print("\nSource GCP grid:")
print(f"  Columns: {sorted(set(cols))}")
print(f"  Rows:    {sorted(set(rows))}")

print("\nGeographic extent:")
print(f"  Longitude: {lons.min():.6f} -> {lons.max():.6f}")
print(f"  Latitude:  {lats.min():.6f} -> {lats.max():.6f}")


# ---------------------------------------------------------
# Build UTM transformation
# ---------------------------------------------------------

to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32644",
    always_xy=True,
)


# ---------------------------------------------------------
# Test forward interpolation at source pixels
# ---------------------------------------------------------

print("\n=== Source Pixel -> Geographic Test ===")

source_test_pixels = [
    (0, 0),
    (25320, 0),
    (0, 16759),
    (25320, 16759),
    (12660, 8380),
]

for col, row in source_test_pixels:

    lon = float(lon_interp(col, row))
    lat = float(lat_interp(col, row))

    print(
        f"Pixel/line ({col:5d}, {row:5d})"
        f" -> Lon/Lat ({lon:.6f}, {lat:.6f})"
    )


# ---------------------------------------------------------
# Find approximate source pixels corresponding to Chennai
# ---------------------------------------------------------

print("\n=== Chennai Geographic -> Approximate Source Pixel ===")

# Create a dense source grid.
# This is NOT the final raster. It is only a diagnostic.
grid_cols = np.linspace(0, WIDTH - 1, 1000)
grid_rows = np.linspace(0, HEIGHT - 1, 700)

cc, rr = np.meshgrid(grid_cols, grid_rows)

grid_lon = lon_interp(cc, rr)
grid_lat = lat_interp(cc, rr)

valid = np.isfinite(grid_lon) & np.isfinite(grid_lat)

print(f"Dense grid points: {grid_lon.size:,}")
print(f"Valid geolocation points: {valid.sum():,}")


for target_lon, target_lat in TEST_POINTS:

    distance = np.full(grid_lon.shape, np.inf)

    distance[valid] = (
        (grid_lon[valid] - target_lon) ** 2
        + (grid_lat[valid] - target_lat) ** 2
    )

    index = np.unravel_index(
        np.argmin(distance),
        distance.shape
    )

    nearest_lon = float(grid_lon[index])
    nearest_lat = float(grid_lat[index])

    nearest_col = float(cc[index])
    nearest_row = float(rr[index])

    error = np.sqrt(
        (nearest_lon - target_lon) ** 2
        + (nearest_lat - target_lat) ** 2
    )

    print(
        f"\nTarget Lon/Lat: "
        f"({target_lon:.4f}, {target_lat:.4f})"
    )

    print(
        f"Nearest source pixel: "
        f"({nearest_col:.1f}, {nearest_row:.1f})"
    )

    print(
        f"Forward mapped Lon/Lat: "
        f"({nearest_lon:.6f}, {nearest_lat:.6f})"
    )

    print(
        f"Approx geographic error: "
        f"{error:.8f} degrees"
    )

    if error < 0.01:
        print("STATUS: Chennai location found in image")
    else:
        print("STATUS: Large interpolation error")


print("\n=== Diagnostic Complete ===")