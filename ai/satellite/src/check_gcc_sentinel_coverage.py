from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GCC_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "chennai"
    / "aoi"
    / "gcc_ward_boundary_2025.geojson"
)

SENTINEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "sentinel1"
    / "sentinel1_vv_vh_chennai_utm44n.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "aoi"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load official GCC wards
# ============================================================

print("=" * 70)
print("1. Loading official GCC 2025 ward boundary")
print("=" * 70)

gcc = gpd.read_file(GCC_FILE)

print(f"Wards loaded: {len(gcc)}")
print(f"GeoJSON CRS: {gcc.crs}")

# The downloaded GeoJSON coordinates are lon/lat.
# The authoritative GCC service identifies this layer as EPSG:32644,
# but the GeoJSON geometry itself is encoded as longitude/latitude.
if gcc.crs is None:
    gcc = gcc.set_crs(epsg=4326, allow_override=True)

# Reproject to the same CRS as Sentinel-1
gcc = gcc.to_crs(epsg=32644)

print(f"Reprojected CRS: {gcc.crs}")


# ============================================================
# 2. Validate and dissolve wards
# ============================================================

print("\n" + "=" * 70)
print("2. Creating single GCC boundary")
print("=" * 70)

valid_count = int(gcc.geometry.is_valid.sum())
print(f"Valid ward geometries: {valid_count}/{len(gcc)}")

if valid_count != len(gcc):
    print("WARNING: Some ward geometries are invalid. Repairing them...")
    gcc["geometry"] = gcc.geometry.make_valid()

gcc_boundary = gcc.geometry.union_all()

gcc_area_m2 = gcc_boundary.area
gcc_area_km2 = gcc_area_m2 / 1_000_000

print(f"GCC geometry type: {gcc_boundary.geom_type}")
print(f"GCC area: {gcc_area_km2:.6f} km²")
print(f"GCC bounds: {gcc_boundary.bounds}")


# ============================================================
# 3. Read Sentinel-1 raster
# ============================================================

print("\n" + "=" * 70)
print("3. Reading Sentinel-1 raster")
print("=" * 70)

with rasterio.open(SENTINEL_FILE) as src:

    print(f"CRS: {src.crs}")
    print(f"Resolution: {src.res}")
    print(f"Raster size: {src.width} x {src.height}")
    print(f"Bounds: {src.bounds}")

    if src.crs != gcc.crs:
        raise RuntimeError(
            f"CRS mismatch: Sentinel-1={src.crs}, GCC={gcc.crs}"
        )

    # Read VV band.
    # Our processed stack uses zero outside the valid SAR footprint.
    vv = src.read(1)

    transform = src.transform

    # Valid SAR pixels are non-zero.
    valid_mask = np.isfinite(vv) & (vv > 0)

    valid_pixel_count = int(valid_mask.sum())

    print(f"Valid Sentinel-1 pixels: {valid_pixel_count:,}")

    # Convert valid pixels into polygons.
    print("\nBuilding valid Sentinel-1 footprint...")

    footprint_parts = []

    for geom, value in shapes(
        valid_mask.astype(np.uint8),
        mask=valid_mask,
        transform=transform,
    ):
        if value == 1:
            footprint_parts.append(shape(geom))

    print(f"Footprint pieces: {len(footprint_parts):,}")

    sentinel_footprint = unary_union(footprint_parts)

    print(f"Sentinel footprint geometry: {sentinel_footprint.geom_type}")
    print(f"Sentinel footprint area: {sentinel_footprint.area / 1e6:.6f} km²")


# ============================================================
# 4. Intersect GCC with Sentinel-1 valid footprint
# ============================================================

print("\n" + "=" * 70)
print("4. Calculating GCC / Sentinel-1 coverage")
print("=" * 70)

covered_geometry = gcc_boundary.intersection(sentinel_footprint)

covered_area_m2 = covered_geometry.area
covered_area_km2 = covered_area_m2 / 1_000_000

coverage_percent = (
    covered_area_m2 / gcc_area_m2
) * 100.0

uncovered_area_km2 = gcc_area_km2 - covered_area_km2


print(f"GCC total area:        {gcc_area_km2:.6f} km²")
print(f"GCC covered area:      {covered_area_km2:.6f} km²")
print(f"GCC uncovered area:    {uncovered_area_km2:.6f} km²")
print(f"GCC coverage:          {coverage_percent:.4f}%")
print(f"GCC uncovered:         {100.0 - coverage_percent:.4f}%")


# ============================================================
# 5. Save boundary outputs
# ============================================================

print("\n" + "=" * 70)
print("5. Saving GIS outputs")
print("=" * 70)

gcc_gdf = gpd.GeoDataFrame(
    {"name": ["Greater Chennai Corporation"]},
    geometry=[gcc_boundary],
    crs="EPSG:32644",
)

covered_gdf = gpd.GeoDataFrame(
    {"name": ["GCC area covered by Sentinel-1"]},
    geometry=[covered_geometry],
    crs="EPSG:32644",
)

uncovered_geometry = gcc_boundary.difference(sentinel_footprint)

uncovered_gdf = gpd.GeoDataFrame(
    {"name": ["GCC area without valid Sentinel-1 coverage"]},
    geometry=[uncovered_geometry],
    crs="EPSG:32644",
)

gcc_output = OUTPUT_DIR / "gcc_boundary_2025_utm44n.geojson"
covered_output = OUTPUT_DIR / "gcc_sentinel1_covered.geojson"
uncovered_output = OUTPUT_DIR / "gcc_sentinel1_uncovered.geojson"

gcc_gdf.to_file(gcc_output, driver="GeoJSON")
covered_gdf.to_file(covered_output, driver="GeoJSON")
uncovered_gdf.to_file(uncovered_output, driver="GeoJSON")

print(f"Saved: {gcc_output}")
print(f"Saved: {covered_output}")
print(f"Saved: {uncovered_output}")


# ============================================================
# 6. Final summary
# ============================================================

print("\n" + "=" * 70)
print("FINAL COVERAGE RESULT")
print("=" * 70)

print(f"""
Greater Chennai Corporation:
    Area                : {gcc_area_km2:.4f} km²

Sentinel-1:
    Valid coverage     : {covered_area_km2:.4f} km²

Coverage:
    Covered            : {coverage_percent:.4f}%
    Uncovered          : {100.0 - coverage_percent:.4f}%

This percentage is calculated from the actual valid Sentinel-1
footprint, NOT merely from the rectangular raster bounds.
""")