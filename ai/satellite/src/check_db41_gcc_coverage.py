from pathlib import Path
import geopandas as gpd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]

GCC_UNCOVERED = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "aoi"
    / "gcc_sentinel1_uncovered.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chennai"
    / "aoi"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# DB41 exact catalogue footprint
# ---------------------------------------------------------

db41_coords = [[
    [80.737976, 13.045553],
    [81.049484, 14.557140],
    [78.741203, 14.995490],
    [78.445488, 13.487610],
    [80.737976, 13.045553],
]]

db41 = gpd.GeoDataFrame(
    {"product": ["DB41"]},
    geometry=[
        gpd.GeoSeries.from_wkt([
            "POLYGON (("
            "80.737976 13.045553,"
            "81.049484 14.557140,"
            "78.741203 14.995490,"
            "78.445488 13.487610,"
            "80.737976 13.045553"
            "))"
        ])[0]
    ],
    crs="EPSG:4326",
)

# ---------------------------------------------------------
# Load exact GCC area that B07B does NOT cover
# ---------------------------------------------------------

uncovered = gpd.read_file(GCC_UNCOVERED)

print("\n=== INPUT ===")
print(f"Uncovered GCC geometry: {GCC_UNCOVERED}")
print(f"CRS: {uncovered.crs}")

# Make sure both are in common UTM 44N grid
uncovered = uncovered.to_crs("EPSG:32644")
db41 = db41.to_crs("EPSG:32644")

# ---------------------------------------------------------
# Calculate intersection
# ---------------------------------------------------------

intersection = gpd.overlay(
    uncovered,
    db41,
    how="intersection"
)

uncovered_area = uncovered.geometry.area.sum() / 1_000_000
covered_area = intersection.geometry.area.sum() / 1_000_000

remaining = uncovered_area - covered_area

coverage_pct = (
    covered_area / uncovered_area * 100
    if uncovered_area > 0
    else 0
)

print("\n=== DB41 COVERAGE OF MISSING GCC AREA ===")
print(f"B07B uncovered GCC area : {uncovered_area:.6f} km²")
print(f"DB41 covers              : {covered_area:.6f} km²")
print(f"Remaining uncovered      : {remaining:.6f} km²")
print(f"DB41 coverage            : {coverage_pct:.4f}%")

# ---------------------------------------------------------
# Save geometries
# ---------------------------------------------------------

if not intersection.empty:
    intersection.to_file(
        OUTPUT_DIR / "db41_covers_b07b_gap.geojson",
        driver="GeoJSON"
    )

# Calculate the part still not covered by DB41
remaining_geom = gpd.overlay(
    uncovered,
    db41,
    how="difference"
)

if not remaining_geom.empty:
    remaining_geom.to_file(
        OUTPUT_DIR / "gcc_remaining_gap_after_db41.geojson",
        driver="GeoJSON"
    )

print("\n=== OUTPUTS ===")

if not intersection.empty:
    print(
        OUTPUT_DIR
        / "db41_covers_b07b_gap.geojson"
    )

if not remaining_geom.empty:
    print(
        OUTPUT_DIR
        / "gcc_remaining_gap_after_db41.geojson"
    )

print("\nDone.")