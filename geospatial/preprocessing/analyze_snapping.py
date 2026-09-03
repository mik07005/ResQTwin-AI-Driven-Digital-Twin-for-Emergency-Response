import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import cKDTree


ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"


# --------------------------------------------------
# LOAD ROADS
# --------------------------------------------------

roads = gpd.read_file(ROAD_FILE)

print("=" * 60)
print("ENDPOINT GAP ANALYSIS")
print("=" * 60)

print("Original features:", len(roads))
print("CRS:", roads.crs)


# --------------------------------------------------
# EXPLODE MULTILINESTRINGS
# --------------------------------------------------

segments = roads.explode(index_parts=False).copy()
segments = segments.reset_index(drop=True)

print("Segments after explode:", len(segments))


# --------------------------------------------------
# PROJECT TO METRIC CRS
# --------------------------------------------------

# UTM Zone 44N is appropriate for Chennai
segments = segments.to_crs("EPSG:32644")

print("Projected CRS:", segments.crs)


# --------------------------------------------------
# EXTRACT ENDPOINTS
# --------------------------------------------------

points = []

for idx, geom in enumerate(segments.geometry):

    if geom is None or geom.is_empty:
        continue

    coords = list(geom.coords)

    start = coords[0]
    end = coords[-1]

    points.append((idx, start[0], start[1]))
    points.append((idx, end[0], end[1]))


points_df = pd.DataFrame(
    points,
    columns=["segment", "x", "y"]
)

print("Total endpoints:", len(points_df))


# --------------------------------------------------
# BUILD KD-TREE
# --------------------------------------------------

coordinates = points_df[["x", "y"]].values

tree = cKDTree(coordinates)


# --------------------------------------------------
# FIND NEAREST OTHER ENDPOINT
# --------------------------------------------------

distances, indices = tree.query(
    coordinates,
    k=2
)

# k=2 because nearest point is the point itself
nearest_distances = distances[:, 1]


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("NEAREST ENDPOINT DISTANCE")
print("=" * 60)

print(
    pd.Series(nearest_distances).describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)


# --------------------------------------------------
# TEST TOLERANCES
# --------------------------------------------------

print("\n" + "=" * 60)
print("SNAPPING TOLERANCE ANALYSIS")
print("=" * 60)

tolerances = [
    0.5,
    1,
    2,
    3,
    5,
    10
]

for tolerance in tolerances:

    count = (nearest_distances <= tolerance).sum()

    percentage = (
        count / len(nearest_distances)
    ) * 100

    print(
        f"{tolerance:>5} m : "
        f"{count:>6} endpoints "
        f"({percentage:.2f}%)"
    )


print("\nAnalysis complete.")