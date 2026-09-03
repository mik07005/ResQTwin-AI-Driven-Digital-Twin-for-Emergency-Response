import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union


ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"


print("=" * 60)
print("UNIQUE ROAD INTERSECTION ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

roads = gpd.read_file(ROAD_FILE)

segments = roads.explode(index_parts=False).copy()
segments = segments.reset_index(drop=True)

print("Segments:", len(segments))


# --------------------------------------------------
# PROJECT
# --------------------------------------------------

segments = segments.to_crs("EPSG:32644")


# --------------------------------------------------
# SPATIAL INDEX
# --------------------------------------------------

sindex = segments.sindex


# --------------------------------------------------
# FIND INTERSECTION POINTS
# --------------------------------------------------

intersection_points = []

pair_count = 0

for i, geom in enumerate(segments.geometry):

    if geom is None or geom.is_empty:
        continue

    candidates = sindex.query(
        geom,
        predicate="intersects"
    )

    for j in candidates:

        if i >= j:
            continue

        geom2 = segments.geometry.iloc[j]

        intersection = geom.intersection(geom2)

        if intersection.is_empty:
            continue

        pair_count += 1

        if intersection.geom_type == "Point":

            intersection_points.append(intersection)

        elif intersection.geom_type == "MultiPoint":

            intersection_points.extend(
                list(intersection.geoms)
            )


# --------------------------------------------------
# CREATE GEODATAFRAME
# --------------------------------------------------

print("\nIntersection pairs:", pair_count)
print("Raw intersection points:", len(intersection_points))


points_gdf = gpd.GeoDataFrame(
    geometry=intersection_points,
    crs="EPSG:32644"
)


# --------------------------------------------------
# EXACT UNIQUE POINTS
# --------------------------------------------------

points_gdf["x"] = points_gdf.geometry.x
points_gdf["y"] = points_gdf.geometry.y

exact_unique = points_gdf.drop_duplicates(
    subset=["x", "y"]
)

print(
    "Exact unique intersection points:",
    len(exact_unique)
)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

output = exact_unique[
    ["geometry"]
].to_crs("EPSG:4326")

output.to_file(
    "data/processed/gcc/intersection_points.geojson",
    driver="GeoJSON"
)


print("\nSaved:")
print(
    "data/processed/gcc/intersection_points.geojson"
)

print("\nAnalysis complete.")