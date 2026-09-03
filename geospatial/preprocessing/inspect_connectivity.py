import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from collections import Counter

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"

roads = gpd.read_file(ROAD_FILE)

print("=" * 60)
print("ROAD NETWORK CONNECTIVITY INSPECTION")
print("=" * 60)

print(f"Features: {len(roads)}")
print(f"CRS: {roads.crs}")


# --------------------------------------------------
# 1. EXACT DUPLICATE GEOMETRY
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXACT DUPLICATE GEOMETRY")
print("=" * 60)

geometry_wkt = roads.geometry.to_wkt()

duplicate_mask = geometry_wkt.duplicated(keep=False)

duplicates = roads[duplicate_mask]

print(f"Features involved: {len(duplicates)}")

if len(duplicates) > 0:
    print("\nDuplicate objectids:")
    print(duplicates["objectid"].tolist())


# --------------------------------------------------
# 2. MULTILINESTRING COUNT
# --------------------------------------------------

print("\n" + "=" * 60)
print("GEOMETRY TYPES")
print("=" * 60)

print(roads.geometry.geom_type.value_counts())


# --------------------------------------------------
# 3. CONVERT TO LINESTRING PARTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXPLODING MULTILINESTRINGS")
print("=" * 60)

segments = roads.explode(index_parts=False).copy()

segments = segments.reset_index(drop=True)

print(f"Original features: {len(roads)}")
print(f"After explode: {len(segments)}")


# --------------------------------------------------
# 4. GET START / END POINTS
# --------------------------------------------------

segments["start_point"] = segments.geometry.apply(
    lambda geom: Point(geom.coords[0])
)

segments["end_point"] = segments.geometry.apply(
    lambda geom: Point(geom.coords[-1])
)


# --------------------------------------------------
# 5. ENDPOINT CONNECTION CHECK
# --------------------------------------------------

print("\n" + "=" * 60)
print("ENDPOINT CONNECTION CHECK")
print("=" * 60)

points = []

for idx, row in segments.iterrows():

    points.append(
        ("start", idx, row.start_point.x, row.start_point.y)
    )

    points.append(
        ("end", idx, row.end_point.x, row.end_point.y)
    )


point_df = pd.DataFrame(
    points,
    columns=["type", "segment", "x", "y"]
)

# Exact coordinate grouping
point_counts = (
    point_df
    .groupby(["x", "y"])
    .size()
)

connected = point_counts[point_counts > 1]
isolated = point_counts[point_counts == 1]

print(f"Total endpoints: {len(point_df)}")
print(f"Unique endpoint locations: {len(point_counts)}")
print(f"Connected endpoint locations: {len(connected)}")
print(f"Isolated endpoint locations: {len(isolated)}")


# --------------------------------------------------
# 6. BASIC NETWORK STATISTICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("NETWORK SUMMARY")
print("=" * 60)

print(f"Segments: {len(segments)}")
print(f"Endpoint locations: {len(point_counts)}")
print(f"Locations shared by multiple segments: {len(connected)}")
print(f"Locations belonging to only one segment: {len(isolated)}")


# --------------------------------------------------
# 7. SAVE ENDPOINTS FOR VISUAL INSPECTION
# --------------------------------------------------

endpoint_gdf = gpd.GeoDataFrame(
    point_df,
    geometry=gpd.points_from_xy(
        point_df.x,
        point_df.y
    ),
    crs=roads.crs
)

endpoint_gdf.to_file(
    "data/processed/gcc/road_endpoints.geojson",
    driver="GeoJSON"
)

print("\nSaved:")
print("data/processed/gcc/road_endpoints.geojson")

print("\nInspection complete.")

print("\n" + "=" * 60)
print("EXACT DUPLICATE DETAILS")
print("=" * 60)

if len(duplicates) > 0:

    print(
        duplicates[
            [
                "objectid",
                "road_id",
                "road_name",
                "area_name",
                "locality",
                "zone",
                "ward",
                "department"
            ]
        ].to_string(index=False)
    )