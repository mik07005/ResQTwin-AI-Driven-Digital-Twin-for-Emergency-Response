import geopandas as gpd
import pandas as pd


ROAD_FILE = "data/raw/gcc/roads_centreline.geojson"


print("=" * 60)
print("ROAD INTERSECTION ANALYSIS")
print("=" * 60)


# --------------------------------------------------
# LOAD
# --------------------------------------------------

roads = gpd.read_file(ROAD_FILE)

print("Original features:", len(roads))
print("CRS:", roads.crs)


# --------------------------------------------------
# EXPLODE MULTILINESTRINGS
# --------------------------------------------------

segments = roads.explode(index_parts=False).copy()

segments = segments.reset_index(drop=True)

print("Segments:", len(segments))


# --------------------------------------------------
# PROJECT TO METRIC CRS
# --------------------------------------------------

segments = segments.to_crs("EPSG:32644")

print("Projected CRS:", segments.crs)


# --------------------------------------------------
# SPATIAL INDEX
# --------------------------------------------------

print("\nBuilding spatial index...")

sindex = segments.sindex


# --------------------------------------------------
# FIND INTERSECTIONS
# --------------------------------------------------

intersection_pairs = []

for i, geom in enumerate(segments.geometry):

    if geom is None or geom.is_empty:
        continue

    candidates = list(
        sindex.query(
            geom,
            predicate="intersects"
        )
    )

    for j in candidates:

        # Avoid self
        if i == j:
            continue

        # Store each pair only once
        if i < j:
            intersection_pairs.append((i, j))


print("\nCandidate/intersecting pairs:", len(intersection_pairs))


# --------------------------------------------------
# CLASSIFY INTERSECTIONS
# --------------------------------------------------

crossings = []
touches = []
overlaps = []

for i, j in intersection_pairs:

    geom1 = segments.geometry.iloc[i]
    geom2 = segments.geometry.iloc[j]

    intersection = geom1.intersection(geom2)

    if intersection.is_empty:
        continue

    if intersection.geom_type in ["Point", "MultiPoint"]:
        crossings.append((i, j))

    elif intersection.geom_type in [
        "LineString",
        "MultiLineString"
    ]:
        overlaps.append((i, j))

    else:
        touches.append((i, j))


print("\n" + "=" * 60)
print("INTERSECTION RESULTS")
print("=" * 60)

print("Point/MultiPoint intersections:", len(crossings))
print("Line overlaps:", len(overlaps))
print("Other intersections:", len(touches))


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nAnalysis complete.")