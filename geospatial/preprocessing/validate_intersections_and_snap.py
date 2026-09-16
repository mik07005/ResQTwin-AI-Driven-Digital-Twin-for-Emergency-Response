import geopandas as gpd
import pandas as pd
import numpy as np

from shapely.geometry import Point, LineString
from shapely.ops import split
from shapely.strtree import STRtree
from collections import defaultdict
import networkx as nx


# ============================================================
# PATHS
# ============================================================

EDGE_FILE = "data/processed/gcc/road_edges_clean.geojson"
NODE_FILE = "data/processed/gcc/road_nodes_clean.geojson"
INTERSECTION_FILE = "data/processed/gcc/intersection_points.geojson"

EDGE_OUTPUT = "data/processed/gcc/road_edges_intersection_aware.geojson"
NODE_OUTPUT = "data/processed/gcc/road_nodes_intersection_aware.geojson"
SNAP_OUTPUT = "data/processed/gcc/snapped_connections.geojson"


# ============================================================
# PARAMETERS
# ============================================================

# Very small gaps are highly likely to be digitization errors.
STRONG_SNAP_TOLERANCE = 0.5

# Candidate snapping tolerance.
MAX_SNAP_TOLERANCE = 2.0

# Intersection point -> edge tolerance.
INTERSECTION_TOLERANCE = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("INTERSECTION-AWARE + CONTROLLED SNAPPING VALIDATION")
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading clean road graph...")

edges = gpd.read_file(EDGE_FILE)
nodes = gpd.read_file(NODE_FILE)
intersections = gpd.read_file(INTERSECTION_FILE)

print("Clean edges:", len(edges))
print("Clean nodes:", len(nodes))
print("Intersection points:", len(intersections))


# ============================================================
# PROJECT TO METRIC CRS
# ============================================================

edges = edges.to_crs("EPSG:32644")
nodes = nodes.to_crs("EPSG:32644")
intersections = intersections.to_crs("EPSG:32644")

print("Working CRS: EPSG:32644")


# ============================================================
# VALIDATE GEOMETRIES
# ============================================================

edges = edges[
    edges.geometry.notna()
    & ~edges.geometry.is_empty
].copy()

nodes = nodes[
    nodes.geometry.notna()
    & ~nodes.geometry.is_empty
].copy()

intersections = intersections[
    intersections.geometry.notna()
    & ~intersections.geometry.is_empty
].copy()


# ============================================================
# 1. INTERSECTION → EXISTING NODE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INTERSECTION NODE VALIDATION")
print("=" * 60)

node_tree = STRtree(nodes.geometry.values)

intersection_existing = 0
intersection_missing = 0

intersection_node_distances = []

for point in intersections.geometry:

    candidates = node_tree.query(
        point.buffer(INTERSECTION_TOLERANCE)
    )

    if len(candidates) == 0:
        intersection_missing += 1
        continue

    min_distance = min(
        point.distance(nodes.geometry.iloc[i])
        for i in candidates
    )

    intersection_node_distances.append(min_distance)

    if min_distance <= INTERSECTION_TOLERANCE:
        intersection_existing += 1
    else:
        intersection_missing += 1


print(
    "Intersections already represented by nodes:",
    intersection_existing
)

print(
    "Intersections not represented by nodes:",
    intersection_missing
)

if intersection_node_distances:
    print(
        "Average intersection-node distance:",
        round(np.mean(intersection_node_distances), 3),
        "m"
    )


# ============================================================
# 2. FIND INTERSECTIONS LYING ON EDGES
# ============================================================

print("\n" + "=" * 60)
print("INTERSECTION → EDGE VALIDATION")
print("=" * 60)

edge_tree = STRtree(edges.geometry.values)

intersection_on_edge = 0
intersection_not_on_edge = 0

edge_split_candidates = []

for idx, point in enumerate(intersections.geometry):

    candidates = edge_tree.query(
        point.buffer(INTERSECTION_TOLERANCE)
    )

    found = False

    for edge_idx in candidates:

        edge_geom = edges.geometry.iloc[edge_idx]

        distance = point.distance(edge_geom)

        if distance <= INTERSECTION_TOLERANCE:

            # Check whether point is actually inside
            # the edge rather than simply at its endpoint.
            projected = edge_geom.project(point)

            if (
                projected > INTERSECTION_TOLERANCE
                and
                projected < edge_geom.length - INTERSECTION_TOLERANCE
            ):
                edge_split_candidates.append(
                    (
                        edge_idx,
                        point
                    )
                )

                found = True
                break

    if found:
        intersection_on_edge += 1
    else:
        intersection_not_on_edge += 1


print(
    "Intersections requiring edge splitting:",
    intersection_on_edge
)

print(
    "Intersections not requiring edge splitting:",
    intersection_not_on_edge
)


# ============================================================
# 3. CONTROLLED ENDPOINT SNAP ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CONTROLLED ENDPOINT SNAP ANALYSIS")
print("=" * 60)

endpoint_records = []

for edge_idx, geom in enumerate(edges.geometry):

    if geom is None or geom.is_empty:
        continue

    start = Point(geom.coords[0])
    end = Point(geom.coords[-1])

    endpoint_records.append(
        {
            "edge_idx": edge_idx,
            "side": "start",
            "geometry": start
        }
    )

    endpoint_records.append(
        {
            "edge_idx": edge_idx,
            "side": "end",
            "geometry": end
        }
    )


endpoint_gdf = gpd.GeoDataFrame(
    endpoint_records,
    geometry="geometry",
    crs="EPSG:32644"
)

endpoint_tree = STRtree(endpoint_gdf.geometry.values)


# ============================================================
# FIND SNAP CANDIDATES
# ============================================================

snap_records = []

strong_candidates = 0
weak_candidates = 0

for idx, endpoint in enumerate(endpoint_gdf.geometry):

    candidates = endpoint_tree.query(
        endpoint.buffer(MAX_SNAP_TOLERANCE)
    )

    best_candidate = None
    best_distance = float("inf")

    for candidate_idx in candidates:

        if candidate_idx == idx:
            continue

        candidate = endpoint_gdf.iloc[candidate_idx]

        # Never snap an endpoint to another endpoint
        # belonging to the same edge.
        if candidate.edge_idx == endpoint_gdf.iloc[idx].edge_idx:
            continue

        distance = endpoint.distance(
            candidate.geometry
        )

        if (
            distance > 0
            and
            distance < best_distance
        ):
            best_distance = distance
            best_candidate = candidate_idx

    if best_candidate is None:
        continue

    candidate = endpoint_gdf.iloc[best_candidate]

    if best_distance <= STRONG_SNAP_TOLERANCE:

        strong_candidates += 1

        snap_records.append(
            {
                "endpoint_idx": idx,
                "candidate_idx": best_candidate,
                "edge_a": endpoint_gdf.iloc[idx].edge_idx,
                "edge_b": candidate.edge_idx,
                "distance_m": best_distance,
                "confidence": "strong",
                "geometry": endpoint
            }
        )

    elif best_distance <= MAX_SNAP_TOLERANCE:

        weak_candidates += 1

        snap_records.append(
            {
                "endpoint_idx": idx,
                "candidate_idx": best_candidate,
                "edge_a": endpoint_gdf.iloc[idx].edge_idx,
                "edge_b": candidate.edge_idx,
                "distance_m": best_distance,
                "confidence": "candidate",
                "geometry": endpoint
            }
        )


print(
    "Strong snap candidates (<= 0.5 m):",
    strong_candidates
)

print(
    "Additional candidates (0.5–2 m):",
    weak_candidates
)

print(
    "Total snap candidates:",
    len(snap_records)
)


# ============================================================
# 4. REMOVE DUPLICATE SNAP PAIRS
# ============================================================

unique_snap_pairs = {}

for record in snap_records:

    edge_a = record["edge_a"]
    edge_b = record["edge_b"]

    pair = tuple(
        sorted([edge_a, edge_b])
    )

    if pair not in unique_snap_pairs:

        unique_snap_pairs[pair] = record


snap_records = list(
    unique_snap_pairs.values()
)


print(
    "Unique candidate connections:",
    len(snap_records)
)


# ============================================================
# 5. SAVE SNAP CANDIDATES
# ============================================================

if snap_records:

    snap_gdf = gpd.GeoDataFrame(
        snap_records,
        geometry="geometry",
        crs="EPSG:32644"
    )

else:

    snap_gdf = gpd.GeoDataFrame(
        columns=[
            "endpoint_idx",
            "candidate_idx",
            "edge_a",
            "edge_b",
            "distance_m",
            "confidence",
            "geometry"
        ],
        geometry="geometry",
        crs="EPSG:32644"
    )


snap_gdf.to_crs("EPSG:4326").to_file(
    SNAP_OUTPUT,
    driver="GeoJSON"
)

print("\nSaved snap candidates:")
print(SNAP_OUTPUT)


# ============================================================
# 6. GRAPH CONNECTIVITY BEFORE CHANGES
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY BASELINE")
print("=" * 60)

G_before = nx.MultiGraph()

for _, row in edges.iterrows():

    G_before.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        edge_id=int(row["edge_id"])
    )


components_before = list(
    nx.connected_components(G_before)
)

components_before.sort(
    key=len,
    reverse=True
)

largest_before = len(
    components_before[0]
)

print(
    "Connected components:",
    len(components_before)
)

print(
    "Largest component:",
    largest_before
)

print(
    "Largest component coverage:",
    round(
        largest_before / len(nodes) * 100,
        2
    ),
    "%"
)


# ============================================================
# 7. INTERSECTION SPLIT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INTERSECTION SPLIT SUMMARY")
print("=" * 60)

print(
    "Valid intersection points:",
    len(intersections)
)

print(
    "Intersection points lying inside edges:",
    intersection_on_edge
)

print(
    "Potential edge-splitting locations:",
    len(edge_split_candidates)
)


# ============================================================
# IMPORTANT SAFETY CHECK
# ============================================================

print("\n" + "=" * 60)
print("SAFETY CHECK")
print("=" * 60)

print(
    "No automatic snapping has been applied."
)

print(
    "No road geometry has been modified."
)

print(
    "Candidate connections were exported for validation."
)

print(
    "\nReason:"
)

print(
    "Distance-only snapping can incorrectly connect "
    "parallel roads, divided roads, service lanes, or "
    "nearby but unrelated road segments."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("INTERSECTION + SNAPPING VALIDATION COMPLETE")
print("=" * 60)

print("Clean edges:", len(edges))
print("Clean nodes:", len(nodes))

print(
    "Intersections:",
    len(intersections)
)

print(
    "Intersections already represented by nodes:",
    intersection_existing
)

print(
    "Intersections requiring potential splitting:",
    intersection_on_edge
)

print(
    "Strong snap candidates:",
    strong_candidates
)

print(
    "Additional snap candidates:",
    weak_candidates
)

print(
    "Current connected components:",
    len(components_before)
)

print(
    "Current largest component:",
    largest_before
)

print(
    "Largest component coverage:",
    round(
        largest_before / len(nodes) * 100,
        2
    ),
    "%"
)

print("\nOutput:")
print(SNAP_OUTPUT)

print("\nNext step:")
print(
    "Review candidate snapping + split edges, "
    "then build the final routing graph."
)