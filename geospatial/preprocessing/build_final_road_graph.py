import geopandas as gpd
import pandas as pd
import networkx as nx

from shapely.geometry import LineString


# ============================================================
# PATHS
# ============================================================

EDGE_FILE = "data/processed/gcc/road_edges_clean.geojson"
NODE_FILE = "data/processed/gcc/road_nodes_clean.geojson"

SNAP_FILE = "data/processed/gcc/snapped_connections.geojson"

FINAL_NODE_OUTPUT = "data/processed/gcc/road_nodes_final.geojson"
FINAL_EDGE_OUTPUT = "data/processed/gcc/road_edges_final.geojson"

ACCEPTED_OUTPUT = "data/processed/gcc/accepted_snaps.geojson"
REJECTED_OUTPUT = "data/processed/gcc/rejected_snaps.geojson"


WORKING_CRS = "EPSG:32644"

# ------------------------------------------------------------
# SNAP THRESHOLDS
# ------------------------------------------------------------

STRONG_THRESHOLD = 0.5

# Never accept a candidate above this distance automatically.
MAX_AUTO_SNAP = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("FINAL GCC ROAD GRAPH CONSTRUCTION")
print("=" * 60)


# ============================================================
# LOAD CLEAN GRAPH
# ============================================================

print("\nLoading clean road graph...")

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

print("Clean edges:", len(edges))
print("Clean nodes:", len(nodes))
print("CRS:", edges.crs)


# ============================================================
# LOAD SNAP CANDIDATES
# ============================================================

print("\nLoading snapping candidates...")

snaps = gpd.read_file(SNAP_FILE)

print("Snap candidates:", len(snaps))


# ============================================================
# PROJECT EVERYTHING
# ============================================================

nodes = nodes.to_crs(WORKING_CRS)
edges = edges.to_crs(WORKING_CRS)
snaps = snaps.to_crs(WORKING_CRS)


# ============================================================
# BASIC EDGE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INPUT EDGE VALIDATION")
print("=" * 60)

invalid_geometry = (
    edges.geometry.isna()
    | edges.geometry.is_empty
)

invalid_from = (
    edges["from_node"].isna()
)

invalid_to = (
    edges["to_node"].isna()
)

node_ids = set(nodes["node_id"].astype(int))

invalid_from_refs = ~edges["from_node"].astype(int).isin(node_ids)
invalid_to_refs = ~edges["to_node"].astype(int).isin(node_ids)

print(
    "Invalid/empty geometries:",
    int(invalid_geometry.sum())
)

print(
    "Invalid from_node references:",
    int((invalid_from | invalid_from_refs).sum())
)

print(
    "Invalid to_node references:",
    int((invalid_to | invalid_to_refs).sum())
)


# ============================================================
# CLEAN EDGE TABLE
# ============================================================

edges = edges[
    ~invalid_geometry
    & ~invalid_from
    & ~invalid_to
    & ~invalid_from_refs
    & ~invalid_to_refs
].copy()

edges["from_node"] = edges["from_node"].astype(int)
edges["to_node"] = edges["to_node"].astype(int)

edges = edges.reset_index(drop=True)

print("Valid edges:", len(edges))


# ============================================================
# NODE COORDINATE LOOKUP
# ============================================================

print("\nBuilding node coordinate lookup...")

node_lookup = {}

for _, row in nodes.iterrows():

    node_id = int(row["node_id"])

    x = row.geometry.x
    y = row.geometry.y

    node_lookup[node_id] = (x, y)

print("Node coordinate mappings:", len(node_lookup))


# ============================================================
# EXISTING GRAPH CONNECTION LOOKUP
# ============================================================

print("\nBuilding existing edge connection lookup...")

existing_pairs = set()

for _, row in edges.iterrows():

    a = int(row["from_node"])
    b = int(row["to_node"])

    if a == b:
        continue

    pair = tuple(sorted((a, b)))

    existing_pairs.add(pair)

print("Existing unique node connections:", len(existing_pairs))



# ============================================================
# EDGE ID → NODE ENDPOINT LOOKUP
# ============================================================

edge_id_to_nodes = {}

for _, row in edges.iterrows():

    edge_id = int(row["edge_id"])

    edge_id_to_nodes[edge_id] = (
        int(row["from_node"]),
        int(row["to_node"])
    )

print(
    "Edge endpoint mappings:",
    len(edge_id_to_nodes)
)


# ============================================================
# SNAP CANDIDATE CLASSIFICATION
# ============================================================

print("\n" + "=" * 60)
print("SNAP CANDIDATE CLASSIFICATION")
print("=" * 60)

accepted_records = []
rejected_records = []

invalid_candidates = 0
already_connected = 0
same_node = 0
distance_rejected = 0

seen_pairs = set()


for idx, row in snaps.iterrows():

    distance = float(row["distance_m"])

    edge_a = row.get("edge_a")
    edge_b = row.get("edge_b")

    # --------------------------------------------------------
    # Validate edge references
    # --------------------------------------------------------

    try:

        edge_a = int(edge_a)
        edge_b = int(edge_b)

    except (TypeError, ValueError):

        invalid_candidates += 1

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "distance_m": distance,
            "reason": "invalid_edge_reference",
            "geometry": row.geometry
        })

        continue


    # Edge IDs are stored in edge_id column.
    # Build lookup lazily below.
    if edge_a not in edge_id_to_nodes or edge_b not in edge_id_to_nodes:

        invalid_candidates += 1

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "distance_m": distance,
            "reason": "edge_id_not_found",
            "geometry": row.geometry
        })

        continue


    # --------------------------------------------------------
    # Candidate endpoint nodes
    # --------------------------------------------------------

    a_nodes = edge_id_to_nodes[edge_a]
    b_nodes = edge_id_to_nodes[edge_b]


    # Candidate may involve any endpoint combination.
    possible_pairs = []

    for a in a_nodes:

        for b in b_nodes:

            if a == b:
                same_node += 1
                continue

            possible_pairs.append((a, b))


    if not possible_pairs:

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "distance_m": distance,
            "reason": "same_node",
            "geometry": row.geometry
        })

        continue


    # --------------------------------------------------------
    # Find the closest endpoint pair
    # --------------------------------------------------------

    best_pair = None
    best_distance = float("inf")

    for a, b in possible_pairs:

        if a not in node_lookup or b not in node_lookup:
            continue

        ax, ay = node_lookup[a]
        bx, by = node_lookup[b]

        dx = ax - bx
        dy = ay - by

        d = (dx * dx + dy * dy) ** 0.5

        if d < best_distance:

            best_distance = d
            best_pair = (a, b)


    if best_pair is None:

        invalid_candidates += 1

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "distance_m": distance,
            "reason": "no_valid_endpoint_pair",
            "geometry": row.geometry
        })

        continue


    node_a, node_b = best_pair

    pair = tuple(sorted((node_a, node_b)))


    # --------------------------------------------------------
    # Distance validation
    # --------------------------------------------------------

    if best_distance > MAX_AUTO_SNAP:

        distance_rejected += 1

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "node_a": node_a,
            "node_b": node_b,
            "distance_m": best_distance,
            "reason": "distance_above_auto_threshold",
            "geometry": row.geometry
        })

        continue


    # --------------------------------------------------------
    # Existing connection
    # --------------------------------------------------------

    if pair in existing_pairs:

        already_connected += 1

        rejected_records.append({
            "candidate_id": idx,
            "edge_a": edge_a,
            "edge_b": edge_b,
            "node_a": node_a,
            "node_b": node_b,
            "distance_m": best_distance,
            "reason": "already_connected",
            "geometry": row.geometry
        })

        continue


    # --------------------------------------------------------
    # Duplicate candidate
    # --------------------------------------------------------

    if pair in seen_pairs:
        continue

    seen_pairs.add(pair)


    # --------------------------------------------------------
    # ACCEPT
    # --------------------------------------------------------

    accepted_records.append({
        "candidate_id": idx,
        "edge_a": edge_a,
        "edge_b": edge_b,
        "node_a": node_a,
        "node_b": node_b,
        "distance_m": best_distance,
        "confidence": "strong",
        "geometry": row.geometry
    })


# ============================================================
# CLASSIFICATION SUMMARY
# ============================================================

print("Raw snap candidates:", len(snaps))
print("Strong accepted candidates:", len(accepted_records))
print(
    "Rejected/review candidates:",
    len(rejected_records)
)
print("Invalid candidates:", invalid_candidates)
print("Already-connected candidates:", already_connected)
print("Same-node candidates:", same_node)
print("Distance-rejected candidates:", distance_rejected)


# ============================================================
# CREATE ACCEPTED / REJECTED GEODATAFRAMES
# ============================================================

accepted = gpd.GeoDataFrame(
    accepted_records,
    geometry="geometry",
    crs=WORKING_CRS
)

rejected = gpd.GeoDataFrame(
    rejected_records,
    geometry="geometry",
    crs=WORKING_CRS
)


# ============================================================
# SAVE CLASSIFICATION
# ============================================================

print("\nSaving snap classification files...")

if len(accepted) > 0:

    accepted.to_crs("EPSG:4326").to_file(
        ACCEPTED_OUTPUT,
        driver="GeoJSON"
    )

else:

    gpd.GeoDataFrame(
        columns=[
            "candidate_id",
            "edge_a",
            "edge_b",
            "node_a",
            "node_b",
            "distance_m",
            "confidence",
            "geometry"
        ],
        geometry="geometry",
        crs="EPSG:4326"
    ).to_file(
        ACCEPTED_OUTPUT,
        driver="GeoJSON"
    )


if len(rejected) > 0:

    rejected.to_crs("EPSG:4326").to_file(
        REJECTED_OUTPUT,
        driver="GeoJSON"
    )


# ============================================================
# BUILD SNAP CONNECTOR EDGES
# ============================================================

print("\n" + "=" * 60)
print("BUILDING SNAP CONNECTOR EDGES")
print("=" * 60)

connector_records = []

for connector_id, row in accepted.iterrows():

    node_a = int(row["node_a"])
    node_b = int(row["node_b"])

    if node_a not in node_lookup:
        continue

    if node_b not in node_lookup:
        continue

    ax, ay = node_lookup[node_a]
    bx, by = node_lookup[node_b]

    geometry = LineString([
        (ax, ay),
        (bx, by)
    ])

    length = geometry.length

    if length <= 0:
        continue

    connector_records.append({
        "edge_id": f"snap_{connector_id}",
        "from_node": node_a,
        "to_node": node_b,
        "length_m": length,
        "edge_type": "snap_connector",
        "snap_distance_m": length,
        "geometry": geometry
    })


connectors = gpd.GeoDataFrame(
    connector_records,
    geometry="geometry",
    crs=WORKING_CRS
)

print("Snap connectors created:", len(connectors))


# ============================================================
# PREPARE ORIGINAL EDGES
# ============================================================

original_edges = edges.copy()

original_edges["edge_type"] = "road"

original_edges["snap_distance_m"] = 0.0


# ============================================================
# COMBINE ORIGINAL + CONNECTOR EDGES
# ============================================================

final_edges = pd.concat(
    [
        original_edges[
            [
                "edge_id",
                "from_node",
                "to_node",
                "length_m",
                "edge_type",
                "snap_distance_m",
                "geometry"
            ]
        ],
        connectors[
            [
                "edge_id",
                "from_node",
                "to_node",
                "length_m",
                "edge_type",
                "snap_distance_m",
                "geometry"
            ]
        ]
    ],
    ignore_index=True
)

final_edges = gpd.GeoDataFrame(
    final_edges,
    geometry="geometry",
    crs=WORKING_CRS
)


# ============================================================
# FINAL EDGE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL EDGE VALIDATION")
print("=" * 60)

invalid = (
    final_edges.geometry.isna()
    | final_edges.geometry.is_empty
)

zero_length = (
    final_edges.geometry.length <= 0
)

short_edges = (
    (final_edges["edge_type"] == "road")
    & (final_edges["length_m"] <= 0.5)
)

self_loops = (
    final_edges["from_node"]
    == final_edges["to_node"]
)

print("Invalid/empty edges:", int(invalid.sum()))
print("Zero-length edges:", int(zero_length.sum()))
print("Original road edges <= 0.5 m:", int(short_edges.sum()))
print("Self-loop edges:", int(self_loops.sum()))


# ============================================================
# REMOVE ONLY INVALID CONNECTORS
# ============================================================

final_edges = final_edges[
    ~invalid
    & ~zero_length
    & ~self_loops
].copy()

final_edges = final_edges.reset_index(drop=True)


# ============================================================
# REASSIGN EDGE IDS
# ============================================================

final_edges["edge_id"] = range(len(final_edges))


# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================

print("\nBuilding NetworkX graph...")

G = nx.Graph()

for node_id in nodes["node_id"]:
    G.add_node(int(node_id))

for _, row in final_edges.iterrows():

    G.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        edge_id=int(row["edge_id"]),
        length_m=float(row["length_m"])
    )


# ============================================================
# CONNECTIVITY
# ============================================================

components = list(nx.connected_components(G))

components.sort(
    key=len,
    reverse=True
)

largest_component = len(components[0])

coverage = (
    largest_component
    / len(nodes)
    * 100
)


# ============================================================
# DEGREE
# ============================================================

degrees = dict(G.degree())

degree_0 = sum(
    1 for d in degrees.values()
    if d == 0
)

degree_1 = sum(
    1 for d in degrees.values()
    if d == 1
)

degree_2 = sum(
    1 for d in degrees.values()
    if d == 2
)

degree_3_plus = sum(
    1 for d in degrees.values()
    if d >= 3
)

max_degree = max(degrees.values())


# ============================================================
# SAVE FINAL GRAPH
# ============================================================

print("\n" + "=" * 60)
print("SAVING FINAL GRAPH")
print("=" * 60)

nodes_out = nodes.to_crs("EPSG:4326")
edges_out = final_edges.to_crs("EPSG:4326")

nodes_out.to_file(
    FINAL_NODE_OUTPUT,
    driver="GeoJSON"
)

edges_out.to_file(
    FINAL_EDGE_OUTPUT,
    driver="GeoJSON"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL ROUTING GRAPH SUMMARY")
print("=" * 60)

print("Final nodes:", len(nodes_out))

print(
    "Final edges:",
    len(edges_out)
)

print(
    "Original clean edges:",
    len(original_edges)
)

print(
    "Accepted snap connections:",
    len(accepted)
)

print(
    "Snap connector edges added:",
    len(connectors)
)

print(
    "Total network length (km):",
    round(
        final_edges["length_m"].sum() / 1000,
        2
    )
)

print(
    "Average edge length (m):",
    round(
        final_edges["length_m"].mean(),
        2
    )
)

print(
    "Minimum edge length (m):",
    round(
        final_edges["length_m"].min(),
        4
    )
)

print(
    "Maximum edge length (m):",
    round(
        final_edges["length_m"].max(),
        2
    )
)


# ============================================================
# CONNECTIVITY SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY")
print("=" * 60)

print(
    "Connected components:",
    len(components)
)

print(
    "Largest component:",
    largest_component
)

print(
    "Largest component coverage:",
    round(coverage, 2),
    "%"
)

print(
    "Components with <= 10 nodes:",
    sum(
        1
        for c in components
        if len(c) <= 10
    )
)


# ============================================================
# DEGREE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NODE DEGREE")
print("=" * 60)

print("Degree 0:", degree_0)
print("Degree 1:", degree_1)
print("Degree 2:", degree_2)
print("Degree >= 3:", degree_3_plus)
print("Maximum degree:", max_degree)


# ============================================================
# OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print(FINAL_NODE_OUTPUT)
print(FINAL_EDGE_OUTPUT)
print(ACCEPTED_OUTPUT)
print(REJECTED_OUTPUT)

print("\n" + "=" * 60)
print("FINAL ROAD GRAPH CONSTRUCTION COMPLETE")
print("=" * 60)