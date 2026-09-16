import os
import geopandas as gpd
import networkx as nx
import pandas as pd


# ============================================================
# PATHS
# ============================================================

NODE_FILE = "data/processed/gcc/road_nodes_final.geojson"
EDGE_FILE = "data/processed/gcc/road_edges_final.geojson"

OUTPUT_DIR = "data/processed/gcc"
ROUTE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_route.geojson"
)


# ============================================================
# CONFIGURATION
# ============================================================

CRS = "EPSG:4326"

# Number of nodes to use from the largest component
TEST_ROUTE_NODE_COUNT = 2


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("RESQTWIN ROUTING ENGINE")
print("=" * 60)


# ============================================================
# LOAD GRAPH DATA
# ============================================================

print("\nLoading final road graph...")

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

print(f"Nodes loaded: {len(nodes)}")
print(f"Edges loaded: {len(edges)}")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("INPUT VALIDATION")
print("=" * 60)

required_node_columns = [
    "node_id"
]

required_edge_columns = [
    "edge_id",
    "from_node",
    "to_node",
    "length_m"
]


missing_nodes = [
    col for col in required_node_columns
    if col not in nodes.columns
]

missing_edges = [
    col for col in required_edge_columns
    if col not in edges.columns
]


if missing_nodes:
    raise ValueError(
        f"Missing node columns: {missing_nodes}"
    )

if missing_edges:
    raise ValueError(
        f"Missing edge columns: {missing_edges}"
    )


# Geometry checks

invalid_nodes = (
    nodes.geometry.isna()
    | nodes.geometry.is_empty
)

invalid_edges = (
    edges.geometry.isna()
    | edges.geometry.is_empty
    | ~edges.geometry.is_valid
)


print(
    "Invalid/empty nodes:",
    int(invalid_nodes.sum())
)

print(
    "Invalid/empty edges:",
    int(invalid_edges.sum())
)


if invalid_nodes.any():
    raise ValueError(
        "Invalid node geometries detected."
    )

if invalid_edges.any():
    raise ValueError(
        "Invalid edge geometries detected."
    )


# ============================================================
# NODE ID VALIDATION
# ============================================================

node_ids = set(
    nodes["node_id"].astype(int)
)

from_ids = set(
    edges["from_node"].astype(int)
)

to_ids = set(
    edges["to_node"].astype(int)
)

referenced_ids = from_ids | to_ids

invalid_references = (
    referenced_ids - node_ids
)


print(
    "Invalid node references:",
    len(invalid_references)
)


if invalid_references:
    raise ValueError(
        "Edges contain references to missing nodes."
    )


# ============================================================
# EDGE LENGTH VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("EDGE WEIGHT VALIDATION")
print("=" * 60)


missing_lengths = int(
    edges["length_m"].isna().sum()
)

non_positive_lengths = int(
    (edges["length_m"] <= 0).sum()
)


print(
    "Missing length values:",
    missing_lengths
)

print(
    "Non-positive lengths:",
    non_positive_lengths
)


if missing_lengths > 0:
    raise ValueError(
        "Missing edge lengths detected."
    )

if non_positive_lengths > 0:
    raise ValueError(
        "Non-positive edge lengths detected."
    )


# ============================================================
# BUILD NETWORKX GRAPH
# ============================================================

print("\n" + "=" * 60)
print("BUILDING ROUTING GRAPH")
print("=" * 60)

print("Creating NetworkX MultiGraph...")


G = nx.MultiGraph()


# ------------------------------------------------------------
# ADD NODES
# ------------------------------------------------------------

for node_id in nodes["node_id"]:

    G.add_node(
        int(node_id)
    )


# ------------------------------------------------------------
# ADD EDGES
# ------------------------------------------------------------

for _, row in edges.iterrows():

    from_node = int(row["from_node"])
    to_node = int(row["to_node"])

    edge_id = int(row["edge_id"])

    length_m = float(row["length_m"])

    G.add_edge(
        from_node,
        to_node,
        key=edge_id,
        edge_id=edge_id,
        length_m=length_m,
        weight=length_m
    )


print(
    "Graph nodes:",
    G.number_of_nodes()
)

print(
    "Graph edges:",
    G.number_of_edges()
)


# ============================================================
# CONNECTIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY")
print("=" * 60)


components = list(
    nx.connected_components(G)
)

components.sort(
    key=len,
    reverse=True
)


component_count = len(
    components
)

largest_component = components[0]


print(
    "Connected components:",
    component_count
)

print(
    "Largest component:",
    len(largest_component)
)

print(
    "Largest component coverage:",
    round(
        len(largest_component)
        / G.number_of_nodes()
        * 100,
        2
    ),
    "%"
)


# ============================================================
# SELECT TEST ROUTE NODES
# ============================================================

print("\n" + "=" * 60)
print("TEST ROUTE SELECTION")
print("=" * 60)


largest_nodes = sorted(
    largest_component
)


if len(largest_nodes) < 2:
    raise ValueError(
        "Largest connected component has fewer than 2 nodes."
    )


# ------------------------------------------------------------
# Deterministic node selection
# ------------------------------------------------------------
#
# We deliberately avoid random nodes.
#
# Pick nodes separated through the component ordering.
# This gives a reproducible routing test.
# ------------------------------------------------------------

source_index = 0

target_index = len(largest_nodes) - 1

source_node = largest_nodes[source_index]

target_node = largest_nodes[target_index]


print(
    "Source node:",
    source_node
)

print(
    "Target node:",
    target_node
)


if source_node == target_node:
    raise ValueError(
        "Source and target nodes are identical."
    )


# ============================================================
# SHORTEST PATH
# ============================================================

print("\n" + "=" * 60)
print("SHORTEST-PATH TEST")
print("=" * 60)

print(
    f"Routing {source_node} → {target_node}..."
)


try:

    route_nodes = nx.shortest_path(
        G,
        source=source_node,
        target=target_node,
        weight="weight"
    )

except nx.NetworkXNoPath:

    raise RuntimeError(
        "No route exists between selected nodes."
    )


# ============================================================
# CALCULATE ROUTE DISTANCE
# ============================================================

route_edges = []

route_distance = 0.0


for u, v in zip(
    route_nodes[:-1],
    route_nodes[1:]
):

    edge_data = G.get_edge_data(
        u,
        v
    )

    if edge_data is None:
        raise RuntimeError(
            f"Missing graph edge between {u} and {v}"
        )


    # MultiGraph may contain multiple edges.
    # Select the edge with minimum length.
    best_key = min(
        edge_data,
        key=lambda k:
        edge_data[k]["length_m"]
    )


    edge_info = edge_data[
        best_key
    ]

    edge_id = edge_info[
        "edge_id"
    ]

    length_m = edge_info[
        "length_m"
    ]


    route_edges.append(
        edge_id
    )

    route_distance += length_m


# ============================================================
# ROUTE SUMMARY
# ============================================================

print("\nRoute found: YES")

print(
    "Route nodes:",
    len(route_nodes)
)

print(
    "Route edges:",
    len(route_edges)
)

print(
    "Route distance:",
    round(
        route_distance,
        2
    ),
    "m"
)

print(
    "Route distance:",
    round(
        route_distance / 1000,
        3
    ),
    "km"
)


# ============================================================
# CREATE ROUTE GEOMETRY
# ============================================================

print("\n" + "=" * 60)
print("CREATING ROUTE GEOMETRY")
print("=" * 60)


route_edge_gdf = edges[
    edges["edge_id"].isin(
        route_edges
    )
].copy()


# Preserve route traversal order

edge_order = {
    edge_id: index
    for index, edge_id
    in enumerate(route_edges)
}


route_edge_gdf[
    "route_order"
] = route_edge_gdf[
    "edge_id"
].map(edge_order)


route_edge_gdf = route_edge_gdf.sort_values(
    "route_order"
)


# ============================================================
# ADD ROUTE METADATA
# ============================================================

route_edge_gdf[
    "route_source"
] = source_node

route_edge_gdf[
    "route_target"
] = target_node

route_edge_gdf[
    "route_distance_m"
] = route_distance


# ============================================================
# SAVE ROUTE
# ============================================================

print("\nSaving test route...")


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


route_edge_gdf.to_file(
    ROUTE_OUTPUT,
    driver="GeoJSON"
)


print(
    "Saved:",
    ROUTE_OUTPUT
)


# ============================================================
# FINAL ROUTING TEST
# ============================================================

print("\n" + "=" * 60)
print("ROUTING ENGINE TEST RESULT")
print("=" * 60)

print(
    "Graph loaded: PASS"
)

print(
    "Graph validation: PASS"
)

print(
    "Largest component identified: PASS"
)

print(
    "Source/target connectivity: PASS"
)

print(
    "Shortest path: PASS"
)

print(
    "Route geometry: PASS"
)

print(
    "Route export: PASS"
)

print("\n" + "=" * 60)

print(
    "ROUTING ENGINE READY: YES"
)

print("=" * 60)