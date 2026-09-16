import geopandas as gpd
import pandas as pd
import networkx as nx


# ============================================================
# PATHS
# ============================================================

EDGE_FILE = "data/processed/gcc/road_edges.geojson"
NODE_FILE = "data/processed/gcc/road_nodes.geojson"

CLEAN_EDGE_OUTPUT = "data/processed/gcc/road_edges_clean.geojson"
CLEAN_NODE_OUTPUT = "data/processed/gcc/road_nodes_clean.geojson"

REMOVED_EDGE_OUTPUT = "data/processed/gcc/removed_graph_edges.geojson"


# ============================================================
# PARAMETERS
# ============================================================

# Only remove edges that are effectively geometric artifacts.
SHORT_EDGE_THRESHOLD_M = 0.5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CLEANING GCC ROAD GRAPH")
print("=" * 60)

edges = gpd.read_file(EDGE_FILE)
nodes = gpd.read_file(NODE_FILE)

print("Input edges:", len(edges))
print("Input nodes:", len(nodes))
print("CRS:", edges.crs)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_edge_columns = [
    "edge_id",
    "from_node",
    "to_node",
    "length_m",
    "geometry"
]

required_node_columns = [
    "node_id",
    "geometry"
]

for col in required_edge_columns:
    if col not in edges.columns:
        raise ValueError(f"Missing edge column: {col}")

for col in required_node_columns:
    if col not in nodes.columns:
        raise ValueError(f"Missing node column: {col}")


# ============================================================
# REMOVE INVALID GEOMETRIES
# ============================================================

print("\n" + "=" * 60)
print("GEOMETRY VALIDATION")
print("=" * 60)

invalid_geometry = (
    edges.geometry.isna()
    | edges.geometry.is_empty
)

print("Invalid/empty geometries:", invalid_geometry.sum())

removed_invalid = edges[invalid_geometry].copy()

edges = edges[~invalid_geometry].copy()


# ============================================================
# SELF-LOOPS
# ============================================================

print("\n" + "=" * 60)
print("SELF-LOOP CLEANING")
print("=" * 60)

self_loop_mask = (
    edges["from_node"]
    == edges["to_node"]
)

self_loops = edges[self_loop_mask].copy()

print("Self-loop edges found:", len(self_loops))

if len(self_loops) > 0:
    print(
        self_loops[
            [
                "edge_id",
                "from_node",
                "to_node",
                "length_m"
            ]
        ].to_string(index=False)
    )

edges = edges[~self_loop_mask].copy()


# ============================================================
# EXTREMELY SHORT EDGES
# ============================================================

print("\n" + "=" * 60)
print("SHORT EDGE CLEANING")
print("=" * 60)

short_edge_mask = (
    edges["length_m"]
    <= SHORT_EDGE_THRESHOLD_M
)

short_edges = edges[short_edge_mask].copy()

print(
    f"Edges <= {SHORT_EDGE_THRESHOLD_M} m:",
    len(short_edges)
)

edges = edges[~short_edge_mask].copy()


# ============================================================
# NODE REFERENCE VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("NODE REFERENCE VALIDATION")
print("=" * 60)

node_ids = set(nodes["node_id"])

invalid_from = ~edges["from_node"].isin(node_ids)
invalid_to = ~edges["to_node"].isin(node_ids)

print("Invalid from_node references:", invalid_from.sum())
print("Invalid to_node references:", invalid_to.sum())

invalid_reference_mask = (
    invalid_from
    | invalid_to
)

invalid_reference_edges = edges[
    invalid_reference_mask
].copy()

edges = edges[
    ~invalid_reference_mask
].copy()


# ============================================================
# BUILD GRAPH FOR CONNECTIVITY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY ANALYSIS")
print("=" * 60)

G = nx.MultiGraph()

for node_id in nodes["node_id"]:
    G.add_node(int(node_id))

for _, row in edges.iterrows():

    G.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        edge_id=int(row["edge_id"]),
        length_m=float(row["length_m"])
    )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

components = list(nx.connected_components(G))

components = sorted(
    components,
    key=len,
    reverse=True
)

print("Connected components:", len(components))

if components:

    largest_component = components[0]

    print(
        "Largest component:",
        len(largest_component)
    )

    print(
        "Largest component coverage:",
        round(
            len(largest_component)
            / len(nodes)
            * 100,
            2
        ),
        "%"
    )

else:

    largest_component = set()


# ============================================================
# COMPONENT SIZE DISTRIBUTION
# ============================================================

small_components = [
    component
    for component in components
    if len(component) <= 10
]

print(
    "Components with <= 10 nodes:",
    len(small_components)
)


# ============================================================
# KEEP ALL COMPONENTS FOR NOW
# ============================================================

print("\nKeeping all connected components.")

print(
    "Reason: small components may represent legitimate "
    "isolated road structures and will be investigated "
    "separately rather than blindly deleted."
)


# ============================================================
# REMOVE UNUSED NODES
# ============================================================

used_nodes = set(
    edges["from_node"]
).union(
    set(edges["to_node"])
)

clean_nodes = nodes[
    nodes["node_id"].isin(used_nodes)
].copy()


# ============================================================
# RESET INDEX
# ============================================================

edges = edges.reset_index(drop=True)
clean_nodes = clean_nodes.reset_index(drop=True)


# ============================================================
# SAVE REMOVED EDGES
# ============================================================

removed_edges = pd.concat(
    [
        removed_invalid,
        self_loops,
        short_edges,
        invalid_reference_edges
    ],
    ignore_index=True
)

if len(removed_edges) > 0:

    removed_edges = gpd.GeoDataFrame(
        removed_edges,
        geometry="geometry",
        crs=edges.crs
    )

    removed_edges.to_file(
        REMOVED_EDGE_OUTPUT,
        driver="GeoJSON"
    )


# ============================================================
# SAVE CLEAN GRAPH
# ============================================================

print("\n" + "=" * 60)
print("SAVING CLEAN GRAPH")
print("=" * 60)

clean_nodes.to_file(
    CLEAN_NODE_OUTPUT,
    driver="GeoJSON"
)

edges.to_file(
    CLEAN_EDGE_OUTPUT,
    driver="GeoJSON"
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("CLEAN GRAPH SUMMARY")
print("=" * 60)

print("Original edges:", len(edges) + len(removed_edges))
print("Clean edges:", len(edges))

print("Original nodes:", len(nodes))
print("Nodes retained:", len(clean_nodes))

print(
    "Invalid geometry removed:",
    len(removed_invalid)
)

print(
    "Self-loops removed:",
    len(self_loops)
)

print(
    "Short edges removed:",
    len(short_edges)
)

print(
    "Invalid node-reference edges removed:",
    len(invalid_reference_edges)
)

print(
    "Total removed edges:",
    len(removed_edges)
)

if len(edges) > 0:

    print(
        "Clean network length (km):",
        round(
            edges["length_m"].sum() / 1000,
            2
        )
    )

    print(
        "Average edge length (m):",
        round(
            edges["length_m"].mean(),
            2
        )
    )

    print(
        "Minimum edge length (m):",
        round(
            edges["length_m"].min(),
            4
        )
    )

    print(
        "Maximum edge length (m):",
        round(
            edges["length_m"].max(),
            2
        )
    )

print("\nSaved:")
print(CLEAN_EDGE_OUTPUT)
print(CLEAN_NODE_OUTPUT)
print(REMOVED_EDGE_OUTPUT)

print("\nGraph cleaning complete.")