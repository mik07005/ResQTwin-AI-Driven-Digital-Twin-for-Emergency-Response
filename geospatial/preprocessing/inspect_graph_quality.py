import geopandas as gpd
import pandas as pd


# ============================================================
# PATH
# ============================================================

EDGE_FILE = "data/processed/gcc/road_edges.geojson"
NODE_FILE = "data/processed/gcc/road_nodes.geojson"


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("ROAD GRAPH QUALITY INSPECTION")
print("=" * 60)

edges = gpd.read_file(EDGE_FILE)
nodes = gpd.read_file(NODE_FILE)

print("Edges:", len(edges))
print("Nodes:", len(nodes))
print("CRS:", edges.crs)


# ============================================================
# BASIC EDGE CHECKS
# ============================================================

print()
print("=" * 60)
print("EDGE CHECKS")
print("=" * 60)

print(
    "Missing from_node:",
    edges["from_node"].isna().sum()
)

print(
    "Missing to_node:",
    edges["to_node"].isna().sum()
)

print(
    "Missing length:",
    edges["length_m"].isna().sum()
)

print(
    "Zero-length edges:",
    (edges["length_m"] <= 0).sum()
)

print(
    "Negative-length edges:",
    (edges["length_m"] < 0).sum()
)

print(
    "Self-loop edges:",
    (
        edges["from_node"]
        == edges["to_node"]
    ).sum()
)


# ============================================================
# ZERO-LENGTH EDGES
# ============================================================

zero_edges = edges[
    edges["length_m"] <= 0
].copy()

print()
print("=" * 60)
print("ZERO-LENGTH EDGE DETAILS")
print("=" * 60)

if len(zero_edges) == 0:

    print("No zero-length edges found.")

else:

    print(
        "Number of zero-length edges:",
        len(zero_edges)
    )

    print(
        zero_edges[
            [
                "edge_id",
                "from_node",
                "to_node",
                "length_m"
            ]
        ].to_string(index=False)
    )


# ============================================================
# SELF-LOOP DETAILS
# ============================================================

self_loops = edges[
    edges["from_node"]
    == edges["to_node"]
].copy()

print()
print("=" * 60)
print("SELF-LOOP DETAILS")
print("=" * 60)

if len(self_loops) == 0:

    print("No self-loop edges found.")

else:

    print(
        "Number of self-loops:",
        len(self_loops)
    )

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


# ============================================================
# EDGE LENGTH STATISTICS
# ============================================================

print()
print("=" * 60)
print("EDGE LENGTH DISTRIBUTION")
print("=" * 60)

print(
    edges["length_m"].describe()
)


# ============================================================
# VERY SHORT EDGES
# ============================================================

for threshold in [0.01, 0.1, 0.5, 1, 2, 5, 10]:

    count = (
        edges["length_m"] <= threshold
    ).sum()

    print(
        f"Edges <= {threshold} m:",
        count
    )


# ============================================================
# NODE REFERENCE VALIDATION
# ============================================================

print()
print("=" * 60)
print("NODE REFERENCE CHECK")
print("=" * 60)

node_ids = set(
    nodes["node_id"]
)

from_ids = set(
    edges["from_node"]
)

to_ids = set(
    edges["to_node"]
)

invalid_from = from_ids - node_ids
invalid_to = to_ids - node_ids

print(
    "Invalid from_node references:",
    len(invalid_from)
)

print(
    "Invalid to_node references:",
    len(invalid_to)
)


# ============================================================
# NODE DEGREE
# ============================================================

print()
print("=" * 60)
print("NODE DEGREE CHECK")
print("=" * 60)

degree_from = (
    edges["from_node"]
    .value_counts()
)

degree_to = (
    edges["to_node"]
    .value_counts()
)

degree = (
    degree_from
    .add(degree_to, fill_value=0)
)

print(
    "Nodes appearing in graph:",
    len(degree)
)

print(
    "Degree 1 nodes:",
    (degree == 1).sum()
)

print(
    "Degree 2 nodes:",
    (degree == 2).sum()
)

print(
    "Degree >= 3 nodes:",
    (degree >= 3).sum()
)

print(
    "Maximum node degree:",
    int(degree.max())
)


# ============================================================
# ISOLATED NODES
# ============================================================

isolated_nodes = (
    set(nodes["node_id"])
    - set(degree.index)
)

print(
    "Isolated nodes:",
    len(isolated_nodes)
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 60)
print("GRAPH QUALITY INSPECTION COMPLETE")
print("=" * 60)