import geopandas as gpd
import networkx as nx
from collections import Counter


# ============================================================
# PATHS
# ============================================================

NODE_FILE = "data/processed/gcc/road_nodes.geojson"
EDGE_FILE = "data/processed/gcc/road_edges.geojson"


# ============================================================
# LOAD
# ============================================================

print("=" * 60)
print("ROAD GRAPH STRUCTURAL ANALYSIS")
print("=" * 60)

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

print("Nodes:", len(nodes))
print("Edges:", len(edges))
print("CRS:", edges.crs)


# ============================================================
# BUILD MULTIGRAPH
# ============================================================

print("\nBuilding NetworkX MultiGraph...")

G = nx.MultiGraph()

for _, row in nodes.iterrows():

    G.add_node(
        int(row["node_id"]),
        geometry=row.geometry
    )


for _, row in edges.iterrows():

    G.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        key=int(row["edge_id"]),
        edge_id=int(row["edge_id"]),
        length_m=float(row["length_m"])
    )


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())


# ============================================================
# VERIFY EDGE COUNT
# ============================================================

print("\n" + "=" * 60)
print("EDGE COUNT VALIDATION")
print("=" * 60)

if G.number_of_edges() == len(edges):
    print("All GeoJSON edges preserved: YES")
else:
    print("All GeoJSON edges preserved: NO")

    print(
        "Difference:",
        len(edges) - G.number_of_edges()
    )


# ============================================================
# PARALLEL EDGES
# ============================================================

print("\n" + "=" * 60)
print("PARALLEL EDGE ANALYSIS")
print("=" * 60)

pair_counts = Counter()

for _, row in edges.iterrows():

    u = int(row["from_node"])
    v = int(row["to_node"])

    # Treat u-v and v-u as the same pair
    pair = tuple(sorted((u, v)))

    pair_counts[pair] += 1


parallel_pairs = {
    pair: count
    for pair, count in pair_counts.items()
    if count > 1
}

parallel_edge_count = sum(
    count - 1
    for count in parallel_pairs.values()
)

print("Unique node pairs:", len(pair_counts))
print("Node pairs with parallel edges:", len(parallel_pairs))
print("Extra parallel edges:", parallel_edge_count)


if parallel_pairs:

    print("\nTop parallel node pairs:")

    for pair, count in sorted(
        parallel_pairs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:

        print(
            f"Nodes {pair[0]} - {pair[1]} : "
            f"{count} edges"
        )


# ============================================================
# SELF LOOPS
# ============================================================

print("\n" + "=" * 60)
print("SELF-LOOPS")
print("=" * 60)

self_loops = []

for u, v, key, data in G.edges(
    keys=True,
    data=True
):

    if u == v:

        self_loops.append(
            {
                "edge_id": data["edge_id"],
                "node_id": u,
                "length_m": data["length_m"]
            }
        )


print("Self-loop edges:", len(self_loops))

if self_loops:

    print("\nSelf-loop details:")

    for item in self_loops:

        print(
            f"edge={item['edge_id']} "
            f"node={item['node_id']} "
            f"length={item['length_m']:.3f} m"
        )


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY")
print("=" * 60)

components = list(
    nx.connected_components(G)
)

component_sizes = sorted(
    [len(c) for c in components],
    reverse=True
)

print(
    "Connected components:",
    len(component_sizes)
)

print(
    "Largest component:",
    component_sizes[0]
)

print(
    "Largest component coverage:",
    round(
        component_sizes[0]
        / G.number_of_nodes()
        * 100,
        2
    ),
    "%"
)

print(
    "Components with <= 10 nodes:",
    sum(
        1
        for size in component_sizes
        if size <= 10
    )
)


# ============================================================
# COMPONENT SIZE DISTRIBUTION
# ============================================================

print("\nTop 15 component sizes:")

for i, size in enumerate(
    component_sizes[:15],
    start=1
):

    print(
        f"{i:2d}. {size} nodes"
    )


# ============================================================
# NODE DEGREE
# ============================================================

print("\n" + "=" * 60)
print("NODE DEGREE")
print("=" * 60)

degree_values = [
    degree
    for _, degree in G.degree()
]

print(
    "Degree 1:",
    sum(d == 1 for d in degree_values)
)

print(
    "Degree 2:",
    sum(d == 2 for d in degree_values)
)

print(
    "Degree >= 3:",
    sum(d >= 3 for d in degree_values)
)

print(
    "Maximum degree:",
    max(degree_values)
)


# ============================================================
# EDGE LENGTH
# ============================================================

print("\n" + "=" * 60)
print("EDGE LENGTH")
print("=" * 60)

lengths = edges["length_m"]

print(
    "Minimum:",
    lengths.min(),
    "m"
)

print(
    "Median:",
    round(lengths.median(), 2),
    "m"
)

print(
    "Mean:",
    round(lengths.mean(), 2),
    "m"
)

print(
    "Maximum:",
    round(lengths.max(), 2),
    "m"
)

for threshold in [0.01, 0.1, 0.5, 1, 2, 5, 10]:

    print(
        f"Edges <= {threshold} m:",
        int((lengths <= threshold).sum())
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)
print("STRUCTURAL ANALYSIS COMPLETE")
print("=" * 60)