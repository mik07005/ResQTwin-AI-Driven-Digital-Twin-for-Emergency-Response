import geopandas as gpd
import networkx as nx


# ============================================================
# PATHS
# ============================================================

NODE_FILE = "data/processed/gcc/road_nodes.geojson"
EDGE_FILE = "data/processed/gcc/road_edges.geojson"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("ROAD GRAPH QUALITY ANALYSIS")
print("=" * 60)

nodes = gpd.read_file(NODE_FILE)
edges = gpd.read_file(EDGE_FILE)

print("Nodes:", len(nodes))
print("Edges:", len(edges))
print("CRS:", edges.crs)


# ============================================================
# BUILD GRAPH
# ============================================================

print("\nBuilding NetworkX graph...")

G = nx.Graph()

# Add nodes
for _, row in nodes.iterrows():
    G.add_node(
        int(row["node_id"]),
        geometry=row.geometry
    )

# Add edges
for _, row in edges.iterrows():

    G.add_edge(
        int(row["from_node"]),
        int(row["to_node"]),
        edge_id=int(row["edge_id"]),
        length_m=float(row["length_m"])
    )


# ============================================================
# BASIC GRAPH STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

print("\n" + "=" * 60)
print("CONNECTIVITY")
print("=" * 60)

components = list(nx.connected_components(G))

print("Connected components:", len(components))

component_sizes = sorted(
    [len(component) for component in components],
    reverse=True
)

print("Largest component:", component_sizes[0])

if len(component_sizes) > 1:
    print("Second largest component:", component_sizes[1])

print(
    "Nodes in largest component:",
    component_sizes[0]
)

largest_component_percentage = (
    component_sizes[0] /
    G.number_of_nodes()
) * 100

print(
    "Largest component coverage:",
    round(largest_component_percentage, 2),
    "%"
)


# ============================================================
# SMALL DISCONNECTED COMPONENTS
# ============================================================

small_components = [
    size for size in component_sizes
    if size <= 10
]

print(
    "Components with <= 10 nodes:",
    len(small_components)
)


# ============================================================
# NODE DEGREE
# ============================================================

print("\n" + "=" * 60)
print("NODE DEGREE")
print("=" * 60)

degrees = dict(G.degree())

degree_0 = sum(
    1 for degree in degrees.values()
    if degree == 0
)

degree_1 = sum(
    1 for degree in degrees.values()
    if degree == 1
)

degree_2 = sum(
    1 for degree in degrees.values()
    if degree == 2
)

degree_3_plus = sum(
    1 for degree in degrees.values()
    if degree >= 3
)

print("Degree 0:", degree_0)
print("Degree 1:", degree_1)
print("Degree 2:", degree_2)
print("Degree >= 3:", degree_3_plus)
print("Maximum degree:", max(degrees.values()))


# ============================================================
# ISOLATED NODES
# ============================================================

print("\n" + "=" * 60)
print("ISOLATED NODES")
print("=" * 60)

isolated = list(nx.isolates(G))

print("Isolated nodes:", len(isolated))


# ============================================================
# SELF LOOPS
# ============================================================

print("\n" + "=" * 60)
print("SELF-LOOPS")
print("=" * 60)

self_loops = list(nx.selfloop_edges(G))

print("Self-loop edges:", len(self_loops))

if self_loops:
    print("\nSelf-loop node IDs:")
    print([u for u, v in self_loops[:20]])


# ============================================================
# EDGE LENGTH STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("EDGE LENGTH")
print("=" * 60)

lengths = edges["length_m"]

print("Minimum:", round(lengths.min(), 6), "m")
print("Maximum:", round(lengths.max(), 2), "m")
print("Average:", round(lengths.mean(), 2), "m")
print("Median:", round(lengths.median(), 2), "m")

print(
    "Edges <= 1 m:",
    int((lengths <= 1).sum())
)

print(
    "Edges <= 5 m:",
    int((lengths <= 5).sum())
)

print(
    "Edges <= 10 m:",
    int((lengths <= 10).sum())
)


# ============================================================
# TOTAL NETWORK LENGTH
# ============================================================

total_length_km = lengths.sum() / 1000

print("\nTotal network length:")
print(round(total_length_km, 2), "km")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("GRAPH QUALITY ANALYSIS COMPLETE")
print("=" * 60)